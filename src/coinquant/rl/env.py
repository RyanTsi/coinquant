"""Gymnasium trading environment backed by the independent execution core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

try:  # Keep observation/action/reward usable without optional RL packages.
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised only in minimal installs
    class _FallbackEnv:
        metadata: dict[str, Any] = {}

    class _FallbackBox:
        def __init__(self, low: Any, high: Any, shape: tuple[int, ...] | None = None, dtype: Any = np.float32):
            if shape is None:
                shape = np.asarray(low).shape
            self.low = np.broadcast_to(np.asarray(low, dtype=dtype), shape).copy()
            self.high = np.broadcast_to(np.asarray(high, dtype=dtype), shape).copy()
            self.shape = tuple(shape)
            self.dtype = dtype

        def sample(self) -> np.ndarray:
            return np.random.uniform(self.low, self.high).astype(self.dtype)

        def contains(self, value: Any) -> bool:
            values = np.asarray(value)
            return values.shape == self.shape and np.isfinite(values).all() and (values >= self.low).all() and (values <= self.high).all()

    class _FallbackSpaces:
        Box = _FallbackBox

    class _FallbackGym:
        Env = _FallbackEnv

    gym = _FallbackGym()
    spaces = _FallbackSpaces()

from coinquant.backtest.account import Account, AccountConfig
from coinquant.backtest.execution import ExecutionConfig, ExecutionEngine, ExecutionResult
from coinquant.backtest.position import Position
from coinquant.rl.action import ActionAdapter, ActionConfig
from coinquant.rl.observation import ObservationBuilder, ObservationConfig
from coinquant.rl.reward import RewardCalculator, RewardConfig


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class EnvConfig:
    account_config: AccountConfig = AccountConfig()
    execution_config: ExecutionConfig = ExecutionConfig()
    initial_equity: float | None = None
    force_close_at_end: bool = True
    max_episode_steps: int | None = None

    def __post_init__(self) -> None:
        if self.initial_equity is not None:
            initial_equity = _finite(self.initial_equity, "initial_equity")
            if initial_equity <= 0:
                raise ValueError("initial_equity must be greater than 0")
            if not np.isclose(initial_equity, self.account_config.initial_balance):
                raise ValueError("initial_equity must equal account_config.initial_balance")
            object.__setattr__(self, "initial_equity", initial_equity)
        if not isinstance(self.force_close_at_end, bool):
            raise TypeError("force_close_at_end must be a bool")
        if self.max_episode_steps is not None:
            if int(self.max_episode_steps) != self.max_episode_steps or self.max_episode_steps <= 0:
                raise ValueError("max_episode_steps must be a positive integer")
            object.__setattr__(self, "max_episode_steps", int(self.max_episode_steps))

    @property
    def starting_equity(self) -> float:
        return self.account_config.initial_balance


class TradingEnv(gym.Env):
    """Continuous target-exposure environment with T-close/T+1-open timing."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        frame: Any,
        observation_config: ObservationConfig | None = None,
        action_config: ActionConfig | None = None,
        reward_config: RewardConfig | None = None,
        env_config: EnvConfig | None = None,
        observation_normalizer: Any | None = None,
    ):
        if pd is None or not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        self.config = env_config or EnvConfig()
        self.observation_config = observation_config or ObservationConfig()
        self.action_adapter = ActionAdapter(action_config or ActionConfig())
        if self.action_adapter.config.max_leverage > self.config.account_config.leverage:
            raise ValueError("action max_leverage cannot exceed account leverage")
        self.reward_calculator = RewardCalculator(reward_config or RewardConfig())
        self.observation_builder = ObservationBuilder(
            frame,
            self.observation_config,
            observation_normalizer,
        )
        required = {"open_time", "open", "high", "low", "close", "volume"}
        missing = sorted(required - set(self.observation_builder.frame.columns))
        if missing:
            raise ValueError(f"environment frame missing columns: {missing}")
        self.frame = self.observation_builder.frame
        self._validate_frame()
        self._valid_indices = self.observation_builder.valid_indices
        if len(self._valid_indices) < 2:
            raise ValueError("frame must contain at least two actionable observation rows")

        self.action_space = spaces.Box(
            low=np.asarray([-self.action_adapter.config.max_leverage], dtype=np.float32),
            high=np.asarray([self.action_adapter.config.max_leverage], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-self.observation_config.clip_value,
            high=self.observation_config.clip_value,
            shape=(self.observation_builder.observation_size,),
            dtype=np.float32,
        )
        self.account: Account
        self.position: Position
        self.execution: ExecutionEngine
        self._tick = int(self._valid_indices[0])
        self._step_count = 0
        self._peak_equity = self.config.starting_equity
        self._total_reward = 0.0
        self._history: list[dict[str, Any]] = []
        self._has_reset = False

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if hasattr(super(), "reset"):
            try:
                super().reset(seed=seed)
            except (AttributeError, TypeError):
                pass
        self.account = Account(self.config.account_config)
        self.position = Position()
        self.execution = ExecutionEngine(self.config.execution_config)
        self.reward_calculator.reset()
        self._tick = int(self._valid_indices[0])
        self._step_count = 0
        self._peak_equity = self.config.starting_equity
        self._total_reward = 0.0
        self._history = []
        self._has_reset = True
        first_bar = self._bar(self._tick)
        self.execution.mark_to_market(first_bar.close, self.position, self.account, first_bar.timestamp, "RESET")
        timestamp = self.frame.iloc[self._tick]["open_time"]
        return self._observation(), self._get_info(
            None,
            None,
            0.0,
            0.0,
            0.0,
            False,
            decision_time=timestamp,
            entry_time=None,
            exit_time=None,
        )

    def step(self, action: Any):
        if not self._has_reset:
            raise RuntimeError("reset() must be called before step()")
        if self._tick >= int(self._valid_indices[-1]):
            raise RuntimeError("episode is finished; call reset() before step()")

        target_exposure, clipped = self.action_adapter.coerce(action)
        previous_equity = self.account.equity
        previous_exposure = self._current_exposure()
        next_tick = self._tick + 1
        if next_tick > int(self._valid_indices[-1]):
            raise RuntimeError("no next bar is available for this action")
        bar = self._bar(next_tick)
        events: list[ExecutionResult] = []

        opening = self.execution.mark_to_market(
            bar.open, self.position, self.account, bar.timestamp, "OPEN"
        )
        events.append(opening)
        if opening.liquidated:
            events.append(self.execution.force_liquidation(bar.open, self.position, self.account, bar.timestamp))

        funding_rate = self._funding_rate(bar)
        if not self.account.is_liquidated and funding_rate is not None:
            funding = self.execution.settle_funding(
                funding_rate, bar.open, self.position, self.account, bar.timestamp
            )
            events.append(funding)
            if funding.liquidated:
                events.append(self.execution.force_liquidation(bar.open, self.position, self.account, bar.timestamp))

        if not self.account.is_liquidated:
            quantity = self.action_adapter.target_quantity(
                target_exposure,
                self.account.equity,
                bar.open,
                self.execution.cfg.contract_size,
            )
            events.append(
                self.execution.execute_target(
                    quantity,
                    bar.open,
                    bar.open,
                    self.position,
                    self.account,
                    bar.timestamp,
                    "RL_ACTION",
                )
            )

        if not self.account.is_liquidated and self.position.position != 0:
            adverse = bar.low if self.position.position > 0 else bar.high
            intrabar = self.execution.mark_to_market(
                adverse, self.position, self.account, bar.timestamp, "INTRABAR_RISK"
            )
            events.append(intrabar)
            if intrabar.liquidated:
                events.append(self.execution.force_liquidation(adverse, self.position, self.account, bar.timestamp))

        if not self.account.is_liquidated:
            closing = self.execution.mark_to_market(
                bar.close, self.position, self.account, bar.timestamp, "CLOSE"
            )
            events.append(closing)
            if closing.liquidated:
                events.append(self.execution.force_liquidation(bar.close, self.position, self.account, bar.timestamp))

        self.execution.finish_bar(self.position)
        self._tick = next_tick
        self._step_count += 1
        self._peak_equity = max(self._peak_equity, self.account.equity)
        drawdown = self._drawdown()

        force_closed = False
        if self._tick == int(self._valid_indices[-1]) and self.config.force_close_at_end and not self.account.is_liquidated and self.position.position != 0:
            end_event = self.execution.execute_target(
                0.0, bar.close, bar.close, self.position, self.account, bar.timestamp, "END_OF_EPISODE"
            )
            events.append(end_event)
            force_closed = True
            self._peak_equity = max(self._peak_equity, self.account.equity)
            drawdown = self._drawdown()

        # Keep the action exposure as the reward exposure.  It is the only
        # exposure known when the decision is made; the post-close exposure can
        # be zero after an episode-end close or liquidation.
        exposure_during_bar = target_exposure
        actual_exposure = self._current_exposure()
        market_return = bar.close / bar.open - 1.0
        breakdown = self.reward_calculator.calculate(
            previous_equity,
            self.account.equity,
            exposure_during_bar,
            market_return,
            drawdown,
            self.account.is_liquidated,
        )
        self._total_reward += breakdown.reward
        terminated = self.account.is_liquidated or self._tick == int(self._valid_indices[-1])
        truncated = self.config.max_episode_steps is not None and self._step_count >= self.config.max_episode_steps and not terminated
        info = self._get_info(
            events,
            target_exposure,
            breakdown.reward,
            breakdown.gross_return,
            self._turnover(events, previous_equity),
            self.account.is_liquidated,
            decision_time=self.frame.iloc[self._tick - 1]["open_time"],
            entry_time=bar.timestamp,
            exit_time=bar.timestamp,
        )
        info.update(
            {
                "action_was_clipped": clipped,
                "previous_exposure": previous_exposure,
                "actual_exposure": actual_exposure,
                "exposure_during_bar": exposure_during_bar,
                "force_closed_at_end": force_closed,
                "end_of_episode_close": force_closed,
                **breakdown.as_dict(),
            }
        )
        self._history.append(dict(info))
        return self._observation(), float(breakdown.reward), bool(terminated), bool(truncated), info

    def render(self) -> None:
        if self._history:
            print(self._history[-1])

    def close(self) -> None:
        self._has_reset = False

    @property
    def history(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self._history)

    def _observation(self) -> np.ndarray:
        returns = self.reward_calculator.returns
        rolling_volatility = float(np.std(returns, ddof=1)) if len(returns) >= 2 else 0.0
        account_features = {
            "current_exposure": self._current_exposure(),
            "equity_ratio": self.account.equity / self.config.starting_equity,
            "drawdown": self._drawdown(),
            "rolling_volatility": rolling_volatility,
        }
        return self.observation_builder.build(self._tick, account_features)

    def _drawdown(self) -> float:
        if self._peak_equity <= 0:
            return 1.0
        return float(np.clip(1.0 - self.account.equity / self._peak_equity, 0.0, 1.0))

    def _current_exposure(self) -> float:
        return self.action_adapter.exposure_from_position(
            self.position.position,
            self.position.mark_price if self.position.mark_price > 0 else float(self.frame.iloc[self._tick]["close"]),
            self.account.equity,
            self.execution.cfg.contract_size,
        )

    def _bar(self, tick: int):
        from coinquant.backtest.simulation import MarketBar

        row = self.frame.iloc[int(tick)]
        return MarketBar(
            timestamp=row["open_time"],
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            funding_rate=self._funding_rate_from_row(row),
        )

    def _funding_rate(self, bar: Any) -> float | None:
        return None if bar.funding_rate is None else float(bar.funding_rate)

    def _funding_rate_from_row(self, row: Any) -> float | None:
        if "funding_rate" not in self.frame.columns or pd.isna(row.get("funding_rate")):
            return None
        return float(row["funding_rate"])

    def _validate_frame(self) -> None:
        """Validate the immutable market-data contract at construction time."""

        timestamps = self.frame["open_time"]
        if timestamps.isna().any() or not timestamps.is_unique or not timestamps.is_monotonic_increasing:
            raise ValueError("open_time must be non-null, unique and strictly increasing")
        numeric = self.frame.loc[:, ["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError("OHLCV values must be finite")
        if (numeric[:, :4] <= 0).any():
            raise ValueError("OHLC prices must be positive")
        if (numeric[:, 4] < 0).any():
            raise ValueError("volume must be non-negative")
        opens = numeric[:, 0]
        highs = numeric[:, 1]
        lows = numeric[:, 2]
        closes = numeric[:, 3]
        if (lows > np.minimum(opens, closes)).any() or (highs < np.maximum(opens, closes)).any() or (lows > highs).any():
            raise ValueError("OHLC prices must satisfy low <= min(open, close) <= max(open, close) <= high")
        if "funding_rate" in self.frame.columns:
            funding = self.frame["funding_rate"].dropna().to_numpy(dtype=np.float64)
            if not np.isfinite(funding).all():
                raise ValueError("funding_rate values must be finite")

    def _turnover(self, events: list[ExecutionResult], equity: float) -> float:
        if equity <= 0:
            return 0.0
        return float(sum(event.trade_notional for event in events) / equity)

    def _get_info(
        self,
        events: list[ExecutionResult] | None,
        target_exposure: float | None,
        reward: float,
        gross_return: float,
        turnover: float,
        liquidated: bool,
        *,
        decision_time: Any | None = None,
        entry_time: Any | None = None,
        exit_time: Any | None = None,
    ) -> dict[str, Any]:
        event_types = [] if events is None else [event.event_type for event in events]
        trade_events = [
            event for event in events or [] if event.quantity > self.execution.cfg.quantity_epsilon
        ]
        execution_event_type = (
            trade_events[-1].event_type if trade_events else (event_types[-1] if event_types else None)
        )
        timestamp = self.frame.iloc[self._tick]["open_time"]
        return {
            "decision_time": timestamp if decision_time is None else decision_time,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "action": target_exposure,
            "target_exposure": target_exposure,
            "actual_position": self.position.position,
            "gross_return": gross_return,
            "net_return": 0.0,
            "turnover": turnover,
            "fee_cost": float(sum(event.fee for event in events or [])),
            "slippage_cost": float(sum(self._slippage_cost(event) for event in events or [])),
            "funding_payment": float(sum(event.funding_payment for event in events or [])),
            "reward": reward,
            "equity": self.account.equity,
            "peak_equity": self._peak_equity,
            "drawdown": self._drawdown(),
            "liquidated": liquidated,
            "execution_event_type": execution_event_type,
        }

    def _slippage_cost(self, event: ExecutionResult) -> float:
        if event.fill_price is None or event.reference_price is None:
            return 0.0
        return abs(event.fill_price - event.reference_price) * event.quantity * self.execution.cfg.contract_size
