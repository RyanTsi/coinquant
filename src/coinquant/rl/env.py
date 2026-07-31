from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class TradingCosts:
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0002

class Account:
    pass

class TradingEnv(gym.Env):
    """Continuous contract trading environment.

    Observation = [
        Market Features,
        DL Predictions,
        Position Features
    ]
    Action = target position in [-max_leverage, max_leverage].
    Reward = return after fees, slippage, and risk penalties.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 10,
        *,
        feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
        price_column: str = "open",
        time_column: str = "open_time",
        max_leverage: float = 1.0,
        fee_rate: float = 0.0004,
        slippage_rate: float = 0.0002,
        drawdown_penalty: float = 0.0,
        volatility_penalty: float = 0.0,
        position_penalty: float = 0.0,
        reward_scale: float = 1.0,
        reward_mode: str = "log",
        initial_equity: float = 1.0,
        risk_window: int | None = None,
    ):
        super().__init__()

        if window_size <= 0:
            raise ValueError("window_size must be greater than 0")
        if max_leverage <= 0:
            raise ValueError("max_leverage must be greater than 0")
        if fee_rate < 0 or slippage_rate < 0:
            raise ValueError("fee_rate and slippage_rate must be greater than or equal to 0")
        if drawdown_penalty < 0 or volatility_penalty < 0 or position_penalty < 0:
            raise ValueError("risk penalties must be greater than or equal to 0")
        if reward_scale <= 0:
            raise ValueError("reward_scale must be greater than 0")
        if initial_equity <= 0:
            raise ValueError("initial_equity must be greater than 0")
        if reward_mode not in {"log", "simple"}:
            raise ValueError("reward_mode must be 'log' or 'simple'")

        self.window_size = int(window_size)
        self.feature_columns = tuple(feature_columns)
        self.price_column = price_column
        self.time_column = time_column
        self.df = self._prepare_frame(df)
        self.max_leverage = float(max_leverage)
        self.costs = TradingCosts(fee_rate=float(fee_rate), slippage_rate=float(slippage_rate))
        self.drawdown_penalty = float(drawdown_penalty)
        self.volatility_penalty = float(volatility_penalty)
        self.position_penalty = float(position_penalty)
        self.reward_scale = float(reward_scale)
        self.reward_mode = reward_mode
        self.initial_equity = float(initial_equity)
        self.risk_window = int(window_size if risk_window is None else risk_window)
        if self.risk_window <= 0:
            raise ValueError("risk_window must be greater than 0")

        missing_columns = sorted(set(self.feature_columns + (self.price_column,)) - set(self.df.columns))
        if missing_columns:
            raise ValueError(f"missing columns: {missing_columns}")
        if len(self.df) < self.window_size + 2:
            raise ValueError("df must contain at least window_size + 2 rows")

        self._prices = self.df[self.price_column].to_numpy(dtype=np.float64)
        self._observations = self.df.loc[:, self.feature_columns].to_numpy(dtype=np.float32)
        if not np.isfinite(self._prices).all() or np.any(self._prices <= 0):
            raise ValueError(f"{self.price_column} must contain finite positive values")
        if not np.isfinite(self._observations).all():
            raise ValueError("observation columns must contain finite values")
        self._open_times = (
            self.df[self.time_column].to_numpy() if self.time_column in self.df.columns else None
        )
        self._future_returns = self._build_future_returns(self._prices)
        self._past_volatility = self._build_past_volatility(self._future_returns, self.risk_window)

        self._start_tick = self.window_size - 1
        self._last_decision_tick = len(self.df) - 3

        self.action_space = gym.spaces.Box(
            low=-self.max_leverage,
            high=self.max_leverage,
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, len(self.feature_columns)),
            dtype=np.float32,
        )

        self._current_tick: int | None = None
        self._position: float = 0.0
        self._equity: float = self.initial_equity
        self._peak_equity: float = self.initial_equity
        self._total_reward: float = 0.0
        self._terminated: bool = False
        self._truncated: bool = False
        self._history: list[dict[str, Any]] = []

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)
        self._current_tick = self._start_tick
        self._position = 0.0
        self._equity = self.initial_equity
        self._peak_equity = self.initial_equity
        self._total_reward = 0.0
        self._terminated = False
        self._truncated = False
        self._history = []
        observation = self._get_observation()
        info = self._get_info(
            decision_tick=self._current_tick,
            entry_tick=self._current_tick + 1,
            exit_tick=self._current_tick + 2,
            reward=0.0,
            base_reward=0.0,
            risk_penalty=0.0,
            fee_cost=0.0,
            slippage_cost=0.0,
            gross_return=0.0,
            net_return=0.0,
            turnover=0.0,
            market_return=0.0,
            drawdown=0.0,
            target_position=self._position,
            previous_position=self._position,
        )
        return observation, info

    def step(self, action):
        if self._current_tick is None:
            raise RuntimeError("reset() must be called before step()")
        if self._terminated or self._truncated:
            raise RuntimeError("episode is finished; call reset() before step()")
        if self._current_tick > self._last_decision_tick:
            self._truncated = True
            info = self._get_info(
                decision_tick=self._current_tick,
                entry_tick=self._current_tick + 1,
                exit_tick=self._current_tick + 2,
                reward=0.0,
                base_reward=0.0,
                risk_penalty=0.0,
                fee_cost=0.0,
                slippage_cost=0.0,
                gross_return=0.0,
                net_return=0.0,
                turnover=0.0,
                market_return=0.0,
                drawdown=self._drawdown(self._equity),
                target_position=self._position,
                previous_position=self._position,
            )
            return self._get_observation(), 0.0, False, True, info

        previous_tick = self._current_tick
        previous_position = self._position
        target_position = self._coerce_action(action)

        entry_tick = previous_tick + 1
        exit_tick = previous_tick + 2
        market_return = self._market_return(entry_tick, exit_tick)
        turnover = abs(target_position - previous_position)
        gross_return = target_position * market_return
        fee_cost = turnover * self.costs.fee_rate
        slippage_cost = turnover * self.costs.slippage_rate
        net_return = gross_return - fee_cost - slippage_cost

        next_equity = self._equity * (1.0 + net_return)
        terminated = bool(next_equity <= 0 or not np.isfinite(next_equity))
        if terminated:
            next_equity = 0.0

        self._equity = float(next_equity)
        self._peak_equity = max(self._peak_equity, self._equity)
        drawdown = self._drawdown(self._equity)

        base_reward = self._base_reward(net_return)
        risk_penalty = self._risk_penalty(
            target_position=target_position,
            drawdown=drawdown,
            tick=previous_tick,
        )
        reward = (base_reward - risk_penalty) * self.reward_scale
        self._total_reward += reward
        self._position = target_position

        self._current_tick += 1
        self._truncated = self._current_tick > self._last_decision_tick
        self._terminated = terminated

        info = self._get_info(
            decision_tick=previous_tick,
            entry_tick=entry_tick,
            exit_tick=exit_tick,
            reward=reward,
            base_reward=base_reward,
            risk_penalty=risk_penalty,
            fee_cost=fee_cost,
            slippage_cost=slippage_cost,
            gross_return=gross_return,
            net_return=net_return,
            turnover=turnover,
            market_return=market_return,
            drawdown=drawdown,
            target_position=target_position,
            previous_position=previous_position,
        )
        self._history.append(info)

        observation = self._get_observation()
        return observation, reward, self._terminated, self._truncated, info

    def _prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.empty:
            raise ValueError("df must not be empty")
        frame = df.copy().reset_index(drop=True)
        if self.time_column in frame.columns:
            times = frame[self.time_column].to_numpy()
            if len(times) > 1 and not pd.Series(times).is_monotonic_increasing:
                raise ValueError(f"{self.time_column} must be sorted in ascending order")
        return frame

    def _build_future_returns(self, prices: np.ndarray) -> np.ndarray:
        returns = np.zeros(len(prices), dtype=np.float64)
        returns[:-1] = prices[1:] / prices[:-1] - 1.0
        return returns

    def _build_past_volatility(self, future_returns: np.ndarray, window: int) -> np.ndarray:
        series = pd.Series(future_returns[:-1])
        volatility = series.rolling(window=window, min_periods=2).std(ddof=1).shift(1)
        values = np.zeros(len(future_returns), dtype=np.float64)
        if not volatility.empty:
            values[: len(volatility)] = volatility.fillna(0.0).to_numpy(dtype=np.float64)
        return values

    def _get_observation(self) -> np.ndarray:
        if self._current_tick is None:
            raise RuntimeError("environment has not been reset")
        start = self._current_tick - self.window_size + 1
        end = self._current_tick + 1
        return self._observations[start:end].copy()

    def _coerce_action(self, action) -> float:
        value = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        if not np.isfinite(value):
            raise ValueError("action must contain a finite value")
        return float(np.clip(value, -self.max_leverage, self.max_leverage))

    def _market_return(self, entry_tick: int, exit_tick: int) -> float:
        if exit_tick >= len(self._prices):
            return 0.0
        return float(self._prices[exit_tick] / self._prices[entry_tick] - 1.0)

    def _base_reward(self, net_return: float) -> float:
        if self.reward_mode == "simple":
            return net_return
        if net_return <= -1.0:
            return -1.0
        return float(np.log1p(net_return))

    def _risk_penalty(self, target_position: float, drawdown: float, tick: int) -> float:
        volatility = float(self._past_volatility[tick])
        return (
            self.position_penalty * abs(target_position)
            + self.volatility_penalty * abs(target_position) * volatility
            + self.drawdown_penalty * max(0.0, -drawdown)
        )

    def _drawdown(self, equity: float) -> float:
        if self._peak_equity <= 0:
            return 0.0
        return equity / self._peak_equity - 1.0

    def _open_time_at(self, tick: int) -> Any:
        if self._open_times is None or tick < 0 or tick >= len(self._open_times):
            return None
        return self._open_times[tick]

    def _get_info(
        self,
        *,
        decision_tick: int | None = None,
        entry_tick: int | None = None,
        exit_tick: int | None = None,
        reward: float,
        base_reward: float,
        risk_penalty: float,
        fee_cost: float,
        slippage_cost: float,
        gross_return: float,
        net_return: float,
        turnover: float,
        market_return: float,
        drawdown: float,
        target_position: float,
        previous_position: float,
    ) -> dict[str, Any]:
        tick = self._current_tick if self._current_tick is not None else self._start_tick
        decision_tick = tick if decision_tick is None else decision_tick
        entry_tick = tick if entry_tick is None else entry_tick
        exit_tick = tick if exit_tick is None else exit_tick
        info: dict[str, Any] = {
            "tick": tick,
            "decision_tick": decision_tick,
            "entry_tick": entry_tick,
            "exit_tick": exit_tick,
            "from_tick": entry_tick,
            "to_tick": exit_tick,
            "open_time": self._open_time_at(tick),
            "decision_open_time": self._open_time_at(decision_tick),
            "from_open_time": self._open_time_at(entry_tick),
            "to_open_time": self._open_time_at(exit_tick),
            "equity": self._equity,
            "peak_equity": self._peak_equity,
            "position": self._position,
            "target_position": target_position,
            "previous_position": previous_position,
            "turnover": turnover,
            "market_return": market_return,
            "gross_return": gross_return,
            "fee_cost": fee_cost,
            "slippage_cost": slippage_cost,
            "net_return": net_return,
            "reward": reward,
            "base_reward": base_reward,
            "risk_penalty": risk_penalty,
            "drawdown": drawdown,
            "total_reward": self._total_reward,
            "terminated": self._terminated,
            "truncated": self._truncated,
        }
        return info

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)
