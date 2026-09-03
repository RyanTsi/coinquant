"""Reward decomposition for return, risk and drawdown objectives."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class RewardConfig:
    reward_mode: str = "simple"
    reward_scale: float = 100.0
    # Risk shaping is deliberately disabled by default.  Until a stable
    # profitable policy exists, the learning signal should be the account's
    # net return only; the fields remain available for controlled ablations.
    drawdown_penalty_rate: float = 0.0
    volatility_penalty: float = 0.0
    position_penalty: float = 0.0
    turnover_penalty_rate: float = 0.0
    short_penalty_rate: float = 0.0
    action_change_penalty_rate: float = 0.0
    reversal_penalty_rate: float = 0.0
    risk_window: int = 20
    liquidation_penalty: float = 0.0
    enable_penalties: bool = False

    def __post_init__(self) -> None:
        if self.reward_mode not in {"simple", "log"}:
            raise ValueError("reward_mode must be 'simple' or 'log'")
        for name in (
            "reward_scale",
            "drawdown_penalty_rate",
            "volatility_penalty",
            "position_penalty",
            "turnover_penalty_rate",
            "short_penalty_rate",
            "action_change_penalty_rate",
            "reversal_penalty_rate",
            "liquidation_penalty",
        ):
            value = _finite(getattr(self, name), name)
            if name == "reward_scale" and value <= 0:
                raise ValueError("reward_scale must be greater than 0")
            if name != "reward_scale" and value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if int(self.risk_window) != self.risk_window or self.risk_window <= 0:
            raise ValueError("risk_window must be a positive integer")
        if not isinstance(self.enable_penalties, bool):
            raise TypeError("enable_penalties must be a bool")
        object.__setattr__(self, "risk_window", int(self.risk_window))


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    reward: float
    raw_reward: float
    base_reward: float
    net_return: float
    gross_return: float
    risk_penalty: float
    drawdown_penalty: float
    liquidation_penalty: float
    rolling_volatility: float
    drawdown: float
    turnover_penalty: float = 0.0
    short_penalty: float = 0.0
    action_change_penalty: float = 0.0
    reversal_penalty: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


class RewardCalculator:
    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()
        self._returns: list[float] = []

    @property
    def returns(self) -> tuple[float, ...]:
        return tuple(self._returns)

    def reset(self) -> None:
        self._returns.clear()

    def calculate(
        self,
        previous_equity: float,
        current_equity: float,
        target_exposure: float,
        market_return: float = 0.0,
        drawdown: float = 0.0,
        liquidated: bool = False,
        turnover: float = 0.0,
        short_increase: float = 0.0,
        short_opened: bool = False,
        action_change: float = 0.0,
        reversal_exposure: float = 0.0,
    ) -> RewardBreakdown:
        previous_equity = _finite(previous_equity, "previous_equity")
        current_equity = _finite(current_equity, "current_equity")
        target_exposure = _finite(target_exposure, "target_exposure")
        market_return = _finite(market_return, "market_return")
        drawdown = float(np.clip(_finite(drawdown, "drawdown"), 0.0, 1.0))
        turnover = _finite(turnover, "turnover")
        if turnover < 0:
            raise ValueError("turnover must be non-negative")
        short_increase = _finite(short_increase, "short_increase")
        if short_increase < 0:
            raise ValueError("short_increase must be non-negative")
        action_change = _finite(action_change, "action_change")
        reversal_exposure = _finite(reversal_exposure, "reversal_exposure")
        if action_change < 0 or reversal_exposure < 0:
            raise ValueError("action_change and reversal_exposure must be non-negative")
        # Keep the boolean spelling compatible with early experiments while
        # making the production penalty proportional to newly opened shorts.
        if short_opened and short_increase == 0.0:
            short_increase = 1.0

        if previous_equity <= 0:
            net_return = -1.0 if current_equity <= 0 else 0.0
        else:
            net_return = current_equity / previous_equity - 1.0
        net_return = max(net_return, -1.0)
        self._returns.append(net_return)
        if len(self._returns) > self.config.risk_window:
            del self._returns[:-self.config.risk_window]

        if self.config.reward_mode == "log":
            base_reward = float(np.log1p(net_return)) if net_return > -1 else -np.inf
            if not np.isfinite(base_reward):
                base_reward = -1.0
        else:
            base_reward = net_return
        rolling_volatility = (
            float(np.std(self._returns, ddof=1)) if len(self._returns) >= 2 else 0.0
        )
        risk_penalty = (
            self.config.volatility_penalty * abs(target_exposure) * rolling_volatility
            + self.config.position_penalty * target_exposure**2
        )
        turnover_penalty = self.config.turnover_penalty_rate * turnover
        short_penalty = self.config.short_penalty_rate * short_increase
        action_change_penalty = self.config.action_change_penalty_rate * action_change**2
        reversal_penalty = self.config.reversal_penalty_rate * reversal_exposure
        drawdown_penalty = self.config.drawdown_penalty_rate * drawdown
        liquidation_penalty = self.config.liquidation_penalty if liquidated else 0.0
        raw_reward = base_reward
        if self.config.enable_penalties:
            raw_reward -= (
                risk_penalty
                + drawdown_penalty
                + liquidation_penalty
                + turnover_penalty
                + short_penalty
                + action_change_penalty
                + reversal_penalty
            )
        return RewardBreakdown(
            reward=self.config.reward_scale * raw_reward,
            raw_reward=raw_reward,
            base_reward=base_reward,
            net_return=net_return,
            gross_return=target_exposure * market_return,
            risk_penalty=risk_penalty,
            drawdown_penalty=drawdown_penalty,
            liquidation_penalty=liquidation_penalty,
            rolling_volatility=rolling_volatility,
            drawdown=drawdown,
            turnover_penalty=turnover_penalty,
            short_penalty=short_penalty,
            action_change_penalty=action_change_penalty,
            reversal_penalty=reversal_penalty,
        )
