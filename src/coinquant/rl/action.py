"""Continuous target-exposure actions for the RL environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class ActionConfig:
    max_leverage: float = 1.0
    quantity_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        max_leverage = _finite(self.max_leverage, "max_leverage")
        quantity_epsilon = _finite(self.quantity_epsilon, "quantity_epsilon")
        if max_leverage <= 0:
            raise ValueError("max_leverage must be greater than 0")
        if quantity_epsilon < 0:
            raise ValueError("quantity_epsilon must be non-negative")
        object.__setattr__(self, "max_leverage", max_leverage)
        object.__setattr__(self, "quantity_epsilon", quantity_epsilon)


class ActionAdapter:
    """Validate actions and convert target exposure to contract quantity."""

    def __init__(self, config: ActionConfig | None = None):
        self.config = config or ActionConfig()

    @property
    def low(self) -> float:
        return -self.config.max_leverage

    @property
    def high(self) -> float:
        return self.config.max_leverage

    def coerce(self, action: Any) -> tuple[float, bool]:
        values = np.asarray(action, dtype=np.float64)
        if values.size != 1:
            raise ValueError("action must contain exactly one value")
        value = float(values.reshape(-1)[0])
        if not np.isfinite(value):
            raise ValueError("action must contain a finite value")
        clipped = float(np.clip(value, self.low, self.high))
        return clipped, clipped != value

    def target_quantity(
        self,
        target_exposure: float,
        equity: float,
        open_price: float,
        contract_size: float = 1.0,
    ) -> float:
        target_exposure = _finite(target_exposure, "target_exposure")
        equity = _finite(equity, "equity")
        open_price = _finite(open_price, "open_price")
        contract_size = _finite(contract_size, "contract_size")
        if open_price <= 0 or contract_size <= 0:
            raise ValueError("open_price and contract_size must be positive")
        if equity <= 0:
            return 0.0
        return target_exposure * equity / (open_price * contract_size)

    def exposure_from_position(
        self,
        position_quantity: float,
        mark_price: float,
        equity: float,
        contract_size: float = 1.0,
    ) -> float:
        quantity = _finite(position_quantity, "position_quantity")
        mark_price = _finite(mark_price, "mark_price")
        equity = _finite(equity, "equity")
        contract_size = _finite(contract_size, "contract_size")
        if mark_price <= 0 or contract_size <= 0:
            raise ValueError("mark_price and contract_size must be positive")
        if equity <= 0:
            return 0.0
        return float(np.clip(quantity * mark_price * contract_size / equity, self.low, self.high))
