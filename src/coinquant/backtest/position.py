from __future__ import annotations

import math
from dataclasses import dataclass


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    position: float
    entry_price: float
    mark_price: float
    notional: float
    unrealized_pnl: float
    initial_margin: float
    maintenance_margin: float
    holding_steps: int


@dataclass(slots=True)
class Position:
    """Current net position state; account history is intentionally excluded."""

    position: float = 0.0
    entry_price: float = 0.0
    mark_price: float = 0.0
    notional: float = 0.0
    unrealized_pnl: float = 0.0
    initial_margin: float = 0.0
    maintenance_margin: float = 0.0
    holding_steps: int = 0

    def __post_init__(self) -> None:
        self.position = _finite(self.position, "position")
        self.entry_price = _finite(self.entry_price, "entry_price")
        self.mark_price = _finite(self.mark_price, "mark_price")
        self.notional = _finite(self.notional, "notional")
        self.unrealized_pnl = _finite(self.unrealized_pnl, "unrealized_pnl")
        self.initial_margin = _finite(self.initial_margin, "initial_margin")
        self.maintenance_margin = _finite(self.maintenance_margin, "maintenance_margin")
        if isinstance(self.holding_steps, bool) or int(self.holding_steps) != self.holding_steps:
            raise ValueError("holding_steps must be an integer")
        if self.holding_steps < 0:
            raise ValueError("holding_steps must be non-negative")
        self.holding_steps = int(self.holding_steps)
        self._validate_valuation()

    def reset(self) -> None:
        self.position = 0.0
        self.entry_price = 0.0
        self.mark_price = 0.0
        self.notional = 0.0
        self.unrealized_pnl = 0.0
        self.initial_margin = 0.0
        self.maintenance_margin = 0.0
        self.holding_steps = 0

    def commit_trade(self, position: float, entry_price: float) -> None:
        position = _finite(position, "position")
        entry_price = _finite(entry_price, "entry_price")
        if position != 0 and entry_price <= 0:
            raise ValueError("entry_price must be positive for an open position")

        continuing_direction = self.position != 0 and position != 0 and self.position * position > 0
        self.position = position
        self.entry_price = entry_price if position != 0 else 0.0
        self.notional = 0.0
        self.unrealized_pnl = 0.0
        self.initial_margin = 0.0
        self.maintenance_margin = 0.0
        if not continuing_direction:
            self.holding_steps = 0

    def apply_valuation(
        self,
        mark_price: float,
        notional: float,
        unrealized_pnl: float,
        initial_margin: float,
        maintenance_margin: float,
    ) -> None:
        mark_price = _finite(mark_price, "mark_price")
        if mark_price <= 0:
            raise ValueError("mark_price must be positive")
        self.mark_price = mark_price
        if self.position == 0:
            self.notional = 0.0
            self.unrealized_pnl = 0.0
            self.initial_margin = 0.0
            self.maintenance_margin = 0.0
            self.holding_steps = 0
            return

        self.notional = _finite(notional, "notional")
        self.unrealized_pnl = _finite(unrealized_pnl, "unrealized_pnl")
        self.initial_margin = _finite(initial_margin, "initial_margin")
        self.maintenance_margin = _finite(maintenance_margin, "maintenance_margin")
        if self.notional < 0 or self.initial_margin < 0 or self.maintenance_margin < 0:
            raise ValueError("position valuation values must be non-negative")

    def advance_bar(self) -> None:
        if self.position != 0:
            self.holding_steps += 1

    def snapshot(self) -> PositionSnapshot:
        return PositionSnapshot(
            position=self.position,
            entry_price=self.entry_price,
            mark_price=self.mark_price,
            notional=self.notional,
            unrealized_pnl=self.unrealized_pnl,
            initial_margin=self.initial_margin,
            maintenance_margin=self.maintenance_margin,
            holding_steps=self.holding_steps,
        )

    def _validate_valuation(self) -> None:
        if self.position != 0 and self.entry_price <= 0:
            raise ValueError("entry_price must be positive for an open position")
        if self.position != 0 and self.mark_price <= 0:
            raise ValueError("mark_price must be positive for an open position")
        if self.notional < 0 or self.initial_margin < 0 or self.maintenance_margin < 0:
            raise ValueError("position valuation values must be non-negative")
