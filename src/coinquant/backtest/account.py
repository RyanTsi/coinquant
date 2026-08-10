from __future__ import annotations

from dataclasses import dataclass
from coinquant import utils


@dataclass(frozen=True, slots=True)
class AccountConfig:
    initial_balance: float = 10_000.0
    leverage: float = 10.0

    def __post_init__(self) -> None:
        balance = utils.validate_finite(self.initial_balance, "initial_balance")
        leverage = utils.validate_finite(self.leverage, "leverage")
        if balance <= 0:
            raise ValueError("initial_balance must be greater than 0")
        if leverage <= 0:
            raise ValueError("leverage must be greater than 0")
        object.__setattr__(self, "initial_balance", balance)
        object.__setattr__(self, "leverage", leverage)


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    leverage: float
    balance: float
    equity: float
    available_balance: float
    used_margin: float
    maintenance_margin: float
    margin_ratio: float
    realized_pnl: float
    unrealized_pnl: float
    total_fee: float
    total_funding: float
    is_liquidated: bool

    @property
    def liquidated(self) -> bool:
        return self.is_liquidated


class Account:
    """Account-level cash, aggregate margin and risk state."""

    def __init__(self, config: AccountConfig):
        self.cfg = config
        self.leverage = config.leverage
        self.reset()

    def reset(self) -> None:
        self.leverage = self.cfg.leverage
        self.balance = self.cfg.initial_balance
        self.equity = self.cfg.initial_balance
        self.available_balance = self.cfg.initial_balance
        self.used_margin = 0.0
        self.maintenance_margin = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_fee = 0.0
        self.total_funding = 0.0
        self.margin_ratio = 0.0
        self.is_liquidated = False

    def update(
        self,
        unrealized_pnl: float,
        used_margin: float | None = None,
        maintenance_margin: float = 0.0,
        *,
        initial_margin: float | None = None,
    ) -> bool:
        """Apply a complete aggregate valuation and update risk atomically."""
        if initial_margin is not None:
            if used_margin is not None:
                raise TypeError("pass only one of used_margin and initial_margin")
            used_margin = initial_margin
        if used_margin is None:
            raise TypeError("used_margin (or initial_margin) is required")
        unrealized_pnl = utils.validate_finite(unrealized_pnl, "unrealized_pnl")
        used_margin = utils.validate_finite(used_margin, "used_margin")
        maintenance_margin = utils.validate_finite(maintenance_margin, "maintenance_margin")
        if used_margin < 0 or maintenance_margin < 0:
            raise ValueError("margin values must be non-negative")

        self.unrealized_pnl = unrealized_pnl
        self.used_margin = used_margin
        self.maintenance_margin = maintenance_margin
        self.equity = self.balance + self.unrealized_pnl
        self.available_balance = self.equity - self.used_margin

        if maintenance_margin == 0:
            self.margin_ratio = 0.0
        elif self.equity <= 0:
            self.margin_ratio = math.inf
        else:
            self.margin_ratio = maintenance_margin / self.equity

        trigger = maintenance_margin > 0 and self.equity <= maintenance_margin
        self.is_liquidated = self.is_liquidated or trigger
        return trigger

    def pay_fee(self, fee: float) -> None:
        fee = utils.validate_finite(fee, "fee")
        if fee < 0:
            raise ValueError("fee must be non-negative")
        self.balance -= fee
        self.total_fee += fee

    def pay_funding(self, funding_payment: float) -> None:
        funding_payment = utils.validate_finite(funding_payment, "funding_payment")
        self.balance -= funding_payment
        self.total_funding += funding_payment

    def realize_pnl(self, realized_pnl_delta: float) -> None:
        realized_pnl_delta = utils.validate_finite(realized_pnl_delta, "realized_pnl_delta")
        self.balance += realized_pnl_delta
        self.realized_pnl += realized_pnl_delta

    def check_liquidation(self) -> bool:
        trigger = self.maintenance_margin > 0 and self.equity <= self.maintenance_margin
        self.is_liquidated = self.is_liquidated or trigger
        return self.is_liquidated

    def snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            leverage=self.leverage,
            balance=self.balance,
            equity=self.equity,
            available_balance=self.available_balance,
            used_margin=self.used_margin,
            maintenance_margin=self.maintenance_margin,
            margin_ratio=self.margin_ratio,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
            total_fee=self.total_fee,
            total_funding=self.total_funding,
            is_liquidated=self.is_liquidated,
        )

    @property
    def liquidated(self) -> bool:
        return self.is_liquidated
