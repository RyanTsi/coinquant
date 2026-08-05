from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from coinquant.backtest.account import Account, AccountSnapshot
from coinquant.backtest.position import Position, PositionSnapshot


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _price(value: float, name: str) -> float:
    value = _finite(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    contract_size: float = 1.0
    margin_rate: float = 0.005
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0
    quantity_epsilon: float = 1e-12
    liquidation_fee_rate: float = 0.0

    def __post_init__(self) -> None:
        contract_size = _finite(self.contract_size, "contract_size")
        quantity_epsilon = _finite(self.quantity_epsilon, "quantity_epsilon")
        if contract_size <= 0:
            raise ValueError("contract_size must be greater than 0")
        if quantity_epsilon < 0:
            raise ValueError("quantity_epsilon must be non-negative")
        object.__setattr__(self, "contract_size", contract_size)
        object.__setattr__(self, "quantity_epsilon", quantity_epsilon)
        for name in ("margin_rate", "fee_rate", "slippage_rate", "liquidation_fee_rate"):
            value = _finite(getattr(self, name), name)
            if value < 0 or value >= 1:
                raise ValueError(f"{name} must be in [0, 1)")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    event_type: str
    timestamp: Any
    target_quantity: float | None
    before_position: float
    after_position: float
    side: str | None
    quantity: float
    reference_price: float | None
    fill_price: float | None
    trade_notional: float
    realized_pnl_delta: float
    fee: float
    funding_payment: float
    completed: bool
    reason: str | None
    liquidated: bool
    applied_leverage: float
    account: AccountSnapshot
    position: PositionSnapshot

    @property
    def leverage(self) -> float:
        return self.applied_leverage


@dataclass(frozen=True, slots=True)
class _Projection:
    target_position: float
    entry_price: float
    fill_price: float
    quantity: float
    trade_notional: float
    realized_pnl_delta: float
    fee: float
    notional: float
    unrealized_pnl: float
    initial_margin: float
    maintenance_margin: float
    equity: float
    available_balance: float
    mark_price: float


class ExecutionEngine:
    """Stateless transaction coordinator for one account and one net position."""

    def __init__(self, config: ExecutionConfig):
        self.cfg = config

    def execute_target(
        self,
        target_quantity: float,
        reference_price: float,
        mark_price: float,
        position: Position,
        account: Account,
        timestamp: Any = None,
        reason: str = "TARGET",
    ) -> ExecutionResult:
        target_quantity = _finite(target_quantity, "target_quantity")
        reference_price = _price(reference_price, "reference_price")
        mark_price = _price(mark_price, "mark_price")
        before = position.position

        checkpoint = self._checkpoint(position, account)
        self._refresh(mark_price, position, account)
        if account.is_liquidated:
            return self._event(
                event_type="REJECTED",
                timestamp=timestamp,
                target_quantity=target_quantity,
                before_position=before,
                reference_price=reference_price,
                completed=False,
                reason="ACCOUNT_LIQUIDATED",
                account=account,
                position=position,
            )

        dq = target_quantity - before
        if abs(dq) <= self.cfg.quantity_epsilon:
            return self._event(
                event_type="TARGET_NOOP",
                timestamp=timestamp,
                target_quantity=target_quantity,
                before_position=before,
                reference_price=reference_price,
                completed=True,
                reason=reason,
                account=account,
                position=position,
            )

        fill_price = self._fill_price(reference_price, dq)
        projection = self._project(
            target_position=target_quantity,
            executed_delta=dq,
            fill_price=fill_price,
            mark_price=mark_price,
            position=position,
            account=account,
        )

        if self._risk_increases(before, target_quantity) and not self._risk_ok(projection):
            if before != 0 and target_quantity != 0 and before * target_quantity < 0:
                close_delta = -before
                close_projection = self._project(
                    target_position=0.0,
                    executed_delta=close_delta,
                    fill_price=fill_price,
                    mark_price=mark_price,
                    position=position,
                    account=account,
                )
                self._commit(close_projection, position, account)
                return self._event(
                    event_type="TARGET_REDUCED",
                    timestamp=timestamp,
                    target_quantity=target_quantity,
                    before_position=before,
                    reference_price=reference_price,
                    projection=close_projection,
                    completed=False,
                    reason="INSUFFICIENT_MARGIN_FOR_REVERSE",
                    account=account,
                    position=position,
                )

            self._restore(checkpoint, position, account)
            return self._event(
                event_type="REJECTED",
                timestamp=timestamp,
                target_quantity=target_quantity,
                before_position=before,
                reference_price=reference_price,
                completed=False,
                reason="INSUFFICIENT_MARGIN",
                account=account,
                position=position,
            )

        # A rejected risk increase must not leave a partially applied valuation
        # or any other write behind.  Risk-triggered liquidation above is a
        # separate, intentional state transition and is therefore preserved.
        self._restore(checkpoint, position, account)
        self._commit(projection, position, account)
        return self._event(
            event_type="FILLED",
            timestamp=timestamp,
            target_quantity=target_quantity,
            before_position=before,
            reference_price=reference_price,
            projection=projection,
            completed=True,
            reason=reason,
            account=account,
            position=position,
        )

    def mark_to_market(
        self,
        mark_price: float,
        position: Position,
        account: Account,
        timestamp: Any = None,
        reason: str = "MARK",
    ) -> ExecutionResult:
        mark_price = _price(mark_price, "mark_price")
        before = position.position
        self._refresh(mark_price, position, account)
        return self._event(
            event_type="LIQUIDATION_TRIGGER" if account.is_liquidated else "MARK",
            timestamp=timestamp,
            target_quantity=None,
            before_position=before,
            reference_price=mark_price,
            completed=not account.is_liquidated,
            reason="MAINTENANCE_MARGIN" if account.is_liquidated else reason,
            account=account,
            position=position,
        )

    def settle_funding(
        self,
        funding_rate: float,
        mark_price: float,
        position: Position,
        account: Account,
        timestamp: Any = None,
    ) -> ExecutionResult:
        funding_rate = _finite(funding_rate, "funding_rate")
        mark_price = _price(mark_price, "mark_price")
        before = position.position
        self._refresh(mark_price, position, account)
        payment = before * mark_price * self.cfg.contract_size * funding_rate
        account.pay_funding(payment)
        self._refresh(mark_price, position, account)
        return self._event(
            event_type="FUNDING",
            timestamp=timestamp,
            target_quantity=None,
            before_position=before,
            reference_price=mark_price,
            funding_payment=payment,
            completed=not account.is_liquidated,
            reason="MAINTENANCE_MARGIN" if account.is_liquidated else "FUNDING",
            account=account,
            position=position,
        )

    def force_liquidation(
        self,
        reference_price: float,
        position: Position,
        account: Account,
        timestamp: Any = None,
    ) -> ExecutionResult:
        reference_price = _price(reference_price, "reference_price")
        before = position.position
        if before == 0:
            account.is_liquidated = True
            return self._event(
                event_type="LIQUIDATION",
                timestamp=timestamp,
                target_quantity=0.0,
                before_position=0.0,
                reference_price=reference_price,
                completed=True,
                reason="ALREADY_FLAT",
                account=account,
                position=position,
            )

        delta = -before
        fill_price = self._fill_price(reference_price, delta)
        projection = self._project(
            target_position=0.0,
            executed_delta=delta,
            fill_price=fill_price,
            mark_price=reference_price,
            position=position,
            account=account,
        )
        liquidation_fee = projection.trade_notional * self.cfg.liquidation_fee_rate
        self._commit(projection, position, account, fee_override=projection.fee + liquidation_fee)
        account.is_liquidated = True
        return self._event(
            event_type="LIQUIDATION",
            timestamp=timestamp,
            target_quantity=0.0,
            before_position=before,
            reference_price=reference_price,
            projection=projection,
            fee=projection.fee + liquidation_fee,
            completed=True,
            reason="MAINTENANCE_MARGIN",
            account=account,
            position=position,
        )

    def finish_bar(self, position: Position) -> None:
        position.advance_bar()

    def _refresh(self, mark_price: float, position: Position, account: Account) -> None:
        leverage = _finite(account.leverage, "account.leverage")
        if leverage <= 0:
            raise ValueError("account.leverage must be greater than 0")
        account.leverage = leverage
        if position.position == 0:
            notional = unrealized = initial_margin = maintenance_margin = 0.0
        else:
            notional = abs(position.position) * mark_price * self.cfg.contract_size
            unrealized = position.position * (mark_price - position.entry_price) * self.cfg.contract_size
            initial_margin = notional / account.leverage
            maintenance_margin = notional * self.cfg.margin_rate
        position.apply_valuation(mark_price, notional, unrealized, initial_margin, maintenance_margin)
        account.update(unrealized, initial_margin, maintenance_margin)

    def _checkpoint(self, position: Position, account: Account) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        return (
            (
                position.position,
                position.entry_price,
                position.mark_price,
                position.notional,
                position.unrealized_pnl,
                position.initial_margin,
                position.maintenance_margin,
                position.holding_steps,
            ),
            (
                account.balance,
                account.equity,
                account.available_balance,
                account.used_margin,
                account.maintenance_margin,
                account.realized_pnl,
                account.unrealized_pnl,
                account.total_fee,
                account.total_funding,
                account.margin_ratio,
                account.is_liquidated,
            ),
        )

    def _restore(
        self,
        checkpoint: tuple[tuple[Any, ...], tuple[Any, ...]],
        position: Position,
        account: Account,
    ) -> None:
        position_values, account_values = checkpoint
        (
            position.position,
            position.entry_price,
            position.mark_price,
            position.notional,
            position.unrealized_pnl,
            position.initial_margin,
            position.maintenance_margin,
            position.holding_steps,
        ) = position_values
        (
            account.balance,
            account.equity,
            account.available_balance,
            account.used_margin,
            account.maintenance_margin,
            account.realized_pnl,
            account.unrealized_pnl,
            account.total_fee,
            account.total_funding,
            account.margin_ratio,
            account.is_liquidated,
        ) = account_values

    def _project(
        self,
        target_position: float,
        executed_delta: float,
        fill_price: float,
        mark_price: float,
        position: Position,
        account: Account,
    ) -> _Projection:
        old_position = position.position
        old_entry = position.entry_price
        close_qty = min(abs(old_position), abs(executed_delta)) if old_position * executed_delta < 0 else 0.0
        realized_delta = (
            close_qty * math.copysign(1.0, old_position) * (fill_price - old_entry) * self.cfg.contract_size
            if close_qty
            else 0.0
        )
        entry_price = self._entry_after_trade(old_position, old_entry, executed_delta, fill_price, target_position)
        trade_notional = abs(executed_delta) * fill_price * self.cfg.contract_size
        fee = trade_notional * self.cfg.fee_rate
        notional = abs(target_position) * mark_price * self.cfg.contract_size
        unrealized = target_position * (mark_price - entry_price) * self.cfg.contract_size if target_position else 0.0
        initial_margin = notional / account.leverage
        maintenance_margin = notional * self.cfg.margin_rate
        equity = account.balance + realized_delta - fee + unrealized
        return _Projection(
            target_position=target_position,
            entry_price=entry_price,
            fill_price=fill_price,
            quantity=abs(executed_delta),
            trade_notional=trade_notional,
            realized_pnl_delta=realized_delta,
            fee=fee,
            notional=notional,
            unrealized_pnl=unrealized,
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            equity=equity,
            available_balance=equity - initial_margin,
            mark_price=mark_price,
        )

    def _commit(
        self,
        projection: _Projection,
        position: Position,
        account: Account,
        fee_override: float | None = None,
    ) -> None:
        position.commit_trade(projection.target_position, projection.entry_price)
        account.realize_pnl(projection.realized_pnl_delta)
        account.pay_fee(projection.fee if fee_override is None else fee_override)
        self._refresh(projection.mark_price, position, account)

    def _entry_after_trade(
        self,
        old_position: float,
        old_entry: float,
        executed_delta: float,
        fill_price: float,
        target_position: float,
    ) -> float:
        if target_position == 0:
            return 0.0
        if old_position == 0 or old_position * executed_delta < 0 and abs(executed_delta) > abs(old_position):
            return fill_price
        if old_position * executed_delta > 0:
            return (abs(old_position) * old_entry + abs(executed_delta) * fill_price) / abs(target_position)
        return old_entry

    def _risk_increases(self, old_position: float, target_position: float) -> bool:
        if target_position == old_position:
            return False
        if old_position == 0:
            return target_position != 0
        if old_position * target_position > 0:
            return abs(target_position) > abs(old_position)
        return target_position != 0

    def _risk_ok(self, projection: _Projection) -> bool:
        epsilon = max(self.cfg.quantity_epsilon, 1e-12)
        return projection.available_balance >= -epsilon and (
            projection.maintenance_margin <= 0
            or projection.equity > projection.maintenance_margin
        )

    def _fill_price(self, reference_price: float, delta: float) -> float:
        multiplier = 1 + self.cfg.slippage_rate if delta > 0 else 1 - self.cfg.slippage_rate
        return _price(reference_price * multiplier, "fill_price")

    def _event(
        self,
        event_type: str,
        timestamp: Any,
        target_quantity: float | None,
        before_position: float,
        reference_price: float | None,
        account: Account,
        position: Position,
        completed: bool,
        reason: str | None,
        projection: _Projection | None = None,
        fee: float | None = None,
        funding_payment: float = 0.0,
    ) -> ExecutionResult:
        return ExecutionResult(
            event_type=event_type,
            timestamp=timestamp,
            target_quantity=target_quantity,
            before_position=before_position,
            after_position=position.position,
            side=None if projection is None or projection.quantity == 0 else ("BUY" if projection.target_position > before_position else "SELL"),
            quantity=0.0 if projection is None else projection.quantity,
            reference_price=reference_price,
            fill_price=None if projection is None else projection.fill_price,
            trade_notional=0.0 if projection is None else projection.trade_notional,
            realized_pnl_delta=0.0 if projection is None else projection.realized_pnl_delta,
            fee=(0.0 if projection is None else projection.fee) if fee is None else fee,
            funding_payment=funding_payment,
            completed=completed,
            reason=reason,
            liquidated=account.is_liquidated,
            applied_leverage=account.leverage,
            account=account.snapshot(),
            position=position.snapshot(),
        )
