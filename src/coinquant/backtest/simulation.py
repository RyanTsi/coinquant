"""Independent, in-memory bar backtest orchestration.

This module deliberately contains no strategy, model, dataframe or persistence
dependencies.  It coordinates the three trading primitives and keeps immutable
records that can be consumed by the metrics/reporting layers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from coinquant.backtest.account import Account, AccountConfig, AccountSnapshot
from coinquant.backtest.execution import (
    ExecutionConfig,
    ExecutionEngine,
    ExecutionResult,
)
from coinquant.backtest.position import Position, PositionSnapshot
from coinquant.backtest.metrics import BacktestMetrics, calculate_metrics


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive_price(value: float, name: str) -> float:
    value = _finite(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class MarketBar:
    """A validated OHLC bar used by the independent simulator."""

    timestamp: Any
    open: float
    high: float
    low: float
    close: float
    funding_rate: float | None = None

    def __post_init__(self) -> None:
        open_price = _positive_price(self.open, "open")
        high = _positive_price(self.high, "high")
        low = _positive_price(self.low, "low")
        close = _positive_price(self.close, "close")
        if low > min(open_price, close) or high < max(open_price, close) or low > high:
            raise ValueError("OHLC prices must satisfy low <= min(open, close) <= max(open, close) <= high")
        object.__setattr__(self, "open", open_price)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "close", close)
        if self.funding_rate is not None:
            object.__setattr__(self, "funding_rate", _finite(self.funding_rate, "funding_rate"))


@dataclass(frozen=True, slots=True)
class TargetInstruction:
    """A target net quantity generated at a completed bar."""

    generated_at: Any
    target_quantity: float
    reason: str = "TARGET"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_quantity", _finite(self.target_quantity, "target_quantity"))
        if self.reason is None:
            object.__setattr__(self, "reason", "TARGET")
        elif not isinstance(self.reason, str):
            raise TypeError("reason must be a string")
        try:
            metadata = dict(self.metadata)
        except (TypeError, ValueError) as exc:
            raise TypeError("metadata must be a mapping") from exc
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    force_close_at_end: bool = True
    annualization_factor: float | None = 365.0

    def __post_init__(self) -> None:
        if not isinstance(self.force_close_at_end, bool):
            raise TypeError("force_close_at_end must be a bool")
        if self.annualization_factor is not None:
            value = _finite(self.annualization_factor, "annualization_factor")
            if value <= 0:
                raise ValueError("annualization_factor must be greater than 0")
            object.__setattr__(self, "annualization_factor", value)


@dataclass(frozen=True, slots=True)
class AccountLedgerRecord:
    timestamp: Any
    event_type: str
    account: AccountSnapshot
    position: PositionSnapshot
    execution: ExecutionResult | None = None

    @property
    def account_snapshot(self) -> AccountSnapshot:
        return self.account

    @property
    def position_snapshot(self) -> PositionSnapshot:
        return self.position

    @property
    def execution_result(self) -> ExecutionResult | None:
        return self.execution


@dataclass(frozen=True, slots=True)
class BarRecord:
    timestamp: Any
    open: float
    high: float
    low: float
    close: float
    funding_rate: float | None
    target: TargetInstruction | None
    executions: tuple[ExecutionResult, ...]
    account: AccountSnapshot
    position: PositionSnapshot

    @property
    def equity(self) -> float:
        return self.account.equity

    @property
    def target_quantity(self) -> float | None:
        return None if self.target is None else self.target.target_quantity

    @property
    def target_instruction(self) -> TargetInstruction | None:
        return self.target

    @property
    def account_snapshot(self) -> AccountSnapshot:
        return self.account

    @property
    def position_snapshot(self) -> PositionSnapshot:
        return self.position

    @property
    def execution_results(self) -> tuple[ExecutionResult, ...]:
        return self.executions


@dataclass(frozen=True, slots=True)
class BacktestResult:
    status: str
    start_timestamp: Any
    end_timestamp: Any
    account: AccountSnapshot
    position: PositionSnapshot
    bar_ledger: tuple[BarRecord, ...]
    trade_ledger: tuple[ExecutionResult, ...]
    account_ledger: tuple[AccountLedgerRecord, ...]
    discarded_targets: tuple[TargetInstruction, ...]
    equity_curve: tuple[float, ...]
    config: BacktestConfig
    account_config: AccountConfig
    execution_config: ExecutionConfig
    metrics: BacktestMetrics

    @property
    def bars(self) -> tuple[BarRecord, ...]:
        return self.bar_ledger

    @property
    def bar_records(self) -> tuple[BarRecord, ...]:
        return self.bar_ledger

    @property
    def trades(self) -> tuple[ExecutionResult, ...]:
        return self.trade_ledger

    @property
    def trade_records(self) -> tuple[ExecutionResult, ...]:
        return self.trade_ledger

    @property
    def account_records(self) -> tuple[AccountLedgerRecord, ...]:
        return self.account_ledger

    @property
    def final_account(self) -> AccountSnapshot:
        return self.account

    @property
    def final_position(self) -> PositionSnapshot:
        return self.position

    @property
    def final_equity(self) -> float:
        return self.account.equity

    @property
    def initial_balance(self) -> float:
        return self.account_config.initial_balance


@dataclass(slots=True)
class BacktestState:
    config: BacktestConfig
    account: Account
    position: Position
    execution: ExecutionEngine
    pending_target: TargetInstruction | None = None
    current_index: int = -1
    last_bar: MarketBar | None = None
    status: str = "READY"
    bar_ledger: list[BarRecord] = field(default_factory=list)
    trade_ledger: list[ExecutionResult] = field(default_factory=list)
    account_ledger: list[AccountLedgerRecord] = field(default_factory=list)
    discarded_targets: list[TargetInstruction] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    _result: BacktestResult | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.equity_curve:
            self.equity_curve.append(self.account.balance)


def create_backtest_state(
    account_config: AccountConfig | None = None,
    execution_config: ExecutionConfig | None = None,
    backtest_config: BacktestConfig | None = None,
) -> BacktestState:
    """Create the sole mutable state object for one backtest run."""

    account_config = account_config or AccountConfig()
    execution_config = execution_config or ExecutionConfig()
    backtest_config = backtest_config or BacktestConfig()
    return BacktestState(
        config=backtest_config,
        account=Account(account_config),
        position=Position(),
        execution=ExecutionEngine(execution_config),
    )


def advance_bar(state: BacktestState, bar: MarketBar) -> None:
    """Process one complete bar, including pending target execution."""

    if not isinstance(state, BacktestState):
        raise TypeError("state must be a BacktestState")
    if not isinstance(bar, MarketBar):
        raise TypeError("bar must be a MarketBar")
    if state.status in {"COMPLETED", "LIQUIDATED"}:
        raise RuntimeError(f"cannot advance a {state.status.lower()} backtest")
    if state.last_bar is not None and not _is_after(bar.timestamp, state.last_bar.timestamp):
        raise ValueError("bar timestamps must be strictly increasing")

    state.status = "RUNNING"
    events: list[ExecutionResult] = []
    target = state.pending_target
    state.pending_target = None

    opening = state.execution.mark_to_market(bar.open, state.position, state.account, bar.timestamp, "OPEN")
    events.append(opening)
    if opening.liquidated:
        events.append(state.execution.force_liquidation(bar.open, state.position, state.account, bar.timestamp))

    if not state.account.is_liquidated and bar.funding_rate is not None:
        funding = state.execution.settle_funding(
            bar.funding_rate,
            bar.open,
            state.position,
            state.account,
            bar.timestamp,
        )
        events.append(funding)
        if funding.liquidated:
            events.append(state.execution.force_liquidation(bar.open, state.position, state.account, bar.timestamp))

    if target is not None:
        if state.account.is_liquidated:
            state.discarded_targets.append(target)
        else:
            events.append(
                state.execution.execute_target(
                    target.target_quantity,
                    bar.open,
                    bar.open,
                    state.position,
                    state.account,
                    bar.timestamp,
                    target.reason,
                )
            )

    if not state.account.is_liquidated and state.position.position != 0:
        adverse_price = bar.low if state.position.position > 0 else bar.high
        intrabar = state.execution.mark_to_market(
            adverse_price,
            state.position,
            state.account,
            bar.timestamp,
            "INTRABAR_RISK",
        )
        events.append(intrabar)
        if intrabar.liquidated:
            events.append(
                state.execution.force_liquidation(adverse_price, state.position, state.account, bar.timestamp)
            )

    if not state.account.is_liquidated:
        closing = state.execution.mark_to_market(bar.close, state.position, state.account, bar.timestamp, "CLOSE")
        events.append(closing)
        if closing.liquidated:
            events.append(state.execution.force_liquidation(bar.close, state.position, state.account, bar.timestamp))

    state.execution.finish_bar(state.position)
    for event in events:
        _record_event(state, event)

    record = BarRecord(
        timestamp=bar.timestamp,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        funding_rate=bar.funding_rate,
        target=target,
        executions=tuple(events),
        account=state.account.snapshot(),
        position=state.position.snapshot(),
    )
    state.bar_ledger.append(record)
    state.equity_curve.append(record.account.equity)
    state.last_bar = bar
    state.current_index += 1
    state.status = "LIQUIDATED" if state.account.is_liquidated else "RUNNING"


def submit_target(state: BacktestState, target: TargetInstruction) -> None:
    """Queue a target generated by the most recently completed bar."""

    if not isinstance(target, TargetInstruction):
        raise TypeError("target must be a TargetInstruction")
    if state.status in {"COMPLETED", "LIQUIDATED"}:
        raise RuntimeError(f"cannot submit target to a {state.status.lower()} backtest")
    if state.last_bar is None:
        raise RuntimeError("submit_target requires at least one completed bar")
    if target.generated_at != state.last_bar.timestamp:
        raise ValueError("target.generated_at must equal the latest completed bar timestamp")
    if state.pending_target is not None:
        raise ValueError("a target is already pending for the next bar")
    state.pending_target = target


def finalize_backtest(state: BacktestState) -> BacktestResult:
    """Freeze the run and calculate metrics from its immutable ledgers."""

    if state._result is not None:
        return state._result
    if not state.bar_ledger or state.last_bar is None:
        raise ValueError("cannot finalize a backtest with no completed bars")

    if state.pending_target is not None:
        # A last-bar signal has no T+1 bar to fill.  Keep it in the frozen
        # result so callers can distinguish it from a missing signal.
        state.discarded_targets.append(state.pending_target)
        state.pending_target = None

    if state.status not in {"COMPLETED", "LIQUIDATED"}:
        if state.config.force_close_at_end and not state.account.is_liquidated and state.position.position != 0:
            event = state.execution.execute_target(
                0.0,
                state.last_bar.close,
                state.last_bar.close,
                state.position,
                state.account,
                state.last_bar.timestamp,
                "END_OF_TEST",
            )
            _record_event(state, event)
            last = state.bar_ledger[-1]
            state.bar_ledger[-1] = replace(
                last,
                executions=last.executions + (event,),
                account=state.account.snapshot(),
                position=state.position.snapshot(),
            )
            state.equity_curve[-1] = state.account.equity
        state.status = "LIQUIDATED" if state.account.is_liquidated else "COMPLETED"

    metrics = calculate_metrics(
        equity_curve=tuple(state.equity_curve),
        trade_ledger=tuple(state.trade_ledger),
        initial_balance=state.account.cfg.initial_balance,
        annualization_factor=state.config.annualization_factor,
        final_equity=state.account.equity,
        total_fee=state.account.total_fee,
        total_funding=state.account.total_funding,
    )
    state._result = BacktestResult(
        status=state.status,
        start_timestamp=state.bar_ledger[0].timestamp,
        end_timestamp=state.bar_ledger[-1].timestamp,
        account=state.account.snapshot(),
        position=state.position.snapshot(),
        bar_ledger=tuple(state.bar_ledger),
        trade_ledger=tuple(state.trade_ledger),
        account_ledger=tuple(state.account_ledger),
        discarded_targets=tuple(state.discarded_targets),
        equity_curve=tuple(state.equity_curve),
        config=state.config,
        account_config=state.account.cfg,
        execution_config=state.execution.cfg,
        metrics=metrics,
    )
    return state._result


def run_backtest(
    state: BacktestState,
    bars: Iterable[MarketBar],
    targets: Mapping[Any, TargetInstruction] | None = None,
) -> BacktestResult:
    """Run bars through the same API used by an incremental caller."""

    if state.status != "READY":
        raise RuntimeError("run_backtest requires a fresh BacktestState")
    targets = targets or {}
    for key, target in targets.items():
        if not isinstance(target, TargetInstruction):
            raise TypeError("targets must contain TargetInstruction values")
        if target.generated_at != key:
            raise ValueError("target mapping key must equal target.generated_at")

    seen: set[Any] = set()
    for bar in bars:
        if state.status == "LIQUIDATED":
            # Liquidation is terminal for the trading state.  Remaining market
            # data cannot produce additional events, so stop consuming it.
            break
        advance_bar(state, bar)
        if bar.timestamp in targets and state.status != "LIQUIDATED":
            submit_target(state, targets[bar.timestamp])
            seen.add(bar.timestamp)
    missing = [key for key in targets if key not in seen and state.status != "LIQUIDATED"]
    if missing:
        raise ValueError(f"target timestamp does not align to a completed bar: {missing[0]!r}")
    return finalize_backtest(state)


def _record_event(state: BacktestState, event: ExecutionResult) -> None:
    state.account_ledger.append(
        AccountLedgerRecord(
            timestamp=event.timestamp,
            event_type=event.event_type,
            account=event.account,
            position=event.position,
            execution=event,
        )
    )
    if event.quantity > state.execution.cfg.quantity_epsilon:
        state.trade_ledger.append(event)


def _is_after(value: Any, previous: Any) -> bool:
    try:
        return bool(value > previous)
    except (TypeError, ValueError) as exc:
        raise ValueError("bar timestamps must be comparable") from exc
