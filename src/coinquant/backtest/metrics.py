"""Pure metric calculations for frozen backtest ledgers."""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Iterable, Sequence


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    total_return: float | None
    annualized_return: float | None
    annualized_volatility: float | None
    max_drawdown: float
    sharpe: float | None
    trade_count: int
    total_trade_notional: float
    total_fee: float
    total_funding: float

    @property
    def annual_return(self) -> float | None:
        return self.annualized_return

    @property
    def volatility(self) -> float | None:
        return self.annualized_volatility

    @property
    def total_cost(self) -> float:
        return self.total_fee + self.total_funding

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)

    def __getitem__(self, key: str) -> float | int | None:
        return self.as_dict()[key]


def calculate_metrics(
    equity_curve: Sequence[float] | None = None,
    trade_ledger: Iterable[Any] = (),
    initial_balance: float | None = None,
    annualization_factor: float | None = 365.0,
    final_equity: float | None = None,
    result: Any | None = None,
    total_fee: float | None = None,
    total_funding: float | None = None,
) -> BacktestMetrics:
    """Calculate metrics without importing pandas or the simulation module.

    Passing a ``BacktestResult`` through ``result`` is supported as a convenience;
    the explicit arguments remain useful for callers that keep only frozen ledgers.
    """

    if result is None and equity_curve is not None and hasattr(equity_curve, "equity_curve"):
        result = equity_curve
        equity_curve = None
    if result is not None:
        equity_curve = result.equity_curve
        trade_ledger = result.trade_ledger
        initial_balance = getattr(
            result,
            "initial_balance",
            result.account.balance - result.account.realized_pnl + result.account.total_fee + result.account.total_funding,
        )
        annualization_factor = result.config.annualization_factor
        final_equity = result.account.equity
        total_fee = result.account.total_fee
        total_funding = result.account.total_funding
    if equity_curve is None or initial_balance is None:
        raise ValueError("equity_curve and initial_balance are required")
    initial_balance = _finite(initial_balance, "initial_balance")
    if initial_balance <= 0:
        raise ValueError("initial_balance must be greater than 0")
    points = [_finite(value, "equity_curve") for value in equity_curve]
    if not points:
        raise ValueError("equity_curve must not be empty")
    if final_equity is None:
        final_equity = points[-1]
    final_equity = _finite(final_equity, "final_equity")
    if points[-1] != final_equity:
        points[-1] = final_equity

    returns: list[float] = []
    for index in range(1, len(points)):
        previous = points[index - 1]
        if previous == 0:
            # A return from a zero equity base is undefined.  Keep it out of
            # volatility/Sharpe rather than raising or manufacturing infinity.
            continue
        value = points[index] / previous - 1.0
        if math.isfinite(value):
            returns.append(value)
    total_return = final_equity / initial_balance - 1.0
    annualized_return: float | None = None
    if annualization_factor is not None:
        annualization_factor = _finite(annualization_factor, "annualization_factor")
        if annualization_factor <= 0:
            raise ValueError("annualization_factor must be greater than 0")
        bar_count = len(points) - 1
        if bar_count > 0 and final_equity > 0:
            annualized_return = (final_equity / initial_balance) ** (annualization_factor / bar_count) - 1.0

    annualized_volatility: float | None = None
    sharpe: float | None = None
    if len(returns) >= 2:
        mean_return = sum(returns) / len(returns)
        variance = sum((value - mean_return) ** 2 for value in returns) / (len(returns) - 1)
        std = math.sqrt(max(variance, 0.0))
        if annualization_factor is not None:
            annualized_volatility = std * math.sqrt(annualization_factor)
            if std > 0:
                sharpe = mean_return * math.sqrt(annualization_factor) / std

    running_max = points[0]
    max_drawdown = 0.0
    for value in points:
        running_max = max(running_max, value)
        if running_max != 0:
            max_drawdown = min(max_drawdown, value / running_max - 1.0)

    trades = tuple(trade_ledger)
    total_trade_notional = float(sum(float(getattr(item, "trade_notional", 0.0)) for item in trades))
    if total_fee is None:
        total_fee = sum(float(getattr(item, "fee", 0.0)) for item in trades)
    if total_funding is None:
        total_funding = sum(float(getattr(item, "funding_payment", 0.0)) for item in trades)
    return BacktestMetrics(
        total_return=_finite_or_none(total_return),
        annualized_return=_finite_or_none(annualized_return),
        annualized_volatility=_finite_or_none(annualized_volatility),
        max_drawdown=float(max_drawdown),
        sharpe=_finite_or_none(sharpe),
        trade_count=len(trades),
        total_trade_notional=total_trade_notional,
        total_fee=total_fee,
        total_funding=total_funding,
    )


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) if math.isfinite(value) else None
