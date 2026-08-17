"""Independent in-memory backtest primitives and orchestration API."""

from coinquant.backtest.account import Account, AccountConfig, AccountSnapshot
from coinquant.backtest.execution import ExecutionConfig, ExecutionEngine, ExecutionResult
from coinquant.backtest.metrics import BacktestMetrics, calculate_metrics
from coinquant.backtest.position import Position, PositionSnapshot
from coinquant.backtest.simulation import (
    AccountLedgerRecord,
    BacktestConfig,
    BacktestResult,
    BacktestState,
    BarRecord,
    MarketBar,
    TargetInstruction,
    advance_bar,
    create_backtest_state,
    finalize_backtest,
    run_backtest,
    submit_target,
)

__all__ = [
    "Account",
    "AccountConfig",
    "AccountSnapshot",
    "ExecutionConfig",
    "ExecutionEngine",
    "ExecutionResult",
    "Position",
    "PositionSnapshot",
    "BacktestMetrics",
    "calculate_metrics",
    "MarketBar",
    "TargetInstruction",
    "BacktestConfig",
    "BacktestState",
    "BarRecord",
    "AccountLedgerRecord",
    "BacktestResult",
    "create_backtest_state",
    "advance_bar",
    "submit_target",
    "finalize_backtest",
    "run_backtest",
]
