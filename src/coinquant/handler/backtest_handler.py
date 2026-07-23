from __future__ import annotations

from pathlib import Path

from coinquant.backtest.engine import BacktestEngine, BacktestResult
from coinquant.backtest.report import BacktestReport


def run_backtest(
    symbol: str,
    period: str,
    threshold: float = 0.0,
    fee_rate: float = 0.0004,
    output: str | Path | None = None,
) -> tuple[BacktestResult, Path]:
    result = BacktestEngine(
        symbol=symbol,
        period=period,
        threshold=threshold,
        fee_rate=fee_rate,
    ).run()
    report_path = BacktestReport(result).write(output)
    return result, report_path


def format_backtest_metrics(result: BacktestResult) -> str:
    columns = [
        ("model", "model", _fmt_text),
        ("rows", "rows", _fmt_int),
        ("total_return", "total", _fmt_pct),
        ("annual_return", "annual", _fmt_pct),
        ("sharpe", "sharpe", _fmt_float),
        ("max_drawdown", "mdd", _fmt_pct),
        ("calmar", "calmar", _fmt_float),
        ("win_rate", "win", _fmt_pct),
        ("profit_factor", "pf", _fmt_float),
        ("label_ic", "ic", _fmt_float),
        ("label_rank_ic", "rank_ic", _fmt_float),
        ("direction_accuracy", "dir_acc", _fmt_pct),
        ("trade_count", "trades", _fmt_int),
    ]
    rows = []
    for model_name in ("fast", "slow"):
        metrics = result.metrics[model_name]
        rows.append([formatter(metrics.get(key)) for key, _, formatter in columns])

    headers = [header for _, header, _ in columns]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    header_line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    separator = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def _fmt_text(value) -> str:
    if value is None:
        return "-"
    return str(value)


def _fmt_int(value) -> str:
    if value is None:
        return "-"
    return f"{int(value):d}"


def _fmt_float(value) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _fmt_pct(value) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.2f}%"

