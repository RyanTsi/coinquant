from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from coinquant.backtest.alpha_backtester import (
    AlphaBacktester,
    AlphaThresholds,
    BacktestConfig,
    BacktestResult,
)
from coinquant.config import settings


class GridSortKey(str, Enum):
    score = "score"
    total_return = "total_return"
    sharpe = "sharpe"
    calmar = "calmar"
    profit_factor = "profit_factor"
    max_drawdown = "max_drawdown"


@dataclass(frozen=True)
class AlphaGridSearchConfig:
    backtest_config: BacktestConfig
    grid_size: int = 4
    entry_min_quantile: float = 0.70
    entry_max_quantile: float = 0.95
    exit_min_quantile: float = 0.01
    exit_max_quantile: float = 0.30
    top_k: int = 20
    min_trades: int = 1
    max_trades: int | None = None
    max_turnover: float | None = None
    sort_by: GridSortKey = GridSortKey.score
    drawdown_weight: float = 0.5
    turnover_weight: float = 0.0001
    trade_weight: float = 0.0001
    output_dir: str | None = None


@dataclass(frozen=True)
class AlphaGridSearchResult:
    best: dict[str, Any]
    results_path: Path
    summary_path: Path


class AlphaGridSearcher:
    def __init__(self, config: AlphaGridSearchConfig):
        self.config = config
        if self.config.grid_size <= 0:
            raise ValueError("grid_size must be greater than 0")
        if self.config.top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if self.config.min_trades < 0:
            raise ValueError("min_trades must be non-negative")
        self._validate_quantile("entry_min_quantile", self.config.entry_min_quantile)
        self._validate_quantile("entry_max_quantile", self.config.entry_max_quantile)
        self._validate_quantile("exit_min_quantile", self.config.exit_min_quantile)
        self._validate_quantile("exit_max_quantile", self.config.exit_max_quantile)
        if self.config.entry_min_quantile > self.config.entry_max_quantile:
            raise ValueError("entry_min_quantile must be <= entry_max_quantile")
        if self.config.exit_min_quantile > self.config.exit_max_quantile:
            raise ValueError("exit_min_quantile must be <= exit_max_quantile")

    def run(self) -> AlphaGridSearchResult:
        base_backtester = AlphaBacktester(self.config.backtest_config)
        signals = base_backtester.build_signals()
        thresholds = self._build_threshold_grid(signals)
        rows: list[dict[str, Any]] = []

        total = (
            len(thresholds["long_entry"])
            * len(thresholds["short_entry"])
            * len(thresholds["long_exit"])
            * len(thresholds["short_exit"])
        )
        for index, (a_q, a, b_q, b, c_q, c, d_q, d) in enumerate(self._iter_thresholds(thresholds), start=1):
            if index == 1 or index % 100 == 0:
                print(f"grid search {index}/{total}")

            run_config = replace(
                self.config.backtest_config,
                thresholds=AlphaThresholds(
                    long_entry=a,
                    short_entry=b,
                    long_exit=c,
                    short_exit=d,
                ),
            )
            runner = AlphaBacktester(run_config)
            _, _, _, summary = runner._simulate(signals)
            if not self._passes_filters(summary):
                continue

            summary.update(
                {
                    "long_entry": a,
                    "short_entry": b,
                    "long_exit": c,
                    "short_exit": d,
                    "long_entry_quantile": a_q,
                    "short_entry_quantile": b_q,
                    "long_exit_quantile": c_q,
                    "short_exit_quantile": d_q,
                    "score": self._score(summary),
                }
            )
            rows.append(summary)

        if not rows:
            raise ValueError("grid search produced no rows after filters")

        results = pd.DataFrame(rows)
        sort_column = self.config.sort_by.value
        results = results.sort_values(sort_column, ascending=False).reset_index(drop=True)
        return self._save_result(results, thresholds, signals)

    def _build_threshold_grid(self, signals: pd.DataFrame) -> dict[str, list[tuple[float, float]]]:
        entry_quantiles = np.linspace(
            self.config.entry_min_quantile,
            self.config.entry_max_quantile,
            self.config.grid_size,
        )
        exit_quantiles = np.linspace(
            self.config.exit_min_quantile,
            self.config.exit_max_quantile,
            self.config.grid_size,
        )

        return {
            "long_entry": self._quantile_values(signals["alpha_long"], entry_quantiles),
            "short_entry": self._quantile_values(signals["alpha_short"], entry_quantiles),
            "long_exit": self._quantile_values(signals["alpha_long"], exit_quantiles),
            "short_exit": self._quantile_values(signals["alpha_short"], exit_quantiles),
        }

    def _iter_thresholds(self, thresholds: dict[str, list[tuple[float, float]]]):
        for a_q, a in thresholds["long_entry"]:
            for b_q, b in thresholds["short_entry"]:
                for c_q, c in thresholds["long_exit"]:
                    for d_q, d in thresholds["short_exit"]:
                        yield a_q, a, b_q, b, c_q, c, d_q, d

    def _quantile_values(self, series: pd.Series, quantiles: np.ndarray) -> list[tuple[float, float]]:
        values = []
        for quantile in quantiles:
            values.append((float(quantile), float(series.quantile(float(quantile)))))
        return values

    def _passes_filters(self, summary: dict[str, Any]) -> bool:
        if int(summary["closed_trades"]) < self.config.min_trades:
            return False
        if self.config.max_trades is not None and int(summary["closed_trades"]) > self.config.max_trades:
            return False
        if self.config.max_turnover is not None and float(summary["turnover"]) > self.config.max_turnover:
            return False
        return True

    def _score(self, summary: dict[str, Any]) -> float:
        return (
            float(summary["total_return"])
            - self.config.drawdown_weight * abs(float(summary["max_drawdown"]))
            - self.config.turnover_weight * float(summary["turnover"])
            - self.config.trade_weight * int(summary["closed_trades"])
        )

    def _save_result(
        self,
        results: pd.DataFrame,
        thresholds: dict[str, list[tuple[float, float]]],
        signals: pd.DataFrame,
    ) -> AlphaGridSearchResult:
        output_dir = Path(self.config.output_dir or Path(settings.path.data_path) / "backtest")
        output_dir.mkdir(parents=True, exist_ok=True)
        base = self.config.backtest_config
        symbol = base.symbol.replace("/", "_").replace(":", "_")
        stem = f"alpha_grid_{symbol}_{base.period}_{base.split.value}"
        results_path = output_dir / f"{stem}.csv"
        summary_path = output_dir / f"{stem}_summary.json"
        results.to_csv(results_path, index=False)

        best = self._json_safe(results.iloc[0].to_dict())
        summary = {
            "best": best,
            "top_k": self._json_safe(results.head(self.config.top_k).to_dict(orient="records")),
            "grid_config": self._json_safe(asdict(self.config)),
            "threshold_grid": self._json_safe(thresholds),
            "signal_rows": int(len(signals)),
            "result_rows": int(len(results)),
            "results_path": str(results_path),
        }
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=4, allow_nan=False)

        return AlphaGridSearchResult(
            best=best,
            results_path=results_path,
            summary_path=summary_path,
        )

    def _validate_quantile(self, name: str, value: float) -> None:
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")

    def _json_safe(self, value):
        if isinstance(value, dict):
            return {key: self._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [self._json_safe(item) for item in value]
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            value = float(value)
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        return value
