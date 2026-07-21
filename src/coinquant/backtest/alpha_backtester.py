from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from coinquant.config import settings
from coinquant.datasource.ccxt_source import timeframe_to_milliseconds
from coinquant.model.transformer import TransformerModel
from coinquant.trainer.dataset_builder import DatasetBuilder
from coinquant.trainer.model_trainer import LabelMode
from coinquant.trainer.sequence_dataset import SequenceDataset

logger = logging.getLogger(__name__)


class BacktestSplit(str, Enum):
    train = "train"
    valid = "valid"
    test = "test"


@dataclass(frozen=True)
class AlphaThresholds:
    long_entry: float
    short_entry: float
    long_exit: float
    short_exit: float


@dataclass(frozen=True)
class BacktestConfig:
    symbol: str
    period: str
    split: BacktestSplit
    thresholds: AlphaThresholds
    initial_cash: float = 10_000.0
    cash_fraction: float = 1.0 / 3.0
    fee_rate: float = 0.0004
    min_trade_cash: float = 10.0
    max_buy_count: int = 3
    min_hold_bars: int = 0
    cooldown_bars: int = 0
    liquidate_on_end: bool = True
    output_dir: str | None = None


@dataclass(frozen=True)
class BacktestResult:
    summary: dict[str, Any]
    summary_path: Path
    equity_path: Path
    orders_path: Path
    trades_path: Path


class AlphaBacktester:
    def __init__(self, config: BacktestConfig):
        self.config = config
        if self.config.initial_cash <= 0:
            raise ValueError("initial_cash must be greater than 0")
        if not 0 < self.config.cash_fraction <= 1:
            raise ValueError("cash_fraction must be in (0, 1]")
        if self.config.fee_rate < 0:
            raise ValueError("fee_rate must be non-negative")
        if self.config.min_trade_cash < 0:
            raise ValueError("min_trade_cash must be non-negative")
        if self.config.max_buy_count <= 0:
            raise ValueError("max_buy_count must be greater than 0")
        if self.config.min_hold_bars < 0:
            raise ValueError("min_hold_bars must be non-negative")
        if self.config.cooldown_bars < 0:
            raise ValueError("cooldown_bars must be non-negative")

        self.model_name = str(settings.model.name)
        self.model_config = self._load_model_config()
        self.sequence_length = int(settings.data_set.sequence_length)

    def run(self) -> BacktestResult:
        signals = self.build_signals()
        equity_df, orders_df, trades_df, summary = self._simulate(signals)
        summary.update(self._alpha_diagnostics(signals))
        summary.update(
            {
                "symbol": self.config.symbol,
                "period": self.config.period,
                "split": self.config.split.value,
                "thresholds": asdict(self.config.thresholds),
                "initial_cash": self.config.initial_cash,
                "cash_fraction": self.config.cash_fraction,
                "fee_rate": self.config.fee_rate,
                "min_trade_cash": self.config.min_trade_cash,
                "max_buy_count": self.config.max_buy_count,
                "min_hold_bars": self.config.min_hold_bars,
                "cooldown_bars": self.config.cooldown_bars,
                "liquidate_on_end": self.config.liquidate_on_end,
                "model_name": self.model_name,
                "signal_rows": int(len(signals)),
                "start_time": int(signals["execution_open_time"].iloc[0]),
                "end_time": int(signals["execution_open_time"].iloc[-1]),
            }
        )

        return self._save_result(summary, equity_df, orders_df, trades_df)

    def build_signals(self) -> pd.DataFrame:
        splits = DatasetBuilder(self.config.symbol, self.config.period).build_splits_from_db()
        split_df = splits[self.config.split.value]
        if split_df.empty:
            raise ValueError(f"{self.config.split.value} split is empty")

        short_pred = self._predict_alpha(split_df, LabelMode.short)
        long_pred = self._predict_alpha(split_df, LabelMode.long)
        signals = self._build_signal_frame(split_df, short_pred, long_pred)
        if signals.empty:
            raise ValueError("no aligned prediction rows with a next-bar execution price")
        return signals

    def _predict_alpha(self, split_df: pd.DataFrame, label_mode: LabelMode) -> pd.DataFrame:
        checkpoint_path = self._checkpoint_path(label_mode)
        metadata_path = checkpoint_path.with_suffix(".json")
        if not checkpoint_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"missing {label_mode.value} model files: {checkpoint_path} and {metadata_path}. "
                f"Train it first with: coinquant train --symbol {self.config.symbol!r} "
                f"--period {self.config.period!r} --label_mode {label_mode.value}"
            )

        metadata = self._load_metadata(metadata_path)
        feature_columns = metadata["feature_columns"]
        label_column = metadata["label_column"]
        dataset = SequenceDataset(
            split_df,
            feature_columns,
            label_column,
            self.sequence_length,
        )
        if len(dataset) == 0:
            raise ValueError(f"{label_mode.value} prediction dataset is empty")

        model = self._load_model(checkpoint_path, len(feature_columns), label_mode)
        batch_size = int(self._model_params_config().get("batch_size", 1024))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=model.use_gpu,
        )
        preds = model.predict(loader)

        return pd.DataFrame(
            {
                "source_index": dataset.end_indices,
                f"alpha_{label_mode.value}": preds.astype(float),
            }
        )

    def _build_signal_frame(
        self,
        split_df: pd.DataFrame,
        short_pred: pd.DataFrame,
        long_pred: pd.DataFrame,
    ) -> pd.DataFrame:
        aligned = short_pred.merge(long_pred, on="source_index", how="inner")
        aligned["execution_index"] = aligned["source_index"] + 1
        aligned = aligned[aligned["execution_index"] < len(split_df)].copy()
        if aligned.empty:
            return aligned

        source_rows = split_df.iloc[aligned["source_index"].to_numpy()].reset_index(drop=True)
        execution_rows = split_df.iloc[aligned["execution_index"].to_numpy()].reset_index(drop=True)
        aligned = aligned.reset_index(drop=True)
        aligned["source_open_time"] = source_rows["open_time"]
        aligned["execution_open_time"] = execution_rows["open_time"]
        aligned["execution_open"] = execution_rows["open"]
        aligned["execution_close"] = execution_rows["close"]

        for label_column in ["label_close_short", "label_close_long"]:
            if label_column in source_rows:
                aligned[label_column] = source_rows[label_column]

        return aligned.sort_values("execution_open_time").reset_index(drop=True)

    def _simulate(
        self,
        signals: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        thresholds = self.config.thresholds
        cash = float(self.config.initial_cash)
        position_qty = 0.0
        lots: list[dict[str, Any]] = []
        orders: list[dict[str, Any]] = []
        trades: list[dict[str, Any]] = []
        equity_rows: list[dict[str, Any]] = []
        total_fees = 0.0
        traded_notional = 0.0
        last_exit_index: int | None = None

        def current_cost_basis() -> float:
            return float(sum(lot["cost_basis"] for lot in lots))

        def append_order(
            row: pd.Series,
            side: str,
            price: float,
            qty: float,
            notional: float,
            fee: float,
            reason: str,
        ) -> None:
            orders.append(
                {
                    "time": int(row["execution_open_time"]),
                    "source_time": int(row.get("source_open_time", row["execution_open_time"])),
                    "side": side,
                    "price": price,
                    "quantity": qty,
                    "notional": notional,
                    "fee": fee,
                    "reason": reason,
                    "cash_after": cash,
                    "position_qty_after": position_qty,
                    "alpha_long": float(row["alpha_long"]),
                    "alpha_short": float(row["alpha_short"]),
                }
            )

        def sell_position(row: pd.Series, price: float, reason: str) -> None:
            nonlocal cash, position_qty, lots, total_fees, traded_notional, last_exit_index
            if position_qty <= 0:
                return

            qty = position_qty
            notional = qty * price
            fee = notional * self.config.fee_rate
            proceeds = notional - fee
            cost_basis = current_cost_basis()
            cash += proceeds
            total_fees += fee
            traded_notional += notional
            pnl = proceeds - cost_basis
            entry_time = int(lots[0]["entry_time"])
            entry_index = int(lots[0]["entry_index"])
            holding_bars = int(row["execution_index"] - entry_index)
            trades.append(
                {
                    "entry_time": entry_time,
                    "exit_time": int(row["execution_open_time"]),
                    "entry_price_avg": cost_basis / qty,
                    "exit_price": price,
                    "quantity": qty,
                    "buy_count": len(lots),
                    "cost_basis": cost_basis,
                    "exit_notional": notional,
                    "fees": sum(lot["fee"] for lot in lots) + fee,
                    "pnl": pnl,
                    "return": pnl / cost_basis if cost_basis > 0 else 0.0,
                    "holding_bars": holding_bars,
                    "exit_reason": reason,
                    "alpha_long": float(row["alpha_long"]),
                    "alpha_short": float(row["alpha_short"]),
                }
            )
            position_qty = 0.0
            lots = []
            last_exit_index = int(row["execution_index"])
            append_order(row, "sell", price, qty, notional, fee, reason)

        for _, row in signals.iterrows():
            price = float(row["execution_open"])
            mark_price = float(row["execution_close"])
            execution_index = int(row["execution_index"])
            holding_bars = execution_index - int(lots[0]["entry_index"]) if lots else 0
            cooling_down = (
                last_exit_index is not None
                and execution_index - last_exit_index <= self.config.cooldown_bars
            )
            buy_signal = (
                float(row["alpha_long"]) > thresholds.long_entry
                and float(row["alpha_short"]) > thresholds.short_entry
            )
            raw_sell_signal = (
                float(row["alpha_long"]) < thresholds.long_exit
                or float(row["alpha_short"]) < thresholds.short_exit
            )
            sell_signal = (
                position_qty > 0
                and holding_bars >= self.config.min_hold_bars
                and raw_sell_signal
            )

            if sell_signal:
                sell_position(row, price, "threshold")
            elif (
                buy_signal
                and not cooling_down
                and len(lots) < self.config.max_buy_count
            ):
                trade_cash = cash * self.config.cash_fraction
                if trade_cash >= self.config.min_trade_cash:
                    max_notional = cash / (1.0 + self.config.fee_rate)
                    notional = min(trade_cash, max_notional)
                    fee = notional * self.config.fee_rate
                    qty = notional / price
                    cash -= notional + fee
                    position_qty += qty
                    total_fees += fee
                    traded_notional += notional
                    lots.append(
                        {
                            "entry_time": int(row["execution_open_time"]),
                            "entry_index": int(row["execution_index"]),
                            "quantity": qty,
                            "notional": notional,
                            "fee": fee,
                            "cost_basis": notional + fee,
                        }
                    )
                    append_order(row, "buy", price, qty, notional, fee, "threshold")

            position_value = position_qty * mark_price
            equity = cash + position_value
            equity_rows.append(
                {
                    "time": int(row["execution_open_time"]),
                    "cash": cash,
                    "position_qty": position_qty,
                    "position_value": position_value,
                    "equity": equity,
                    "alpha_long": float(row["alpha_long"]),
                    "alpha_short": float(row["alpha_short"]),
                    "long_entry_signal": bool(buy_signal),
                    "exit_signal": bool(sell_signal),
                }
            )

        if position_qty > 0 and self.config.liquidate_on_end:
            last_row = signals.iloc[-1]
            sell_position(last_row, float(last_row["execution_close"]), "end")
            if equity_rows:
                equity_rows[-1]["cash"] = cash
                equity_rows[-1]["position_qty"] = 0.0
                equity_rows[-1]["position_value"] = 0.0
                equity_rows[-1]["equity"] = cash

        equity_df = pd.DataFrame(equity_rows)
        orders_df = pd.DataFrame(orders)
        trades_df = pd.DataFrame(trades)
        summary = self._build_summary(
            signals,
            equity_df,
            orders_df,
            trades_df,
            total_fees,
            traded_notional,
        )
        return equity_df, orders_df, trades_df, summary

    def _build_summary(
        self,
        signals: pd.DataFrame,
        equity_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        total_fees: float,
        traded_notional: float,
    ) -> dict[str, Any]:
        if equity_df.empty:
            final_equity = self.config.initial_cash
            equity_values = np.asarray([self.config.initial_cash], dtype=float)
        else:
            final_equity = float(equity_df["equity"].iloc[-1])
            equity_values = equity_df["equity"].to_numpy(dtype=float)

        total_return = final_equity / self.config.initial_cash - 1.0
        peak = np.maximum.accumulate(equity_values)
        drawdown = equity_values / peak - 1.0
        max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

        returns = pd.Series(equity_values).pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        timeframe_ms = timeframe_to_milliseconds(self.config.period)
        bars_per_year = (365.25 * 24 * 60 * 60 * 1000) / timeframe_ms
        sharpe = 0.0
        if len(returns) > 1 and returns.std(ddof=1) > 0:
            sharpe = float(returns.mean() / returns.std(ddof=1) * math.sqrt(bars_per_year))

        duration_ms = int(signals["execution_open_time"].iloc[-1] - signals["execution_open_time"].iloc[0])
        annual_return = 0.0
        if duration_ms > 0 and final_equity > 0:
            annual_log_return = (
                math.log(final_equity / self.config.initial_cash)
                * (365.25 * 24 * 60 * 60 * 1000)
                / duration_ms
            )
            annual_return = math.exp(annual_log_return) - 1.0 if annual_log_return < 700 else math.inf
        calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

        first_price = float(signals["execution_open"].iloc[0])
        last_price = float(signals["execution_close"].iloc[-1])
        buy_hold_return = last_price / first_price - 1.0 if first_price > 0 else 0.0

        closed_trades = len(trades_df)
        gross_profit = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()) if closed_trades else 0.0
        gross_loss = float(trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()) if closed_trades else 0.0
        profit_factor = (
            gross_profit / abs(gross_loss)
            if gross_loss < 0
            else (math.inf if gross_profit > 0 else 0.0)
        )
        win_rate = float((trades_df["pnl"] > 0).mean()) if closed_trades else 0.0
        avg_trade_return = float(trades_df["return"].mean()) if closed_trades else 0.0
        avg_holding_bars = float(trades_df["holding_bars"].mean()) if closed_trades else 0.0

        exposure = 0.0
        avg_position_fraction = 0.0
        if not equity_df.empty:
            exposure = float((equity_df["position_value"] > 0).mean())
            position_fraction = (
                equity_df["position_value"] / equity_df["equity"].replace(0, np.nan)
            ).replace([np.inf, -np.inf], np.nan)
            avg_position_fraction = float(position_fraction.fillna(0).mean())

        return {
            "final_equity": final_equity,
            "total_return": total_return,
            "annual_return": annual_return,
            "buy_hold_return": buy_hold_return,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "calmar": calmar,
            "closed_trades": int(closed_trades),
            "orders": int(len(orders_df)),
            "buy_orders": int((orders_df["side"] == "buy").sum()) if not orders_df.empty else 0,
            "sell_orders": int((orders_df["side"] == "sell").sum()) if not orders_df.empty else 0,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "avg_trade_return": avg_trade_return,
            "avg_holding_bars": avg_holding_bars,
            "total_fees": total_fees,
            "turnover": traded_notional / self.config.initial_cash,
            "exposure": exposure,
            "avg_position_fraction": avg_position_fraction,
        }

    def _alpha_diagnostics(self, signals: pd.DataFrame) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "alpha_short_mean": float(signals["alpha_short"].mean()),
            "alpha_short_std": float(signals["alpha_short"].std()),
            "alpha_long_mean": float(signals["alpha_long"].mean()),
            "alpha_long_std": float(signals["alpha_long"].std()),
        }
        for alpha_column, label_column in [
            ("alpha_short", "label_close_short"),
            ("alpha_long", "label_close_long"),
        ]:
            if label_column in signals and signals[alpha_column].std() > 0 and signals[label_column].std() > 0:
                diagnostics[f"{alpha_column}_label_corr"] = float(
                    signals[alpha_column].corr(signals[label_column])
                )
        return diagnostics

    def _save_result(
        self,
        summary: dict[str, Any],
        equity_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        trades_df: pd.DataFrame,
    ) -> BacktestResult:
        output_dir = Path(self.config.output_dir or Path(settings.path.data_path) / "backtest")
        output_dir.mkdir(parents=True, exist_ok=True)
        symbol = self.config.symbol.replace("/", "_").replace(":", "_")
        stem = f"alpha_{symbol}_{self.config.period}_{self.config.split.value}"
        summary_path = output_dir / f"{stem}_summary.json"
        equity_path = output_dir / f"{stem}_equity.csv"
        orders_path = output_dir / f"{stem}_orders.csv"
        trades_path = output_dir / f"{stem}_trades.csv"

        safe_summary = self._json_safe(summary)
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(safe_summary, file, indent=4, allow_nan=False)
        equity_df.to_csv(equity_path, index=False)
        orders_df.to_csv(orders_path, index=False)
        trades_df.to_csv(trades_path, index=False)

        return BacktestResult(
            summary=safe_summary,
            summary_path=summary_path,
            equity_path=equity_path,
            orders_path=orders_path,
            trades_path=trades_path,
        )

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

    def _load_model(self, checkpoint_path: Path, d_feat: int, label_mode: LabelMode) -> TransformerModel:
        params = dict(self._model_params_config())
        params["d_feat"] = d_feat
        model = TransformerModel(**params)
        try:
            model.load(checkpoint_path)
        except RuntimeError as exc:
            raise RuntimeError(
                f"failed to load {label_mode.value} model from {checkpoint_path}. "
                "Retrain both short and long models after the latest model/feature changes."
            ) from exc
        return model

    def _checkpoint_path(self, label_mode: LabelMode) -> Path:
        save_dir = Path(settings.path.model_path)
        symbol = self.config.symbol.replace("/", "_").replace(":", "_")
        filename = f"{self.model_name}_{symbol}_{self.config.period}_{label_mode.value}.pt"
        return save_dir / filename

    def _load_model_config(self) -> dict[str, Any]:
        config_path = Path(settings.model.config_dir) / f"{self.model_name}.json"
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
        if not isinstance(config, dict):
            raise ValueError(f"model config must be a JSON object: {config_path}")
        return config

    def _model_params_config(self) -> dict[str, Any]:
        params_config = self.model_config.get("params", {})
        if not isinstance(params_config, dict):
            raise ValueError(f"model config params must be an object for {self.model_name!r}")
        return params_config

    def _load_metadata(self, metadata_path: Path) -> dict[str, Any]:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if not isinstance(metadata.get("feature_columns"), list):
            raise ValueError(f"invalid feature_columns in metadata: {metadata_path}")
        if not isinstance(metadata.get("label_column"), str):
            raise ValueError(f"invalid label_column in metadata: {metadata_path}")
        return metadata
