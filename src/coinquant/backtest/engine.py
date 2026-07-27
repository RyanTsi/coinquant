from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from coinquant.config import settings
from coinquant.backtest.strategy import (
    build_turning_point_signal,
    carry_position_until_next_action,
    shift_signal_to_next_open,
)
from coinquant.trainer.dataset_builder import DatasetBuilder
from coinquant.trainer.model_trainer import LabelMode, MODEL_REGISTRY
from coinquant.trainer.sequence_dataset import SequenceDataset


@dataclass(frozen=True)
class ModelPrediction:
    label_mode: LabelMode
    label_column: str
    pred_column: str
    end_indices: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class BacktestResult:
    symbol: str
    period: str
    threshold: float
    fee_rate: float
    initial_cash: float
    rows: pd.DataFrame
    metrics: dict[str, dict[str, float | int | str | None]]


class BacktestEngine:
    """Run turning-point backtests for the fast and slow trained models."""

    def __init__(
        self,
        symbol: str,
        period: str,
        threshold: float = 0.0,
        fee_rate: float = 0.0004,
        initial_cash: float = 1.0,
        model_dir: str | Path | None = None,
    ):
        if threshold < 0:
            raise ValueError("threshold must be greater than or equal to 0")
        if fee_rate < 0:
            raise ValueError("fee_rate must be greater than or equal to 0")
        if initial_cash <= 0:
            raise ValueError("initial_cash must be greater than 0")

        self.symbol = symbol
        self.period = period
        self.threshold = threshold
        self.fee_rate = fee_rate
        self.initial_cash = initial_cash
        self.model_dir = Path(model_dir or settings.path.model_path)

    def run(self) -> BacktestResult:
        splits = DatasetBuilder(self.symbol, self.period).build_splits_from_db()
        test_df = splits["test"]
        if test_df.empty:
            raise ValueError(f"test dataset is empty for {self.symbol} {self.period}")

        predictions = [
            self._predict_model(LabelMode.fast, test_df),
            self._predict_model(LabelMode.slow, test_df),
        ]
        rows = self._merge_predictions(test_df, predictions)
        rows = self._attach_trading_results(rows)
        metrics = {
            "fast": self._calculate_metrics(rows, LabelMode.fast),
            "slow": self._calculate_metrics(rows, LabelMode.slow),
        }
        return BacktestResult(
            symbol=self.symbol,
            period=self.period,
            threshold=self.threshold,
            fee_rate=self.fee_rate,
            initial_cash=self.initial_cash,
            rows=rows,
            metrics=metrics,
        )

    def _predict_model(self, label_mode: LabelMode, test_df: pd.DataFrame) -> ModelPrediction:
        checkpoint_path = self._checkpoint_path(label_mode)
        metadata_path = checkpoint_path.with_suffix(".json")
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"model checkpoint not found: {checkpoint_path}; train {label_mode.value} first"
            )
        if not metadata_path.exists():
            raise FileNotFoundError(f"model metadata not found: {metadata_path}")

        metadata = self._load_json(metadata_path)
        model_name = str(metadata.get("model_name", settings.model.name))
        feature_columns = list(metadata.get("feature_columns", []))
        label_column = str(metadata.get("label_column", f"label_close_{label_mode.value}"))
        if not feature_columns:
            raise ValueError(f"metadata has no feature_columns: {metadata_path}")

        missing_columns = sorted(set(feature_columns + [label_column]) - set(test_df.columns))
        if missing_columns:
            raise ValueError(f"test dataset missing columns for {label_mode.value}: {missing_columns}")

        sequence_length = int(_get_setting(settings.data_set, "sequence_length", 128))
        dataset = SequenceDataset(test_df, feature_columns, label_column, sequence_length)
        if len(dataset) == 0:
            raise ValueError(f"test dataset has no valid sequences for {label_mode.value}")

        has_saved_model_params = isinstance(metadata.get("model_params"), dict)
        model_params = self._metadata_model_params(metadata, model_name, len(feature_columns))
        if not has_saved_model_params and model_name == "transformer":
            model_params = self._infer_transformer_model_params(checkpoint_path, model_params)
        model = self._build_model(model_name, model_params)
        try:
            model.load(checkpoint_path)
        except RuntimeError:
            if model_name != "transformer":
                raise
            model_params = self._infer_transformer_model_params(checkpoint_path, model_params)
            model = self._build_model(model_name, model_params)
            model.load(checkpoint_path)
        batch_size = int(model_params.get("batch_size", 1024))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        values = np.asarray(model.predict(loader), dtype=np.float64).reshape(-1)

        return ModelPrediction(
            label_mode=label_mode,
            label_column=label_column,
            pred_column=f"pred_{label_mode.value}",
            end_indices=dataset.end_indices,
            values=values,
        )

    def _merge_predictions(
        self,
        test_df: pd.DataFrame,
        predictions: list[ModelPrediction],
    ) -> pd.DataFrame:
        base_columns = [
            "symbol",
            "period",
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "label_z_score_close_fast",
            "label_z_score_close_slow",
        ]
        columns = [column for column in base_columns if column in test_df.columns]
        rows = test_df.loc[:, columns].copy()

        for prediction in predictions:
            rows[prediction.pred_column] = np.nan
            rows.loc[prediction.end_indices, prediction.pred_column] = prediction.values

        pred_columns = [prediction.pred_column for prediction in predictions]
        rows = rows.dropna(subset=pred_columns).sort_values("open_time").reset_index(drop=True)
        if rows.empty:
            raise ValueError("no overlapping fast/slow predictions on test dataset")

        rows["open_datetime"] = (
            pd.to_datetime(rows["open_time"], unit="ms", utc=True)
            .dt.strftime("%Y-%m-%d %H:%M:%S")
        )
        return rows

    def _attach_trading_results(self, rows: pd.DataFrame) -> pd.DataFrame:
        rows = rows.copy()
        rows["next_return"] = rows["open"].shift(-1) / rows["open"] - 1.0
        rows = rows.iloc[:-1].copy()
        if rows.empty:
            raise ValueError("not enough rows to calculate next-bar backtest returns")

        for label_mode in (LabelMode.fast, LabelMode.slow):
            mode = label_mode.value
            pred = rows[f"pred_{mode}"].to_numpy(dtype=np.float64)
            prev_pred, pred_delta, signal = build_turning_point_signal(pred, self.threshold)
            action = shift_signal_to_next_open(signal)
            position = carry_position_until_next_action(action)
            previous_position = np.concatenate(([0], position[:-1]))
            turnover = np.abs(position - previous_position).astype(np.float64)
            gross_return = position * rows["next_return"].to_numpy(dtype=np.float64)
            cost = turnover * self.fee_rate
            net_return = gross_return - cost
            equity = self.initial_cash * np.cumprod(1.0 + net_return)
            running_max = np.maximum.accumulate(equity)
            drawdown = equity / running_max - 1.0

            rows[f"position_{mode}"] = position
            rows[f"signal_{mode}"] = signal
            rows[f"action_{mode}"] = action
            rows[f"pred_prev_{mode}"] = prev_pred
            rows[f"pred_delta_{mode}"] = pred_delta
            rows[f"turnover_{mode}"] = turnover
            rows[f"gross_return_{mode}"] = gross_return
            rows[f"net_return_{mode}"] = net_return
            rows[f"equity_{mode}"] = equity
            rows[f"drawdown_{mode}"] = drawdown

        return rows

    def _calculate_metrics(
        self,
        rows: pd.DataFrame,
        label_mode: LabelMode,
    ) -> dict[str, float | int | str | None]:
        mode = label_mode.value
        pred = rows[f"pred_{mode}"]
        label = rows[f"label_z_score_close_{mode}"]
        returns = rows[f"net_return_{mode}"].to_numpy(dtype=np.float64)
        gross_returns = rows[f"gross_return_{mode}"].to_numpy(dtype=np.float64)
        equity = rows[f"equity_{mode}"].to_numpy(dtype=np.float64)
        position = rows[f"position_{mode}"].to_numpy(dtype=np.float64)
        turnover = rows[f"turnover_{mode}"].to_numpy(dtype=np.float64)
        drawdown = rows[f"drawdown_{mode}"].to_numpy(dtype=np.float64)

        total_return = equity[-1] / self.initial_cash - 1.0
        bars_per_year = _bars_per_year(self.period)
        annual_return = _annualized_return(total_return, len(returns), bars_per_year)
        volatility = _annualized_volatility(returns, bars_per_year)
        sharpe = _safe_divide(float(np.mean(returns)) * math.sqrt(bars_per_year), float(np.std(returns, ddof=1)))
        max_drawdown = float(np.min(drawdown))
        calmar = _safe_divide(annual_return, abs(max_drawdown))
        active_returns = returns[position != 0]
        win_rate = _mean_or_none(active_returns > 0)
        profit_factor = _profit_factor(active_returns)
        label_ic = _corr(pred, label)
        label_rank_ic = _rank_corr(pred, label)
        next_return_pct = rows["next_return"] * 100.0
        next_return_ic = _corr(pred, next_return_pct)
        next_return_rank_ic = _rank_corr(pred, next_return_pct)
        direction_accuracy = _direction_accuracy(pred, label)

        return {
            "model": mode,
            "rows": int(len(rows)),
            "start_time": str(rows["open_datetime"].iloc[0]),
            "end_time": str(rows["open_datetime"].iloc[-1]),
            "threshold": float(self.threshold),
            "fee_rate": float(self.fee_rate),
            "total_return": _finite_or_none(total_return),
            "annual_return": _finite_or_none(annual_return),
            "annual_volatility": _finite_or_none(volatility),
            "sharpe": _finite_or_none(sharpe),
            "max_drawdown": _finite_or_none(max_drawdown),
            "calmar": _finite_or_none(calmar),
            "win_rate": _finite_or_none(win_rate),
            "profit_factor": _finite_or_none(profit_factor),
            "exposure": _finite_or_none(float(np.mean(np.abs(position)))),
            "long_ratio": _finite_or_none(float(np.mean(position > 0))),
            "short_ratio": _finite_or_none(float(np.mean(position < 0))),
            "total_turnover": _finite_or_none(float(np.sum(turnover))),
            "avg_turnover": _finite_or_none(float(np.mean(turnover))),
            "trade_count": int(np.count_nonzero(turnover)),
            "avg_holding_bars": _finite_or_none(_average_holding_bars(position)),
            "mean_bar_return": _finite_or_none(float(np.mean(returns))),
            "gross_total_return": _finite_or_none(float(np.prod(1.0 + gross_returns) - 1.0)),
            "label_ic": _finite_or_none(label_ic),
            "label_rank_ic": _finite_or_none(label_rank_ic),
            "next_return_ic": _finite_or_none(next_return_ic),
            "next_return_rank_ic": _finite_or_none(next_return_rank_ic),
            "direction_accuracy": _finite_or_none(direction_accuracy),
            "prediction_mean": _finite_or_none(float(pred.mean())),
            "prediction_std": _finite_or_none(float(pred.std())),
            "label_mean": _finite_or_none(float(label.mean())),
            "label_std": _finite_or_none(float(label.std())),
        }

    def _checkpoint_path(self, label_mode: LabelMode) -> Path:
        model_name = str(_get_setting(settings.model, "name", "transformer"))
        symbol = self.symbol.replace("/", "_").replace(":", "_")
        return self.model_dir / f"{model_name}_{symbol}_{self.period}_{label_mode.value}.pt"

    def _metadata_model_params(
        self,
        metadata: dict[str, Any],
        model_name: str,
        d_feat: int,
    ) -> dict[str, Any]:
        params = metadata.get("model_params")
        if not isinstance(params, dict):
            params = self._model_params_config(model_name)
        params = dict(params)
        params["d_feat"] = d_feat
        return params

    def _build_model(self, model_name: str, model_params: dict[str, Any]):
        model_class = MODEL_REGISTRY.get(model_name)
        if model_class is None:
            supported = ", ".join(sorted(MODEL_REGISTRY))
            raise ValueError(f"unsupported model {model_name!r}; supported models: {supported}")

        return model_class(**model_params)

    def _infer_transformer_model_params(
        self,
        checkpoint_path: Path,
        base_params: dict[str, Any],
    ) -> dict[str, Any]:
        state = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError(f"checkpoint state must be a dict: {checkpoint_path}")

        params = dict(base_params)
        if "feature_layer.0.weight" in state:
            params["d_linear"] = int(state["feature_layer.0.weight"].shape[0])
        if "feature_layer.2.weight" in state:
            params["d_model"] = int(state["feature_layer.2.weight"].shape[0])
        if "transformer_encoder.layers.0.linear1.weight" in state:
            params["dim_feedforward"] = int(
                state["transformer_encoder.layers.0.linear1.weight"].shape[0]
            )

        layer_indices = set()
        for key in state:
            match = re.match(r"transformer_encoder\.layers\.(\d+)\.", key)
            if match:
                layer_indices.add(int(match.group(1)))
        if layer_indices:
            params["num_layers"] = max(layer_indices) + 1

        return params

    def _model_params_config(self, model_name: str) -> dict[str, Any]:
        model_config = self._load_model_config(model_name)
        params_config = model_config.get("params", {})
        if not isinstance(params_config, dict):
            raise ValueError(f"model config params must be an object for {model_name!r}")
        return params_config

    def _load_model_config(self, model_name: str) -> dict[str, Any]:
        config_path = Path(settings.model.config_dir) / f"{model_name}.json"
        if not config_path.exists():
            raise FileNotFoundError(f"model config not found: {config_path}")
        return self._load_json(config_path)

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            config = json.load(file)
        if not isinstance(config, dict):
            raise ValueError(f"JSON file must contain an object: {path}")
        return config


def _get_setting(section, key: str, default=None):
    if hasattr(section, "get"):
        return section.get(key, default)
    return getattr(section, key, default)


def _bars_per_year(period: str) -> float:
    match = re.fullmatch(r"(\d+)([mhdw])", period.strip().lower())
    if not match:
        return 365.0

    value = int(match.group(1))
    unit = match.group(2)
    minutes_by_unit = {
        "m": 1,
        "h": 60,
        "d": 60 * 24,
        "w": 60 * 24 * 7,
    }
    minutes = value * minutes_by_unit[unit]
    return 365.0 * 24.0 * 60.0 / minutes


def _annualized_return(total_return: float, rows: int, bars_per_year: float) -> float | None:
    if rows <= 0 or total_return <= -1:
        return None
    return (1.0 + total_return) ** (bars_per_year / rows) - 1.0


def _annualized_volatility(returns: np.ndarray, bars_per_year: float) -> float | None:
    if len(returns) < 2:
        return None
    return float(np.std(returns, ddof=1) * math.sqrt(bars_per_year))


def _corr(left: pd.Series, right: pd.Series) -> float | None:
    value = left.corr(right)
    return _finite_or_none(value)


def _rank_corr(left: pd.Series, right: pd.Series) -> float | None:
    value = left.corr(right, method="spearman")
    return _finite_or_none(value)


def _direction_accuracy(pred: pd.Series, label: pd.Series) -> float | None:
    pred_sign = np.sign(pred.to_numpy(dtype=np.float64))
    label_sign = np.sign(label.to_numpy(dtype=np.float64))
    mask = (pred_sign != 0) & (label_sign != 0)
    if not mask.any():
        return None
    return float(np.mean(pred_sign[mask] == label_sign[mask]))


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _mean_or_none(values: np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    return float(np.mean(values))


def _profit_factor(returns: np.ndarray) -> float | None:
    if len(returns) == 0:
        return None
    gains = float(np.sum(returns[returns > 0]))
    losses = float(np.sum(returns[returns < 0]))
    if losses == 0:
        return None if gains == 0 else math.inf
    return gains / abs(losses)


def _average_holding_bars(position: np.ndarray) -> float | None:
    lengths: list[int] = []
    current_position = 0.0
    current_length = 0

    for value in position:
        if value == 0:
            if current_position != 0:
                lengths.append(current_length)
            current_position = 0.0
            current_length = 0
            continue

        if value == current_position:
            current_length += 1
        else:
            if current_position != 0:
                lengths.append(current_length)
            current_position = value
            current_length = 1

    if current_position != 0:
        lengths.append(current_length)

    if not lengths:
        return None
    return float(np.mean(lengths))


def _finite_or_none(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return None
        return float(value)
    return value
