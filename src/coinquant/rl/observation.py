"""Observation and frozen DL feature preparation for the RL environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_BASIC_FEATURE_COLUMNS = (
    "rl_open_return",
    "rl_high_return",
    "rl_low_return",
    "rl_close_return",
    "rl_log_volume_change",
)
DEFAULT_PREDICTION_COLUMNS = ("prediction_fast", "prediction_slow")
DEFAULT_ACCOUNT_FEATURE_COLUMNS = (
    "current_exposure",
    "equity_ratio",
    "drawdown",
    "rolling_volatility",
)


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class ObservationConfig:
    window_size: int = 10
    basic_feature_columns: tuple[str, ...] = DEFAULT_BASIC_FEATURE_COLUMNS
    prediction_columns: tuple[str, str] = DEFAULT_PREDICTION_COLUMNS
    account_feature_columns: tuple[str, ...] = DEFAULT_ACCOUNT_FEATURE_COLUMNS
    clip_value: float = 10.0
    normalize: bool = True

    def __post_init__(self) -> None:
        if int(self.window_size) != self.window_size or self.window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if not self.basic_feature_columns:
            raise ValueError("basic_feature_columns must not be empty")
        if len(self.prediction_columns) != 2:
            raise ValueError("prediction_columns must contain fast and slow columns")
        if not self.account_feature_columns:
            raise ValueError("account_feature_columns must not be empty")
        clip_value = _finite(self.clip_value, "clip_value")
        if clip_value <= 0:
            raise ValueError("clip_value must be greater than 0")
        object.__setattr__(self, "window_size", int(self.window_size))
        object.__setattr__(self, "clip_value", clip_value)
        object.__setattr__(self, "basic_feature_columns", tuple(self.basic_feature_columns))
        object.__setattr__(self, "prediction_columns", tuple(self.prediction_columns))
        object.__setattr__(self, "account_feature_columns", tuple(self.account_feature_columns))

    @property
    def market_columns(self) -> tuple[str, ...]:
        return self.basic_feature_columns + self.prediction_columns

    @property
    def observation_size(self) -> int:
        return self.window_size * len(self.market_columns) + len(self.account_feature_columns)


def build_basic_features(frame: Any) -> Any:
    """Add the five causal OHLCV features used by the RL observation.

    The input is copied and never modified in place.  Only the current row and
    the immediately preceding close/volume are used.
    """

    if pd is None or not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing}")

    result = frame.copy()
    open_price = result["open"].astype(float)
    high = result["high"].astype(float)
    low = result["low"].astype(float)
    close = result["close"].astype(float)
    volume = result["volume"].astype(float)
    numeric = np.column_stack((open_price, high, low, close, volume))
    if not np.isfinite(numeric).all():
        raise ValueError("OHLCV values must be finite")
    if (numeric[:, :4] <= 0).any():
        raise ValueError("OHLC prices must be positive")
    if (numeric[:, 4] < 0).any():
        raise ValueError("volume must be non-negative")
    if (low.to_numpy() > np.minimum(open_price.to_numpy(), close.to_numpy())).any() or (
        high.to_numpy() < np.maximum(open_price.to_numpy(), close.to_numpy())
    ).any() or (low.to_numpy() > high.to_numpy()).any():
        raise ValueError("OHLC prices must satisfy low <= min(open, close) <= max(open, close) <= high")
    previous_close = close.shift(1)
    previous_volume = volume.shift(1)
    result["rl_open_return"] = (open_price / previous_close - 1.0).replace([np.inf, -np.inf], np.nan)
    result["rl_high_return"] = (high / open_price - 1.0).replace([np.inf, -np.inf], np.nan)
    result["rl_low_return"] = (low / open_price - 1.0).replace([np.inf, -np.inf], np.nan)
    result["rl_close_return"] = (close / open_price - 1.0).replace([np.inf, -np.inf], np.nan)
    result["rl_log_volume_change"] = (
        np.log1p(volume) - np.log1p(previous_volume)
    ).replace([np.inf, -np.inf], np.nan)
    result["rl_open_return"] = result["rl_open_return"].fillna(0.0)
    result["rl_log_volume_change"] = result["rl_log_volume_change"].fillna(0.0)
    return result

class ObservationNormalizer:
    """Train-only affine normalizer for flattened observations."""

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = _finite(epsilon, "epsilon")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be greater than 0")
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    def fit(self, observations: np.ndarray) -> "ObservationNormalizer":
        values = np.asarray(observations, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("observations must be a non-empty 2D array")
        if not np.isfinite(values).all():
            raise ValueError("observations must contain finite values")
        self.mean = values.mean(axis=0)
        self.std = values.std(axis=0)
        self.std = np.maximum(self.std, self.epsilon)
        return self

    def transform(self, observation: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("normalizer has not been fitted")
        values = np.asarray(observation, dtype=np.float64)
        if values.shape != self.mean.shape:  # type: ignore[union-attr]
            raise ValueError(f"observation shape mismatch: expected {self.mean.shape}, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("observation must contain finite values")
        return ((values - self.mean) / self.std).astype(np.float32)  # type: ignore[operator]

    def state_dict(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("normalizer has not been fitted")
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "epsilon": self.epsilon}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ObservationNormalizer":
        normalizer = cls(float(state.get("epsilon", 1e-8)))
        normalizer.mean = np.asarray(state["mean"], dtype=np.float64)
        normalizer.std = np.asarray(state["std"], dtype=np.float64)
        if normalizer.mean.ndim != 1 or normalizer.std.shape != normalizer.mean.shape:
            raise ValueError("invalid normalizer state")
        if not np.isfinite(normalizer.mean).all() or not np.isfinite(normalizer.std).all() or (normalizer.std <= 0).any():
            raise ValueError("invalid normalizer state")
        return normalizer


class ObservationBuilder:
    """Build a causal rolling observation from a prepared feature frame."""

    def __init__(
        self,
        frame: Any,
        config: ObservationConfig | None = None,
        normalizer: ObservationNormalizer | None = None,
    ):
        if pd is None or not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        self.config = config or ObservationConfig()
        missing = sorted(set(self.config.market_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"observation frame missing columns: {missing}")
        self.frame = frame.reset_index(drop=True).copy()
        values = self.frame.loc[:, self.config.market_columns].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("observation market columns must contain finite values")
        self._market_values = values.astype(np.float32)
        self.normalizer = normalizer
        self._valid_indices = np.arange(self.config.window_size - 1, len(self.frame), dtype=np.int64)

    @property
    def valid_indices(self) -> np.ndarray:
        return self._valid_indices.copy()

    @property
    def observation_size(self) -> int:
        return self.config.observation_size

    def market_window(self, index: int) -> np.ndarray:
        index = self._validate_index(index)
        start = index - self.config.window_size + 1
        return self._market_values[start : index + 1].copy()

    def build(self, index: int, account_features: Sequence[float] | Mapping[str, float]) -> np.ndarray:
        market = self.market_window(index).reshape(-1).astype(np.float64)
        if isinstance(account_features, Mapping):
            account = np.asarray(
                [account_features[name] for name in self.config.account_feature_columns],
                dtype=np.float64,
            )
        else:
            account = np.asarray(account_features, dtype=np.float64)
        if account.shape != (len(self.config.account_feature_columns),):
            raise ValueError("account_features has an invalid shape")
        result = np.concatenate((market, account))
        if not np.isfinite(result).all():
            raise ValueError("observation must contain finite values")
        if self.normalizer is not None and self.config.normalize:
            result = self.normalizer.transform(result)
        return np.clip(result, -self.config.clip_value, self.config.clip_value).astype(np.float32)

    def _validate_index(self, index: int) -> int:
        if int(index) != index:
            raise TypeError("index must be an integer")
        index = int(index)
        if index < self.config.window_size - 1 or index >= len(self.frame):
            raise IndexError("index does not have a complete observation window")
        return index


class DLFeatureProvider:
    """Small adapter used to attach frozen fast/slow predictions."""

    def __init__(self, predictor_fast: Callable[[Any], Any], predictor_slow: Callable[[Any], Any]):
        if not callable(predictor_fast) or not callable(predictor_slow):
            raise TypeError("both predictors must be callable")
        self.predictor_fast = predictor_fast
        self.predictor_slow = predictor_slow

    def transform(self, frame: Any) -> Any:
        if pd is None or not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        result = frame.copy()
        fast = np.asarray(self.predictor_fast(result), dtype=np.float64).reshape(-1)
        slow = np.asarray(self.predictor_slow(result), dtype=np.float64).reshape(-1)
        if len(fast) != len(result) or len(slow) != len(result):
            raise ValueError("DL predictors must return one value per row")
        if not np.isfinite(fast).all() or not np.isfinite(slow).all():
            raise ValueError("DL predictions must contain finite values")
        result["prediction_fast"] = fast
        result["prediction_slow"] = slow
        return result


def attach_predictions(
    frame: Any,
    fast: Sequence[float],
    slow: Sequence[float],
) -> Any:
    """Attach already aligned predictions without invoking a model."""

    if pd is None or not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if len(fast) != len(frame) or len(slow) != len(frame):
        raise ValueError("prediction lengths must match frame length")
    result = frame.copy()
    result["prediction_fast"] = np.asarray(fast, dtype=np.float64)
    result["prediction_slow"] = np.asarray(slow, dtype=np.float64)
    if not np.isfinite(result[["prediction_fast", "prediction_slow"]].to_numpy()).all():
        raise ValueError("predictions must contain finite values")
    return result
