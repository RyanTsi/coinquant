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
    # A longer causal context is more useful for the multi-scale DL features
    # than the original ten-bar window.  It remains configurable for cheap
    # smoke tests and ablations.
    window_size: int = 32
    basic_feature_columns: tuple[str, ...] = DEFAULT_BASIC_FEATURE_COLUMNS
    prediction_columns: tuple[str, str] = DEFAULT_PREDICTION_COLUMNS
    account_feature_columns: tuple[str, ...] = DEFAULT_ACCOUNT_FEATURE_COLUMNS
    account_history_length: int | None = None
    include_dl_features: bool = True
    dl_feature_prefix: str = "feat_"
    include_prediction_vectors: bool = True
    include_prediction_context: bool = True
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
        if self.account_history_length is not None:
            if (
                int(self.account_history_length) != self.account_history_length
                or self.account_history_length <= 0
            ):
                raise ValueError("account_history_length must be a positive integer or None")
            object.__setattr__(self, "account_history_length", int(self.account_history_length))
        if not isinstance(self.include_dl_features, bool):
            raise TypeError("include_dl_features must be a bool")
        if not isinstance(self.include_prediction_context, bool):
            raise TypeError("include_prediction_context must be a bool")
        if not isinstance(self.include_prediction_vectors, bool):
            raise TypeError("include_prediction_vectors must be a bool")
        if not isinstance(self.dl_feature_prefix, str) or not self.dl_feature_prefix:
            raise ValueError("dl_feature_prefix must be a non-empty string")
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
        account_length = self.account_history_length or self.window_size
        return self.window_size * len(self.market_columns) + account_length * len(self.account_feature_columns)


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
        self._market_columns = self._resolve_market_columns(frame)
        missing = sorted(set(self._market_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"observation frame missing columns: {missing}")
        self.frame = frame.reset_index(drop=True).copy()
        values = self.frame.loc[:, self._market_columns].to_numpy(dtype=np.float64)
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
        account_length = self.config.account_history_length or self.config.window_size
        return self.config.window_size * len(self._market_columns) + account_length * len(self.config.account_feature_columns)

    @property
    def market_columns(self) -> tuple[str, ...]:
        """The concrete market columns selected from the input frame."""

        return self._market_columns

    def market_window(self, index: int) -> np.ndarray:
        index = self._validate_index(index)
        start = index - self.config.window_size + 1
        return self._market_values[start : index + 1].copy()

    def build(self, index: int, account_features: Sequence[float] | Mapping[str, float]) -> np.ndarray:
        market = self.market_window(index).reshape(-1).astype(np.float64)
        account_length = self.config.account_history_length or self.config.window_size
        if isinstance(account_features, Mapping):
            account_row = np.asarray(
                [account_features[name] for name in self.config.account_feature_columns],
                dtype=np.float64,
            )
            account = np.repeat(account_row[None, :], account_length, axis=0)
        else:
            account_values = np.asarray(account_features, dtype=np.float64)
            if account_values.ndim == 1:
                if account_values.shape != (len(self.config.account_feature_columns),):
                    raise ValueError("account_features has an invalid shape")
                account = np.repeat(account_values[None, :], account_length, axis=0)
            elif account_values.ndim == 2:
                if account_values.shape != (
                    account_length,
                    len(self.config.account_feature_columns),
                ):
                    raise ValueError("account_features history has an invalid shape")
                account = account_values
            else:
                raise ValueError("account_features has an invalid shape")
        account = account.reshape(-1)
        result = np.concatenate((market, account))
        if not np.isfinite(result).all():
            raise ValueError("observation must contain finite values")
        if self.normalizer is not None and self.config.normalize:
            result = self.normalizer.transform(result)
        return np.clip(result, -self.config.clip_value, self.config.clip_value).astype(np.float32)

    def _resolve_market_columns(self, frame: Any) -> tuple[str, ...]:
        """Resolve the feature schema without silently dropping DL vectors.

        Existing callers can still provide the two scalar prediction columns.
        When a frame contains vector predictions, all columns using the
        ``prediction_fast_*``/``prediction_slow_*`` convention are appended.
        Likewise, the complete causal DL feature set is picked up from the
        ``feat_`` prefix, preserving the frame's training order.
        """

        columns = list(frame.columns)
        selected: list[str] = []
        for name in self.config.basic_feature_columns:
            if name not in selected:
                selected.append(name)

        configured_predictions = [
            name for name in self.config.prediction_columns if name in columns
        ]
        vector_predictions = []
        if self.config.include_prediction_vectors:
            vector_predictions = [
                name
                for name in columns
                if name.startswith("prediction_fast_") or name.startswith("prediction_slow_")
            ]
        prediction_names = configured_predictions + [
            name for name in vector_predictions if name not in configured_predictions
        ]
        # A frame with only vector columns is valid.  Custom prediction names
        # remain supported when both configured columns are present.
        has_custom_pair = len(configured_predictions) == len(self.config.prediction_columns)
        if not has_custom_pair and not any(name.startswith("prediction_fast") for name in prediction_names):
            raise ValueError("observation frame missing fast prediction columns")
        if not has_custom_pair and not any(name.startswith("prediction_slow") for name in prediction_names):
            raise ValueError("observation frame missing slow prediction columns")
        selected.extend(name for name in prediction_names if name not in selected)

        if self.config.include_dl_features:
            selected.extend(
                name
                for name in columns
                if name.startswith(self.config.dl_feature_prefix) and name not in selected
            )
        if self.config.include_prediction_context:
            selected.extend(
                name
                for name in columns
                if name.startswith("prediction_context_") and name not in selected
            )
        return tuple(selected)

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
        fast = _prediction_matrix(self.predictor_fast(result), len(result), "fast")
        slow = _prediction_matrix(self.predictor_slow(result), len(result), "slow")
        _attach_prediction_matrix(result, fast, "fast")
        _attach_prediction_matrix(result, slow, "slow")
        add_prediction_context_features(result)
        return result


def _prediction_matrix(values: Any, rows: int, mode: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 0:
        matrix = matrix.reshape(1, 1)
    elif matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    elif matrix.ndim != 2:
        raise ValueError(f"{mode} predictor must return a 1D or 2D array")
    if matrix.shape[0] != rows:
        raise ValueError("DL predictors must return one value per row")
    if matrix.shape[1] <= 0 or not np.isfinite(matrix).all():
        raise ValueError("DL predictions must contain finite values")
    return matrix


def _attach_prediction_matrix(frame: pd.DataFrame, values: np.ndarray, mode: str) -> None:
    # Keep the legacy scalar names for old models and reports.  Vector output
    # receives stable numbered columns so ObservationBuilder can discover it.
    if values.shape[1] == 1:
        frame[f"prediction_{mode}"] = values[:, 0]
    else:
        for index in range(values.shape[1]):
            frame[f"prediction_{mode}_{index}"] = values[:, index]
        # A scalar aggregate is useful for compatibility and for causal
        # context features below.
        frame[f"prediction_{mode}"] = values.mean(axis=1)


def add_prediction_context_features(frame: Any) -> Any:
    """Add causal summaries of the fast/slow DL output streams.

    These summaries make scalar checkpoints useful to RL while still allowing
    newer checkpoints to expose a genuine vector embedding.  Every operation
    is backward-looking and initial undefined values are filled with zero.
    """

    if pd is None or not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    result = frame
    required = {"prediction_fast", "prediction_slow"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError(f"missing scalar prediction columns: {missing}")
    # Model windows are unavailable at the beginning of a frame.  They are
    # still excluded by the caller's prediction ``dropna`` step; zero-filling
    # here keeps the causal context columns finite and deterministic.
    fast = result["prediction_fast"].astype(float).fillna(0.0)
    slow = result["prediction_slow"].astype(float).fillna(0.0)
    result["prediction_context_fast_delta"] = fast.diff().fillna(0.0)
    result["prediction_context_slow_delta"] = slow.diff().fillna(0.0)
    result["prediction_context_spread"] = fast - slow
    result["prediction_context_mean"] = (fast + slow) / 2.0
    for name, values in (("fast", fast), ("slow", slow), ("spread", fast - slow)):
        result[f"prediction_context_{name}_mean_4"] = values.rolling(4, min_periods=1).mean()
        result[f"prediction_context_{name}_std_8"] = values.rolling(8, min_periods=1).std().fillna(0.0)
    context_columns = [
        column for column in result.columns if column.startswith("prediction_context_")
    ]
    context_values = result[context_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(context_values).all():
        raise ValueError("prediction context features must contain finite values")
    return result


def attach_predictions(
    frame: Any,
    fast: Sequence[float],
    slow: Sequence[float],
) -> Any:
    """Attach already aligned predictions without invoking a model."""

    if pd is None or not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    result = frame.copy()
    fast_values = _prediction_matrix(fast, len(frame), "fast")
    slow_values = _prediction_matrix(slow, len(frame), "slow")
    _attach_prediction_matrix(result, fast_values, "fast")
    _attach_prediction_matrix(result, slow_values, "slow")
    add_prediction_context_features(result)
    return result
