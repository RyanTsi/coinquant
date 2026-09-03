import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from coinquant.config import settings
from coinquant.model.transformer import TransformerModel
from coinquant.trainer.dataset_builder import DatasetBuilder


class _FeatureWindowDataset(Dataset):
    """Build prediction windows lazily and ignore rows with invalid features."""

    def __init__(self, features: np.ndarray, sequence_length: int):
        self._features = np.ascontiguousarray(features, dtype=np.float32)
        self._sequence_length = sequence_length

        if len(self._features) < sequence_length:
            self._end_indices = np.empty(0, dtype=np.int64)
            return

        finite_rows = np.isfinite(self._features).all(axis=1)
        invalid_prefix = np.concatenate(
            ([0], np.cumsum(~finite_rows, dtype=np.int64))
        )
        invalid_counts = (
            invalid_prefix[sequence_length:] - invalid_prefix[:-sequence_length]
        )
        self._end_indices = (
            np.flatnonzero(invalid_counts == 0) + sequence_length - 1
        )

    def __len__(self) -> int:
        return len(self._end_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        end_idx = self._end_indices[index]
        start_idx = end_idx - self._sequence_length + 1
        feature = self._features[start_idx : end_idx + 1]
        return torch.from_numpy(feature), torch.tensor(0.0, dtype=torch.float32)


def predict_latest(
    feature_frame,
    checkpoint_path: str | Path,
) -> float:
    checkpoint_path = Path(checkpoint_path)
    metadata_path = checkpoint_path.with_suffix(".json")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = list(metadata["feature_columns"])
    model_params = dict(metadata["model_params"])
    sequence_length = int(
        metadata.get(
            "sequence_length",
            settings.data_set.sequence_length,
        )
    )

    missing = sorted(set(feature_columns) - set(feature_frame.columns))
    if missing:
        raise ValueError(f"缺少模型特征列: {missing}")

    # 最后一行就是本次预测对应的 K 线。
    window = feature_frame.sort_values("open_time").tail(sequence_length)
    if len(window) != sequence_length:
        raise ValueError(
            f"至少需要 {sequence_length} 根有效 K 线，当前只有 {len(window)} 根"
        )

    features = np.array(
        window[feature_columns].to_numpy(dtype=np.float32),
        dtype=np.float32,
        copy=True,
    )
    if not np.isfinite(features).all():
        raise ValueError("模型输入包含 NaN 或 inf")

    # Transformer 输入形状：[batch, sequence, feature]
    feature_tensor = torch.from_numpy(features).unsqueeze(0)
    dummy_label = torch.zeros(1, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(feature_tensor, dummy_label),
        batch_size=1,
        shuffle=False,
    )

    model = TransformerModel(**model_params)
    model.load(checkpoint_path)

    predictions = model.predict(loader)
    return float(predictions[0])


def predict_batch(
    feature_frame,
    checkpoint_path: str | Path,
    batch_size: int = 256,
) -> np.ndarray:
    checkpoint_path = Path(checkpoint_path)
    metadata_path = checkpoint_path.with_suffix(".json")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = list(metadata["feature_columns"])
    model_params = dict(metadata["model_params"])
    sequence_length = int(
        metadata.get(
            "sequence_length",
            settings.data_set.sequence_length,
        )
    )

    missing = sorted(set(feature_columns) - set(feature_frame.columns))
    if missing:
        raise ValueError(f"缺少模型特征列: {missing}")

    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    # 每个完整且特征有效的滑动区间都对应一个预测值。
    sorted_frame = feature_frame.sort_values("open_time")
    features = sorted_frame[feature_columns].to_numpy(dtype=np.float32)
    dataset = _FeatureWindowDataset(features, sequence_length)
    if len(dataset) == 0:
        return np.empty(0, dtype=np.float32)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model = TransformerModel(**model_params)
    model.load(checkpoint_path)

    predictions = np.asarray(model.predict(loader)).reshape(-1)
    if len(predictions) != len(dataset):
        raise ValueError(
            f"模型返回了 {len(predictions)} 个预测值，但可预测区间有 {len(dataset)} 个"
        )
    return predictions


def test():
    symbol = "BTC/USDT"
    period = "1h"

    # 构造与训练时完全相同的特征。
    feature_frame = DatasetBuilder(symbol, period).build_from_db()

    fast_prediction = predict_latest(
        feature_frame,
        "data/model/transformer_BTC_USDT_1h_fast.pt",
    )

    slow_prediction = predict_latest(
        feature_frame,
        "data/model/transformer_BTC_USDT_1h_slow.pt",
    )

    prediction_time = feature_frame.sort_values("open_time").iloc[-1]["open_time"]

    print("prediction_time:", prediction_time)
    print("fast_prediction:", fast_prediction)
    print("slow_prediction:", slow_prediction)
