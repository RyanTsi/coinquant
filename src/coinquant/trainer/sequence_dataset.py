import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        label_column: str,
        sequence_length: int,
    ):
        if sequence_length <= 0:
            raise ValueError("sequence_length must be greater than 0")

        missing_columns = set(feature_columns + [label_column]) - set(df.columns)
        if missing_columns:
            raise ValueError(f"missing columns: {sorted(missing_columns)}")

        self._sequence_length = sequence_length
        self._features = df[feature_columns].to_numpy(dtype=np.float32)
        self._labels = df[label_column].to_numpy(dtype=np.float32)
        self._end_indices = self._build_valid_end_indices()

    def _build_valid_end_indices(self) -> np.ndarray:
        end_indices = []
        for end_idx in range(self._sequence_length - 1, len(self._labels)):
            start_idx = end_idx - self._sequence_length + 1
            x = self._features[start_idx : end_idx + 1]
            y = self._labels[end_idx]
            if np.isfinite(x).all() and np.isfinite(y):
                end_indices.append(end_idx)
        return np.asarray(end_indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._end_indices)

    @property
    def end_indices(self) -> np.ndarray:
        return self._end_indices.copy()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        end_idx = self._end_indices[index]
        start_idx = end_idx - self._sequence_length + 1
        x = self._features[start_idx : end_idx + 1]
        y = self._labels[end_idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)
