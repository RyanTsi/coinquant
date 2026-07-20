from enum import Enum
import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from coinquant.config import settings
from coinquant.model.transformer import TransformerModel
from coinquant.trainer.dataset_builder import DatasetBuilder
from coinquant.trainer.sequence_dataset import SequenceDataset

import coinquant.utils as tools

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "transformer": TransformerModel,
}

class LabelMode(str, Enum):
    short = "short"
    long = "long"


class ModelTrainer:
    def __init__(self, symbol: str, period: str, label_mode: LabelMode):
        self.symbol = symbol
        self.period = period
        self.label_mode = label_mode
        self.model_name = tools.get_setting(settings.model, "name", "transformer")
        self.model_config = self._load_model_config(self.model_name)

    def train(self) -> Path:
        splits = DatasetBuilder(self.symbol, self.period).build_splits_from_db()
        train_df = splits["train"]
        valid_df = splits["valid"]
        label_column = "label_close_short" if self.label_mode == LabelMode.short else "label_close_short"

        sequence_length = int(tools.get_setting(settings.data_set, "sequence_length", 128))
        feature_columns = self._resolve_feature_columns(train_df)

        train_dataset = SequenceDataset(train_df, feature_columns, label_column, sequence_length)
        val_dataset = SequenceDataset(valid_df, feature_columns, label_column, sequence_length)
        
        if len(train_dataset) == 0:
            raise ValueError("train dataset is empty after sequence filtering")
        if len(val_dataset) == 0:
            raise ValueError("validation dataset is empty after sequence filtering")

        batch_size = self.model_config.get("params.batch_size", 256)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        model = self._build_model()
        save_path = self._checkpoint_path()
        evals_result: dict[str, list[float]] = {}

        logger.info(
            "training %s for %s %s with %d features, %d train samples, %d val samples",
            self.model_name,
            self.symbol,
            self.period,
            len(feature_columns),
            len(train_dataset),
            len(val_dataset),
        )
        
        model.fit(train_loader, val_loader, evals_result=evals_result, save_path=save_path)
        self._save_metadata(save_path, feature_columns, label_column, evals_result)
        
        return save_path

    def _build_model(self) -> TransformerModel:
        model_class = MODEL_REGISTRY.get(self.model_name)
        if model_class is None:
            supported = ", ".join(sorted(MODEL_REGISTRY))
            raise ValueError(f"unsupported model {self.model_name!r}; supported models: {supported}")

        params_config = self.model_config.get("params", {})
        if not isinstance(params_config, dict):
            raise ValueError(f"model config params must be an object for {self.model_name!r}")

        params = dict(params_config)

        return model_class(**params)

    def _resolve_feature_columns(self, train_df) -> list[str]:
        feature_columns = [column for column in train_df.columns if column.startswith("feat_")]
        if not feature_columns:
            raise ValueError("no feature columns found; expected columns with the 'feat_' prefix")
        return feature_columns

    def _load_model_config(self, model_name: str) -> dict[str, Any]:
        config_path = Path(settings.model.config_dir) / f"{model_name}.json"
        if not config_path.exists():
            raise FileNotFoundError(f"model config not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)

        if not isinstance(config, dict):
            raise ValueError(f"model config must be a JSON object: {config_path}")
        return config

    def _checkpoint_path(self) -> Path:
        save_dir = Path(settings.path.model_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        symbol = self.symbol.replace("/", "_").replace(":", "_")
        filename = f"{self.model_name}_{symbol}_{self.period}.pt"
        return save_dir / filename

    def _save_metadata(
        self,
        save_path: Path,
        feature_columns: list[str],
        label_column: str,
        evals_result: dict[str, list[float]],
    ) -> None:
        metadata = {
            "model_name": self.model_name,
            "symbol": self.symbol,
            "period": self.period,
            "feature_columns": feature_columns,
            "label_column": label_column,
            "evals_result": evals_result,
        }
        metadata_path = save_path.with_suffix(".json")
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4)
