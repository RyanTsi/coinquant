from enum import Enum
from coinquant.trainer.model_trainer import LabelMode, ModelTrainer

def train_model(symbol: str, period: str, label_mode: LabelMode):
    return ModelTrainer(symbol, period, label_mode).train()
