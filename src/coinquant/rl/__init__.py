"""Reinforcement-learning training components."""

from coinquant.rl.action import ActionAdapter, ActionConfig
from coinquant.rl.env import EnvConfig, TradingEnv
from coinquant.rl.observation import (
    DEFAULT_ACCOUNT_FEATURE_COLUMNS,
    DEFAULT_BASIC_FEATURE_COLUMNS,
    DEFAULT_PREDICTION_COLUMNS,
    DLFeatureProvider,
    ObservationBuilder,
    ObservationConfig,
    ObservationNormalizer,
    attach_predictions,
    build_basic_features,
)
from coinquant.rl.reward import RewardBreakdown, RewardCalculator, RewardConfig
from coinquant.rl.trainer import (
    RLTrainingArtifacts,
    RLTrainingConfig,
    build_predicted_frames,
    evaluate_policy,
    train_and_backtest,
    train_rl,
)

__all__ = [
    "ActionAdapter",
    "ActionConfig",
    "EnvConfig",
    "TradingEnv",
    "ObservationConfig",
    "ObservationBuilder",
    "ObservationNormalizer",
    "DLFeatureProvider",
    "build_basic_features",
    "attach_predictions",
    "DEFAULT_BASIC_FEATURE_COLUMNS",
    "DEFAULT_PREDICTION_COLUMNS",
    "DEFAULT_ACCOUNT_FEATURE_COLUMNS",
    "RewardConfig",
    "RewardBreakdown",
    "RewardCalculator",
    "RLTrainingConfig",
    "RLTrainingArtifacts",
    "build_predicted_frames",
    "evaluate_policy",
    "train_and_backtest",
    "train_rl",
]
