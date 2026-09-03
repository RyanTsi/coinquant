"""RL training and deterministic evaluation orchestration.

Both PPO and off-policy SAC share the same frozen-feature environment and
evaluation path.  PPO remains the conservative default for backwards
compatibility; SAC can be selected explicitly for continuous target-exposure
control.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from coinquant import utils
from coinquant.rl.action import ActionConfig
from coinquant.rl.env import EnvConfig, TradingEnv
from coinquant.rl.observation import (
    DEFAULT_BASIC_FEATURE_COLUMNS,
    ObservationConfig,
    add_prediction_context_features,
    build_basic_features,
)
from coinquant.rl.reward import RewardConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RLTrainingConfig:
    symbol: str = "BTC/USDT"
    period: str = "1h"
    window_size: int = 32
    account_history_length: int | None = None
    include_dl_features: bool = True
    include_prediction_vectors: bool = True
    include_prediction_context: bool = True
    dl_vector_dim: int = 8
    total_timesteps: int = 100_000
    eval_freq: int = 20_000
    seed: int = 59_483
    algorithm: str = "ppo"
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0002
    margin_rate: float = 0.005
    liquidation_fee_rate: float = 0.0
    # Contract/account margin leverage.  This is separate from the RL target
    # exposure cap above; 2x is the conservative execution baseline.
    account_leverage: float = 2.0
    # Moderate exposure and explicit action persistence are the current
    # higher-return baseline.  A one-bar continuous policy over-traded and
    # reliably destroyed equity; 0.25x remains available as a defensive mode.
    # Historical field name: this is the RL target exposure cap, while
    # ``account_leverage`` controls initial-margin requirements.
    max_leverage: float = 0.5
    rebalance_interval: int = 48
    min_rebalance_notional_ratio: float = 0.0
    # Keep reward equal to net account return until a stable baseline exists.
    # These knobs are retained for explicit future ablations.
    drawdown_penalty_rate: float = 0.0
    volatility_penalty: float = 0.0
    position_penalty: float = 0.0
    turnover_penalty_rate: float = 0.0
    short_penalty_rate: float = 0.0
    action_change_penalty_rate: float = 0.0
    reversal_penalty_rate: float = 0.0
    reward_scale: float = 100.0
    reward_mode: str = "simple"
    risk_window: int = 20
    liquidation_penalty: float = 0.0
    enable_penalties: bool = False
    initial_equity: float = 1.0
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    gamma: float = 0.999
    gae_lambda: float = 0.95
    ent_coef: float = 0.0
    policy_hidden_sizes: tuple[int, ...] = (256, 256, 128)
    # A temporal extractor preserves the rolling-window structure.  It can be
    # disabled for an apples-to-apples MLP ablation.
    use_temporal_extractor: bool = False
    temporal_conv_channels: int = 64
    temporal_hidden_size: int = 128
    temporal_features_dim: int = 256
    # SAC-specific replay/entropy controls.  They are ignored for PPO.
    sac_buffer_size: int = 100_000
    sac_learning_starts: int = 5_000
    sac_train_freq: int = 1
    sac_gradient_steps: int = 1
    sac_tau: float = 0.005
    sac_ent_coef: str | float = "auto"
    normalize: bool = True
    device: str = "cpu"
    output_dir: str | None = None
    fast_model_path: str | None = None
    slow_model_path: str | None = None

    def __post_init__(self) -> None:
        if not self.symbol or not self.period:
            raise ValueError("symbol and period must not be empty")
        for name in (
            "window_size", "total_timesteps", "eval_freq", "n_steps", "batch_size",
            "risk_window", "dl_vector_dim", "rebalance_interval", "temporal_conv_channels",
            "temporal_hidden_size", "temporal_features_dim", "sac_buffer_size",
            "sac_learning_starts", "sac_train_freq", "sac_gradient_steps",
        ):
            value = getattr(self, name)
            if int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
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
        if not isinstance(self.enable_penalties, bool):
            raise TypeError("enable_penalties must be a bool")
        if not isinstance(self.use_temporal_extractor, bool):
            raise TypeError("use_temporal_extractor must be a bool")
        algorithm = str(self.algorithm).strip().lower()
        if algorithm not in {"ppo", "sac"}:
            raise ValueError("algorithm must be 'ppo' or 'sac'")
        object.__setattr__(self, "algorithm", algorithm)
        if not self.policy_hidden_sizes:
            raise ValueError("policy_hidden_sizes must not be empty")
        hidden_sizes = tuple(self.policy_hidden_sizes)
        if any(isinstance(size, bool) or int(size) != size or size <= 0 for size in hidden_sizes):
            raise ValueError("policy_hidden_sizes must contain positive integers")
        object.__setattr__(self, "policy_hidden_sizes", tuple(int(size) for size in hidden_sizes))
        for name in (
            "fee_rate",
            "slippage_rate",
            "margin_rate",
            "liquidation_fee_rate",
            "account_leverage",
            "max_leverage",
            "drawdown_penalty_rate",
            "volatility_penalty",
            "position_penalty",
            "turnover_penalty_rate",
            "short_penalty_rate",
            "action_change_penalty_rate",
            "reversal_penalty_rate",
            "reward_scale",
            "liquidation_penalty",
            "initial_equity",
            "learning_rate",
            "gamma",
            "gae_lambda",
            "ent_coef",
            "min_rebalance_notional_ratio",
            "sac_tau",
        ):
            value = utils.validate_finite(getattr(self, name), name)
            if name in {"fee_rate", "slippage_rate", "margin_rate", "liquidation_fee_rate", "drawdown_penalty_rate", "volatility_penalty", "position_penalty", "turnover_penalty_rate", "short_penalty_rate", "action_change_penalty_rate", "reversal_penalty_rate", "ent_coef", "min_rebalance_notional_ratio"} and value < 0:
                raise ValueError(f"{name} must be non-negative")
            if name in {"account_leverage", "max_leverage", "reward_scale", "initial_equity", "learning_rate"} and value <= 0:
                raise ValueError(f"{name} must be greater than 0")
            if name in {"margin_rate", "fee_rate", "slippage_rate", "liquidation_fee_rate"} and value >= 1:
                raise ValueError(f"{name} must be less than 1")
            if name in {"gamma", "gae_lambda"} and not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
            if name == "sac_tau" and not 0 < value <= 1:
                raise ValueError("sac_tau must be in (0, 1]")
            object.__setattr__(self, name, value)
        if not isinstance(self.sac_ent_coef, (str, int, float)) or isinstance(self.sac_ent_coef, bool):
            raise TypeError("sac_ent_coef must be 'auto' or a non-negative number")
        if isinstance(self.sac_ent_coef, str):
            if self.sac_ent_coef.strip().lower() != "auto":
                raise ValueError("sac_ent_coef string must be 'auto'")
            object.__setattr__(self, "sac_ent_coef", "auto")
        else:
            ent_coef = utils.validate_finite(self.sac_ent_coef, "sac_ent_coef")
            if ent_coef < 0:
                raise ValueError("sac_ent_coef must be non-negative")
            object.__setattr__(self, "sac_ent_coef", ent_coef)
        if self.reward_mode not in {"simple", "log"}:
            raise ValueError("reward_mode must be 'simple' or 'log'")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        if self.max_leverage > self.account_leverage:
            raise ValueError("max_leverage cannot exceed account_leverage")


@dataclass(frozen=True, slots=True)
class RLTrainingArtifacts:
    run_dir: Path
    final_model_path: Path
    best_model_path: Path
    final_vecnormalize_path: Path | None
    best_vecnormalize_path: Path | None
    metrics_path: Path
    train_metrics: dict[str, Any]
    valid_metrics: dict[str, Any]
    test_metrics: dict[str, Any]


def build_predicted_frames(
    symbol: str,
    period: str,
    fast_model_path: str | Path | None = None,
    slow_model_path: str | Path | None = None,
    vector_dim: int = 8,
) -> dict[str, Any]:
    """Build chronological train/valid/test frames with frozen DL predictions."""

    from coinquant.trainer.dataset_builder import DatasetBuilder

    splits = DatasetBuilder(symbol, period).build_splits_from_db()
    result: dict[str, Any] = {}
    for split_name, frame in splits.items():
        prepared = build_basic_features(frame)
        prepared = _attach_model_prediction(
            prepared,
            fast_model_path or _default_model_path(symbol, period, "fast"),
            "fast",
            vector_dim=vector_dim,
        )
        prepared = _attach_model_prediction(
            prepared,
            slow_model_path or _default_model_path(symbol, period, "slow"),
            "slow",
            vector_dim=vector_dim,
        )
        prepared = prepared.dropna(
            subset=["prediction_fast", "prediction_slow"]
        ).reset_index(drop=True)
        # Build rolling prediction context only after the unavailable prefix
        # has been removed, so its first statistics are not diluted by
        # artificial zero predictions.
        add_prediction_context_features(prepared)
        result[split_name] = prepared
    return result


def train_rl(
    config: RLTrainingConfig,
    frames: Mapping[str, Any] | None = None,
) -> RLTrainingArtifacts:
    """Train the configured RL algorithm and evaluate it on valid/test frames.

    Stable-Baselines3 is imported lazily so feature and reward unit tests do not
    require the optional training stack.
    """

    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError("train_rl requires stable-baselines3 and gymnasium") from exc

    _configure_seeds(config.seed)
    frames = dict(
        frames
        or build_predicted_frames(
            config.symbol,
            config.period,
            config.fast_model_path,
            config.slow_model_path,
            config.dl_vector_dim,
        )
    )
    frames = {
        name: _prepare_frame_for_env(frame)
        for name, frame in frames.items()
    }
    for name in ("train", "valid", "test"):
        if name not in frames or len(frames[name]) < config.window_size + 2:
            raise ValueError(f"{name} frame is missing or too short")

    run_dir = _make_run_dir(config)
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    observation_config, action_config, reward_config, env_config = _configs(config)

    def make_train_env():
        return Monitor(
            TradingEnv(
                frames["train"], observation_config, action_config, reward_config, env_config
            )
        )

    train_env = DummyVecEnv([make_train_env])
    vec_env = VecNormalize(train_env, norm_obs=config.normalize, norm_reward=config.normalize, clip_obs=observation_config.clip_value) if config.normalize else train_env

    def make_valid_env():
        return Monitor(
            TradingEnv(
                frames["valid"], observation_config, action_config, reward_config, env_config
            )
        )

    valid_env = DummyVecEnv([make_valid_env])
    eval_env = valid_env
    if config.normalize:
        eval_env = VecNormalize(
            valid_env,
            training=False,
            norm_obs=True,
            norm_reward=False,
            clip_obs=observation_config.clip_value,
        )
        eval_env.obs_rms = vec_env.obs_rms
    import torch

    policy_kwargs = {
        "net_arch": list(config.policy_hidden_sizes),
        "activation_fn": torch.nn.Tanh,
    }
    if config.use_temporal_extractor:
        # Import only when requested so the observation/reward utilities still
        # work in installations without stable-baselines3.
        from coinquant.rl.features import TemporalFeatureExtractor

        train_observation_builder = TradingEnv(
            frames["train"], observation_config, action_config, reward_config, env_config
        ).observation_builder
        policy_kwargs["features_extractor_class"] = TemporalFeatureExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "market_feature_count": len(train_observation_builder.market_columns),
            "window_size": config.window_size,
            "account_feature_count": len(observation_config.account_feature_columns),
            "account_history_length": observation_config.account_history_length or config.window_size,
            "conv_channels": config.temporal_conv_channels,
            "recurrent_hidden_size": config.temporal_hidden_size,
            "features_dim": config.temporal_features_dim,
        }

    if config.algorithm == "ppo":
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            ent_coef=config.ent_coef,
            policy_kwargs=policy_kwargs,
            seed=config.seed,
            device=config.device,
            verbose=0,
        )
        model_class = PPO
    else:
        model = SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=config.learning_rate,
            buffer_size=config.sac_buffer_size,
            learning_starts=config.sac_learning_starts,
            batch_size=config.batch_size,
            tau=config.sac_tau,
            gamma=config.gamma,
            train_freq=config.sac_train_freq,
            gradient_steps=config.sac_gradient_steps,
            ent_coef=config.sac_ent_coef,
            policy_kwargs=policy_kwargs,
            seed=config.seed,
            device=config.device,
            verbose=0,
        )
        model_class = SAC
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(run_dir / "eval"),
        eval_freq=config.eval_freq,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
    )
    model.learn(total_timesteps=config.total_timesteps, callback=eval_callback)
    final_model_path = model_dir / "final_model.zip"
    model.save(final_model_path)
    final_vec_path = model_dir / "final_vecnormalize.pkl" if config.normalize else None
    if isinstance(vec_env, VecNormalize):
        vec_env.save(final_vec_path)

    best_model_path = model_dir / "best_model.zip"
    if not best_model_path.exists():
        model.save(best_model_path)
    best_model = model_class.load(best_model_path, env=vec_env, device=config.device)
    best_vec_path = model_dir / "best_vecnormalize.pkl" if config.normalize else None
    if isinstance(vec_env, VecNormalize):
        vec_env.save(best_vec_path)
    train_metrics, train_history = _run_policy(best_model, frames["train"], config, vec_env)
    valid_metrics, valid_history = _run_policy(best_model, frames["valid"], config, vec_env)
    test_metrics, test_history = _run_policy(best_model, frames["test"], config, vec_env)
    _write_ledger(run_dir / "train_ledger.jsonl", train_history)
    _write_ledger(run_dir / "valid_ledger.jsonl", valid_history)
    _write_ledger(run_dir / "test_ledger.jsonl", test_history)
    metrics_payload = {
        "config": asdict(config),
        "metrics": {"train": train_metrics, "valid": valid_metrics, "test": test_metrics},
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(_jsonable(metrics_payload), indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(_jsonable(asdict(config)), indent=2), encoding="utf-8")
    vec_env.close()
    eval_env.close()
    return RLTrainingArtifacts(
        run_dir=run_dir,
        final_model_path=Path(final_model_path),
        best_model_path=Path(best_model_path),
        final_vecnormalize_path=final_vec_path,
        best_vecnormalize_path=best_vec_path,
        metrics_path=metrics_path,
        train_metrics=train_metrics,
        valid_metrics=valid_metrics,
        test_metrics=test_metrics,
    )


def evaluate_policy(
    model: Any,
    frame: Any,
    config: RLTrainingConfig,
    training_vec_env: Any | None = None,
) -> dict[str, Any]:
    """Run a deterministic policy on one frame and return RL metrics."""

    metrics, _ = _run_policy(model, frame, config, training_vec_env)
    return metrics


def _run_policy(
    model: Any,
    frame: Any,
    config: RLTrainingConfig,
    training_vec_env: Any | None = None,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Run one episode and return both metrics and its immutable history."""

    env = TradingEnv(frame, *_configs(config))
    observation, _ = env.reset(seed=config.seed)
    done = False
    normalize_env = training_vec_env if training_vec_env is not None and hasattr(training_vec_env, "normalize_obs") else None
    previous_training_flag = None if normalize_env is None else getattr(normalize_env, "training", None)
    if normalize_env is not None and previous_training_flag is not None:
        # Evaluation must never update train-time observation statistics.
        normalize_env.training = False
    try:
        while not done:
            policy_observation = observation
            if normalize_env is not None:
                policy_observation = normalize_env.normalize_obs(
                    np.asarray([observation], dtype=np.float32)
                )[0]
            action, _ = model.predict(policy_observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
    finally:
        if normalize_env is not None and previous_training_flag is not None:
            normalize_env.training = previous_training_flag
    history = env.history
    return _history_metrics(
        history, config.initial_equity, env.account.equity, config.period, config.algorithm
    ), history


def train_and_backtest(config: RLTrainingConfig) -> RLTrainingArtifacts:
    """Compatibility name for the complete train/valid/test pipeline."""

    return train_rl(config)


def _configs(config: RLTrainingConfig):
    from coinquant.backtest.account import AccountConfig
    from coinquant.backtest.execution import ExecutionConfig

    observation_config = ObservationConfig(
        window_size=config.window_size,
        account_history_length=config.account_history_length,
        include_dl_features=config.include_dl_features,
        include_prediction_vectors=config.include_prediction_vectors,
        include_prediction_context=config.include_prediction_context,
        normalize=False,
    )
    action_config = ActionConfig(max_leverage=config.max_leverage)
    reward_config = RewardConfig(
        reward_mode=config.reward_mode,
        reward_scale=config.reward_scale,
        drawdown_penalty_rate=config.drawdown_penalty_rate,
        volatility_penalty=config.volatility_penalty,
        position_penalty=config.position_penalty,
        turnover_penalty_rate=config.turnover_penalty_rate,
        short_penalty_rate=config.short_penalty_rate,
        action_change_penalty_rate=config.action_change_penalty_rate,
        reversal_penalty_rate=config.reversal_penalty_rate,
        risk_window=config.risk_window,
        liquidation_penalty=config.liquidation_penalty,
        enable_penalties=config.enable_penalties,
    )
    env_config = EnvConfig(
        account_config=AccountConfig(config.initial_equity, config.account_leverage),
        execution_config=ExecutionConfig(
            margin_rate=config.margin_rate,
            fee_rate=config.fee_rate,
            slippage_rate=config.slippage_rate,
            liquidation_fee_rate=config.liquidation_fee_rate,
        ),
        force_close_at_end=True,
        rebalance_interval=config.rebalance_interval,
        min_rebalance_notional_ratio=config.min_rebalance_notional_ratio,
    )
    return observation_config, action_config, reward_config, env_config


def _attach_model_prediction(
    frame: Any,
    model_path: str | Path,
    mode: str,
    vector_dim: int | None = 8,
):
    from torch.utils.data import DataLoader

    model_path = Path(model_path)
    metadata_path = model_path.with_suffix(".json")
    if not model_path.exists():
        raise FileNotFoundError(f"DL model checkpoint not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"DL model metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = list(metadata.get("feature_columns", []))
    label_column = str(metadata.get("label_column", f"label_z_score_close_{mode}"))
    params = dict(metadata.get("model_params", {}))
    sequence_length_value = metadata.get("sequence_length", params.pop("sequence_length", None))
    if sequence_length_value is None:
        try:
            from coinquant.config import settings

            sequence_length_value = getattr(settings.data_set, "sequence_length", 128)
        except ImportError:
            # Existing checkpoint metadata predates an explicit sequence length;
            # the project default is 128 and keeps feature-only tests usable in
            # a minimal installation.
            sequence_length_value = 128
    sequence_length = int(sequence_length_value)
    if sequence_length <= 0:
        raise ValueError("DL sequence_length must be greater than 0")
    if not feature_columns:
        raise ValueError(f"metadata has no feature_columns: {metadata_path}")
    if "d_feat" in params and int(params["d_feat"]) != len(feature_columns):
        raise ValueError(
            f"DL metadata d_feat={params['d_feat']} does not match {len(feature_columns)} feature columns"
        )
    from coinquant.model.transformer import TransformerModel
    from coinquant.trainer.sequence_dataset import SequenceDataset

    dataset = SequenceDataset(frame, feature_columns, label_column, sequence_length)
    if len(dataset) == 0:
        raise ValueError(f"no valid {mode} DL sequences")
    model = TransformerModel(**params)
    model.load(model_path)
    loader = DataLoader(dataset, batch_size=int(params.get("batch_size", 512)), shuffle=False)
    if hasattr(model, "predict_with_features"):
        predictions, vectors = model.predict_with_features(loader, feature_dim=vector_dim)
    else:  # pragma: no cover - compatibility with external model adapters
        predictions = model.predict(loader)
        raw_predictions = np.asarray(predictions, dtype=np.float32)
        if raw_predictions.ndim == 1:
            raw_predictions = raw_predictions[:, None]
        if raw_predictions.ndim != 2:
            raise ValueError(f"{mode} model predictions have an invalid shape")
        vectors = raw_predictions[:, :vector_dim] if vector_dim is not None else raw_predictions
        predictions = raw_predictions[:, 0]
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] != len(predictions):
        raise ValueError(f"{mode} DL vectors have an invalid shape")
    result = frame.copy()
    result[f"prediction_{mode}"] = np.nan
    result.loc[dataset.end_indices, f"prediction_{mode}"] = predictions
    for index in range(vectors.shape[1]):
        result[f"prediction_{mode}_{index}"] = np.nan
        result.loc[dataset.end_indices, f"prediction_{mode}_{index}"] = vectors[:, index]
    finite_predictions = result[f"prediction_{mode}"].dropna().to_numpy()
    if not np.isfinite(finite_predictions).all() or not np.isfinite(vectors).all():
        raise ValueError(f"{mode} predictions contain non-finite values")
    return result


def _prepare_frame_for_env(frame: Any) -> Any:
    """Normalize caller-supplied frames to the RL feature contract."""

    if not hasattr(frame, "columns"):
        raise TypeError("RL frames must be pandas DataFrames")
    required = set(DEFAULT_BASIC_FEATURE_COLUMNS)
    if not required.issubset(frame.columns):
        frame = build_basic_features(frame)
    if {
        "prediction_fast",
        "prediction_slow",
    }.issubset(frame.columns) and not any(
        column.startswith("prediction_context_") for column in frame.columns
    ):
        frame = frame.copy()
        add_prediction_context_features(frame)
    return frame


def _default_model_path(symbol: str, period: str, mode: str) -> Path:
    from coinquant.config import settings

    model_dir = Path(settings.path.model_path)
    slug = symbol.replace("/", "_").replace(":", "_")
    return model_dir / f"transformer_{slug}_{period}_{mode}.pt"


def _make_run_dir(config: RLTrainingConfig) -> Path:
    root = Path(config.output_dir) if config.output_dir else Path("data/model/rl")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = config.symbol.replace("/", "_").replace(":", "_")
    run_dir = root / f"{config.algorithm}_{slug}_{config.period}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _configure_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _history_metrics(
    history: Any,
    initial_equity: float,
    final_equity: float | None = None,
    period: str = "1h",
    algorithm: str = "ppo",
) -> dict[str, Any]:
    records = list(history)
    rewards = np.asarray([float(item.get("reward", 0.0)) for item in records], dtype=np.float64)
    returns = np.asarray([float(item.get("net_return", 0.0)) for item in records], dtype=np.float64)
    equities = np.asarray([float(item.get("equity", initial_equity)) for item in records], dtype=np.float64)
    if final_equity is None:
        final_equity = float(equities[-1]) if len(equities) else initial_equity
    equity_points = np.concatenate(([initial_equity], equities))
    running_max = np.maximum.accumulate(equity_points)
    drawdown = np.divide(equity_points, running_max, out=np.ones_like(equity_points), where=running_max != 0) - 1.0
    total_return = final_equity / initial_equity - 1.0
    bars_per_year = _bars_per_year(period)
    period_volatility = float(np.std(returns, ddof=1)) if len(returns) >= 2 else None
    volatility = None if period_volatility is None else period_volatility * math.sqrt(bars_per_year)
    sharpe = (
        None
        if period_volatility in (None, 0.0)
        else float(np.mean(returns) / period_volatility * math.sqrt(bars_per_year))
    )
    annualized_return = _annualized_return(total_return, len(returns), bars_per_year)
    turnovers = np.asarray([float(item.get("turnover", 0.0)) for item in records], dtype=np.float64)
    exposures = np.asarray([float(item.get("exposure_during_bar", item.get("actual_exposure", 0.0))) for item in records], dtype=np.float64)
    long_ratio = float(np.mean(exposures > 1e-12)) if len(exposures) else 0.0
    short_ratio = float(np.mean(exposures < -1e-12)) if len(exposures) else 0.0
    flat_ratio = float(np.mean(np.isclose(exposures, 0.0))) if len(exposures) else 0.0
    return {
        "model": f"rl_{algorithm}",
        "rows": len(records),
        "total_return": float(total_return),
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(np.min(drawdown)) if len(drawdown) else 0.0,
        "trade_count": int(sum(float(item.get("turnover", 0.0)) > 0 for item in records)),
        "total_turnover": float(sum(float(item.get("turnover", 0.0)) for item in records)),
        "avg_turnover": float(turnovers.mean()) if len(turnovers) else 0.0,
        "mean_abs_exposure": float(np.mean(np.abs(exposures))) if len(exposures) else 0.0,
        "long_exposure_ratio": long_ratio,
        "short_exposure_ratio": short_ratio,
        "flat_exposure_ratio": flat_ratio,
        "total_fee": float(sum(float(item.get("fee_cost", 0.0)) for item in records)),
        "total_funding": float(sum(float(item.get("funding_payment", 0.0)) for item in records)),
        "liquidation_count": int(sum(bool(item.get("liquidated", False)) for item in records)),
        "total_reward": float(rewards.sum()) if len(rewards) else 0.0,
        "mean_reward": float(rewards.mean()) if len(rewards) else 0.0,
        "final_equity": float(final_equity),
        "start_time": records[0].get("decision_time") if records else None,
        "end_time": records[-1].get("exit_time") if records else None,
    }


def _bars_per_year(period: str) -> float:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([mhdw])\s*", str(period).lower())
    if match is None:
        raise ValueError(f"unsupported period for annualization: {period!r}")
    amount = float(match.group(1))
    unit = match.group(2)
    minutes = {"m": amount, "h": amount * 60.0, "d": amount * 24.0 * 60.0, "w": amount * 7.0 * 24.0 * 60.0}[unit]
    if minutes <= 0:
        raise ValueError("period must be greater than 0")
    return 365.0 * 24.0 * 60.0 / minutes


def _annualized_return(total_return: float, periods: int, periods_per_year: float) -> float | None:
    if periods <= 0 or 1.0 + total_return <= 0:
        return None
    return float((1.0 + total_return) ** (periods_per_year / periods) - 1.0)


def _write_ledger(path: Path, history: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in history:
            handle.write(json.dumps(_jsonable(dict(record)), ensure_ascii=False) + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat") and not isinstance(value, (str, bytes)):
        try:
            return value.isoformat()
        except (AttributeError, TypeError, ValueError):
            pass
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value
