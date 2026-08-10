"""PPO training and deterministic evaluation orchestration."""

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
    ObservationConfig,
    build_basic_features,
)
from coinquant.rl.reward import RewardConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RLTrainingConfig:
    symbol: str = "BTC/USDT"
    period: str = "1h"
    window_size: int = 10
    total_timesteps: int = 100_000
    eval_freq: int = 20_000
    seed: int = 59_483
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0002
    margin_rate: float = 0.005
    liquidation_fee_rate: float = 0.0
    account_leverage: float = 10.0
    max_leverage: float = 1.0
    drawdown_penalty_rate: float = 0.005
    volatility_penalty: float = 0.05
    position_penalty: float = 0.00002
    reward_scale: float = 100.0
    reward_mode: str = "simple"
    risk_window: int = 20
    liquidation_penalty: float = 0.0
    initial_equity: float = 1.0
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    gamma: float = 0.999
    gae_lambda: float = 0.95
    ent_coef: float = 0.0
    normalize: bool = True
    device: str = "cpu"
    output_dir: str | None = None
    fast_model_path: str | None = None
    slow_model_path: str | None = None

    def __post_init__(self) -> None:
        if not self.symbol or not self.period:
            raise ValueError("symbol and period must not be empty")
        for name in ("window_size", "total_timesteps", "eval_freq", "n_steps", "batch_size", "risk_window"):
            value = getattr(self, name)
            if int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
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
            "reward_scale",
            "liquidation_penalty",
            "initial_equity",
            "learning_rate",
            "gamma",
            "gae_lambda",
            "ent_coef",
        ):
            value = utils.validate_finite(getattr(self, name), name)
            if name in {"fee_rate", "slippage_rate", "margin_rate", "liquidation_fee_rate", "drawdown_penalty_rate", "volatility_penalty", "position_penalty", "ent_coef"} and value < 0:
                raise ValueError(f"{name} must be non-negative")
            if name in {"account_leverage", "max_leverage", "reward_scale", "initial_equity", "learning_rate"} and value <= 0:
                raise ValueError(f"{name} must be greater than 0")
            if name in {"margin_rate", "fee_rate", "slippage_rate", "liquidation_fee_rate"} and value >= 1:
                raise ValueError(f"{name} must be less than 1")
            if name in {"gamma", "gae_lambda"} and not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
            object.__setattr__(self, name, value)
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
        )
        prepared = _attach_model_prediction(
            prepared,
            slow_model_path or _default_model_path(symbol, period, "slow"),
            "slow",
        )
        prepared = prepared.dropna(
            subset=["prediction_fast", "prediction_slow"]
        ).reset_index(drop=True)
        result[split_name] = prepared
    return result


def train_rl(
    config: RLTrainingConfig,
    frames: Mapping[str, Any] | None = None,
) -> RLTrainingArtifacts:
    """Train PPO and evaluate it on valid/test frames.

    Stable-Baselines3 is imported lazily so feature and reward unit tests do not
    require the optional training stack.
    """

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError("train_rl requires stable-baselines3 and gymnasium") from exc

    _configure_seeds(config.seed)
    frames = dict(frames or build_predicted_frames(config.symbol, config.period, config.fast_model_path, config.slow_model_path))
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
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        ent_coef=config.ent_coef,
        seed=config.seed,
        device=config.device,
        verbose=0,
    )
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
    best_model = PPO.load(best_model_path, env=vec_env, device=config.device)
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
    while not done:
        policy_observation = observation
        if training_vec_env is not None and hasattr(training_vec_env, "normalize_obs"):
            policy_observation = training_vec_env.normalize_obs(
                np.asarray([observation], dtype=np.float32)
            )[0]
        action, _ = model.predict(policy_observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
    history = env.history
    return _history_metrics(history, config.initial_equity, env.account.equity, config.period), history


def train_and_backtest(config: RLTrainingConfig) -> RLTrainingArtifacts:
    """Compatibility name for the complete train/valid/test pipeline."""

    return train_rl(config)


def _configs(config: RLTrainingConfig):
    from coinquant.backtest.account import AccountConfig
    from coinquant.backtest.execution import ExecutionConfig

    observation_config = ObservationConfig(window_size=config.window_size, normalize=False)
    action_config = ActionConfig(max_leverage=config.max_leverage)
    reward_config = RewardConfig(
        reward_mode=config.reward_mode,
        reward_scale=config.reward_scale,
        drawdown_penalty_rate=config.drawdown_penalty_rate,
        volatility_penalty=config.volatility_penalty,
        position_penalty=config.position_penalty,
        risk_window=config.risk_window,
        liquidation_penalty=config.liquidation_penalty,
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
    )
    return observation_config, action_config, reward_config, env_config


def _attach_model_prediction(frame: Any, model_path: str | Path, mode: str):
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
    predictions = np.asarray(model.predict(loader), dtype=np.float64).reshape(-1)
    result = frame.copy()
    result[f"prediction_{mode}"] = np.nan
    result.loc[dataset.end_indices, f"prediction_{mode}"] = predictions
    finite_predictions = result[f"prediction_{mode}"].dropna().to_numpy()
    if not np.isfinite(finite_predictions).all():
        raise ValueError(f"{mode} predictions contain non-finite values")
    return result


def _default_model_path(symbol: str, period: str, mode: str) -> Path:
    from coinquant.config import settings

    model_dir = Path(settings.path.model_path)
    slug = symbol.replace("/", "_").replace(":", "_")
    return model_dir / f"transformer_{slug}_{period}_{mode}.pt"


def _make_run_dir(config: RLTrainingConfig) -> Path:
    root = Path(config.output_dir) if config.output_dir else Path("data/model/rl")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = config.symbol.replace("/", "_").replace(":", "_")
    run_dir = root / f"ppo_{slug}_{config.period}_{timestamp}"
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
        "model": "rl_ppo",
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
