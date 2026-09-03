"""Inference-time ensemble for independently trained RL policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


class RLPolicyEnsemble:
    """Average deterministic target-exposure actions from several RL runs.

    Each member keeps its own frozen VecNormalize statistics.  This is
    important because normalization statistics are part of a trained policy;
    sharing one member's statistics can silently change the action distribution.
    The class intentionally implements SB3's small ``predict`` surface so it
    can be used anywhere a loaded PPO/SAC policy is expected.
    """

    def __init__(self, members: Sequence[tuple[Any, Any]], weights: Sequence[float] | None = None, scale: float = 1.0, max_action: float | None = None):
        if not members:
            raise ValueError("members must not be empty")
        self.members = tuple(members)
        if weights is None:
            weights = np.ones(len(self.members), dtype=np.float64)
        values = np.asarray(weights, dtype=np.float64).reshape(-1)
        if values.shape != (len(self.members),) or not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
            raise ValueError("weights must be finite, non-negative and non-zero")
        self.weights = values / values.sum()
        self.scale = float(scale)
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be positive and finite")
        self.max_action = None if max_action is None else float(max_action)
        if self.max_action is not None and (not np.isfinite(self.max_action) or self.max_action <= 0):
            raise ValueError("max_action must be positive and finite")

    def predict(self, observation: Any, state: Any | None = None, episode_start: Any | None = None, deterministic: bool = True):
        """Return a weighted, clipped action and a null recurrent state."""

        values = np.asarray(observation, dtype=np.float32)
        single = values.ndim == 1
        batch = values[None, :] if single else values
        actions: list[np.ndarray] = []
        for model, normalizer in self.members:
            normalized = normalizer.normalize_obs(batch)
            action, _ = model.predict(normalized, deterministic=deterministic)
            actions.append(np.asarray(action, dtype=np.float64).reshape(len(batch), -1)[:, 0])
        action = self.scale * np.tensordot(self.weights, np.stack(actions, axis=0), axes=(0, 0))
        if self.max_action is not None:
            action = np.clip(action, -self.max_action, self.max_action)
        if single:
            return np.asarray([float(action[0])], dtype=np.float32), None
        return np.asarray(action, dtype=np.float32).reshape(-1, 1), None


def load_ensemble(manifest_path: str | Path, device: str = "cpu") -> RLPolicyEnsemble:
    """Load an ensemble from an ``ensemble.json`` manifest.

    The manifest stores member run directories, so model and normalization
    files remain in their normal SB3 layout and can be audited independently.
    """

    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_dirs = []
    for item in payload.get("run_dirs", []):
        path = Path(item)
        if not path.is_absolute():
            path = manifest_path.parent / path
        run_dirs.append(path)
    if not run_dirs:
        raise ValueError("ensemble manifest has no run_dirs")
    # Imports stay lazy for users who only need the environment utilities.
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from coinquant.rl.trainer import RLTrainingConfig, _configs, _prepare_frame_for_env, build_predicted_frames
    from coinquant.rl.env import TradingEnv

    configs = []
    for run_dir in run_dirs:
        config_payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        config_payload["policy_hidden_sizes"] = tuple(config_payload.get("policy_hidden_sizes", (256, 256, 128)))
        config_payload.pop("output_dir", None)
        configs.append(RLTrainingConfig(**{k: v for k, v in config_payload.items() if k in RLTrainingConfig.__dataclass_fields__}))
    first = configs[0]
    frames = build_predicted_frames(first.symbol, first.period, first.fast_model_path, first.slow_model_path, first.dl_vector_dim)
    frames = {name: _prepare_frame_for_env(frame) for name, frame in frames.items()}
    observation_config, action_config, reward_config, env_config = _configs(first)
    def make_env():
        return TradingEnv(frames["train"], observation_config, action_config, reward_config, env_config)
    members = []
    for run_dir, config in zip(run_dirs, configs):
        dummy = DummyVecEnv([make_env])
        norm_path = run_dir / "model" / "best_vecnormalize.pkl"
        normalizer = VecNormalize.load(norm_path, dummy)
        normalizer.training = False
        normalizer.norm_reward = False
        model_cls = SAC if config.algorithm == "sac" else PPO
        model = model_cls.load(run_dir / "model" / "best_model.zip", device=device)
        members.append((model, normalizer))
    return RLPolicyEnsemble(
        members,
        weights=payload.get("weights"),
        scale=float(payload.get("scale", 1.0)),
        max_action=payload.get("max_action", first.max_leverage),
    )
