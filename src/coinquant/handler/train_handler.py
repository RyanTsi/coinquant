from coinquant.trainer.model_trainer import LabelMode, ModelTrainer
from coinquant.rl.trainer import RLTrainingArtifacts, RLTrainingConfig, train_rl


def train_model(symbol: str, period: str, label_mode: LabelMode):
    return ModelTrainer(symbol, period, label_mode).train()


def train_rl_model(symbol: str, period: str, total_timesteps: int, algorithm: str = "ppo") -> RLTrainingArtifacts:
    """Train PPO or SAC using a compact public parameter surface."""

    if isinstance(total_timesteps, bool) or total_timesteps < 2:
        raise ValueError("total_timesteps must be at least 2")
    n_steps = min(2048, total_timesteps)
    batch_size = min(256, n_steps)
    while n_steps % batch_size != 0:
        batch_size -= 1
    config = RLTrainingConfig(
        symbol=symbol,
        period=period,
        total_timesteps=total_timesteps,
        algorithm=algorithm,
        n_steps=n_steps,
        batch_size=batch_size,
        eval_freq=min(20_000, total_timesteps),
    )
    return train_rl(config)
