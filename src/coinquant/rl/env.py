from __future__ import annotations

import gymnasium as gym

class TradingEnv(gym.Env):
    """Continuous contract trading environment.

    Observation = [
        Market Features,
        DL Predictions,
        Position Features
    ]
    Action = target position.
    Reward = return after fees, slippage, and risk penalties.
    """
