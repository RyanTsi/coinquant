import gymnasium as gym
import numpy as np

class Actions:
    target_postion: float
    
class accound:
    cash:float

class TradingEnv(gym.Env):
        
    def __init__(self, df, window_size):
        super().__init__()
        self.df = df
        self.window_size = window_size

        # spaces
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(df) # todo
        self._truncated = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self.history = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

    def step(self, action):

        return obs, reward, done, info