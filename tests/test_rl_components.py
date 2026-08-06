import numpy as np
import pandas as pd
import pytest

from coinquant.backtest.account import AccountConfig
from coinquant.rl.action import ActionAdapter, ActionConfig
from coinquant.rl.env import EnvConfig, TradingEnv
from coinquant.rl.observation import (
    ObservationBuilder,
    ObservationConfig,
    attach_predictions,
    build_basic_features,
)
from coinquant.rl.reward import RewardCalculator, RewardConfig


def make_frame(rows: int = 16) -> pd.DataFrame:
    close = np.linspace(100.0, 110.0, rows)
    frame = pd.DataFrame(
        {
            "open_time": pd.date_range("2025-01-01", periods=rows, freq="h"),
            "open": close - 0.1,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.arange(rows, dtype=float) + 1.0,
        }
    )
    return attach_predictions(
        build_basic_features(frame),
        np.linspace(-1.0, 1.0, rows),
        np.linspace(1.0, -1.0, rows),
    )


def test_observation_is_causal_and_has_default_shape():
    frame = make_frame()
    config = ObservationConfig(window_size=10, normalize=False)
    builder = ObservationBuilder(frame, config)
    account = {name: 0.0 for name in config.account_feature_columns}
    observation = builder.build(9, account)
    assert observation.shape == (74,)
    assert observation.dtype == np.float32

    changed = frame.copy()
    changed.loc[10:, "prediction_fast"] = 999.0
    changed_builder = ObservationBuilder(changed, config)
    np.testing.assert_array_equal(observation, changed_builder.build(9, account))
    with pytest.raises(IndexError):
        builder.build(8, account)


def test_action_validation_clipping_and_quantity_conversion():
    adapter = ActionAdapter(ActionConfig(max_leverage=2.0))
    assert adapter.coerce([3.0]) == (2.0, True)
    assert adapter.target_quantity(-1.0, 1000.0, 100.0, 10.0) == pytest.approx(-1.0)
    assert adapter.exposure_from_position(-1.0, 100.0, 1000.0, 10.0) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        adapter.coerce([np.nan])
    with pytest.raises(ValueError):
        adapter.coerce([0.0, 1.0])


def test_reward_breakdown_uses_equity_return_once():
    calculator = RewardCalculator(
        RewardConfig(
            reward_scale=10.0,
            volatility_penalty=0.0,
            position_penalty=0.0,
            drawdown_penalty_rate=0.5,
        )
    )
    breakdown = calculator.calculate(100.0, 110.0, 1.0, market_return=0.2, drawdown=0.1)
    assert breakdown.net_return == pytest.approx(0.1)
    assert breakdown.gross_return == pytest.approx(0.2)
    assert breakdown.raw_reward == pytest.approx(0.1 - 0.05)
    assert breakdown.reward == pytest.approx(0.5)


def test_environment_executes_at_next_open_and_closes_at_last_close():
    frame = make_frame()
    # Make the first next bar gap higher so the result cannot be mistaken for
    # an action filled at the decision bar's close.
    frame.loc[10, ["open", "high", "low", "close"]] = [110.0, 121.0, 109.0, 121.0]
    frame.loc[9, ["open", "high", "low", "close"]] = [100.0, 101.0, 99.0, 100.0]
    env = TradingEnv(
        frame,
        action_config=ActionConfig(max_leverage=1.0),
        env_config=EnvConfig(
            account_config=AccountConfig(initial_balance=1000.0, leverage=10.0),
            force_close_at_end=True,
        ),
    )
    _, reset_info = env.reset(seed=7)
    assert reset_info["decision_time"] == frame.loc[9, "open_time"]
    _, _, terminated, truncated, info = env.step([1.0])
    assert not terminated
    assert not truncated
    assert info["decision_time"] == frame.loc[9, "open_time"]
    assert info["entry_time"] == frame.loc[10, "open_time"]
    assert info["actual_position"] > 0
    assert info["equity"] > 1000.0

    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step([1.0])
    assert info["end_of_episode_close"] is True
    assert env.position.position == 0.0
    assert env.account.equity > 0.0


def test_environment_rejects_bad_time_or_ohlc():
    frame = make_frame()
    frame.loc[2, "open_time"] = frame.loc[1, "open_time"]
    with pytest.raises(ValueError, match="open_time"):
        TradingEnv(frame)
    frame = make_frame()
    frame.loc[3, "low"] = frame.loc[3, "high"] + 1.0
    with pytest.raises(ValueError, match="OHLC"):
        TradingEnv(frame)
