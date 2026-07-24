from __future__ import annotations

import numpy as np
import pandas as pd

from coinquant.backtest.engine import BacktestEngine
from coinquant.backtest.strategy import (
    build_turning_point_signal,
    carry_position_until_next_action,
    shift_signal_to_next_open,
)


def test_turning_point_signal_uses_previous_and_current_predictions() -> None:
    predictions = np.array([0.8, 0.6, 0.7, -0.6, -0.4, -0.5], dtype=np.float64)

    previous_pred, pred_delta, signal = build_turning_point_signal(predictions, threshold=0.5)
    action = shift_signal_to_next_open(signal)

    assert np.isnan(previous_pred[0])
    np.testing.assert_allclose(pred_delta[1:], np.array([-0.2, 0.1, -1.3, 0.2, -0.1]))
    np.testing.assert_array_equal(signal, np.array([0, -1, 0, -1, 1, 0], dtype=np.int8))
    np.testing.assert_array_equal(action, np.array([0, 0, -1, 0, -1, 1], dtype=np.int8))
    np.testing.assert_array_equal(
        carry_position_until_next_action(action),
        np.array([0, 0, -1, -1, -1, 1], dtype=np.int8),
    )


def test_backtest_engine_shifts_execution_to_next_open() -> None:
    engine = BacktestEngine(
        symbol="BTC/USDT",
        period="15m",
        threshold=0.5,
        fee_rate=0.0,
        initial_cash=1.0,
    )
    rows = pd.DataFrame(
        {
            "open_time": [1, 2, 3, 4, 5, 6, 7],
            "open": [100.0, 102.0, 101.0, 103.0, 99.0, 98.0, 100.0],
            "pred_fast": [0.8, 0.6, 0.7, -0.6, -0.4, -0.5, -0.3],
            "pred_slow": [0.8, 0.6, 0.7, -0.6, -0.4, -0.5, -0.3],
            "label_close_fast": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "label_close_slow": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    result = engine._attach_trading_results(rows)

    np.testing.assert_array_equal(
        result["signal_fast"].to_numpy(),
        np.array([0, -1, 0, -1, 1, 0], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        result["action_fast"].to_numpy(),
        np.array([0, 0, -1, 0, -1, 1], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        result["position_fast"].to_numpy(),
        np.array([0, 0, -1, -1, -1, 1], dtype=np.int8),
    )
    np.testing.assert_allclose(
        result["next_return"].to_numpy(),
        np.array(
            [
                0.02,
                -0.00980392156862745,
                0.01980198019801982,
                -0.03883495145631066,
                -0.010101010101010055,
                0.020408163265306145,
            ]
        ),
    )
    np.testing.assert_allclose(
        result["net_return_fast"].to_numpy(),
        np.array(
            [
                0.0,
                0.0,
                -0.01980198019801982,
                0.03883495145631066,
                0.010101010101010055,
                0.020408163265306145,
            ]
        ),
    )
