from __future__ import annotations

import numpy as np


def build_turning_point_signal(
    predictions: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build T signals from prediction[T - 1] and prediction[T].

    A signal is generated at bar T, and should be executed at the next bar open.
    Short: prediction[T - 1] > threshold and prediction[T] is lower.
    Long: prediction[T - 1] < -threshold and prediction[T] is higher.
    """
    if threshold < 0:
        raise ValueError("threshold must be greater than or equal to 0")

    pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
    previous_pred = np.concatenate(([np.nan], pred[:-1]))
    pred_delta = pred - previous_pred

    signal = np.zeros(len(pred), dtype=np.int8)
    signal[(previous_pred > threshold) & (pred_delta < 0)] = -1
    signal[(previous_pred < -threshold) & (pred_delta > 0)] = 1

    return previous_pred, pred_delta, signal


def shift_signal_to_next_open(signal: np.ndarray) -> np.ndarray:
    """Shift a bar-close signal to the next bar's open action."""
    values = np.asarray(signal, dtype=np.int8).reshape(-1)
    action = np.zeros(len(values), dtype=np.int8)
    if len(values) > 1:
        action[1:] = values[:-1]
    return action


def carry_position_until_next_action(action: np.ndarray) -> np.ndarray:
    """Carry the latest non-zero action as position until the next action."""
    values = np.asarray(action, dtype=np.int8).reshape(-1)
    position = np.zeros(len(values), dtype=np.int8)
    current_position = 0

    for index, value in enumerate(values):
        if value != 0:
            current_position = int(value)
        position[index] = current_position

    return position
