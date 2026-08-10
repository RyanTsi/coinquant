import json

import numpy as np
import pandas as pd

from coinquant.model import predict as predict_module


class _FakeTransformerModel:
    batch_lengths: list[int] = []
    loaded_path = None

    def __init__(self, **model_params):
        self.model_params = model_params

    def load(self, checkpoint_path):
        type(self).loaded_path = checkpoint_path

    def predict(self, loader):
        predictions = []
        type(self).batch_lengths = []
        for features, _ in loader:
            type(self).batch_lengths.append(len(features))
            predictions.append(features[:, -1, 0].numpy())
        return np.concatenate(predictions)


def _write_metadata(tmp_path, sequence_length=3):
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "feature_columns": ["feat_a", "feat_b"],
                "model_params": {"d_feat": 2},
                "sequence_length": sequence_length,
            }
        ),
        encoding="utf-8",
    )
    return checkpoint_path


def test_predict_batch_predicts_every_complete_window_in_time_order(
    tmp_path, monkeypatch
):
    checkpoint_path = _write_metadata(tmp_path)
    monkeypatch.setattr(predict_module, "TransformerModel", _FakeTransformerModel)
    frame = pd.DataFrame(
        {
            "open_time": [3, 1, 5, 2, 4],
            "feat_a": [3.0, 1.0, 5.0, 2.0, 4.0],
            "feat_b": [30.0, 10.0, 50.0, 20.0, 40.0],
        }
    )

    predictions = predict_module.predict_batch(frame, checkpoint_path, batch_size=2)

    np.testing.assert_array_equal(predictions, np.array([3.0, 4.0, 5.0]))
    assert _FakeTransformerModel.batch_lengths == [2, 1]
    assert _FakeTransformerModel.loaded_path == checkpoint_path


def test_predict_batch_skips_only_windows_containing_invalid_features(
    tmp_path, monkeypatch
):
    checkpoint_path = _write_metadata(tmp_path)
    monkeypatch.setattr(predict_module, "TransformerModel", _FakeTransformerModel)
    frame = pd.DataFrame(
        {
            "open_time": [1, 2, 3, 4, 5, 6],
            "feat_a": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
            "feat_b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )

    predictions = predict_module.predict_batch(frame, checkpoint_path)

    np.testing.assert_array_equal(predictions, np.array([5.0, 6.0]))


def test_predict_batch_returns_empty_array_when_no_complete_window(
    tmp_path, monkeypatch
):
    checkpoint_path = _write_metadata(tmp_path, sequence_length=3)
    monkeypatch.setattr(predict_module, "TransformerModel", _FakeTransformerModel)
    frame = pd.DataFrame(
        {
            "open_time": [1, 2],
            "feat_a": [1.0, 2.0],
            "feat_b": [10.0, 20.0],
        }
    )

    predictions = predict_module.predict_batch(frame, checkpoint_path)

    assert predictions.dtype == np.float32
    assert predictions.size == 0
