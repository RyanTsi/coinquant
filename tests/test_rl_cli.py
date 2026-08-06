from types import SimpleNamespace

import pytest

pytest.importorskip("dynaconf")

from typer.testing import CliRunner

import coinquant.cli as cli
import coinquant.handler.train_handler as train_handler


def test_rl_handler_derives_compact_training_schedule(monkeypatch):
    captured = {}

    def fake_train(config):
        captured["config"] = config
        return config

    monkeypatch.setattr(train_handler, "train_rl", fake_train)
    config = train_handler.train_rl_model("ETH/USDT", "4h", 300)
    assert config.symbol == "ETH/USDT"
    assert config.period == "4h"
    assert config.total_timesteps == 300
    assert config.n_steps == 300
    assert config.batch_size == 150
    assert config.eval_freq == 300
    assert captured["config"] is config


def test_train_rl_cli_prints_split_summary(monkeypatch):
    artifacts = SimpleNamespace(
        run_dir="data/model/rl/example",
        train_metrics={"total_return": 0.1, "max_drawdown": -0.05, "sharpe": 1.2},
        valid_metrics={"total_return": 0.02, "max_drawdown": -0.03, "sharpe": None},
        test_metrics={"total_return": -0.01, "max_drawdown": -0.04, "sharpe": -0.2},
    )
    monkeypatch.setattr(cli, "train_rl_model", lambda symbol, period, timesteps: artifacts)
    result = CliRunner().invoke(
        cli.app,
        ["train-rl", "--symbol", "ETH/USDT", "--period", "4h", "--timesteps", "300"],
    )
    assert result.exit_code == 0, result.stdout
    assert "RL run saved to data/model/rl/example" in result.stdout
    assert "train: return=10.00%" in result.stdout
    assert "valid: return=2.00%" in result.stdout
    assert "test: return=-1.00%" in result.stdout
