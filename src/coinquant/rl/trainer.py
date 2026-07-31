from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from coinquant.backtest.engine import BacktestEngine
from coinquant.config import settings
from coinquant.rl.env import DEFAULT_FEATURE_COLUMNS, TradingEnv
from coinquant.trainer.dataset_builder import DatasetBuilder
from coinquant.trainer.model_trainer import LabelMode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RLTrainingConfig:
    symbol: str = "BTC/USDT"
    period: str = "1h"
    window_size: int = 10
    total_timesteps: int = 100_000
    eval_freq: int = 20_000
    seed: int = 59483
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0002
    drawdown_penalty: float = 0.005
    volatility_penalty: float = 0.05
    position_penalty: float = 0.00002
    reward_scale: float = 100.0
    reward_mode: str = "simple"
    max_leverage: float = 1.0
    initial_equity: float = 1.0
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    gamma: float = 0.999
    gae_lambda: float = 0.95
    ent_coef: float = 0.005
    normalize: bool = True
    output_dir: str | None = None


@dataclass(frozen=True)
class RLTrainingArtifacts:
    run_dir: Path
    final_model_path: Path
    best_model_path: Path
    final_vecnormalize_path: Path | None
    best_vecnormalize_path: Path | None
    metrics_path: Path
    test_report_path: Path
    train_metrics: dict[str, float | int | str | None]
    valid_metrics: dict[str, float | int | str | None]
    test_metrics: dict[str, float | int | str | None]


def train_and_backtest(config: RLTrainingConfig) -> RLTrainingArtifacts:
    """Train PPO on the train split, validate on valid, and backtest test."""
    _configure_seeds(config.seed)
    run_dir = _make_run_dir(config)
    model_dir = run_dir / "model"
    log_dir = run_dir / "logs"
    report_dir = Path(settings.path.data_path) / "backtest"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    logger.info("building %s %s predicted train/valid/test frames", config.symbol, config.period)
    frames = build_predicted_frames(config.symbol, config.period)
    for split, frame in frames.items():
        logger.info("%s rows: %d", split, len(frame))

    logger.info("training PPO for %d timesteps", config.total_timesteps)
    model, final_model_path, best_model_path, final_vecnormalize_path, best_vecnormalize_path = _train_ppo(
        config=config,
        train_frame=frames["train"],
        valid_frame=frames["valid"],
        run_dir=run_dir,
        model_dir=model_dir,
        log_dir=log_dir,
    )

    logger.info("running deterministic train/valid/test policy backtests")
    vecnormalize_path = best_vecnormalize_path if best_vecnormalize_path and best_vecnormalize_path.exists() else final_vecnormalize_path
    policy_path = best_model_path if best_model_path.exists() else final_model_path
    split_results = {
        split: run_policy_backtest(
            model_path=policy_path,
            frame=frame,
            config=config,
            vecnormalize_path=vecnormalize_path,
        )
        for split, frame in frames.items()
    }

    test_rows, test_metrics = split_results["test"]
    test_report_path = report_dir / f"rl_backtest_{_slug(config.symbol)}_{config.period}_{_date_suffix(test_rows)}.html"
    write_rl_backtest_report(
        output_path=test_report_path,
        config=config,
        rows=test_rows,
        metrics=test_metrics,
        split_metrics={split: metrics for split, (_, metrics) in split_results.items()},
        model_path=policy_path,
    )

    metrics_path = run_dir / "metrics.json"
    metrics_payload = {
        "config": _jsonable(asdict(config)),
        "model_path": str(policy_path),
        "final_model_path": str(final_model_path),
        "best_model_path": str(best_model_path) if best_model_path.exists() else None,
        "vecnormalize_path": str(vecnormalize_path) if vecnormalize_path else None,
        "test_report_path": str(test_report_path),
        "metrics": {split: metrics for split, (_, metrics) in split_results.items()},
    }
    metrics_path.write_text(json.dumps(_jsonable(metrics_payload), indent=4), encoding="utf-8")

    return RLTrainingArtifacts(
        run_dir=run_dir,
        final_model_path=final_model_path,
        best_model_path=best_model_path,
        final_vecnormalize_path=final_vecnormalize_path,
        best_vecnormalize_path=best_vecnormalize_path,
        metrics_path=metrics_path,
        test_report_path=test_report_path,
        train_metrics=split_results["train"][1],
        valid_metrics=split_results["valid"][1],
        test_metrics=test_metrics,
    )


def build_predicted_frames(symbol: str, period: str) -> dict[str, pd.DataFrame]:
    splits = DatasetBuilder(symbol, period).build_splits_from_db()
    engine = BacktestEngine(symbol=symbol, period=period)
    return {
        name: _attach_model_predictions(engine, split)
        for name, split in splits.items()
    }


def run_policy_backtest(
    *,
    model_path: str | Path,
    frame: pd.DataFrame,
    config: RLTrainingConfig,
    vecnormalize_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int | str | None]]:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    base_env = DummyVecEnv([lambda: Monitor(_make_env(frame, config))])
    if config.normalize and vecnormalize_path is not None:
        env = VecNormalize.load(str(vecnormalize_path), base_env)
        env.training = False
        env.norm_reward = False
    else:
        env = base_env

    model = PPO.load(str(model_path), env=env, device="cpu")
    observation = env.reset()
    records: list[dict[str, Any]] = []
    done = False

    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, dones, infos = env.step(action)
        info = dict(infos[0])
        records.append(_record_from_info(frame, info, float(np.asarray(action).reshape(-1)[0])))
        done = bool(dones[0])

    env.close()
    rows = pd.DataFrame.from_records(records)
    metrics = calculate_rl_metrics(rows, config.period, config.initial_equity)
    return rows, metrics


def calculate_rl_metrics(
    rows: pd.DataFrame,
    period: str,
    initial_equity: float,
) -> dict[str, float | int | str | None]:
    if rows.empty:
        raise ValueError("rows must not be empty")

    returns = rows["net_return"].to_numpy(dtype=np.float64)
    gross_returns = rows["gross_return"].to_numpy(dtype=np.float64)
    equity = rows["equity"].to_numpy(dtype=np.float64)
    position = rows["position"].to_numpy(dtype=np.float64)
    turnover = rows["turnover"].to_numpy(dtype=np.float64)
    drawdown = rows["drawdown"].to_numpy(dtype=np.float64)
    reward = rows["reward"].to_numpy(dtype=np.float64)

    total_return = equity[-1] / initial_equity - 1.0
    bars_per_year = _bars_per_year(period)
    annual_return = _annualized_return(total_return, len(returns), bars_per_year)
    annual_volatility = _annualized_volatility(returns, bars_per_year)
    sharpe = _safe_divide(float(np.mean(returns)) * math.sqrt(bars_per_year), float(np.std(returns, ddof=1)))
    max_drawdown = float(np.min(drawdown))
    calmar = _safe_divide(annual_return, abs(max_drawdown))
    active_returns = returns[np.abs(position) > 1e-8]

    return {
        "model": "rl_ppo",
        "rows": int(len(rows)),
        "start_time": str(rows["from_datetime"].iloc[0]),
        "end_time": str(rows["to_datetime"].iloc[-1]),
        "total_return": _finite_or_none(total_return),
        "annual_return": _finite_or_none(annual_return),
        "annual_volatility": _finite_or_none(annual_volatility),
        "sharpe": _finite_or_none(sharpe),
        "max_drawdown": _finite_or_none(max_drawdown),
        "calmar": _finite_or_none(calmar),
        "win_rate": _finite_or_none(_mean_or_none(active_returns > 0)),
        "profit_factor": _finite_or_none(_profit_factor(active_returns)),
        "exposure": _finite_or_none(float(np.mean(np.abs(position)))),
        "long_ratio": _finite_or_none(float(np.mean(position > 1e-8))),
        "short_ratio": _finite_or_none(float(np.mean(position < -1e-8))),
        "flat_ratio": _finite_or_none(float(np.mean(np.abs(position) <= 1e-8))),
        "max_abs_position": _finite_or_none(float(np.max(np.abs(position)))),
        "total_turnover": _finite_or_none(float(np.sum(turnover))),
        "avg_turnover": _finite_or_none(float(np.mean(turnover))),
        "trade_count": int(np.count_nonzero(turnover > 1e-8)),
        "mean_bar_return": _finite_or_none(float(np.mean(returns))),
        "gross_total_return": _finite_or_none(float(np.prod(1.0 + gross_returns) - 1.0)),
        "total_reward": _finite_or_none(float(np.sum(reward))),
        "mean_reward": _finite_or_none(float(np.mean(reward))),
        "final_equity": _finite_or_none(float(equity[-1])),
    }


def write_rl_backtest_report(
    *,
    output_path: str | Path,
    config: RLTrainingConfig,
    rows: pd.DataFrame,
    metrics: dict[str, float | int | str | None],
    split_metrics: dict[str, dict[str, float | int | str | None]],
    model_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbol": config.symbol,
        "period": config.period,
        "model_path": str(model_path),
        "config": _jsonable(asdict(config)),
        "metrics": _jsonable(metrics),
        "split_metrics": _jsonable(split_metrics),
        "rows": _frame_records(rows),
    }
    payload_json = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    path.write_text(_HTML_TEMPLATE.replace("__RL_PAYLOAD__", payload_json), encoding="utf-8")
    return path


def _train_ppo(
    *,
    config: RLTrainingConfig,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    run_dir: Path,
    model_dir: Path,
    log_dir: Path,
):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    class SaveVecNormalizeCallback(BaseCallback):
        def __init__(self, path: Path):
            super().__init__()
            self.path = path

        def _on_step(self) -> bool:
            env = self.model.get_env()
            if isinstance(env, VecNormalize):
                env.save(str(self.path))
            return True

    train_env = DummyVecEnv([lambda: Monitor(_make_env(train_frame, config))])
    valid_env = DummyVecEnv([lambda: Monitor(_make_env(valid_frame, config))])

    best_vecnormalize_path = None
    final_vecnormalize_path = None
    if config.normalize:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        valid_env = VecNormalize(valid_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        best_vecnormalize_path = model_dir / "best_vecnormalize.pkl"
        final_vecnormalize_path = model_dir / "final_vecnormalize.pkl"
        valid_env.training = False
        valid_env.norm_reward = False

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        ent_coef=config.ent_coef,
        seed=config.seed,
        device="cpu",
        verbose=1,
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
    )

    eval_callback = EvalCallback(
        valid_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=max(1, config.eval_freq),
        n_eval_episodes=1,
        deterministic=True,
        render=False,
        callback_on_new_best=SaveVecNormalizeCallback(best_vecnormalize_path) if best_vecnormalize_path else None,
    )
    model.learn(total_timesteps=config.total_timesteps, callback=eval_callback, progress_bar=False)

    final_model_path = model_dir / "final_model.zip"
    best_model_path = model_dir / "best_model.zip"
    model.save(str(final_model_path))
    if final_vecnormalize_path is not None:
        train_env.save(str(final_vecnormalize_path))
    if best_vecnormalize_path is not None and not best_vecnormalize_path.exists():
        train_env.save(str(best_vecnormalize_path))

    train_env.close()
    valid_env.close()
    (run_dir / "config.json").write_text(json.dumps(_jsonable(asdict(config)), indent=4), encoding="utf-8")
    return model, final_model_path, best_model_path, final_vecnormalize_path, best_vecnormalize_path


def _make_env(frame: pd.DataFrame, config: RLTrainingConfig) -> TradingEnv:
    return TradingEnv(
        frame,
        window_size=config.window_size,
        feature_columns=DEFAULT_FEATURE_COLUMNS,
        max_leverage=config.max_leverage,
        fee_rate=config.fee_rate,
        slippage_rate=config.slippage_rate,
        drawdown_penalty=config.drawdown_penalty,
        volatility_penalty=config.volatility_penalty,
        position_penalty=config.position_penalty,
        reward_scale=config.reward_scale,
        reward_mode=config.reward_mode,
        initial_equity=config.initial_equity,
    )


def _attach_model_predictions(engine: BacktestEngine, frame: pd.DataFrame) -> pd.DataFrame:
    predictions = [
        engine._predict_model(LabelMode.fast, frame),
        engine._predict_model(LabelMode.slow, frame),
    ]
    return engine._merge_predictions(frame, predictions)


def _record_from_info(frame: pd.DataFrame, info: dict[str, Any], action: float) -> dict[str, Any]:
    decision_tick = int(info["decision_tick"])
    entry_tick = int(info["entry_tick"])
    exit_tick = int(info["exit_tick"])
    decision_row = frame.iloc[decision_tick]
    entry_row = frame.iloc[entry_tick]
    exit_row = frame.iloc[exit_tick]
    return {
        "decision_tick": decision_tick,
        "entry_tick": entry_tick,
        "exit_tick": exit_tick,
        "from_tick": entry_tick,
        "to_tick": exit_tick,
        "decision_open_time": decision_row.get("open_time"),
        "from_open_time": entry_row.get("open_time"),
        "to_open_time": exit_row.get("open_time"),
        "decision_datetime": decision_row.get("open_datetime"),
        "from_datetime": entry_row.get("open_datetime"),
        "to_datetime": exit_row.get("open_datetime"),
        "decision_open": float(decision_row["open"]),
        "decision_high": float(decision_row["high"]),
        "decision_low": float(decision_row["low"]),
        "decision_close": float(decision_row["close"]),
        "open": float(entry_row["open"]),
        "high": float(entry_row["high"]),
        "low": float(entry_row["low"]),
        "close": float(entry_row["close"]),
        "next_open": float(exit_row["open"]),
        "volume": float(entry_row["volume"]),
        "pred_fast": float(decision_row["pred_fast"]),
        "pred_slow": float(decision_row["pred_slow"]),
        "raw_action": action,
        "previous_position": float(info["previous_position"]),
        "position": float(info["position"]),
        "turnover": float(info["turnover"]),
        "market_return": float(info["market_return"]),
        "gross_return": float(info["gross_return"]),
        "fee_cost": float(info["fee_cost"]),
        "slippage_cost": float(info["slippage_cost"]),
        "net_return": float(info["net_return"]),
        "reward": float(info["reward"]),
        "base_reward": float(info["base_reward"]),
        "risk_penalty": float(info["risk_penalty"]),
        "equity": float(info["equity"]),
        "peak_equity": float(info["peak_equity"]),
        "drawdown": float(info["drawdown"]),
    }


def _make_run_dir(config: RLTrainingConfig) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = Path(config.output_dir) if config.output_dir else Path(settings.path.model_path) / "rl"
    return root / f"ppo_{_slug(config.symbol)}_{config.period}_{timestamp}"


def _configure_seeds(seed: int) -> None:
    np.random.seed(seed)


def _date_suffix(rows: pd.DataFrame) -> str:
    start = str(rows["from_datetime"].iloc[0])[:10].replace("-", "")
    end = str(rows["to_datetime"].iloc[-1])[:10].replace("-", "")
    return f"{start}_{end}"


def _slug(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def _frame_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    clean = df.replace([np.inf, -np.inf], np.nan)
    return json.loads(clean.to_json(orient="records"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if np.isfinite(value):
            return value
        return None
    return value


def _bars_per_year(period: str) -> float:
    import re

    match = re.fullmatch(r"(\d+)([mhdw])", period.strip().lower())
    if not match:
        return 365.0
    value = int(match.group(1))
    unit = match.group(2)
    minutes_by_unit = {
        "m": 1,
        "h": 60,
        "d": 60 * 24,
        "w": 60 * 24 * 7,
    }
    minutes = value * minutes_by_unit[unit]
    return 365.0 * 24.0 * 60.0 / minutes


def _annualized_return(total_return: float, rows: int, bars_per_year: float) -> float | None:
    if rows <= 0 or total_return <= -1:
        return None
    return (1.0 + total_return) ** (bars_per_year / rows) - 1.0


def _annualized_volatility(returns: np.ndarray, bars_per_year: float) -> float | None:
    if len(returns) < 2:
        return None
    return float(np.std(returns, ddof=1) * math.sqrt(bars_per_year))


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _mean_or_none(values: np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    return float(np.mean(values))


def _profit_factor(returns: np.ndarray) -> float | None:
    if len(returns) == 0:
        return None
    gains = float(np.sum(returns[returns > 0]))
    losses = float(np.sum(returns[returns < 0]))
    if losses == 0:
        return None if gains == 0 else math.inf
    return gains / abs(losses)


def _finite_or_none(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return None
        return float(value)
    return value


def _format_metrics(metrics: dict[str, float | int | str | None]) -> str:
    keys = [
        ("rows", "rows", lambda value: f"{int(value):d}" if value is not None else "-"),
        ("total_return", "total", _fmt_pct),
        ("annual_return", "annual", _fmt_pct),
        ("sharpe", "sharpe", _fmt_float),
        ("max_drawdown", "mdd", _fmt_pct),
        ("calmar", "calmar", _fmt_float),
        ("win_rate", "win", _fmt_pct),
        ("profit_factor", "pf", _fmt_float),
        ("exposure", "exposure", _fmt_pct),
        ("trade_count", "trades", lambda value: f"{int(value):d}" if value is not None else "-"),
    ]
    headers = [label for _, label, _ in keys]
    values = [formatter(metrics.get(key)) for key, _, formatter in keys]
    widths = [max(len(headers[index]), len(values[index])) for index in range(len(headers))]
    header_line = "  ".join(headers[index].ljust(widths[index]) for index in range(len(headers)))
    separator = "  ".join("-" * width for width in widths)
    value_line = "  ".join(values[index].ljust(widths[index]) for index in range(len(values)))
    return "\n".join([header_line, separator, value_line])


def _fmt_float(value) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _fmt_pct(value) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.2f}%"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and backtest a BTC 1h PPO contract policy.")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--period", default="1h")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=59483)
    parser.add_argument("--fee-rate", type=float, default=0.0004)
    parser.add_argument("--slippage-rate", type=float, default=0.0002)
    parser.add_argument("--drawdown-penalty", type=float, default=0.005)
    parser.add_argument("--volatility-penalty", type=float, default=0.05)
    parser.add_argument("--position-penalty", type=float, default=0.00002)
    parser.add_argument("--reward-scale", type=float, default=100.0)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--initial-equity", type=float, default=1.0)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    config = RLTrainingConfig(
        symbol=args.symbol,
        period=args.period,
        window_size=args.window_size,
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        seed=args.seed,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        drawdown_penalty=args.drawdown_penalty,
        volatility_penalty=args.volatility_penalty,
        position_penalty=args.position_penalty,
        reward_scale=args.reward_scale,
        max_leverage=args.max_leverage,
        initial_equity=args.initial_equity,
        normalize=not args.no_normalize,
        output_dir=args.output_dir,
    )
    artifacts = train_and_backtest(config)
    print("train")
    print(_format_metrics(artifacts.train_metrics))
    print("valid")
    print(_format_metrics(artifacts.valid_metrics))
    print("test")
    print(_format_metrics(artifacts.test_metrics))
    print(f"best model: {artifacts.best_model_path}")
    print(f"metrics: {artifacts.metrics_path}")
    print(f"test report: {artifacts.test_report_path}")


_HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CoinQuant RL Backtest</title>
<style>
body { margin: 0; background: #f6f7f9; color: #17202a; font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.page { width: min(1320px, calc(100vw - 32px)); margin: 0 auto; padding: 18px 0 28px; }
h1 { margin: 0 0 4px; font-size: 22px; line-height: 1.2; letter-spacing: 0; }
.meta { color: #687382; margin-bottom: 14px; }
.grid { display: grid; grid-template-columns: repeat(5, minmax(130px, 1fr)); gap: 10px; margin-bottom: 12px; }
.metric, .panel, table { background: white; border: 1px solid #d7dee7; border-radius: 8px; box-shadow: 0 10px 30px rgba(25,35,52,0.08); }
.metric { padding: 10px 12px; min-width: 0; }
.metric-label { color: #687382; font-size: 12px; }
.metric-value { margin-top: 4px; font-size: 19px; font-weight: 700; line-height: 1.2; }
.stack { display: grid; gap: 10px; }
.panel-head { padding: 9px 12px 7px; border-bottom: 1px solid #d7dee7; font-weight: 700; }
canvas { display: block; width: 100%; height: 230px; }
#equityCanvas { height: 280px; }
table { width: 100%; border-collapse: collapse; margin-top: 12px; overflow: hidden; }
th, td { padding: 8px 10px; border-bottom: 1px solid #d7dee7; text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: #687382; font-weight: 600; background: #eef2f5; }
@media (max-width: 900px) { .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } table { display: block; overflow-x: auto; } }
</style>
</head>
<body>
<main class="page">
  <h1 id="title">CoinQuant RL Backtest</h1>
  <div class="meta" id="meta"></div>
  <section class="grid" id="metricGrid"></section>
  <section class="stack">
    <div class="panel"><div class="panel-head">权益曲线</div><canvas id="equityCanvas"></canvas></div>
    <div class="panel"><div class="panel-head">仓位</div><canvas id="positionCanvas"></canvas></div>
    <div class="panel"><div class="panel-head">Fast / Slow 预测值</div><canvas id="predictionCanvas"></canvas></div>
  </section>
  <table id="splitTable"></table>
  <table id="recentTable"></table>
</main>
<script>
const payload = __RL_PAYLOAD__;
const rows = payload.rows;
const colors = { equity: "#111827", position: "#2563eb", fast: "#118a67", slow: "#d97706", grid: "#e7ecf2", axis: "#687382" };
function fmtNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}
function fmtPct(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}
function drawLine(canvasId, seriesDefs, formatter) {
  const canvas = document.getElementById(canvasId);
  const rectBox = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rectBox.width * dpr));
  canvas.height = Math.max(1, Math.round(rectBox.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const width = rectBox.width;
  const height = rectBox.height;
  const rect = { left: 58, right: width - 16, top: 18, bottom: height - 30 };
  const visible = rows;
  const allValues = [];
  seriesDefs.forEach(def => visible.forEach(row => {
    const value = Number(row[def.key]);
    if (Number.isFinite(value)) allValues.push(value);
  }));
  if (!visible.length || !allValues.length) return;
  let min = Math.min(...allValues);
  let max = Math.max(...allValues);
  if (min === max) { min -= 1; max += 1; }
  const pad = (max - min) * 0.1;
  min -= pad; max += pad;
  const y = value => rect.bottom - ((value - min) / (max - min)) * (rect.bottom - rect.top);
  const x = index => rect.left + (rect.right - rect.left) * (index / Math.max(1, visible.length - 1));
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = colors.grid;
  ctx.fillStyle = colors.axis;
  ctx.font = "12px Inter, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i += 1) {
    const yy = rect.top + (rect.bottom - rect.top) * (i / 4);
    const value = max - (max - min) * (i / 4);
    ctx.beginPath(); ctx.moveTo(rect.left, yy); ctx.lineTo(rect.right, yy); ctx.stroke();
    ctx.fillText(formatter(value), rect.left - 8, yy);
  }
  seriesDefs.forEach(def => {
    ctx.strokeStyle = def.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    visible.forEach((row, index) => {
      const value = Number(row[def.key]);
      const xx = x(index);
      const yy = y(value);
      if (index === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
    });
    ctx.stroke();
  });
}
function renderMetrics() {
  document.getElementById("title").textContent = `${payload.symbol} ${payload.period} RL 回测`;
  document.getElementById("meta").textContent = `${payload.metrics.start_time} UTC 至 ${payload.metrics.end_time} UTC · ${rows.length} 根K线 · ${payload.model_path}`;
  const defs = [
    ["total_return", "总收益", fmtPct],
    ["annual_return", "年化", fmtPct],
    ["sharpe", "Sharpe", value => fmtNumber(value, 2)],
    ["max_drawdown", "最大回撤", fmtPct],
    ["calmar", "Calmar", value => fmtNumber(value, 2)],
    ["win_rate", "胜率", fmtPct],
    ["profit_factor", "盈亏比", value => fmtNumber(value, 2)],
    ["exposure", "平均暴露", fmtPct],
    ["total_turnover", "总换手", value => fmtNumber(value, 1)],
    ["trade_count", "调仓次数", value => fmtNumber(value, 0)]
  ];
  const grid = document.getElementById("metricGrid");
  grid.innerHTML = "";
  defs.forEach(([key, label, format]) => {
    const item = document.createElement("div");
    item.className = "metric";
    item.innerHTML = `<div class="metric-label">${label}</div><div class="metric-value">${format(payload.metrics[key])}</div>`;
    grid.appendChild(item);
  });
}
function renderSplitTable() {
  const defs = [
    ["split", "区间", value => value],
    ["rows", "样本", value => fmtNumber(value, 0)],
    ["total_return", "总收益", fmtPct],
    ["annual_return", "年化", fmtPct],
    ["sharpe", "Sharpe", value => fmtNumber(value, 2)],
    ["max_drawdown", "MDD", fmtPct],
    ["profit_factor", "盈亏比", value => fmtNumber(value, 2)],
    ["exposure", "暴露", fmtPct]
  ];
  const names = ["train", "valid", "test"];
  const header = `<thead><tr>${defs.map(([, label]) => `<th>${label}</th>`).join("")}</tr></thead>`;
  const body = names.map(name => {
    const metrics = payload.split_metrics[name];
    const cells = defs.map(([key, , format]) => `<td>${key === "split" ? name : format(metrics[key])}</td>`).join("");
    return `<tr>${cells}</tr>`;
  }).join("");
  document.getElementById("splitTable").innerHTML = `${header}<tbody>${body}</tbody>`;
}
function renderRecentTable() {
  const sample = rows.slice(-80);
  const defs = [
    ["to_datetime", "时间", value => value],
    ["position", "仓位", value => fmtNumber(value, 3)],
    ["turnover", "换手", value => fmtNumber(value, 3)],
    ["market_return", "市场收益", fmtPct],
    ["net_return", "净收益", fmtPct],
    ["reward", "Reward", value => fmtNumber(value, 4)],
    ["equity", "权益", value => fmtNumber(value, 4)],
    ["drawdown", "回撤", fmtPct]
  ];
  const header = `<thead><tr>${defs.map(([, label]) => `<th>${label}</th>`).join("")}</tr></thead>`;
  const body = sample.map(row => {
    const cells = defs.map(([key, , format]) => `<td>${format(row[key])}</td>`).join("");
    return `<tr>${cells}</tr>`;
  }).join("");
  document.getElementById("recentTable").innerHTML = `${header}<tbody>${body}</tbody>`;
}
function drawAll() {
  renderMetrics();
  renderSplitTable();
  renderRecentTable();
  drawLine("equityCanvas", [{ key: "equity", color: colors.equity }], value => fmtNumber(value, 3));
  drawLine("positionCanvas", [{ key: "position", color: colors.position }], value => fmtNumber(value, 2));
  drawLine("predictionCanvas", [{ key: "pred_fast", color: colors.fast }, { key: "pred_slow", color: colors.slow }], value => fmtNumber(value, 3));
}
window.addEventListener("resize", drawAll);
drawAll();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
