import logging
from typing import Annotated

import typer

from coinquant.handler.fetch_handler import FetchMode, FetchHandler
from coinquant.handler.train_handler import train_model, train_rl_model
from coinquant.handler.export_handler import export_samples_to_csv

from coinquant.config import settings
from coinquant.trainer.model_trainer import LabelMode
from coinquant.rl.report import render_rl_report

app = typer.Typer(help="BTC quantitative research command line tools.")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

@app.command(help="database status")
def status():
    handler = FetchHandler(FetchMode.incremental)
    count = handler.count()
    print(f"total rows: {count}")

@app.command(help="fetch historical data")
def fetch(
    mode: Annotated[
        FetchMode,
        typer.Option("--mode", "-m", help="获取模式：incremental 只获取缺口，full 重新获取指定区间。"),
    ] = FetchMode.incremental,
):
    handler = FetchHandler(mode)
    handler.fetch()

@app.command(help="export historical data to CSV")
def export(
    symbol: Annotated[
        str,
        typer.Option("--symbol", "-s", help="导出的合约种类"),
    ] = "BTC/USDT",
    period: Annotated[
        str,
        typer.Option("--period", "-p", help="导出周期"),
    ] = "15m",
    begin: Annotated[
        str,
        typer.Option("--begin", help="开始时间"),
    ] = "2020-01-01T00:00:00Z",
    end: Annotated[
        str,
        typer.Option("--end", help="结束时间"),
    ] = settings.data.end_date,
):
    export_samples_to_csv(
        settings.path.sample_path,
        symbol,
        period,
        begin,
        end,
    )

@app.command(help="train configured model")
def train(
    symbol: Annotated[
        str,
        typer.Option("--symbol", "-s", help="训练的合约种类"),
    ] = "BTC/USDT",
    period: Annotated[
        str,
        typer.Option("--period", "-p", help="训练周期"),
    ] = "15m",
    label_mode: Annotated[
        LabelMode,
        typer.Option("--label_mode", '-l', help="预测快周期/慢周期"),
    ] = LabelMode.fast
):
    save_path = train_model(symbol, period, label_mode)
    typer.echo(f"model saved to {save_path}")


@app.command("train-rl", help="train and evaluate a PPO or SAC trading agent")
def train_rl_command(
    symbol: Annotated[
        str,
        typer.Option("--symbol", "-s", help="训练的合约种类"),
    ] = "BTC/USDT",
    period: Annotated[
        str,
        typer.Option("--period", "-p", help="训练周期"),
    ] = "1h",
    timesteps: Annotated[
        int,
        typer.Option("--timesteps", "-n", min=2, help="RL 训练总步数"),
    ] = 100_000,
    algorithm: Annotated[
        str,
        typer.Option("--algorithm", help="RL 算法：ppo 或 sac"),
    ] = "ppo",
):
    artifacts = train_rl_model(symbol, period, timesteps, algorithm)
    typer.echo(f"RL run saved to {artifacts.run_dir}")
    for split, metrics in (
        ("train", artifacts.train_metrics),
        ("valid", artifacts.valid_metrics),
        ("test", artifacts.test_metrics),
    ):
        total_return = float(metrics["total_return"])
        max_drawdown = float(metrics["max_drawdown"])
        sharpe = metrics.get("sharpe")
        sharpe_text = "n/a" if sharpe is None else f"{float(sharpe):.3f}"
        typer.echo(
            f"{split}: return={total_return:.2%}, "
            f"max_drawdown={max_drawdown:.2%}, sharpe={sharpe_text}"
        )


@app.command("report-rl", help="生成 RL 交易 K 线标注报告")
def report_rl_command(
    run_dir: Annotated[
        str,
        typer.Argument(help="RL run 目录，例如 data/model/rl_ensemble_x025_v1/某个 run"),
    ],
    split: Annotated[
        str,
        typer.Option("--split", help="数据段：train、valid 或 test"),
    ] = "test",
    output: Annotated[
        str | None,
        typer.Option("--output", "-o", help="输出 HTML 路径，默认写入 run 目录"),
    ] = None,
):
    path = render_rl_report(run_dir, split=split, output_path=output)
    typer.echo(f"RL report saved to {path}")


def main() -> None:
    """Run the Typer application."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)
    app()

if __name__ == "__main__":
    raise SystemExit(main())
