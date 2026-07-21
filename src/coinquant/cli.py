import logging
import json
from typing import Annotated

import typer

from coinquant.handler.fetch_handler import FetchMode, FetchHandler
from coinquant.handler.train_handler import train_model
from coinquant.handler.export_handler import export_samples_to_csv
from coinquant.handler.backtest_handler import (
    run_alpha_backtest,
    run_alpha_grid_search,
)
from coinquant.config import settings
from coinquant.backtest.alpha_backtester import BacktestSplit
from coinquant.trainer.model_trainer import LabelMode

app = typer.Typer(help="BTC quantitative research command line tools.")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

@app.command(help="database staust")
def staust():
    handler = FetchHandler(FetchMode.incremental)
    count = handler.count()
    print(f"totle rows: {count}")

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
        typer.Option("--label_mode", '-l', help="预测长短线"),
    ] = LabelMode.short
):
    save_path = train_model(symbol, period, label_mode)
    typer.echo(f"model saved to {save_path}")

@app.command(help="backtest long/short alpha threshold strategy")
def backtest(
    symbol: Annotated[
        str | None,
        typer.Option("--symbol", "-s", help="回测的合约种类，默认读取配置"),
    ] = None,
    period: Annotated[
        str | None,
        typer.Option("--period", "-p", help="回测周期，默认读取配置"),
    ] = None,
    split: Annotated[
        BacktestSplit | None,
        typer.Option("--split", help="回测数据切分，默认读取配置"),
    ] = None,
):
    try:
        result = run_alpha_backtest(
            symbol=symbol,
            period=period,
            split=split,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        typer.echo(f"backtest failed: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(json.dumps(result.summary, indent=2))
    typer.echo(f"summary saved to {result.summary_path}")
    typer.echo(f"equity saved to {result.equity_path}")
    typer.echo(f"orders saved to {result.orders_path}")
    typer.echo(f"trades saved to {result.trades_path}")

@app.command(help="grid search alpha threshold backtest parameters")
def backtest_grid(
    symbol: Annotated[
        str | None,
        typer.Option("--symbol", "-s", help="回测的合约种类，默认读取配置"),
    ] = None,
    period: Annotated[
        str | None,
        typer.Option("--period", "-p", help="回测周期，默认读取配置"),
    ] = None,
    split: Annotated[
        BacktestSplit | None,
        typer.Option("--split", help="搜索数据切分，默认读取配置"),
    ] = None,
):
    try:
        result = run_alpha_grid_search(
            symbol=symbol,
            period=period,
            split=split,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        typer.echo(f"backtest grid failed: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(json.dumps(result.best, indent=2))
    typer.echo(f"grid results saved to {result.results_path}")
    typer.echo(f"grid summary saved to {result.summary_path}")

def main() -> None:
    """Run the Typer application."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)
    app()

if __name__ == "__main__":
    raise SystemExit(main())
