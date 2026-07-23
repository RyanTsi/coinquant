import logging
from typing import Annotated

import typer

from coinquant.handler.fetch_handler import FetchMode, FetchHandler
from coinquant.handler.train_handler import train_model
from coinquant.handler.export_handler import export_samples_to_csv
from coinquant.handler.backtest_handler import format_backtest_metrics, run_backtest

from coinquant.config import settings
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
        typer.Option("--label_mode", '-l', help="预测快周期/慢周期"),
    ] = LabelMode.fast
):
    save_path = train_model(symbol, period, label_mode)
    typer.echo(f"model saved to {save_path}")

@app.command(help="run fast/slow model backtest on test split")
def backtest(
    symbol: Annotated[
        str,
        typer.Option("--symbol", "-s", help="回测的合约种类"),
    ] = "BTC/USDT",
    period: Annotated[
        str,
        typer.Option("--period", "-p", help="回测周期"),
    ] = "15m",
    threshold: Annotated[
        float,
        typer.Option("--threshold", help="预测阈值；高于阈值做多，低于负阈值做空。"),
    ] = 0.0,
    fee_rate: Annotated[
        float,
        typer.Option("--fee-rate", help="单边手续费率，例如 0.0004 表示 4 bps。"),
    ] = 0.0004,
    output: Annotated[
        str | None,
        typer.Option("--output", "-o", help="回测网页输出路径"),
    ] = None,
):
    result, report_path = run_backtest(symbol, period, threshold, fee_rate, output)
    typer.echo(format_backtest_metrics(result))
    typer.echo(f"backtest report saved to {report_path}")

def main() -> None:
    """Run the Typer application."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)
    app()

if __name__ == "__main__":
    raise SystemExit(main())
