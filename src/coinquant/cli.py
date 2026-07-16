import logging
from typing import Annotated

import typer

from coinquant.handler.fetch_handler import FetchMode, FetchHandler
from coinquant.handler.export_handler import export_samples_to_csv
from coinquant.config import settings

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

def main() -> None:
    """Run the Typer application."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)
    app()

if __name__ == "__main__":
    raise SystemExit(main())
