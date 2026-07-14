import logging
from typing import Annotated

import typer

from coinquant.handler.fetch_handler import FetchMode, FetchHandler
from coinquant.config import settings

app = typer.Typer(help="BTC quantitative research command line tools.")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

@app.command(help="fetch historical data")
def fetch(
    mode: Annotated[
        FetchMode,
        typer.Option("--mode", "-m", help="获取模式：incremental 只获取缺口，full 重新获取指定区间。"),
    ] = FetchMode.incremental,
):
    handler = FetchHandler(mode)
    handler.fetch()

@app.command(help="hello")
def hello(name: Annotated[str, typer.Option("--name", "-n", help="Your name")]):
    typer.echo(f"Hello {name}!")
    typer.echo(f"Database path: {settings.path.db_path}")

def main() -> None:
    """Run the Typer application."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)
    app()


if __name__ == "__main__":
    raise SystemExit(main())
