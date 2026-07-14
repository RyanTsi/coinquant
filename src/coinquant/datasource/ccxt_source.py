from datetime import datetime, timezone
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_exchange():
    import ccxt

    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {
            "defaultType": "future"
        }
    })
    exchange.load_markets()
    return exchange


def fetch_ohlcv(symbol, timeframe, begin_time):
    exchange = _get_exchange()
    ohlcv = exchange.fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        since=begin_time,
    )
    return ohlcv


def convert_datetime_to_timestamp(dt_str) -> int:
    if dt_str.endswith("Z"):
        dt_str = f"{dt_str[:-1]}+00:00"
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def test():
    import pandas as pd

    a = fetch_ohlcv("ETH/USDT", "15m", convert_datetime_to_timestamp("2026-07-14T00:00:00Z"))
    for row in a:
        row[0] = pd.to_datetime(row[0], unit='ms')
        print(row)

# test()
