import ccxt
import pandas as pd

exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {
        "defaultType": "future"
    }
})

exchange.load_markets()

def fetch_ohlcv(symbol, timeframe, begin_time):
    ohlcv = exchange.fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        since=begin_time,
    )
    return ohlcv

def convert_datetime_to_timestamp(dt_str) -> int:
    return exchange.parse8601(dt_str)

def test():
    a = fetch_ohlcv("ETH/USDT", "15m", convert_datetime_to_timestamp("2026-07-14T00:00:00Z"))
    for row in a:
        row[0] = pd.to_datetime(row[0], unit='ms')
        print(row)

# test()