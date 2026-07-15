import logging
import time
from datetime import datetime, timezone
from functools import lru_cache
import ccxt
import time

logger = logging.getLogger(__name__)

OHLCV_LIMIT = 1000

_TIMEFRAME_UNITS_IN_MS = {
    "m": 60 * 1000,
    "h": 60 * 60 * 1000,
    "d": 24 * 60 * 60 * 1000,
    "w": 7 * 24 * 60 * 60 * 1000,
}

def timeframe_to_milliseconds(timeframe: str) -> int:
    unit = timeframe[-1]
    if unit not in _TIMEFRAME_UNITS_IN_MS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return int(timeframe[:-1]) * _TIMEFRAME_UNITS_IN_MS[unit]

class ccxtSourceFetcher:
    def __init__(self):
        self._exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {
                "defaultType": "future"
            }
        })
        self._exchange.load_markets()
    
    def convert_datetime_to_timestamp(self, dt_str) -> int:
        return self._exchange.parse8601(dt_str)
    
    def fetch_ohlcv(self, symbol, timeframe, begin_datetime):
        since = self.convert_datetime_to_timestamp(begin_datetime)
        all_data = []
        while True:
            candles = self._exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=OHLCV_LIMIT,
            )
            if not candles:
                break
            all_data.extend(candles)
            since = candles[-1][0] + timeframe_to_milliseconds(timeframe)
            logger.info(f"Fetched {len(candles)} candles for {symbol} at {timeframe}, next since: {since}")
            time.sleep(0.2)
        return all_data

def test():
    fetcher = ccxtSourceFetcher()
    data = fetcher.fetch_ohlcv("BTC/USDT", "1h", "2020-01-01T00:00:00Z")
    with open("test_data.csv", "w") as f:
        for candle in data:
            f.write(",".join(map(str, candle)) + "\n")
    
# test()