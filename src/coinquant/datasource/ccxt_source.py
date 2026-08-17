import logging
from collections.abc import Iterator

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

    def convert_datetime_to_timestamp(self, value: str | int | float) -> int:
        if isinstance(value, (int, float)):
            return int(value)

        timestamp = self._exchange.parse8601(value)
        if timestamp is None:
            raise ValueError(f"Invalid datetime: {value}")
        return timestamp

    def fetch_ohlcv_pages(
        self,
        symbol,
        timeframe,
        begin_datetime,
        end_datetime=None,
    ) -> Iterator[list[list]]:
        since = self.convert_datetime_to_timestamp(begin_datetime)
        end_timestamp = (
            self.convert_datetime_to_timestamp(end_datetime)
            if end_datetime is not None
            else None
        )
        timeframe_ms = timeframe_to_milliseconds(timeframe)

        while end_timestamp is None or since <= end_timestamp:
            candles = self._exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=OHLCV_LIMIT,
            )
            if not candles:
                break

            if end_timestamp is not None:
                candles = [candle for candle in candles if candle[0] <= end_timestamp]
                if not candles:
                    break

            next_since = candles[-1][0] + timeframe_ms
            logger.info(
                "Fetched %s candles for %s at %s, next since: %s",
                len(candles),
                symbol,
                timeframe,
                next_since,
            )
            yield candles

            if next_since <= since:
                raise RuntimeError(
                    f"fetch_ohlcv did not advance for {symbol} at {timeframe}: {since}"
                )
            since = next_since
            time.sleep(0.35)

    def fetch_ohlcv(self, symbol, timeframe, begin_datetime, end_datetime=None):
        all_data = []
        for candles in self.fetch_ohlcv_pages(
            symbol,
            timeframe,
            begin_datetime,
            end_datetime,
        ):
            all_data.extend(candles)
        return all_data

def test():
    fetcher = ccxtSourceFetcher()
    data = fetcher.fetch_ohlcv("BTC/USDT", "1h", "2020-01-01T00:00:00Z")
    with open("test_data.csv", "w") as f:
        for candle in data:
            f.write(",".join(map(str, candle)) + "\n")
    
# test()
