import logging
import time
from datetime import datetime, timezone
from functools import lru_cache

logger = logging.getLogger(__name__)

OHLCV_LIMIT = 1000
MAX_FETCH_RETRIES = 5
RATE_LIMIT_BACKOFF_SECONDS = 60
NETWORK_BACKOFF_SECONDS = 10
_TIMEFRAME_UNITS_IN_MS = {
    "m": 60 * 1000,
    "h": 60 * 60 * 1000,
    "d": 24 * 60 * 60 * 1000,
    "w": 7 * 24 * 60 * 60 * 1000,
}


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


def timeframe_to_milliseconds(timeframe: str) -> int:
    unit = timeframe[-1]
    if unit not in _TIMEFRAME_UNITS_IN_MS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return int(timeframe[:-1]) * _TIMEFRAME_UNITS_IN_MS[unit]


def iter_ohlcv_batches(
    symbol,
    timeframe,
    begin_time,
    end_time=None,
    limit=OHLCV_LIMIT,
):
    exchange = _get_exchange()
    timeframe_ms = timeframe_to_milliseconds(timeframe)
    since = begin_time

    while end_time is None or since <= end_time:
        batch = _fetch_ohlcv_page(exchange, symbol, timeframe, since, limit)
        if not batch:
            break

        next_since = batch[-1][0] + timeframe_ms
        filtered_batch = [
            row for row in batch if end_time is None or row[0] <= end_time
        ]
        if filtered_batch:
            yield filtered_batch

        if next_since <= since:
            break
        if end_time is not None and next_since > end_time:
            break
        if len(batch) < limit:
            break

        since = next_since


def fetch_ohlcv(symbol, timeframe, begin_time, end_time=None):
    rows = []
    for batch in iter_ohlcv_batches(symbol, timeframe, begin_time, end_time):
        rows.extend(batch)
    return rows


def _fetch_ohlcv_page(exchange, symbol, timeframe, since, limit):
    import ccxt

    for attempt in range(1, MAX_FETCH_RETRIES + 1):
        try:
            return exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit,
            )
        except (ccxt.RateLimitExceeded, ccxt.DDoSProtection) as exc:
            if attempt == MAX_FETCH_RETRIES:
                raise
            delay_seconds = RATE_LIMIT_BACKOFF_SECONDS * attempt
            logger.warning(
                "Rate limited fetching %s %s since=%s; retrying in %s seconds "
                "(attempt %s/%s): %s",
                symbol,
                timeframe,
                since,
                delay_seconds,
                attempt,
                MAX_FETCH_RETRIES,
                exc,
            )
            time.sleep(delay_seconds)
        except (
            ccxt.ExchangeNotAvailable,
            ccxt.NetworkError,
            ccxt.RequestTimeout,
        ) as exc:
            if attempt == MAX_FETCH_RETRIES:
                raise
            delay_seconds = NETWORK_BACKOFF_SECONDS * attempt
            logger.warning(
                "Network error fetching %s %s since=%s; retrying in %s seconds "
                "(attempt %s/%s): %s",
                symbol,
                timeframe,
                since,
                delay_seconds,
                attempt,
                MAX_FETCH_RETRIES,
                exc,
            )
            time.sleep(delay_seconds)


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
