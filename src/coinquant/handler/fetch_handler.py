import logging
from enum import Enum
from typing import List

from coinquant.datasource.database import dataBase
from coinquant.datasource.ccxt_source import (
    convert_datetime_to_timestamp,
    fetch_ohlcv,
    iter_ohlcv_batches,
)
from coinquant.config import settings

logger = logging.getLogger(__name__)


class FetchMode(str, Enum):
    """Execution modes supported by the fetch workflow."""

    incremental = "incremental"
    full = "full"


class FetchHandler:
    """Handler for fetching historical data."""

    def __init__(self, mode: FetchMode):
        self._mode = mode
        self._db = dataBase(settings.path.db_path)
        self._begin_time = convert_datetime_to_timestamp(settings.data.begin_date)
        self._end_time = convert_datetime_to_timestamp(settings.data.end_date)

    def fetch(self):
        if self._mode == FetchMode.incremental:
            self._fetch_incremental()
        elif self._mode == FetchMode.full:
            self._fetch_full()
        else:
            raise ValueError(f"Unsupported fetch mode: {self._mode}")

    def _fetch_incremental(self):
        # Logic to fetch only missing data
        logger.info("Fetching incremental data...")
        for period in settings.data.period_list:
            for symbol in settings.data.coin_list:
                (_, max_time) = self._db.time_range(period, symbol)
                begin_time = max_time if max_time is not None else self._begin_time
                self._fetch_and_store(symbol, period, begin_time)

    def _fetch_full(self):
        # Logic to fetch all data for the specified range
        logger.info("Fetching full data...")
        for symbol in settings.data.coin_list:
            for period in settings.data.period_list:
                self._fetch_and_store(symbol, period, self._begin_time)

    def _fetch_and_store(self, symbol: str, period: str, begin_time: int) -> int:
        if begin_time > self._end_time:
            logger.info(
                "Skipping %s %s because begin_time is after end_date",
                symbol,
                period,
            )
            return 0

        total_rows = 0
        
        for data in iter_ohlcv_batches(symbol, period, begin_time, self._end_time):
            stored_rows = self._store_data(symbol, period, data)
            total_rows += stored_rows
            logger.info(
                "Stored %s rows for %s %s",
                stored_rows,
                symbol,
                period,
            )
        logger.info("Finished %s %s: %s rows", symbol, period, total_rows)
        return total_rows

    def _store_data(self, symbol: str, period: str, data: List[list]) -> int:
        # Logic to store fetched data into the database
        rows = [[symbol, period, *row] for row in data]
        return self._db.insert_rows(rows)


def test():
    print(settings.data.coin_list)

# test()
