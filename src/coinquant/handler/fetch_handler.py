import logging
from enum import Enum
from typing import List

from coinquant.datasource.database import dataBase
from coinquant.datasource.ccxt_source import fetch_ohlcv, convert_datetime_to_timestamp
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
            (_, max_time) = self._db.time_range(period)
            begin_time = max_time if max_time is not None else convert_datetime_to_timestamp(settings.data.begin_date)
            for symbol in settings.data.coin_list:
                data = fetch_ohlcv(symbol, period, begin_time)
                self._store_data(symbol, period, data)


    def _fetch_full(self):
        # Logic to fetch all data for the specified range
        logger.info("Fetching full data...")
        for symbol in settings.data.coin_list:
            for period in settings.data.period_list:
                data = fetch_ohlcv(symbol, period, convert_datetime_to_timestamp(settings.data.begin_date))
                self._store_data(symbol, period, data)


    def _fetch_by_timestamp(self, symbol: str, period: str, timestamp: int):
        return fetch_ohlcv(symbol, period, timestamp)

    def _store_data(self, symbol: str, period: str, data: List[list]):
        # Logic to store fetched data into the database
        for row in data:
            row.insert(0, symbol)  # Insert symbol at the beginning
            row.insert(1, period)  # Insert period after symbol
        self._db.insert_rows(data)

def test():
    print(settings.data.coin_list)

# test()
