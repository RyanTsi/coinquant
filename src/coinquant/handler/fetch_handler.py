import logging
from enum import Enum

from coinquant.datasource.database import dataBase
from coinquant.datasource.ccxt_source import ccxtSourceFetcher, timeframe_to_milliseconds
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
        self._begin_time = settings.data.begin_date
        self._end_time = settings.data.end_date
        self._fetcher = ccxtSourceFetcher()

    def fetch(self):
        if self._mode == FetchMode.incremental:
            self._fetch_incremental()
        elif self._mode == FetchMode.full:
            self._fetch_full()
        else:
            raise ValueError(f"Unsupported fetch mode: {self._mode}")

    def count(self) -> int:
        """Count the number of rows in the database."""
        return self._db.count()

    def _fetch_incremental(self):
        # Logic to fetch only missing data
        logger.info("Fetching incremental data...")
        for period in settings.data.period_list:
            for symbol in settings.data.coin_list:
                (_, max_time) = self._db.time_range(period, symbol)
                begin_time = max_time if max_time is not None else self._begin_time
                total_rows = self._fetch_and_store(symbol, period, begin_time)
                logger.info(f"{total_rows} rows fetched for {symbol} at {period}")

    def _fetch_full(self):
        # Logic to fetch all data for the specified range
        logger.info("Fetching full data...")
        for symbol in settings.data.coin_list:
            for period in settings.data.period_list:
                total_rows = self._fetch_and_store(symbol, period, self._begin_time)
                logger.info(f"{total_rows} rows fetched for {symbol} at {period}")

    def _fetch_and_store(
        self,
        symbol: str,
        period: str,
        begin_time: str | int | float,
    ) -> int:
        begin_timestamp = self._fetcher.convert_datetime_to_timestamp(begin_time)
        end_timestamp = self._fetcher.convert_datetime_to_timestamp(self._end_time)

        if begin_timestamp > end_timestamp:
            logger.info(
                "Skipping %s %s because begin_time is after end_date",
                symbol,
                period,
            )
            return 0

        total_rows = 0
        for candles in self._fetcher.fetch_ohlcv_pages(
            symbol,
            period,
            begin_timestamp,
            end_timestamp,
        ):
            total_rows += self._store_data(symbol, period, candles)
        return total_rows

    def _store_data(self, symbol: str, period: str, data: list[list]) -> int:
        # Logic to store fetched data into the database
        rows = ((symbol, period, *row) for row in data)
        return self._db.insert_rows(rows)
    


def test():
    print(settings.data.coin_list)

# test()
