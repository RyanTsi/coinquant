from typing import Any

import duckdb
from pathlib import Path
from coinquant.config import settings

KLINE_COLUMNS = (
    "symbol",
    "period",
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS klines (
    symbol VARCHAR NOT NULL,
    period VARCHAR NOT NULL,
    open_time BIGINT NOT NULL,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    PRIMARY KEY (symbol, period, open_time)
);
"""

class dataBase:
    def __init__(self, db_path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(db_path)
        self.create_table()

    def create_table(self):
        self._conn.execute(_CREATE_TABLE_SQL)

    def insert_rows(self, rows: list[tuple[Any, ...]]) -> int:
        if not rows:
            return 0
        self._conn.executemany(
            f"""
            INSERT OR REPLACE INTO klines ({", ".join(KLINE_COLUMNS)})
            VALUES ({", ".join("?" for _ in KLINE_COLUMNS)})
            """,
            rows,
        )
        return len(rows)

    def count(self, period: str | None = None, symbol: str | None = None) -> int:
        """Return row count, optionally filtered by ``period`` and ``symbol``."""
        if period is None and symbol is None:
            result = self._conn.execute("SELECT COUNT(*) FROM klines").fetchone()
            return int(result[0]) if result else 0
        if period is None and symbol is not None:
            result = self._conn.execute(
                "SELECT COUNT(*) FROM klines WHERE symbol = ?",
                [symbol],
            ).fetchone()
            return int(result[0]) if result else 0
        if period is not None and symbol is None:
            result = self._conn.execute(
                "SELECT COUNT(*) FROM klines WHERE period = ?",
                [period],
            ).fetchone()
            return int(result[0]) if result else 0
        if period is not None and symbol is not None:
            result = self._conn.execute(
                "SELECT COUNT(*) FROM klines WHERE period = ? AND symbol = ?",
                [period, symbol],
            ).fetchone()
            return int(result[0]) if result else 0

    def query(self, period: str | None = None, symbol: str | None = None, start_time: int = 0) -> list[tuple]:
        if period is None and symbol is None:
            result = self._conn.execute("SELECT * FROM klines").fetchall()
        elif period is None and symbol is not None:
            result = self._conn.execute(
                "SELECT * FROM klines WHERE symbol = ? AND open_time >= ?",
                [symbol, start_time],
            ).fetchall()
        elif period is not None and symbol is None:
            result = self._conn.execute(
                "SELECT * FROM klines WHERE period = ? AND open_time >= ?",
                [period, start_time],
            ).fetchall()
        else:
            result = self._conn.execute(
                "SELECT * FROM klines WHERE period = ? AND symbol = ? AND open_time >= ?",
                [period, symbol, start_time],
            ).fetchall()
        return result

    def time_range(self, period: str) -> tuple[int | None, int | None]:
        """Return (min_open_time, max_open_time) for a period."""
        result = self._conn.execute(
            "SELECT MIN(open_time), MAX(open_time) FROM klines WHERE period = ?",
            [period],
        ).fetchone()
        if not result:
            return None, None
        return result[0], result[1]
    
def test():
    db = dataBase(settings.path.db_path)
    print(db.count())
    print(db.query(symbol="BTCUSDT"))
    print(db.time_range("1m"))

test()