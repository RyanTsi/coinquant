from pathlib import Path

import pandas as pd

from coinquant.config import settings
from coinquant.datasource.database import KLINE_COLUMNS, DataBase
from coinquant.utils import convert_datetime_to_timestamp


def export_samples_to_csv(
    output_dir: str,
    symbol: str | None = None,
    period: str | None = None,
    begin_time: str = settings.data.begin_date,
    end_time: str = settings.data.end_date,
) -> None:
    db = DataBase()
    data = db.query(
        period,
        symbol,
        convert_datetime_to_timestamp(begin_time),
        convert_datetime_to_timestamp(end_time),
    )
    output_path = Path(output_dir) / "exported_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame.from_records(data, columns=KLINE_COLUMNS)
    frame.to_csv(output_path, index=False)
