from coinquant.datasource.database import dataBase
from coinquant.config import settings
from coinquant.utils import convert_datetime_to_timestamp

def export_samples_to_csv(
    db_path: str, 
    output_dir: str, 
    symbol: str | None = None,
    period: str | None = None,
    begin_time: str = settings.data.begin_date,
    end_time: str = settings.data.begin_date,
) -> None:
    db = dataBase(db_path)
    data = db.query(period, symbol, convert_datetime_to_timestamp(begin_time), convert_datetime_to_timestamp(end_time))
    data.to_csv(f"{output_dir}/exported_data.csv", index=False)