from datetime import datetime, timezone

def convert_datetime_to_timestamp(dt_str) -> int:
    if dt_str.endswith("Z"):
        dt_str = f"{dt_str[:-1]}+00:00"
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)
