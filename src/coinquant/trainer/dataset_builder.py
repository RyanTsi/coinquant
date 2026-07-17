import numpy as np
import pandas as pd

from coinquant.datasource.database import DataBase, KLINE_COLUMNS
from coinquant.config import settings
from coinquant.utils import convert_datetime_to_timestamp

class DatasetBuilder:
    def __init__(self, symbol, period):
        self._begin_time = convert_datetime_to_timestamp(settings.data.begin_date)
        self._end_time = convert_datetime_to_timestamp(settings.data.end_date)
        self._symbol = symbol
        self._period = period
        self._window = settings.data_set.rolling_window
        self._future_length = settings.data_set.future_length
        self._train_end_time = convert_datetime_to_timestamp(settings.data_set.split.train_end_date)
        self._val_end_time = convert_datetime_to_timestamp(settings.data_set.split.val_end_date)
        self._test_end_time = convert_datetime_to_timestamp(settings.data_set.split.test_end_date)
        self._short_rate = settings.data_set.short_rate
        self._long_rate  = settings.data_set.long_rate

    def build_from_db(self):
        db = DataBase()
        data = db.query(self._period, self._symbol, self._begin_time, self._end_time)
        df = pd.DataFrame(data, columns=KLINE_COLUMNS)
        df = self._build_features(df)
        self._build_labels(df)
        
        return df

    def build_splits_from_db(self):
        df = self.build_from_db()
        return self._split_by_time(df)

    def _build_features(self, df):
        df['feat_ret_high']  = 100 * (df['high'] / df['open'] - 1)
        df['feat_ret_low']   = 100 * (df['low'] / df['open'] - 1)
        df['feat_ret_close'] = 100 * (df['close'] / df['open'] - 1)
        for ma_window in [20, 50, 100, 200]:
            df[f'ma{ma_window}'] = df['close'].rolling(ma_window).mean()
            df[f'ma{ma_window}_slope'] = df[f'ma{ma_window}'].pct_change()
            self._z_rolling_score(df, f'ma{ma_window}_slope')
            # df[f'feat_ma{ma_window}'] = 100 * (df[f'ma{ma_window}'] / df['open'] - 1)
        df['log_open']  = np.log(df['open'])
        df['log1p_volume'] = np.log(df['volume'])
        self._z_rolling_score(df, 'log_open')
        self._z_rolling_score(df, 'log1p_volume')
        df = df.dropna()
        return df

    def _build_labels(self, df):
        df['label_close_short'] = 0
        df['label_close_long'] = 0
        short_weight = 1.0
        long_weight  = 1.0 

        for i in range(1, self._future_length + 1):
            df['label_close_short'] += short_weight * df['feat_ret_close'].shift(-i)
            short_weight *= self._short_rate

        for i in range(1, self._future_length ** 2 + 1):
            df['label_close_long'] += long_weight * df['feat_ret_close'].shift(-i)
            long_weight *= self._long_rate

        return df
    
    def _z_rolling_score(self, df, column_name):
        res_column_name = f"feat_z_score_{column_name}"
        df[res_column_name] = (df[column_name] - df[column_name].rolling(self._window).mean()) / df[column_name].rolling(self._window).std()

    def _split_by_time(self, df):
        if self._train_end_time >= self._val_end_time:
            raise ValueError("train_end_date must be earlier than val_end_date")
        if self._val_end_time >= self._test_end_time:
            raise ValueError("val_end_date must be earlier than test_end_date")

        df = df.sort_values("open_time").reset_index(drop=True)

        train_df = df[df["open_time"] < self._train_end_time]
        val_df   = df[(df["open_time"] >= self._train_end_time) & (df["open_time"] < self._val_end_time)]
        test_df  = df[(df["open_time"] >= self._val_end_time)  & (df["open_time"] < self._test_end_time)]

        train_df = self._trim_future_boundary(train_df)
        val_df = self._trim_future_boundary(val_df)
        test_df = self._trim_future_boundary(test_df)

        return {
            "train": train_df.reset_index(drop=True),
            "val": val_df.reset_index(drop=True),
            "test": test_df.reset_index(drop=True),
        }

    def _trim_future_boundary(self, df):
        if len(df) <= self._future_length ** 2:
            return df.iloc[:0].copy()
        return df.iloc[:-self._future_length ** 2].dropna(subset=["label_close_long"]).copy()

def test():
    datasetBuilder = DatasetBuilder("BTC/USDT", "4h")
    all_dp = datasetBuilder.build_from_db()
    all_dp.to_csv("totle.csv")
    splits = datasetBuilder.build_splits_from_db()
    for name, df in splits.items():
        df.to_csv(f"{name}.csv")
        print(name, len(df))

test()
