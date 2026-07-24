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
        self._valid_end_time = convert_datetime_to_timestamp(settings.data_set.split.valid_end_date)
        self._test_end_time = convert_datetime_to_timestamp(settings.data_set.split.test_end_date)
        self._fast_rate = settings.data_set.fast_rate
        self._slow_rate = settings.data_set.slow_rate

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

        candle_high_body = df[['open', 'close']].max(axis=1)
        candle_low_body = df[['open', 'close']].min(axis=1)
        df['feat_range_high_low'] = 100 * (df['high'] / df['low'] - 1)
        df['feat_body_abs'] = df['feat_ret_close'].abs()
        df['feat_upper_shadow'] = 100 * (df['high'] / candle_high_body - 1)
        df['feat_lower_shadow'] = 100 * (candle_low_body / df['low'] - 1)

        for return_window in [1, 2, 4, 8, 16, 32]:
            df[f'feat_log_ret_{return_window}'] = 100 * np.log(df['close'] / df['close'].shift(return_window))

        for volatility_window in [8, 16, 32, 64]:
            df[f'volatility_{volatility_window}'] = df['feat_ret_close'].rolling(volatility_window).std()
            self._z_rolling_score(df, f'volatility_{volatility_window}')

        for ma_window in [3, 5, 20, 50, 100, 200]:
            df[f'ma{ma_window}'] = df['close'].rolling(ma_window).mean()
            df[f'feat_ma{ma_window}_distance'] = 100 * (df['close'] / df[f'ma{ma_window}'] - 1)
            df[f'ma{ma_window}_slope'] = df[f'ma{ma_window}'].pct_change()
            self._z_rolling_score(df, f'ma{ma_window}_slope')

        df['log_open']  = np.log(df['open'])
        df['log1p_volume'] = np.log1p(df['volume'])
        self._z_rolling_score(df, 'log_open')
        self._z_rolling_score(df, 'log1p_volume')

        for volume_window in [1, 4, 16]:
            df[f'volume_change_{volume_window}'] = df['log1p_volume'].diff(volume_window)
            self._z_rolling_score(df, f'volume_change_{volume_window}')

        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        return df

    def _build_labels(self, df):
        df['label_close_fast'] = 100 * np.log(df['ma3'].shift(-self._future_length) / df['ma3'])
        df['label_close_slow'] = 100 * np.log(df['ma5'].shift(-self._future_length ** 2) / df['ma5'])
        # fast_weight = 1.0
        # slow_weight = 1.0
        # total_fast_weight = 0.0
        # total_slow_weight = 0.0
        # for i in range(1, self._future_length + 1):
        #     df['label_close_fast'] += fast_weight * df['feat_ret_close'].shift(-i)
        #     total_fast_weight += fast_weight
        #     fast_weight *= self._fast_rate

        # for i in range(1, self._future_length ** 2 + 1):
        #     df['label_close_slow'] += slow_weight * df['feat_ret_close'].shift(-i)
        #     total_slow_weight += slow_weight
        #     slow_weight *= self._slow_rate

        # df['label_close_fast'] /= total_fast_weight
        # df['label_close_slow'] /= total_slow_weight
        
        return df
    
    def _z_rolling_score(self, df, column_name):
        res_column_name = f"feat_z_score_{column_name}"
        df[res_column_name] = (df[column_name] - df[column_name].rolling(self._window).mean()) / df[column_name].rolling(self._window).std()

    def _split_by_time(self, df):
        if self._train_end_time >= self._valid_end_time:
            raise ValueError("train_end_date must be earlier than valid_end_date")
        if self._valid_end_time >= self._test_end_time:
            raise ValueError("valid_end_date must be earlier than test_end_date")

        df = df.sort_values("open_time").reset_index(drop=True)

        train_df = df[df["open_time"] < self._train_end_time]
        valid_df  = df[(df["open_time"] >= self._train_end_time) & (df["open_time"] < self._valid_end_time)]
        test_df  = df[(df["open_time"] >= self._valid_end_time)  & (df["open_time"] < self._test_end_time)]

        train_df = self._trim_future_boundary(train_df)
        valid_df = self._trim_future_boundary(valid_df)
        test_df  = self._trim_future_boundary(test_df)

        return {
            "train": train_df.reset_index(drop=True),
            "valid": valid_df.reset_index(drop=True),
            "test": test_df.reset_index(drop=True),
        }

    def _trim_future_boundary(self, df):
        if len(df) <= self._future_length ** 2:
            return df.iloc[:0].copy()
        return df.iloc[:-self._future_length ** 2].dropna(subset=["label_close_slow"]).copy()

def test():
    datasetBuilder = DatasetBuilder("BTC/USDT", "4h")
    all_dp = datasetBuilder.build_from_db()
    all_dp.to_csv("total.csv")
    splits = datasetBuilder.build_splits_from_db()
    for name, df in splits.items():
        df.to_csv(f"{name}.csv")
        print(name, len(df))

# test()
