import sys

sys.path.append('/mnf/mfs')
from work_whs.loc_lib.pre_load import *


class FutIndex:
    @staticmethod
    def CCI_fun(high, low, close, n):
        tp = sum([high, low, close]) / 3
        ma_n = bt.AZ_Rolling_mean(tp, n)
        md_n = bt.AZ_Rolling_mean((ma_n - tp).abs(), n)
        tmp_df = (tp - ma_n) / md_n
        return tmp_df

    @staticmethod
    def boll_fun(close, n):
        ma_n = bt.AZ_Rolling_mean(close, n)
        md_n = bt.AZ_Rolling(close, n).std()
        tmp_df = (close - ma_n)/md_n
        return tmp_df


class Signal:
    @staticmethod
    def fun_1(tmp_df, limit):
        target_df_up = (tmp_df > limit).astype(int)
        target_df_dn = (tmp_df < -limit).astype(int)
        target_df = target_df_up - target_df_dn
        return target_df


class Position:
    @staticmethod
    def fun_1(signal_df, method='ffill', limit=None):
        pos_df = signal_df.replace(0, np.nan).fillna(method=method, limit=limit).fillna(0)
        return pos_df
