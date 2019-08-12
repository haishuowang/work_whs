import sys

sys.path.append('/mnf/mfs')
from work_whs.loc_lib.pre_load import *


class FutIndex:
    @staticmethod
    def CCI_fun(high, low, close, n, cap=5, return_line=False):
        tp = sum([high, low, close]) / 3
        ma_n = bt.AZ_Rolling_mean(tp, n)
        md_n = bt.AZ_Rolling_mean((ma_n - tp).abs(), n)
        tmp_df = (tp - ma_n) / md_n
        if cap:
            tmp_df[tmp_df > cap] = cap
            tmp_df[tmp_df < -cap] = -cap
        if return_line:
            return tmp_df, tp, ma_n, md_n
        else:
            return tmp_df

    @staticmethod
    def boll_fun(close, n, cap=5, return_line=False):
        ma_n = bt.AZ_Rolling_mean(close, n)
        md_n = bt.AZ_Rolling(close, n).std()
        tmp_df = (close - ma_n) / md_n
        if cap:
            tmp_df[tmp_df > cap] = cap
            tmp_df[tmp_df < -cap] = -cap
        if return_line:
            return tmp_df, ma_n, md_n
        else:
            return tmp_df

    @staticmethod
    def test_fun(close, n, cap=5, num=4, return_line=False):
        ma_n = bt.AZ_Rolling_mean(close, n)
        md_n = bt.AZ_Rolling(close, n).apply(lambda x: np.mean(abs(((x - x.mean()) ** num))) ** (1/num), raw=False)
        tmp_df = (close - ma_n) / md_n
        if cap:
            tmp_df[tmp_df > cap] = cap
            tmp_df[tmp_df < -cap] = -cap
        if return_line:
            return tmp_df, ma_n, md_n
        else:
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

    @staticmethod
    def fun_2(signal_df, hold_time):
        pos_df = bt.AZ_Rolling(signal_df, hold_time).mean()
        return pos_df

    @staticmethod
    def fun_3(signal_df, stop_df, method='ffill', limit=None):
        # stop_df为1, -1
        signal_df = signal_df.replace(0, np.nan)
        tmp_pos_df = signal_df.fillna(method=method, limit=limit).fillna(0)

        target_stop_df = stop_df

        tmp_df = (tmp_pos_df * target_stop_df)

        # x = tmp_df[tmp_df<0].astype(int).replace(0, np.nan).replace(1, 0)
        x = (tmp_df < 0).astype(int).replace(0, np.nan).replace(1, 0)
        pos_df = signal_df.add(x, fill_value=0).fillna(method=method, limit=limit).fillna(0)
        return pos_df
