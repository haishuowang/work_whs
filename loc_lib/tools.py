import pandas as pd
import os
import numpy as np


def AZ_Col_zscore(df, n, cap=float('nan')):
    target = df.rolling(window=n).apply(lambda x: (x[-1] - x.mean()) / x.std())
    target[target > cap] = cap
    target[target < cap] = -cap
    return target


def AZ_Row_zscore(df, cap=float('nan')):
    # target = df.apply(lambda x: (x - df.mean().values.ravel()) / df.std().values.ravel(), axis=0)
    target = (df.sub(df.sum(axis=1), axis=0)).div(df.std(axis=1), axis=0)
    target[target > cap] = cap
    target[target < cap] = -cap
    return target


def AZ_Rolling_sharpe(pnl_df, roll_year=1, year_len=250, cut_point_list=None, output=False):
    """
    rolling sharpe
    :param pnl_df:
    :param roll_year:
    :param year_len:
    :param cut_point_list:
    :param output:
    :return:
    """
    if cut_point_list is None:
        cut_point_list = [0.05, 0.33, 0.5, 0.66, 0.95]
    rolling_sharpe = pnl_df.rolling(int(roll_year * year_len))\
        .apply(lambda x: np.sqrt(year_len) * x.mean() / x.std())
    cut_sharpe = rolling_sharpe.quantile(cut_point_list)
    if output:
        return cut_sharpe
    else:
        return rolling_sharpe, cut_sharpe


def AZ_Normal_IR(signal, pct_n, lag=1):
    """
    计算IR
    :param signal:输入signal
    :param pct_n: 未来n天的return
    :param lag:
    :return:
    """
    signal = signal.shift(lag)
    signal = signal.replace(0, np.nan)
    corr_df = signal.corrwith(pct_n, axis=1).dropna()

    IC_mean = corr_df.mean()
    IC_std = corr_df.std()
    IR = IC_mean / IC_std
    return IC_mean, IC_std, IR


def AZ_Leverage_ratio(pnl_df):
    """
    返回250天的return/(负的 一个月的return)
    :param pnl_df:
    :return:
    """
    pnl_df = pd.Series(pnl_df)
    pnl_df_20 = pnl_df - pnl_df.shift(20)
    pnl_df_250 = pnl_df - pnl_df.shift(250)
    return pnl_df_250.mean()/(-pnl_df_20.min())


def AZ_Path_create(target_path):
    """
    添加路径
    :param target_path:
    :return:
    """
    if not os.path.exists(target_path):
        os.makedirs(target_path)


def AZ_split_stock(stock_list):
    """
    在所有代码中寻找A股代码
    :param stock_list:
    :return:
    """
    eqa = [x for x in stock_list if (x.startswith('0') or x.startswith('3')) and x.endwith('SZ')
           or x.startswith('6') and x.endwith('SH')]
    return eqa


def AZ_add_stock_suffix(stock_list):
    """
    给只有数字的 A股代码 添加后缀
    如 000001 运行后 000001.SZ
    :param stock_list:
    :return:
    """
    return list(map(lambda x: x + '.SH' if x.startswith('6') else x + '.SZ', stock_list))
