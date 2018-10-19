import pandas as pd
import numpy as np
from datetime import datetime
import os


def sort_fun(x, n=1000):
    target = np.array([float('nan')] * len(x))
    target[x <= n] = 1
    target[x > n] = np.nan
    return target


def market_top_n_fun(begin_date, end_date, data_df, n, save_way):
    data_mean = data_df.rolling(window=120, min_periods=1).mean()
    data_mean.iloc[:119] = np.nan
    data_mean_cut = data_mean[(data_mean.index >= begin_date) & (data_mean.index < end_date)]
    data_sort = data_mean_cut.rank(axis=1, ascending=False, na_option='bottom')
    market_top_n = data_sort.apply(sort_fun, args=(n,), axis=1)
    market_top_n = market_top_n.dropna(how='all', axis='columns')
    # market_top_n.to_pickle(os.path.join(save_way, 'market_top_{}.pkl'.format(n)))
    return market_top_n


if __name__ == '__main__':
    # for n in [100, 300, 500, 800, 1000]:
    n = 800
    print(n)
    begin_date = pd.to_datetime('20080401')
    end_date = datetime.now()
    data_df = pd.read_csv('/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TVALCNY.csv',
                          sep='|', index_col=0, parse_dates=True)
    # data_df = pd.DataFrame(data['values'], columns=data['columns'], index=data['index'])
    data_df.index = pd.to_datetime(data_df.index)
    save_way = '/mnt/mfs/DAT_EQT/STK_Groups1'
    turnover_top_800 = market_top_n_fun(begin_date, end_date, data_df, n, save_way)

    HS300 = pd.read_csv('/mnt/mfs/DAT_EQT/EM_Funda/IDEX_YS_WEIGHT_A/SECURITYNAME_000300.csv',
                        index_col=0, parse_dates=True, sep='|', low_memory=False)

    ZZ500 = pd.read_csv('/mnt/mfs/DAT_EQT/EM_Funda/IDEX_YS_WEIGHT_A/SECURITYNAME_000905.csv',
                        index_col=0, parse_dates=True, sep='|', low_memory=False)

    xnms = sorted(set(turnover_top_800.columns) | set(ZZ500.columns) | set(HS300.columns))
    xinx = sorted(set(turnover_top_800.index) | set(ZZ500.index) | set(HS300.index))

    HS300 = HS300.reindex(columns=xnms, index=xinx)
    ZZ500 = ZZ500.reindex(columns=xnms, index=xinx)
    turnover_top_800 = turnover_top_800.reindex(columns=xnms, index=xinx)

    ZZ500_mask = ZZ500.notna()
    HS300_mask = HS300.notna()
    turnover_top_800_mask = turnover_top_800.notna()

    a = HS300_mask | ZZ500_mask | turnover_top_800_mask
    b = a.astype(float).replace(0, np.nan)
