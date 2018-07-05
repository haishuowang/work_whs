import pandas as pd
import numpy as np
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
    market_top_n.to_pickle(os.path.join(save_way, 'market_top_{}.pkl'.format(n)))
    return market_top_n


if __name__ == '__main__':
    for n in [100, 300, 500, 800, 1000]:
        print(n)
        begin_date = pd.to_datetime('20080401')
        end_date = pd.to_datetime('20180401')
        data = pd.read_pickle('/mnt/mfs/DAT_EQT/EM_Funda/daily/R_MarketCap_Only.pkl')
        data_df = pd.DataFrame(data['values'], columns=data['columns'], index=data['index'])
        data_df.index = pd.to_datetime(data_df.index)
        save_way = '/mnt/mfs/DAT_EQT/STK_Groups1'
        market_top_n = market_top_n_fun(begin_date, end_date, data_df, n, save_way)
