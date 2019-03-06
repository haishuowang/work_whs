import pandas as pd
import numpy as np


def intra_vwap_tab_f1d(factor_df):
    for i in range(12):
        vwap_load_path = '/mnt/mfs/dat_whs/data/base_data/intra_vwap_tab_{}.pkl'.format(i+1)
        vwap_df = pd.read_pickle(vwap_load_path)
        vwap_df.columns = [x[2:] + '.' + x[:2] for x in vwap_df.columns]

        xnms = sorted(list(set(factor_df.columns) & set(vwap_df.columns)))
        xinx = sorted(list(set(factor_df.index) & set(vwap_df.index)))

        factor_df = factor_df.reindex(index=xinx, columns=xnms)
        vwap_df = vwap_df.reindex(index=xinx, columns=xnms)

        adj_vwap = factor_df * vwap_df
        adj_vwap_f_return = adj_vwap.shift(-1)/adj_vwap - 1
        adj_vwap_f_return.to_pickle('/mnt/mfs/dat_whs/data/return_data/intra_vwap_tab_{}_f1d.pkl'.format(i+1))


def intra_vwap_tab_fnd_to_open_fun(factor_df, n=1):
    for i in range(12)[:1]:
        vwap_load_path = '/mnt/mfs/dat_whs/data/base_data/intra_vwap_tab_{}.pkl'.format(i + 1)
        vwap_df = pd.read_pickle(vwap_load_path)
        vwap_df.columns = [x[2:] + '.' + x[:2] for x in vwap_df.columns]
        vwap_df_1 = pd.read_pickle('/mnt/mfs/dat_whs/data/base_data/intra_vwap_tab_1.pkl')
        vwap_df_1.columns = [x[2:] + '.' + x[:2] for x in vwap_df_1.columns]

        xnms = sorted(list(set(factor_df.columns) & set(vwap_df.columns)))
        xinx = sorted(list(set(factor_df.index) & set(vwap_df.index)))

        vwap_df = vwap_df.reindex(index=xinx, columns=xnms)
        vwap_df_1 = vwap_df_1.reindex(index=xinx, columns=xnms)
        factor_df = factor_df.reindex(index=xinx, columns=xnms)

        adj_vwap = factor_df * vwap_df
        adj_vwap_1 = factor_df * vwap_df_1

        open_sell_return = adj_vwap_1.shift(-n) / adj_vwap - 1
        open_sell_return.to_pickle('/mnt/mfs/dat_whs/data/return_data/intra_vwap_tab_{}_f1d_open.pkl'.format(i + 1))


def open_1_hour_vwap_fun(factor_df):
    for i in range(4):
        data = pd.read_pickle('/mnt/mfs/dat_whs/data/base_data/intra_vwap_60_tab_{}.pkl'.format(i+1))
        data.columns = [x[2:] + '.' + x[:2] for x in data.columns]

        xnms = sorted(list(set(factor_df.columns) & set(data.columns)))
        xinx = sorted(list(set(factor_df.index) & set(data.index)))

        data = data.reindex(index=xinx, columns=xnms)
        factor_df = factor_df.reindex(index=xinx, columns=xnms)

        adj_data = data * factor_df
        open_1h_vwap = adj_data.shift(-1) / adj_data - 1
        open_1h_vwap.to_pickle('/mnt/mfs/dat_whs/data/return_data/intra_vwap_60_tab_{}_f1d.pkl'.format(i + 1))


if __name__ == '__main__':
    factor_df = pd.read_csv('/mnt/mfs/dat_whs/data/AllStock/all_tafactor.csv', index_col=0)
    factor_df.index = pd.to_datetime(factor_df.index.astype(str))
    open_1_hour_vwap_fun(factor_df.copy())
    # intra_vwap_tab_fnd_to_open_fun(factor_df.copy(), n=1)
    # intra_vwap_tab_f1d(factor_df.copy())
