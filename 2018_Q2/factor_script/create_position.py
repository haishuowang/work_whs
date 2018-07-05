import pandas as pd
import numpy as np
import os
from functools import reduce
import open_lib.shared_tools.back_test as bt

root_path = '/mnt/mfs/dat_whs'
# load_path = '/media/hdd0/whs/data/AllStock'
load_path = root_path + '/data/AllStock'
all_open = pd.read_csv(os.path.join(load_path, 'all_open.csv'), index_col=0)
all_high = pd.read_csv(os.path.join(load_path, 'all_high.csv'), index_col=0)
all_low = pd.read_csv(os.path.join(load_path, 'all_low.csv'), index_col=0)
all_close = pd.read_csv(os.path.join(load_path, 'all_close.csv'), index_col=0)
all_amount = pd.read_csv(os.path.join(load_path, 'all_amount.csv'), index_col=0)

all_volume = pd.read_csv(os.path.join(load_path, 'all_volume.csv'), index_col=0)
all_adj_r = pd.read_csv(os.path.join(load_path, 'all_adj_r.csv'), index_col=0)


def split_stock(stock_list):
    eqa = [x for x in stock_list if (x.startswith('0') or x.startswith('3')) and x.endwith('SZ')
           or x.startswith('6') and x.endwith('SH')]
    return eqa


def add_stock_suffix(stock_list):
    return list(map(lambda x: x + '.SH' if x.startswith('6') else x + '.SZ', stock_list))


def split_fun(df, n=3):
    split_list = [0.] + [1 / n * (i + 1) for i in range(n)]
    mid = (n + 1) / 2
    df_sort = df.rank(axis=1, method='first', pct=True)
    target_df = pd.DataFrame(0, index=df.index, columns=df.columns)
    for i in range(n):
        target_df[(split_list[i + 1] >= df_sort) & (df_sort > split_list[i])] = i + 1 - mid
    return target_df


# def fnd_pct(close, n=5):
#     pct_n = close / close.shift(n) - 1
#     return pct_n


def fnd_pct_adj(adj_r, n=5):
    adj_r_fillna = adj_r.fillna(0)
    return (adj_r_fillna + 1).rolling(window=n, min_periods=1) \
        .apply(lambda a: reduce(lambda x, y: x * y, a) - 1).fillna(0).shift(-n)


###############################################################################################################


def pnd_hl(high, low, close, n):
    high_n = high.rolling(window=n, min_periods=1).max().shift(1)
    low_n = low.rolling(window=n, min_periods=1).min().shift(1)
    h_diff = (close - high_n)
    l_diff = (close - low_n)

    h_diff[h_diff > 0] = 1
    h_diff[h_diff <= 0] = 0

    l_diff[l_diff >= 0] = 0
    l_diff[l_diff < 0] = -1

    pos = h_diff + l_diff
    return pos


def pnd_vol(close, n=5, n_split=3):
    vol = close.rolling(window=n, min_periods=1).std() / close.rolling(window=n, min_periods=1).mean()
    return split_fun(vol, n_split)


def pnd_volume(volume, n=5, n_split=3):
    volume_n = volume.rolling(window=n, min_periods=1).sum()
    return split_fun(volume_n, n_split)


def pnd_std(close, n=5, limit=2.5):
    signal = (close - close.rolling(window=n, min_periods=1).mean()) / close.rolling(window=n, min_periods=1).std()
    signal[(signal < limit) & (signal < limit)] = 0
    signal[signal >= limit] = 1
    signal[signal <= -limit] = -1
    return signal


def return_r(sector_df, n=5):
    pct_n = pd.read_pickle(root_path + '/data/adj_data/fnd_pct/pct_f{0}d.pkl'.format(n))
    pct_n = pct_n.shift(n)[sector_df.columns] * sector_df
    r_r_zscore = bt.AZ_Row_zscore(pct_n, n)
    return r_r_zscore


def return_c(sector_df, n=5):
    pct_n = pd.read_pickle(root_path + '/data/adj_data/fnd_pct/pct_f{0}d.pkl'.format(n))
    pct_n = pct_n.shift(n)[sector_df.columns] * sector_df
    r_c_zscore = bt.AZ_Row_zscore(pct_n)
    return r_c_zscore


def volume_r(volume, n=5):
    volume_n = volume.rolling(window=n, min_periods=1).sum()
    v_r_zscore = bt.AZ_Row_zscore(volume_n, n)
    return v_r_zscore


def volume_c(volume, n=5):
    volume_n = volume.rolling(window=n, min_periods=1).sum()
    v_c_zscore = bt.AZ_Row_zscore(volume_n)
    return v_c_zscore


def extreme_data(zscore_df, limit=2):
    zscore_df[(zscore_df <= limit) & (zscore_df >= -limit)] = 0
    zscore_df[zscore_df > limit] = 1
    zscore_df[zscore_df < -limit] = -1
    return zscore_df


def pnd_continue_ud(close, n=3):
    return close.rolling(window=n).apply(lambda x: 1 if (np.diff(x) >= 0).all() and sum(np.diff(x)) > 0
    else (-1 if (np.diff(x) <= 0).all() and sum(np.diff(x)) < 0 else 0))


def pnd_sep_1d_ud(close, n=5):
    use_list = [i for i in range(n) if i % 2 == 0]
    target_df = close.rolling(window=n) \
        .apply(lambda x: 1 if (np.diff(x[use_list]) >= 0).all() and sum(np.diff(x[use_list])) > 0
               else (-1 if (np.diff(x[use_list]) <= 0).all() and sum(np.diff(x[use_list])) < 0 else 0))
    return target_df


def p1d_jump_hl(close, open, split_float=0.05):
    jump_df = open / close.shift(1) - 1
    target_df = pd.DataFrame(index=jump_df.index, columns=jump_df.columns)
    target_df[(jump_df > 0.101) | (jump_df < -0.101)] = 0
    target_df[(split_float >= jump_df) & (jump_df >= -split_float)] = 0
    target_df[jump_df > split_float] = 1
    target_df[jump_df < -split_float] = -1
    return target_df


def pnnd_moment(close, n_short=10, n_long=60):
    ma_long = close.rolling(window=n_long, min_periods=1).mean()
    ma_short = close.rolling(window=n_short, min_periods=1).mean()
    ma_dif = ma_short - ma_long
    ma_dif[ma_dif == 0] = 0
    ma_dif[ma_dif > 0] = 1
    ma_dif[ma_dif < 0] = -1
    return ma_dif


def pnnd_liquid(amount, n_short=10, n_long=60):
    ma_long = amount.rolling(window=n_long, min_periods=1).mean()
    ma_short = amount.rolling(window=n_short, min_periods=1).mean()
    ma_dif = ma_short - ma_long
    ma_dif[ma_dif == 0] = 0
    ma_dif[ma_dif > 0] = 1
    ma_dif[ma_dif < 0] = -1
    return ma_dif


####################################################################################################################
# def fnd_pct_adj_set(para_list):
#     for n in para_list:
#         print('pct_f{0}d'.format(n))
#         index_save_path = root_path + '/data/adj_data/fnd_pct/pct_f{0}d.pkl'.format(n)
#         fnd_pct_adj_df = fnd_pct_adj(EQA_adj_r, n)
#         fnd_pct_adj_df.to_pickle(index_save_path)


def pnd_hl_set(para_list, index_root_path):
    for n in para_list:
        print('hl_p{0}d'.format(n))
        index_save_path = os.path.join(index_root_path, 'hl_p{0}d.pkl'.format(n))
        pnd_hl_df = pnd_hl(EQA_high, EQA_low, EQA_close, n)
        pnd_hl_df.to_pickle(index_save_path)


def pnd_vol_set(para_list, index_root_path):
    for n in para_list:
        print('vol_p{0}d'.format(n))
        index_save_path = os.path.join(index_root_path, 'vol_p{0}d.pkl'.format(n))
        pnd_vol_df = pnd_vol(EQA_close, n)
        pnd_vol_df.to_pickle(index_save_path)


def pnd_volume_set(para_list, index_root_path):
    for n in para_list:
        print('volume_p{0}d'.format(n))
        index_save_path = os.path.join(index_root_path, 'volume_p{0}d.pkl'.format(n))
        pnd_volume_df = pnd_volume(EQA_volume, n)
        pnd_volume_df.to_pickle(index_save_path)


def return_r_set(para_list, index_root_path, limit_list, sector_df):
    for n in para_list:
        rrn_df = return_r(sector_df, n)
        rrn_df = rrn_df[sector_df.columns]
        for limit in limit_list:
            print('rr{}_ext_{}'.format(n, limit))
            index_save_path = os.path.join(index_root_path, 'rr{}_ext_{}.pkl'.format(n, limit))
            rrn_ext_df = extreme_data(rrn_df * sector_df, limit)
            rrn_ext_df.to_pickle(index_save_path)


def return_c_set(para_list, index_root_path, limit_list, sector_df):
    for n in para_list:
        rcn_df = return_c(sector_df, n)
        rcn_df = rcn_df[sector_df.columns]
        for limit in limit_list:
            print('rc{}_ext_{}'.format(n, limit))
            index_save_path = os.path.join(index_root_path, 'rc{}_ext_{}.pkl'.format(n, limit))
            rcn_ext_df = extreme_data(rcn_df, limit)
            rcn_ext_df.to_pickle(index_save_path)


def volume_r_set(para_list, index_root_path, limit_list, sector_df):
    for n in para_list:
        vrn_df = volume_r(EQA_close, n)
        for limit in limit_list:
            print('vr{}_ext_{}'.format(n, limit))
            index_save_path = os.path.join(index_root_path, 'vr{}_ext_{}.pkl'.format(n, limit))
            vrn_ext_df = extreme_data(vrn_df * sector_df, limit)
            vrn_ext_df.to_pickle(index_save_path)


def volume_c_set(para_list, index_root_path, limit_list):
    for n in para_list:
        vcn_df = volume_c(EQA_close, n)
        for limit in limit_list:
            index_save_path = os.path.join(index_root_path, 'vc{}_ext_{}.pkl'.format(n, limit))
            print('vc{}_ext_{}'.format(n, limit))
            vcn_ext_df = extreme_data(vcn_df, limit)
            vcn_ext_df.to_pickle(index_save_path)


def pnd_continue_ud_set(continue_list, index_root_path):
    for n in continue_list:
        print('continue_ud_p{}d'.format(n))
        index_save_path = os.path.join(index_root_path, 'continue_ud_p{}d.pkl'.format(n))
        continue_ud_df = pnd_continue_ud(EQA_close, n)
        continue_ud_df.to_pickle(index_save_path)


def pnd_sep_1d_ud_set(sep_1d_list, index_root_path):
    for n in sep_1d_list:
        print('sep_1d_ud_p{}d'.format(n))
        index_save_path = os.path.join(index_root_path, 'sep_1d_ud_p{}d.pkl'.format(n))
        sep_1d_ud_df = pnd_sep_1d_ud(EQA_close, n)
        sep_1d_ud_df.to_pickle(index_save_path)


def p1d_jump_hl_set(index_root_path, split_float_list):
    for split_float in split_float_list:
        print('jump_hl_split_{}_p1d.pkl'.format(split_float))
        index_save_path = os.path.join(index_root_path, 'jump_hl_split_{}_p1d.pkl'.format(split_float))
        p1d_jump_hl_df = p1d_jump_hl(EQA_close, EQA_open, split_float)
        p1d_jump_hl_df.to_pickle(index_save_path)


def pnnd_moment_set(short_long_list):
    for n_short, n_long in short_long_list:
        print('moment_s{}_l{}'.format(n_short, n_long))
        index_save_path = os.path.join(index_root_path, 'moment_s{}_l{}.pkl'.format(n_short, n_long))
        pnnd_moment_df = pnnd_moment(EQA_close, n_short, n_long)
        pnnd_moment_df.to_pickle(index_save_path)


def pnnd_liquid_set(short_long_list):
    for n_short, n_long in short_long_list:
        print('liquid_s{}_l{}'.format(n_short, n_long))
        index_save_path = os.path.join(index_root_path, 'liquid_s{}_l{}.pkl'.format(n_short, n_long))
        pnnd_liquid_df = pnnd_liquid(EQA_close, n_short, n_long)
        pnnd_liquid_df.to_pickle(index_save_path)


####################################################################################################################
if __name__ == '__main__':
    root_save_path = '/mnt/mfs/dat_whs/data/factor_data'
    for sector in sorted([x[:-4] for x in os.listdir('/mnt/mfs/dat_whs/data/sector_data')]):
        print(sector)
        sector_path = os.path.join('/mnt/mfs/dat_whs/data/sector_data', sector + '.pkl')
        sector_df = pd.read_pickle(sector_path).dropna(how='all', axis='columns')
        sector_set = sector_df.columns

        global EQA_open, EQA_high, EQA_low, EQA_close, EQA_volume, EQA_adj_r, EQA_amount

        date_index = pd.to_datetime(all_open.index.astype(str))
        all_open.index = date_index
        all_high.index = date_index
        all_low.index = date_index
        all_close.index = date_index
        all_amount.index = date_index

        all_volume.index = date_index
        all_adj_r.index = date_index

        # sector_list = [x[:-4] for x in os.listdir('/mnt/mfs/dat_whs/data/sector_data')]

        index_root_path = os.path.join(root_save_path, sector)
        bt.AZ_Path_create(index_root_path)

        EQA_open = all_open[sorted(list(set(all_open.columns) - set(sector_set)))]
        EQA_high = all_high[sorted(list(set(all_high.columns) - set(sector_set)))]
        EQA_low = all_low[sorted(list(set(all_low.columns) - set(sector_set)))]
        EQA_close = all_close[sorted(list(set(all_close.columns) - set(sector_set)))]
        EQA_volume = all_volume[sorted(list(set(all_volume.columns) - set(sector_set)))]
        EQA_adj_r = all_adj_r[sorted(list(set(all_adj_r.columns) - set(sector_set)))]
        EQA_amount = all_amount[sorted(list(set(all_amount.columns) - set(sector_set)))]

        para_list = [5, 20, 60]
        # fnd_pct_adj_set(para_list)
        # pnd_hl_set(para_list, index_root_path)
        # pnd_vol_set(para_list, index_root_path)
        # pnd_volume_set(para_list, index_root_path)
        #
        # limit_list = [2, 2.5]
        # return_r_set(para_list, index_root_path, limit_list, sector_df)
        # return_c_set(para_list, index_root_path, limit_list, sector_df)
        # volume_r_set(para_list, index_root_path, limit_list, sector_df)
        # volume_c_set(para_list, index_root_path, limit_list)
        #
        # continue_list = [3, 4, 5]
        # pnd_continue_ud_set(continue_list, index_root_path)

        sep_1d_list = [5, 7, 9]
        pnd_sep_1d_ud_set(sep_1d_list, index_root_path)

        split_float_list = [0.03, 0.02, 0.01]
        p1d_jump_hl_set(index_root_path, split_float_list)

        short_long_list = [(5, 10), (10, 60), (10, 100), (20, 100), (20, 200), (40, 200)]
        pnnd_moment_set(short_long_list)
        pnnd_liquid_set(short_long_list)
