import pandas as pd
import numpy as np
import os
from functools import reduce
from itertools import product, permutations, combinations
import loc_lib.shared_tools.back_test as bt
from multiprocessing import Pool


def AZ_Rolling_mean(df, n, min_periods=1):
    target = df.rolling(window=n, min_periods=min_periods).mean()
    target.iloc[:n - 1] = np.nan
    return target


def AZ_Col_zscore(df, n, cap=None, min_periods=1):
    df_mean = AZ_Rolling_mean(df, n, min_periods=min_periods)
    df_std = df.rolling(window=n, min_periods=min_periods).std()
    df_std.replace(0, np.nan, inplace=True)
    target = (df - df_mean) / df_std
    if cap is not None:
        target[target > cap] = cap
        target[target < -cap] = -cap
    return target


def AZ_Factor_info(factor_df, sector_mean):
    """

    :param factor_df: 因子df
    :param sector_mean: sector每天的股票数量
    :return:
    """
    signal_mean = factor_df.abs().sum(axis=1).mean()
    hedge_mean = factor_df.sum(axis=1).abs().mean()
    signal_info = signal_mean / sector_mean
    hedge_info = hedge_mean / sector_mean
    return signal_info, hedge_info, signal_mean


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


def fnd_pct_adj(adj_r, n=1):
    adj_r_fillna = adj_r.fillna(0)
    return (adj_r_fillna + 1).rolling(window=n, min_periods=1) \
        .apply(lambda a: reduce(lambda x, y: x * y, a) - 1).fillna(0).shift(-n)


#################################################################################################################
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


def extreme_data(zscore_df, limit=2):
    zscore_df_copy = zscore_df.copy()
    zscore_df_copy[(zscore_df <= limit) & (zscore_df >= -limit)] = 0
    zscore_df_copy[zscore_df > limit] = 1
    zscore_df_copy[zscore_df < -limit] = -1
    return zscore_df_copy


def pnd_continue_ud(target_df, n=3):
    return target_df.rolling(window=n).apply(lambda x: 1 if (np.diff(x) >= 0).all() and sum(np.diff(x)) > 0
    else (-1 if (np.diff(x) <= 0).all() and sum(np.diff(x)) < 0 else 0))


def pnd_continue_ud_pct(target_df, n=3):
    return target_df.rolling(window=n).apply(lambda x: 1 if (x >= 0).all() and sum(x) > 0
    else (-1 if (x <= 0).all() and sum(x) < 0 else 0))


def p1d_jump_hl(close, open, split_float=0.05):
    jump_df = open.shift(-1) / close - 1
    target_df = pd.DataFrame(index=jump_df.index, columns=jump_df.columns)
    target_df[(jump_df > 0.101) | (jump_df < -0.101)] = 0
    target_df[(split_float >= jump_df) & (jump_df >= -split_float)] = 0
    target_df[jump_df > split_float] = 1
    target_df[jump_df < -split_float] = -1
    return target_df


def pnnd_moment(aadj_r, n_short=10, n_long=60):
    ma_long = aadj_r.rolling(window=n_long, min_periods=1).mean()
    ma_short = aadj_r.rolling(window=n_short, min_periods=1).mean()
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


def pnd_hl_set(para_list, index_root_path, sector_high, sector_low, sector_close, sector_df):
    for n in para_list:
        print('hl_p{0}d'.format(n))
        index_save_path = os.path.join(index_root_path, 'hl_p{0}d.pkl'.format(n))
        pnd_hl_df = pnd_hl(sector_high, sector_low, sector_close, n)
        pnd_hl_df = pnd_hl_df * sector_df
        signal_info, hedge_info, signal_mean = AZ_Factor_info(pnd_hl_df, sector_mean)
        if signal_info > 0.01 and signal_mean > 2:
            pnd_hl_df.to_pickle(index_save_path)


def return_r_set(index_root_path, limit_list, sector_adj_r, sector_df):
    rrn_df = bt.AZ_Row_zscore(sector_adj_r * sector_df)
    for limit in limit_list:
        print('rr{}_ext_{}'.format(n, limit))
        index_save_path = os.path.join(index_root_path, 'rr{}_ext_{}.pkl'.format(n, limit))
        rrn_ext_df = extreme_data(rrn_df, limit)
        signal_info, hedge_info, signal_mean = AZ_Factor_info(rrn_ext_df, sector_mean)
        if signal_info > 0.01 and signal_mean > 2:
            rrn_ext_df.to_pickle(index_save_path)


def return_c_set(para_list, index_root_path, limit_list, sector_adj_r, sector_df):
    for n in para_list:

        # rcn_df = bt.AZ_Col_zscore(sector_adj_r, n)
        rcn_df = AZ_Col_zscore(sector_adj_r, n)
        for limit in limit_list:
            print('rc{}_ext_{}'.format(n, limit))
            index_save_path = os.path.join(index_root_path, 'rc{}_ext_{}.pkl'.format(n, limit))
            rcn_ext_df = extreme_data(rcn_df, limit)
            signal_info, hedge_info, signal_mean = AZ_Factor_info(rcn_ext_df, sector_mean)
            rcn_ext_df = rcn_ext_df * sector_df
            if signal_info > 0.01 and signal_mean > 2:
                rcn_ext_df.to_pickle(index_save_path)


def volume_r_set(index_root_path, limit_list, sector_volume, sector_df):
    vrn_df = bt.AZ_Row_zscore(sector_volume*sector_df)
    for limit in limit_list:
        print('vr{}_ext_{}'.format(n, limit))
        index_save_path = os.path.join(index_root_path, 'volume_r_ext_{}.pkl'.format(limit))
        vrn_ext_df = extreme_data(vrn_df, limit)
        signal_info, hedge_info, signal_mean = AZ_Factor_info(vrn_ext_df, sector_mean)
        if signal_info > 0.01 and signal_mean > 2:
            vrn_ext_df.to_pickle(index_save_path)


def volume_c_set(para_list, index_root_path, limit_list, sector_volume, sector_df):
    for n in para_list:
        vcn_df = AZ_Col_zscore(sector_volume, n)
        for limit in limit_list:
            index_save_path = os.path.join(index_root_path, 'volume_c_p{}d_ext_{}.pkl'.format(n, limit))
            print('vc{}_ext_{}'.format(n, limit))
            vcn_ext_df = extreme_data(vcn_df, limit)
            vcn_ext_df = vcn_ext_df * sector_df
            signal_info, hedge_info, signal_mean = AZ_Factor_info(vcn_ext_df, sector_mean)
            if signal_info > 0.01 and signal_mean > 2:
                vcn_ext_df.to_pickle(index_save_path)


def pnd_continue_ud_set(continue_list, index_root_path, sector_aadj_r, sector_df):
    target_df = pd.DataFrame()
    index_save_path = os.path.join(index_root_path, 'continue_ud_p{}d.pkl'
                                   .format(''.join([str(x) for x in continue_list])))
    for n in continue_list:
        print('continue_ud_p{}d'.format(n))
        continue_ud_df = pnd_continue_ud_pct(sector_aadj_r, n)
        continue_ud_df = continue_ud_df * sector_df
        signal_info, hedge_info, signal_mean = AZ_Factor_info(continue_ud_df, sector_mean)
        if signal_info > 0.01 and signal_mean > 2:
            target_df = target_df.add(continue_ud_df, fill_value=0)
            print('p{}d_continue_ud_set {} success!'.format(n, round(signal_info, 4)))
        else:
            print('p{}d_continue_ud_set {} error!'.format(n, round(signal_info, 4)))
    target_df.to_pickle(index_save_path)


def p1d_jump_hl_set(index_root_path, split_float_list, sector_close, sector_open, sector_df):
    target_df = pd.DataFrame()
    index_save_path = os.path.join(index_root_path, 'jump_hl_split_{}_p1d.pkl'
                                   .format(''.join([str(x) for x in split_float_list])))
    for split_float in split_float_list:
        print('jump_hl_split_{}_p1d.pkl'.format(split_float))
        p1d_jump_hl_df = p1d_jump_hl(sector_close, sector_open, split_float)
        p1d_jump_hl_df = p1d_jump_hl_df * sector_df
        signal_info, hedge_info, signal_mean = AZ_Factor_info(p1d_jump_hl_df, sector_mean)
        if signal_info > 0.01 and signal_mean > 2:
            target_df = target_df.add(p1d_jump_hl_df, fill_value=0)
    target_df.to_pickle(index_save_path)


def pnd_volitality_set(index_root_path, para_list, sector_adj_r):
    def sort_x_fun(x):
        sort_x = sorted(x)
        up_line = sort_x[int(len(x) * 0.98)]
        dn_line = sort_x[int(len(x) * 0.02)]
        x[x > up_line] = up_line
        x[x < dn_line] = dn_line
        return x

    for n in para_list:
        target_df_save_path = os.path.join(index_root_path, 'volitality_p{}n'.format(n))
        evol_df_save_path = os.path.join(index_root_path, 'evol_30_p{}n'.format(n))
        vol_continue_ud_path = os.path.join(index_root_path, 'vol_p{}n_continue_3ud'.format(n))
        target_df = bt.AZ_Rolling(sector_adj_r, n).std()

        evol_df = bt.AZ_Rolling(target_df, 30).apply(lambda x: 1 if x[-1] > 2 * x.mean() else 0)
        evol_df.to_pickle(evol_df_save_path)

        vol_continue_ud_df = pnd_continue_ud(target_df, n=3)
        vol_continue_ud_df.to_pickle(vol_continue_ud_path)
        # 剔除极值
        target_df = target_df.apply(sort_x_fun, axis=1)
        target_df.to_pickle(target_df_save_path)


def pnnd_moment_set(index_root_path, short_long_list, sector_adj_r, sector_df):
    for n_short, n_long in short_long_list:
        print('moment_s{}_l{}'.format(n_short, n_long))
        index_save_path = os.path.join(index_root_path, 'moment_s{}_l{}.pkl'.format(n_short, n_long))
        pnnd_moment_df = pnnd_moment(sector_adj_r, n_short, n_long)
        pnnd_moment_df = pnnd_moment_df * sector_df
        signal_info, hedge_info, signal_mean = AZ_Factor_info(pnnd_moment_df, sector_mean)
        if signal_info > 0.01 and signal_mean > 2:
            pnnd_moment_df.to_pickle(index_save_path)
            print('pnnd_moment_set long={} short={} {} success!'.format(n_short, n_long, round(signal_info, 4)))
        else:
            print('pnnd_moment_set long={} short={} {} error!'.format(n_short, n_long, round(signal_info, 4)))


def pnnd_liquid_set(index_root_path, short_long_list, sector_amount, sector_df):
    for n_short, n_long in short_long_list:
        print('liquid_s{}_l{}'.format(n_short, n_long))
        index_save_path = os.path.join(index_root_path, 'liquid_s{}_l{}.pkl'.format(n_short, n_long))
        pnnd_liquid_df = pnnd_liquid(sector_amount, n_short, n_long)
        pnnd_liquid_df = pnnd_liquid_df * sector_df
        signal_info, hedge_info, signal_mean = AZ_Factor_info(pnnd_liquid_df, sector_mean)

        if signal_info > 0.01 and signal_mean > 2:
            pnnd_liquid_df.to_pickle(index_save_path)
            print('pnnd_liquid_set long={} short={} {} success!'.format(n_short, n_long, round(signal_info, 4)))
        else:
            print('pnnd_liquid_set long={} short={} {} error!'.format(n_short, n_long, round(signal_info, 4)))


# 负偏度系数
def NCSKEW_set(para_list, sector_adj_r):
    for n in para_list:
        up_line = -(n(n - 1)) ** 1.5 * bt.AZ_Rolling(sector_adj_r, n).apply(lambda x: sum((x - x.mean()) ** 3))
        dn_line = (n - 1) * (n - 2) * (bt.AZ_Rolling(sector_adj_r, n).apply(lambda x: sum((x - x.mean()) ** 2))) ** 1.5
        NCSKEW_factor = up_line / dn_line
        NCSKEW_factor.to_pickle()


# 融资融券数据
def pnd_roll_mean_row_extre_fun(tab_name, data, n_list, limit_list, index_root_path, sector_df):
    for n in n_list:
        data_roll_mean = bt.AZ_Rolling_mean(data * sector_df, n)
        data_roll_mean.replace(0, np.nan, inplace=True)
        data_roll_mean = data_roll_mean * sector_df
        data_pnd_roll_mean_row_extre = bt.AZ_Row_zscore(data_roll_mean)
        for limit in limit_list:
            target_df = extreme_data(data_pnd_roll_mean_row_extre, limit=limit)
            signal_info, hedge_info, signal_mean = AZ_Factor_info(target_df, sector_mean)
            if signal_info > 0.01 and signal_mean > 2:
                target_df.to_pickle(os.path.join(index_root_path,
                                                 '{}_p{}d_roll_mean_row_ext_{}.pkl'.format(tab_name, n, limit)))
                print('{}_p{}d_roll_mean_row_ext_{} {} success!'.format(tab_name, n, limit, round(signal_info, 4)))
            else:
                print('{}_p{}d_roll_mean_row_ext_{} {} error!'.format(tab_name, n, limit, round(signal_info, 4)))


def pnd_row_extre_fun(tab_name, data, limit_list, index_root_path, sector_df):
    data_pnd_row_extre = bt.AZ_Row_zscore(data*sector_df)
    target_df = pd.DataFrame()
    for limit in limit_list:
        tmp_df = extreme_data(data_pnd_row_extre, limit=limit)
        signal_info, hedge_info, signal_mean = AZ_Factor_info(target_df, sector_mean)
        if signal_info > 0.01 and signal_mean > 2:
            target_df = target_df.add(tmp_df, fill_value=0)
            print('{}_row_ext_{} {} success!'.format(tab_name, limit, round(signal_info, 4)))
        else:
            print('{}_row_ext_{} {} error!'.format(tab_name, limit, round(signal_info, 4)))
    target_df.to_pickle(os.path.join(index_root_path,
                                     '{}_row_ext_{}.pkl'.format(tab_name,
                                                                ''.join([str(x) for x in limit_list]))))


def pnd_col_extre_fun(tab_name, data, n_list, limit_list, index_root_path, sector_df):
    for n in n_list:
        data_pnd_col_extre = AZ_Col_zscore(data, n)
        # data_pnd_col_extre = bt.AZ_Col_zscore(data, n)
        target_df = pd.DataFrame()
        for limit in limit_list:
            tmp_df = extreme_data(data_pnd_col_extre, limit=limit)
            tmp_df = tmp_df * sector_df
            target_df = target_df.add(tmp_df, fill_value=0)
            # if if_filter:
            #     signal_info, hedge_info, signal_mean = AZ_Factor_info(target_df, sector_mean)
            #     if signal_info > 0.01 and signal_mean > 2:
            #         target_df = target_df.add(tmp_df, fill_value=0)
            #         print('{}_p{}d_col_ext_{} {} success!'.format(tab_name, n, limit, round(signal_info, 4)))
            #     else:
            #         print('{}_p{}d_col_ext_{} {} error!'.format(tab_name, n, limit, round(signal_info, 4)))
            # else:

        target_df.to_pickle(os.path.join(index_root_path, '{}_p{}d_col_ext_{}.pkl'
                                         .format(tab_name, n, ''.join([str(x) for x in limit_list]))))


def pnd_continue_up_dn_fun(tab_name, data, n_list, index_root_path, sector_df):
    target_df = pd.DataFrame()
    for n in n_list:
        tmp_df = pnd_continue_ud(data, n)
        tmp_df = tmp_df * sector_df
        target_df = target_df.add(tmp_df, fill_value=0)
    target_df.to_pickle(os.path.join(index_root_path,
                                     '{}_p{}d_continue_ud.pkl'.format(tab_name, ''.join([str(x) for x in n_list]))))


def rzrq_create_factor(index_root_path, sector_df):
    # 融资融券数据
    rzrq_root_path = '/mnt/mfs/DAT_EQT/EM_Funda/TRAD_MT_MARGIN'
    name_list = ['RZRQYE', 'RZMRE', 'RZYE', 'RQMCL', 'RQYE', 'RQYL', 'RQCHL', 'RZCHE']
    # 均值
    rolling_mean_list = [5, 10, 20, 60]
    limit_list = [1, 1.5, 2]
    updn_list = [3, 4, 5]
    # 单个数据　简单的z-score
    for tab_name in name_list:
        print(tab_name)
        data = bt.AZ_Load_csv(os.path.join(rzrq_root_path, tab_name + '.csv'))
        data = data.reindex(index=sector_df.index, columns=sector_df.columns)
        data.replace(0, np.nan, inplace=True)
        pnd_roll_mean_row_extre_fun(tab_name, data, rolling_mean_list, limit_list, index_root_path, sector_df)
        pnd_col_extre_fun(tab_name, data, rolling_mean_list, limit_list, index_root_path, sector_df)
        pnd_row_extre_fun(tab_name, data, limit_list, index_root_path, sector_df)
        pnd_continue_up_dn_fun(tab_name, data, updn_list, index_root_path, sector_df)

        data_p5d_chg = data.div(data.shift(5), fill_value=0) - 1
        data_p5d_chg.replace(np.inf, 0, inplace=True)
        pnd_col_extre_fun(tab_name + '_chg5', data_p5d_chg, rolling_mean_list, limit_list, index_root_path, sector_df)
        pnd_row_extre_fun(tab_name + '_chg5', data_p5d_chg, limit_list, index_root_path, sector_df)
        pnd_continue_up_dn_fun(tab_name + '_chg5', data_p5d_chg, updn_list, index_root_path, sector_df)

    # 两个数据相除
    # for tab_name_1, tab_name_2 in combinations(name_list, 2):
    #     data_1 = pd.read_pickle(os.path.join(rzrq_root_path, tab_name_1 + '.pkl'))
    #     data_2 = pd.read_pickle(os.path.join(rzrq_root_path, tab_name_2 + '.pkl'))
    #     data_df = data_1.div(data_2, fill_value=0) - 1
    #     data_df.replace(np.inf, 0, inplace=True)
    #     pnd_col_extre_fun(tab_name_1 + '_' + tab_name_2, data_df, rolling_mean_list, limit_list, index_root_path)
    #     pnd_row_extre_fun(tab_name_1 + '_' + tab_name_2, data_df, limit_list, index_root_path)
    #     # pnd_continue_up_dn_fun(tab_name_1 + tab_name_2, data_df, updn_list, index_root_path)


# # intraday factor
# def intraday_reverse_volume_factor(begin_str, end_str, sector_set, index_root_path):
#     print('intraday_reverse_volume_factor')
#     begin_year, begin_month, begin_day = begin_str[:4], begin_str[:6], begin_str
#     end_year, end_month, end_day = end_str[:4], end_str[:6], end_str
#     target_df = pd.DataFrame()
#     intraday_path = '/mnt/mfs/DAT_PUBLIC/intraday/eqt_5mbar'
#     year_list = [x for x in os.listdir(intraday_path) if (x >= begin_year) & (x <= end_year)]
#     for year in sorted(year_list):
#         year_path = os.path.join(intraday_path, year)
#         month_list = [x for x in os.listdir(year_path) if (x >= begin_month) & (x <= end_month)]
#         for month in sorted(month_list):
#             month_path = os.path.join(year_path, month)
#             day_list = [x for x in os.listdir(month_path) if (x >= begin_day) & (x <= end_day)]
#             for day in sorted(day_list):
#                 # print(day)
#                 day_path = os.path.join(month_path, day)
#                 close = pd.read_csv(os.path.join(day_path, 'Close.csv'), index_col=0)
#                 volume = pd.read_csv(os.path.join(day_path, 'Volume.csv'), index_col=0)
#
#                 reverse_df = close.rolling(window=3).apply(lambda x: -1 if (x[0] > x[1]) & (x[1] < x[2])
#                 else (1 if (x[0] < x[1]) & (x[1] > x[2]) else 0)).shift(-1).astype(float)
#                 volume_up = (volume.astype(float) * reverse_df[reverse_df > 0].astype(float)).sum()
#                 volume_dn = (volume.astype(float) * reverse_df[reverse_df < 0].astype(float)).sum()
#
#                 part_df = pd.DataFrame(volume_up / volume_dn, columns=[pd.to_datetime(day)]).T
#
#                 part_df[part_df > 1] = 1
#                 part_df[part_df < -1] = -1
#
#                 target_df = target_df.append(part_df)
#
#     # 选出Ａ股
#     filter_columns = [x for x in target_df.columns if (x.startswith('SH') and x[2] == '6') or
#                       (x.startswith('SZ') and (x[2] == '0' or x[2] == '3'))]
#     filter_columns_c = [x[2:] + '.' + x[:2] for x in filter_columns]
#     target_df.columns = [x[2:] + '.' + x[:2] for x in target_df.columns]
#     use_set = sorted(list(set(filter_columns_c) & set(sector_set)))
#     target_df = target_df[use_set]
#
#     target_df.to_pickle(os.path.join(index_root_path, 'intra_long_short_volume.pkl'))
#     return target_df
#
#
# def intraday_most_volume_twap(begin_str, end_str, sector_set, index_root_path):
#     print('intraday_most_volume_twap')
#     begin_year, begin_month, begin_day = begin_str[:4], begin_str[:6], begin_str
#     end_year, end_month, end_day = end_str[:4], end_str[:6], end_str
#     target_df = pd.DataFrame()
#     intraday_path = '/mnt/mfs/DAT_PUBLIC/intraday/eqt_5mbar'
#     year_list = [x for x in os.listdir(intraday_path) if (x >= begin_year) & (x <= end_year)]
#     for year in sorted(year_list):
#         year_path = os.path.join(intraday_path, year)
#         month_list = [x for x in os.listdir(year_path) if (x >= begin_month) & (x <= end_month)]
#         for month in sorted(month_list):
#             month_path = os.path.join(year_path, month)
#             day_list = [x for x in os.listdir(month_path) if (x >= begin_day) & (x <= end_day)]
#             for day in sorted(day_list):
#                 # print(day)
#                 day_path = os.path.join(month_path, day)
#                 close = pd.read_csv(os.path.join(day_path, 'Close.csv'), index_col=0)
#                 volume = pd.read_csv(os.path.join(day_path, 'Volume.csv'), index_col=0)
#
#                 volume_rank = volume.rank(ascending=False, na_option='bottom')
#                 volume_rank[volume_rank <= 18] = 1
#                 volume_rank[volume_rank > 18] = np.nan
#                 volume_most = volume * volume_rank
#                 vwap_most_volume = (volume_most.astype(float) * close.astype(float)).sum() / volume_most.sum()
#                 vwap_today = (volume.astype(float) * close.astype(float)).sum() / volume.sum()
#                 compare_today_vwap = vwap_most_volume / vwap_today
#                 compare_today_vwap[compare_today_vwap < 1] = -1
#                 compare_today_vwap[compare_today_vwap > 1] = 1
#                 target_df = target_df.append(pd.DataFrame(compare_today_vwap, columns=[pd.to_datetime(day)]).T)
#
#     # 选出Ａ股
#     filter_columns = [x for x in target_df.columns if (x.startswith('SH') and x[2] == '6') or
#                       (x.startswith('SZ') and (x[2] == '0' or x[2] == '3'))]
#     filter_columns_c = [x[2:] + '.' + x[:2] for x in filter_columns]
#     target_df.columns = [x[2:] + '.' + x[:2] for x in target_df.columns]
#     use_set = sorted(list(set(filter_columns_c) & set(sector_set)))
#     target_df = target_df[use_set]
#     target_df.to_pickle(os.path.join(index_root_path, 'intra_most_volume_vwap_compare.pkl'))
#     return target_df
#
#
# def intraday_open_1_hour_vwap(begin_str, end_str):
#     print('intraday_open_1_hour_vwap')
#     begin_year, begin_month, begin_day = begin_str[:4], begin_str[:6], begin_str
#     end_year, end_month, end_day = end_str[:4], end_str[:6], end_str
#     target_df = pd.DataFrame()
#     intraday_path = '/mnt/mfs/DAT_PUBLIC/intraday/eqt_5mbar'
#     year_list = [x for x in os.listdir(intraday_path) if (x >= begin_year) & (x <= end_year)]
#     for year in sorted(year_list):
#         year_path = os.path.join(intraday_path, year)
#         month_list = [x for x in os.listdir(year_path) if (x >= begin_month) & (x <= end_month)]
#         for month in sorted(month_list):
#             month_path = os.path.join(year_path, month)
#             day_list = [x for x in os.listdir(month_path) if (x >= begin_day) & (x <= end_day)]
#             for day in sorted(day_list):
#                 # print(day)
#                 day_path = os.path.join(month_path, day)
#                 close = pd.read_csv(os.path.join(day_path, 'Close.csv'), index_col=0)
#                 volume = pd.read_csv(os.path.join(day_path, 'Volume.csv'), index_col=0)
#                 part_close = close.iloc[:12]
#                 part_volume = volume.iloc[:12]
#                 vwap_1_hour = (part_close.astype(float) * part_volume.astype(float)).sum() / part_volume.sum()
#
#                 target_df = target_df.append(pd.DataFrame(vwap_1_hour, columns=[pd.to_datetime(day)]).T)
#     filter_columns = [x for x in target_df.columns if (x.startswith('SH') and x[2] == '6') or
#                       (x.startswith('SZ') and (x[2] == '0' or x[2] == '3'))]
#     filter_columns_c = [x[2:] + '.' + x[:2] for x in filter_columns]
#     target_df.columns = [x[2:] + '.' + x[:2] for x in target_df.columns]
#     adj_factor = pd.read_csv('/mnt/mfs/dat_whs/data/AllStock/all_tafactor.csv', index_col=0)
#
#     adj_factor.index = pd.to_datetime(adj_factor.index.astype(str))
#
#     target_df_adj = target_df * adj_factor
#     return_vwap_f1d = target_df_adj.shift(-1) / target_df_adj - 1
#     return_vwap_f1d.to_pickle('/mnt/mfs/dat_whs/data/return_data/open_1_hour_vwap.pkl')
#     return return_vwap_f1d
#
#
# def intraday_create_factor(begin_str, end_str, factor_save_path):
#     intra_raw_path = '/mnt/mfs/dat_whs/data/base_data'
#     volume_list = [x[:-4] for x in os.listdir(intra_raw_path) if x.startswith('intra') and 'volume' in x]
#     vwap_list = [x[:-4] for x in os.listdir(intra_raw_path) if x.startswith('intra') and 'vwap' in x]
#     for tab_name in volume_list:
#         print(tab_name)
#         intra_data = pd.read_pickle(os.path.join(intra_raw_path, tab_name + '.pkl'))
#         filter_columns = [x for x in intra_data.columns if (x.startswith('SH') and x[2] == '6') or
#                           (x.startswith('SZ') and (x[2] == '0' or x[2] == '3'))]
#         intra_data = intra_data[filter_columns]
#         intra_data.columns = [x[2:] + '.' + x[:2] for x in intra_data.columns]
#         intra_data = intra_data[sorted(intra_data.columns)]
#         limit_list = [1, 1.5, 2]
#         para_list = [10, 20, 60]
#         # pnd_row_extre_fun(tab_name, intra_data, limit_list, index_root_path)
#         pnd_col_extre_fun(tab_name, intra_data, para_list, limit_list, factor_save_path, sector_df)


####################################################################################################################
if __name__ == '__main__':

    sector_data_path = '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_10'

    base_data_path = '/mnt/mfs/DAT_EQT/EM_Tab14/TRAD_SK_DAILY_JC'

    factor_save_path = '/media/hdd1/dat_whs/data/new_factor_data'

    EQA_open = bt.AZ_Load_csv(os.path.join(base_data_path, 'OPEN.csv'))
    EQA_high = bt.AZ_Load_csv(os.path.join(base_data_path, 'HIGH.csv'))
    EQA_low = bt.AZ_Load_csv(os.path.join(base_data_path, 'LOW.csv'))
    EQA_close = bt.AZ_Load_csv(os.path.join(base_data_path, 'NEW.csv'))
    EQA_volume = bt.AZ_Load_csv(os.path.join(base_data_path, 'TVOL.csv'))
    EQA_amount = bt.AZ_Load_csv(os.path.join(base_data_path, 'TVALCHY.csv'))
    EQA_adj_r = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Tab14/DERIVED/aadj_r.csv')
    begin_str = '20100101'
    end_str = '20180401'

    # pool = Pool(20)
    # intraday_open_1_hour_vwap(begin_str, end_str)
    # intraday_create_factor(begin_str, end_str, factor_save_path)

    for sector in ['market_top_500']:
        print('********************************************************************')
        print(sector)
        sector_path = os.path.join(sector_data_path, sector + '.csv')
        sector_df = bt.AZ_Load_csv(sector_path).dropna(how='all', axis='columns')

        sector_mean = sector_df.sum(axis=1).mean()

        sector_df = sector_df[(sector_df.index >= pd.to_datetime(begin_str)) &
                              (sector_df.index < pd.to_datetime(end_str))]

        xinx = sector_df.index
        xnms = sector_df.columns

        sector_open = EQA_open.reindex(index=xinx, columns=xnms)
        sector_high = EQA_high.reindex(index=xinx, columns=xnms)
        sector_low = EQA_low.reindex(index=xinx, columns=xnms)
        sector_close = EQA_close.reindex(index=xinx, columns=xnms)
        sector_volume = EQA_volume.reindex(index=xinx, columns=xnms)
        sector_adj_r = EQA_adj_r.reindex(index=xinx, columns=xnms)
        sector_amount = EQA_amount.reindex(index=xinx, columns=xnms)

        index_root_path = os.path.join(factor_save_path, sector)
        bt.AZ_Path_create(index_root_path)
        # 技术指标
        para_list = [5, 20, 60, 120]
        pnd_hl_set(para_list, index_root_path, sector_high, sector_low, sector_close, sector_df)
        pnd_volitality_set(index_root_path, para_list, sector_adj_r)

        limit_list = [1, 1.5, 2]
        return_r_set(index_root_path, limit_list, sector_adj_r, sector_df)
        return_c_set(para_list, index_root_path, limit_list, sector_adj_r, sector_df)

        volume_r_set(index_root_path, limit_list, sector_volume, sector_df)
        volume_c_set(para_list, index_root_path, limit_list, sector_volume, sector_df)

        continue_list = [3, 4, 5]
        pnd_continue_ud_set(continue_list, index_root_path, sector_close, sector_df)

        split_float_list = [0.03, 0.02, 0.01]
        p1d_jump_hl_set(index_root_path, split_float_list, sector_close, sector_open, sector_df)

        short_long_list = [(5, 10), (10, 60), (10, 100), (20, 100), (20, 200), (40, 200)]

        pnnd_moment_set(index_root_path, short_long_list, sector_adj_r, sector_df)
        pnnd_liquid_set(index_root_path, short_long_list, sector_amount, sector_df)

        # # 融资融券factor
        rzrq_create_factor(index_root_path, sector_df)

        # 日内factor

# pool.apply_async(intraday_reverse_volume_factor, args=(begin_str, end_str, sector_set, index_root_path))
# pool.apply_async(intraday_most_volume_twap, args=(begin_str, end_str, sector_set, index_root_path))

# intraday_reverse_volume_factor(begin_str, end_str, sector_set, index_root_path)
# intraday_most_volume_twap(begin_str, end_str, sector_set, index_root_path)

# pool.close()
# pool.join()
