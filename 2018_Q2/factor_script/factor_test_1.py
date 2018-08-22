import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import product, permutations, combinations
from multiprocessing import Pool, Lock, cpu_count
import time
from datetime import datetime
import open_lib.shared_tools.back_test as bt
import random
# 读取数据的函数 以及
from factor_script.script_load_data import load_index_data, load_sector_data, load_locked_data, load_pct, \
    load_part_factor, create_log_save_path

from factor_script.script_filter_fun import pos_daily_fun, out_sample_perf, \
    filter_ic, filter_ic_sharpe, filter_ic_leve, filter_pot_sharpe, filter_all

from open_lib.shared_tools import send_email
# def AZ_Factor_info(factor_df, sector_mean):
#     """
#
#     :param factor_df: 因子df
#     :param sector_mean: sector每天的股票数量
#     :return:
#     """
#     signal_mean_fun = factor_df.abs().sum(axis=1).mean()
#     hedge_mean = factor_df.sum(axis=1).abs().mean()
#     signal_info = signal_mean_fun / sector_mean
#     hedge_info = hedge_mean / sector_mean
#     return signal_info, hedge_info


def intra_factor_test(begin_date, cut_date, end_date, locked_df):
    intra_factor_path = '/mnt/mfs/dat_whs/data/intra_factor_data'
    file_name_list = sorted(os.listdir(intra_factor_path))
    for file_name in file_name_list:
        factor_df = pd.read_pickle('/mnt/mfs/dat_whs/data/intra_factor_data/' + file_name)
        factor_df = factor_df[factor_df > 0]
        factor_df = factor_df * locked_df
        factor_df.fillna(method='ffill', inplace=True)

        return_choose = load_pct(begin_date, end_date, factor_df.columns)

        in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
        fit_ratio, leve_ratio, sharpe_q_out, pnl_df = filter_all(cut_date, factor_df, return_choose, None, lag=0,
                                                                 hold_time=2, if_hedge=False, hedge_ratio=1,
                                                                 if_return_pnl=True)

        plt.figure(figsize=[12, 6])
        plt.plot(pnl_df.cumsum())
        plt.savefig('/mnt/mfs/dat_whs/result/tmp/' + file_name[:-4] + '.png')
        print(sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, sharpe_q_out, file_name[:-4])


# def test_index_1():


if __name__ == '__main__':
    begin_date = pd.to_datetime('20100101')
    cut_date = pd.to_datetime('20160401')
    end_date = pd.to_datetime('20180401')

    if_save = True
    if_new_program = True
    if_hedge = True

    # sector_name = 'market_top_100'
    # index_name = '000016'
    # for sector_name in ['market_top_100', 'market_top_500', 'market_top_1000']:
    sector_name = 'market_top_300'
    index_name = '000300'

    # sector
    sector_df = load_sector_data(begin_date, end_date, sector_name)
    sector_mean = sector_df.sum(axis=1).mean()
    sector_set = sector_df.columns
    # suspend or limit up_dn
    locked_df = load_locked_data(begin_date, end_date, sector_set)
    # return

    # index data
    # index_df = load_index_data(begin_date, end_date, index_name)

    index_df = load_index_data(begin_date, end_date, index_name)
    for file_name in sorted(os.listdir('/mnt/mfs/dat_whs/data/intra_factor_data')):
        vwap_file_path = os.path.join('/mnt/mfs/dat_whs/data/base_data/intra_vwap_tab_{}.pkl'
                                      .format(file_name.split('_')[3]))
        vwap_df = pd.read_pickle(vwap_file_path)
        vwap_df.columns = [x[2:] + '.' + x[:2] for x in vwap_df.columns]
        factor_df = pd.read_pickle('/mnt/mfs/dat_whs/data/intra_factor_data/' + file_name)

        # return_choose = load_pct(begin_date, end_date, sector_set)

        xnms = sorted(list(set(vwap_df.columns) & set(factor_df.columns) & set(sector_set)))
        xidx = sorted(list(set(vwap_df.index) & set(factor_df.index) & set(sector_df.index)))
        sector_df = sector_df.reindex(index=xidx, columns=xnms)

        vwap_df = vwap_df.reindex(index=xidx, columns=xnms) * sector_df
        factor_df = factor_df.reindex(index=xidx, columns=xnms) * sector_df
        locked_df = locked_df.reindex(index=xidx, columns=xnms) * sector_df
        if file_name.split('_')[3] != '12':
            return_file = 'intra_vwap_tab_{}_f1d_open'.format(int(file_name.split('_')[3]) + 1)
            return_choose = load_pct(begin_date, end_date, xnms, return_file)
        else:
            return_file = 'pct_n'
            return_choose = load_pct(begin_date, end_date, xnms)
        # return_choose = load_pct(begin_date, end_date, xnms)
        return_choose = return_choose.reindex(index=xidx, columns=xnms) * sector_df
        index_df = index_df.reindex(index=xidx)

        vwap_df_pct = vwap_df / vwap_df.shift(1) - 1
        vwap_df_pct_cdist = bt.AZ_Col_zscore(vwap_df_pct, 30)
        vwap_df_pct_cdist_up = (vwap_df_pct_cdist > 2)*(vwap_df_pct_cdist < 4)
        vwap_df_pct_cdist_dn = (vwap_df_pct_cdist < -2)*(vwap_df_pct_cdist > (-4))

        volume_up = (factor_df >= 2).astype(int)
        volume_dn = (factor_df <= 0).astype(int)

        factor_p_up_v_up = volume_up[vwap_df_pct_cdist_up] > 0
        factor_p_dn_v_up = volume_up[vwap_df_pct_cdist_dn] > 0

        factor_p_up_v_dn = volume_dn[vwap_df_pct_cdist_up] > 0
        factor_p_dn_v_dn = volume_dn[vwap_df_pct_cdist_dn] > 0

        plt.figure(figsize=[12, 6])
        factor_set = {'p_up_v_up': factor_p_up_v_up, 'p_dn_v_up': factor_p_dn_v_up,
                      'p_up_v_dn': factor_p_up_v_dn, 'p_dn_v_dn': factor_p_dn_v_dn}
        hold_time = 1
        hold_time_list = [1, 3, 10, 20]
        for key in factor_set.keys():
            # key = 'p_dn_v_dn'
            factor_use = factor_set[key]
            factor_use = factor_use * locked_df
            factor_use.fillna(method='ffill', inplace=True)
            factor_use = factor_use * sector_df

            in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
            fit_ratio, leve_ratio, sharpe_q_out, pnl_df = filter_all(cut_date, factor_use, return_choose, index_df,
                                                                     lag=0, hold_time=hold_time, if_hedge=True,
                                                                     hedge_ratio=1, if_return_pnl=True)

            # fig_save_path = '/mnt/mfs/dat_whs/result/tmp/{}_{}_dn_h_vlm_u.png'.format(sector_name, file_name[:-4])
            print(file_name[:-4], return_file, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in,
                  sharpe_q_out, factor_use.sum(axis=1).mean())
            asset_df = pnl_df.cumsum()
            plt.plot(asset_df, label='{},hold_time={},sharpe_in_m={}, asset={}'
                     .format(key, hold_time, sharpe_q_in_df_m, round(asset_df.iloc[-1], 4)))
        plt.legend()
        fig_save_path = '/mnt/mfs/dat_whs/result/tmp/{}_{}.png'.format(sector_name, file_name[:-4])
        plt.savefig(fig_save_path)
        plt.close()
        subject = os.path.split(fig_save_path)[-1][:-4]
        send_email.send_email('', ['whs@yingpei.com'], [fig_save_path], subject)
