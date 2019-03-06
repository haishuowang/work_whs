import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import product, permutations, combinations
import random
from loc_lib.shared_tools import send_email
import loc_lib.shared_tools.back_test as bt


def plot_send_result(pnl_df, sharpe_ratio, subject):
    figure_save_path = '/mnt/mfs/dat_whs/tmp_figure'
    plt.figure(figsize=[16, 8])
    plt.plot(pnl_df.index, pnl_df.cumsum(), label='sharpe_ratio={}'.format(sharpe_ratio))
    plt.legend()
    plt.savefig(os.path.join(figure_save_path, '{}.png'.format(subject)))
    text = ''
    to = ['whs@yingpei.com']
    filepath = [os.path.join(figure_save_path, '{}.png'.format(subject))]
    send_email.send_email(text, to, filepath, subject)


def AZ_Load_csv(target_path, index_time_type=True):
    if index_time_type:
        target_df = pd.read_table(target_path, sep='|', index_col=0, low_memory=False, parse_dates=True)
    else:
        target_df = pd.read_table(target_path, sep='|', index_col=0, low_memory=False)
    return target_df


def load_index_data(xinx, index_name):
    data = bt.AZ_Load_csv(os.path.join('/mnt/mfs/DAT_EQT/EM_Tab09/INDEX_TD_DAILYSYS/CHG.csv'))
    target_df = data[index_name].reindex(index=xinx)
    return target_df * 0.01


def get_corr_matrix(pos_file_list, cut_date=None):
    return_df = AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv').astype(float)
    index_df_1 = load_index_data(return_df.index, '000300').fillna(0)
    index_df_2 = load_index_data(return_df.index, '000905').fillna(0)
    hedge_df = 0.5 * index_df_1 + 0.5 * index_df_2
    return_df = return_df.sub(hedge_df, axis=0)
    sum_pnl_df = pd.DataFrame()
    for pos_file_name in pos_file_list:
        pos_df = AZ_Load_csv('/mnt/mfs/AAPOS/{}'.format(pos_file_name))
        pnl_df = (pos_df.shift(2) * return_df).sum(axis=1).replace(0, np.nan).fillna(method='ffill').dropna()
        pnl_df.name = pos_file_name
        sum_pnl_df = pd.concat([sum_pnl_df, pnl_df], axis=1)
        plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df), pos_file_name)
    if cut_date is not None:
        sum_pnl_df = sum_pnl_df[sum_pnl_df.index > cut_date]
    return sum_pnl_df, sum_pnl_df.corr()


def plot_all_alpha():
    all_pnl_df = pd.read_csv('/mnt/mfs/AATST/corr_tst_pnls', sep='|', index_col=0, parse_dates=True)
    for col in all_pnl_df.columns:
        pnl = all_pnl_df[col]
        pnl = pnl.replace(0, np.nan).fillna(method='ffill').dropna()
        plot_send_result(pnl, bt.AZ_Sharpe_y(pnl), f'Alpha_{col}')


if __name__ == '__main__':
    plot_all_alpha()

    # all_pos_file = os.listdir('/mnt/mfs/AAPOS')
    # self_pos_file = [x for x in all_pos_file if x.startswith('WHS')]
    # sum_pnl_df, corr_matrix = get_corr_matrix(self_pos_file, pd.to_datetime('20150901'))
    # print(corr_matrix.values)
    # data = bt.AZ_Load_csv('/mnt/mfs/AATST/corr_tst_pnls')
    # pnl_df = pd.concat([data, sum_pnl_df], axis=1)
