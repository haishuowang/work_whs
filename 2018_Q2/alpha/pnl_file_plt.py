import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from collections import OrderedDict
import matplotlib.pyplot as plt
from itertools import product, permutations, combinations
import random
from loc_lib.shared_tools import send_email
import loc_lib.shared_tools.back_test as bt


def plot_send_result(pnl_df, sharpe_ratio, subject):
    figure_save_path = '/mnt/mfs/dat_whs/tmp_figure'
    plt.figure(figsize=[16, 8])
    plt.plot(pnl_df.index, pnl_df.cumsum(), label='sharpe_ratio='.format(sharpe_ratio))
    plt.legend()
    plt.savefig(os.path.join(figure_save_path, '{}.png'.format(subject)))
    text = ''
    to = ['whs@yingpei.com']
    filepath = [os.path.join(figure_save_path, '{}.png'.format(subject))]
    send_email.send_email(text, to, filepath, subject)


if __name__ == '__main__':
    pos_file_name = 'RZJNORMAL10.pos'
    pos_df = bt.AZ_Load_csv('/mnt/mfs/AAPOS/{}'.format(pos_file_name)).iloc[-100:]
    xnms = pos_df.columns
    xinx = pos_df.index
    return_df = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv').astype(float)
    return_df = return_df.reindex(index=xinx, columns=xnms)
    pnl_df = (pos_df.shift(2) * return_df).sum(axis=1)
    plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df), 'mix_factor')
    sharpe_ratio = bt.AZ_Sharpe_y(pnl_df)
    plot_send_result(pnl_df, sharpe_ratio, pos_file_name)
