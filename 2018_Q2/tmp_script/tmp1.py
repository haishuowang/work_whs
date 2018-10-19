import pandas as pd
import numpy as np
import os
from itertools import product, permutations, combinations
from multiprocessing import Pool, Lock, cpu_count
import time
from collections import OrderedDict
import random
from datetime import datetime
import sys
import matplotlib.pyplot as plt

sys.path.append('/mnt/mfs/work_whs')
sys.path.append('/mnt/mfs/work_whs/2018_Q2')
import loc_lib.shared_tools.back_test as bt
from open_lib.shared_tools import send_email


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


def load_index_data(index_name):
    data = bt.AZ_Load_csv(os.path.join('/mnt/mfs/DAT_EQT/EM_Tab09/INDEX_TD_DAILYSYS/CHG.csv'))
    target_df = data[index_name].reindex(index=xinx)
    return target_df * 0.01


def pos_daily_fun(df, n=5):
    return df.rolling(window=n, min_periods=1).sum()


sector_name = 'market_top_2000'
root_path = '/mnt/mfs/DAT_EQT'
begin_date = pd.to_datetime('20100101')
end_date = pd.to_datetime('20180901')

market_top_n = bt.AZ_Load_csv(os.path.join(root_path, 'EM_Funda/DERIVED_10/' + sector_name + '.csv'))
market_top_n = market_top_n[(market_top_n.index >= begin_date) & (market_top_n.index < end_date)]
market_top_n.dropna(how='all', axis='columns', inplace=True)
xnms = market_top_n.columns
xinx = market_top_n.index

return_df = bt.AZ_Load_csv(os.path.join(root_path, 'EM_Funda/DERIVED_14/aadj_r.csv')).astype(float)

signal_df = (return_df < -0.097).astype(int)
signal_df = signal_df.reindex(columns=xnms, index=xinx, fill_value=1)

signal_df = (signal_df.fillna(0).diff() < 0).astype(int)
signal_df.replace(0, np.nan, inplace=True)
pos_df = signal_df.fillna(method='ffill', limit=5)

# pos_df = pos_daily_fun(signal_df, n=20)

pos_df = pos_df.div(pos_df.abs().sum(axis=1).replace(0, np.nan), axis=0)
pos_df[pos_df > 0.1] = 0.1

index_df_1 = load_index_data('000300').fillna(0)
index_df_2 = load_index_data('000905').fillna(0)
hedge_df = 0.5 * index_df_1 + 0.5 * index_df_2

return_choose = bt.AZ_Load_csv(os.path.join(root_path, 'EM_Funda/DERIVED_14/aadj_r.csv'))
return_choose = return_choose.reindex(index=xinx, columns=xnms)
return_choose = return_choose.sub(hedge_df, axis=0)

pnl_df = (-return_choose * pos_df.shift(2)).sum(axis=1) - 0.003 * pos_df.shift(2).fillna(0).diff().abs().sum(axis=1)
# pnl_df = (-return_choose * pos_df.shift(2)).sum(axis=1)
print(1)
plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df), 'aaa_b')
