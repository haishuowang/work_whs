import numpy as np
import pandas as pd
import os
import sys
from sqlalchemy import create_engine
from itertools import product, permutations, combinations
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
import random
import matplotlib
from collections import OrderedDict
from multiprocessing import Pool, Lock
from multiprocessing.dummy import Pool as ThreadPool
import re
import sys

sys.path.append("/mnt/mfs/LIB_ROOT")
sys.path.append('/mnt/mfs')
import open_lib.shared_paths.path as pt
from work_whs.loc_lib.shared_tools import send_email
import work_whs.loc_lib.shared_tools.back_test as bt
from collections import Counter


def plot_send_result(pnl_df, sharpe_ratio, subject, text=''):
    figure_save_path = os.path.join('/mnt/mfs/dat_whs', 'tmp_figure')
    plt.figure(figsize=[16, 8])
    plt.plot(pnl_df.index, pnl_df.cumsum(), label='sharpe_ratio={}'.format(sharpe_ratio))
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(figure_save_path, '{}.png'.format(subject)))
    plt.close()
    to = ['whs@yingpei.com']
    filepath = [os.path.join(figure_save_path, '{}.png'.format(subject))]
    send_email.send_email(text, to, filepath, subject)


def plot_send_result_mul(pnl_df, subject, text=''):
    assert type(pnl_df) == pd.DataFrame
    figure_save_path = os.path.join('/mnt/mfs/dat_whs', 'tmp_figure')
    pnl_num = len(pnl_df.columns)
    plt.figure(figsize=[16, 8*pnl_num])
    for i, col in enumerate(pnl_df.columns):
        ax = plt.subplot(pnl_num, 1, i+1)
        ax.plot(pnl_df[col].index, pnl_df[col].cumsum(), label=f'{col}, sharpe_ratio={bt.AZ_Sharpe_y(pnl_df[col])}')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(figure_save_path, '{}.png'.format(subject)))
    plt.close()
    to = ['whs@yingpei.com']
    filepath = [os.path.join(figure_save_path, '{}.png'.format(subject))]
    send_email.send_email(text, to, filepath, subject)


def mysql_select(select_col_list, table_name, conn, key_col=None, cond=None, cpu_num=20, step=100):
    key_col_list = pd.read_sql(f'SELECT DISTINCT {key_col} '
                               f'FROM {table_name}', conn).values.ravel()
    select_col_str = ', '.join(select_col_list)

    def fetch_data(sids):
        print(f"SELECT {select_col_str} "
              f"FROM {table_name} "
              f"WHERE {key_col} in {str(tuple(sids))} AND BulletinType = 'lsgg'")
        lsgg_df = pd.read_sql(f"SELECT {select_col_str} "
                              f"FROM {table_name} "
                              f"WHERE {key_col} in {str(tuple(sids))} AND BulletinType = 'lsgg'", conn)
        return lsgg_df

    p = ThreadPool(cpu_num)
    res = pd.concat(p.map(fetch_data, [key_col_list[i: i + step] for i in range(0, len(key_col_list), step)]))
    return res


def plot_send_data(raw_df, subject, text=''):
    figure_save_path = os.path.join('/mnt/mfs/dat_whs', 'tmp_figure')
    raw_df.plot(legend=True)
    plt.savefig(f'{figure_save_path}/{subject}.png')
    plt.close()
    to = ['whs@yingpei.com']
    filepath = [f'{figure_save_path}/{subject}.png']
    send_email.send_email(text, to, filepath, subject)

