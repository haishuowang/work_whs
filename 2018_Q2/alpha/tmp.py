import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append('/mnt/mfs/work_whs/2018_Q2')
sys.path.append('/mnt/mfs/work_whs')
from loc_lib.shared_tools import send_email
import loc_lib.shared_tools.back_test as bt


def AZ_Load_csv(target_path, index_time_type=True):
    if index_time_type:
        target_df = pd.read_table(target_path, sep='|', index_col=0, low_memory=False, parse_dates=True)
    else:
        target_df = pd.read_table(target_path, sep='|', index_col=0, low_memory=False)
    return target_df


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


def plot_pnl():
    data = pd.read_csv('/mnt/mfs/AATST/corr_tst_pnls', sep='|', index_col=0, parse_dates=True)
    for x in data.columns:
        print(x)
        pnl_df = data[x].replace(0, np.nan).fillna(method='ffill').dropna()
        pnl_df = pnl_df[pnl_df.index > pd.to_datetime('20180101')]
        sp = bt.AZ_Sharpe_y(data[x])
        plot_send_result(pnl_df, sp, f'ALPHA {x}')


# if __name__ == '__main__':
#     plot_pnl()
