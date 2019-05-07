import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/mnt/mfs')
from work_whs.loc_lib.shared_tools import send_email
import work_whs.loc_lib.shared_tools.back_test as bt
figure_save_path = '/mnt/mfs/dat_whs/tmp_figure'


def plot_send_result(pnl_df, sharpe_ratio, subject, text=''):
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


def plot_send_data(raw_df, subject, text=''):
    raw_df.plot(legend=True)
    plt.savefig(f'{figure_save_path}/{subject}.png')
    plt.close()
    to = ['whs@yingpei.com']
    filepath = [f'{figure_save_path}/{subject}.png']
    send_email.send_email(text, to, filepath, subject)


def savfig_send(subject='tmp', text='', to=None, filepath=None):
    target_save_path = f'{figure_save_path}/{subject}.png'
    if to is None:
        to = ['whs@yingpei.com']
    if filepath is None:
        filepath = [target_save_path]
    plt.savefig(target_save_path)
    send_email.send_email(text, to, filepath, subject)
    plt.close()



