import numpy as np
import pandas as pd
import os
import sys
from itertools import product, permutations, combinations
from datetime import datetime
import time
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
import random
import matplotlib
import sys

sys.path.append("/mnt/mfs/LIB_ROOT")
sys.path.append('/mnt/mfs/work_whs/AZ_2018_Q2')
sys.path.append('/mnt/mfs/work_whs')
import open_lib.shared_paths.path as pt
from open_lib.shared_tools import send_email


def plot_send_result(pnl_df, sharpe_ratio, subject):
    figure_save_path = os.path.join('/mnt/mfs/dat_whs', 'tmp_figure')
    plt.figure(figsize=[16, 8])
    plt.plot(pnl_df.index, pnl_df.cumsum(), label='sharpe_ratio={}'.format(sharpe_ratio))
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(figure_save_path, '{}.png'.format(subject)))
    text = ''
    to = ['whs@yingpei.com']
    filepath = [os.path.join(figure_save_path, '{}.png'.format(subject))]
    send_email.send_email(text, to, filepath, subject)