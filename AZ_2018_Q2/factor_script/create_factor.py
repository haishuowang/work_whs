import pandas as pd
import numpy as np
import os
from functools import reduce
import open_lib_c.shared_tools.back_test as bt
bt.AZ_Rolling()

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

EQA = [x for x in all_close.columns if x.startswith('0') or x.startswith('3') or x.startswith('6')]
global EQA_open, EQA_high, EQA_low, EQA_close, EQA_volume, EQA_adj_r, EQA_amount

EQA_open = all_open[EQA]
EQA_high = all_high[EQA]
EQA_low = all_low[EQA]
EQA_close = all_close[EQA]
EQA_amount = all_amount[EQA]

EQA_volume = all_volume[EQA]
EQA_adj_r = all_adj_r[EQA]

date_index = pd.to_datetime(EQA_open.index.astype(str))
EQA_open.index = date_index
EQA_high.index = date_index
EQA_low.index = date_index
EQA_close.index = date_index
EQA_amount.index = date_index

EQA_volume.index = date_index
EQA_adj_r.index = date_index


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


