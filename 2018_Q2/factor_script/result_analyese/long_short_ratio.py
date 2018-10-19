import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import product, permutations, combinations
from sklearn.cluster import KMeans
import random
import sys

sys.path.append('/mnt/mfs/work_whs/2018_Q2')
sys.path.append('/mnt/mfs/work_whs')
from loc_lib.shared_tools import send_email
import loc_lib.shared_tools.back_test as bt


def load_index_data(root_path, index_name, xinx):
    data = bt.AZ_Load_csv(os.path.join(root_path, 'EM_Tab09/INDEX_TD_DAILYSYS/CHG.csv'))
    target_df = data[index_name].reindex(index=xinx)
    return target_df * 0.01


def main(root_path, pos_df):
    xnms = pos_df.coulmns
    xinx = pos_df.index
    index_df_1 = load_index_data(root_path, '000300', xinx).fillna(0)
    index_df_2 = load_index_data(root_path, '000905', xinx).fillna(0)
    hedge_df = 0.5 * index_df_1 + 0.5 * index_df_2

    return_choose = bt.AZ_Load_csv(os.path.join(root_path, 'EM_Funda/DERIVED_14/aadj_r.csv'))
    return_choose = return_choose.reindex(index=xinx, columns=xnms)
    return_choose = return_choose.sub(hedge_df, axis=0)
    pnl_long = return_choose * pos_df[pos_df > 0].shift(2)
    pos_long = pos_df[pos_df > 0]
    asset_long = pnl_long.cumsum().iloc[-1]
    pos_sum = pos_long.abs().sum(axis=1).sum()
    return_long = asset_long/pos_sum * 250
    pos_long_sum = pos_long.abs().sum(axis=1).sum()
    annual_long = return_long/pos_long_sum * 250

    pnl_short = return_choose * pos_df[pos_df < 0].shift(2)
    pos_short = pos_df[pos_df < 0]
    return_short = pnl_short.cumsum().iloc[-1]
    pos_long_sum = pos_short.abs().sum(axis=1).mean()
    annual_short = return_short / pos_long_sum * 250

    print(f'annual_short: {annual_short}, annual_long: {annual_long}')
    return annual_short, annual_long
