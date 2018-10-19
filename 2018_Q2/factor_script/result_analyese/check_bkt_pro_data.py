import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import product, permutations, combinations
from sklearn.cluster import KMeans
import random
import sys
from datetime import datetime

sys.path.append('/mnt/mfs/work_whs/2018_Q2')
sys.path.append('/mnt/mfs/work_whs')
from loc_lib.shared_tools import send_email
import loc_lib.shared_tools.back_test as bt

from factor_script.script_load_data import load_sector_data, load_locked_data, load_pct, \
    load_part_factor, create_log_save_path, load_index_data, deal_mix_factor

from factor_script.main_file import main_file_sector_3 as mf

sector_name = 'market_top_1000_industry_10_15'
bkt_path = f'/mnt/mfs/dat_whs/data/new_factor_data/{sector_name}'
pro_path = f'/media/hdd1/DAT_PreCalc/PreCalc_whs/{sector_name}'

test_file_name = sorted(os.listdir(pro_path))
test_file_name.remove('vol_p50d.pkl')
test_file_name.remove('turn_p30d_0.24.pkl')
test_file_name.remove('vol_p100d.pkl')
test_file_name.remove('return_p90d_0.2.pkl')
test_file_name.remove('EBIT_and_asset_QYOY_Y3YGR_0.3.pkl')

begin_date = pd.to_datetime('20140101')
end_date = pd.to_datetime('20180901')

for file_name in test_file_name:
    print(file_name)
    bkt_data = pd.read_pickle(f'{bkt_path}/{file_name}')
    pro_data = pd.read_pickle(f'{pro_path}/{file_name}')

    part_bkt_data = bkt_data.loc[begin_date:end_date].replace(np.nan, 0).round(10)
    part_pro_data = pro_data.loc[begin_date:end_date].replace(np.nan, 0).round(10)

    col = sorted(list(set(part_bkt_data.columns) & set(part_pro_data.columns)))
    d = part_bkt_data[col]
    e = part_pro_data[col]
    print((d != e).sum().sum())
    if (d != e).sum().sum() != 0:
        (d - e).loc[pd.to_datetime('2018-03-22')].replace(0, np.nan).dropna()
        print((d - e).abs().sum(axis=1).replace(0, np.nan).dropna())
        break
