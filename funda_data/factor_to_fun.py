import pandas as pd
import numpy as np
import funda_data.funda_data_deal as fdd
from funda_data.funda_data_deal import SectorData
import open_lib.shared_paths.path as pt
import loc_lib.shared_tools.back_test as bt
import os
from datetime import datetime, timedelta


def find_fun(fun_list):
    target_fun = fdd
    for a in fun_list:
        target_fun = getattr(target_fun, a)
    return target_fun


def load_raw_data(root_path, raw_data_path):
    raw_data_list = []
    for target_path in raw_data_path:
        tmp_data = bt.AZ_Load_csv(os.path.join(root_path.str(), target_path+'.csv'))
        raw_data_list += tmp_data
    return raw_data_list


def create_data(root_path, info_path, sector_df, mode='bkt'):
    info = pd.read_pickle(info_path)
    root_path = pt._BinFiles(mode)
    args = info['args']
    fun_list = info['fun'].split('.')
    raw_data_path = info['raw_data_path']
    raw_data_list = load_raw_data(root_path, raw_data_path)

    target_fun = find_fun(fun_list)
    target_df = target_fun(*raw_data_list, sector_df, *args)
    return target_df


# if __name__ == '__main__':
#     mode = 'bkt'
#
#     begin_date = pd.to_datetime('20100101')
#     end_date = pd.to_datetime('20180801')
#     sector_name = 'market_top_500'
#     save_root_path = '/mnt/mfs/dat_whs/data/new_factor_data/market_top_500_tmp'
#     bt.AZ_Path_create(save_root_path)
#     root_path = pt._BinFiles(mode)
#
#     sector_data_class = SectorData(root_path)
#     sector_df = sector_data_class.load_sector_data(begin_date, end_date, sector_name)
#
#     info_path = '/mnt/mfs/dat_whs/data/factor_to_fun/RQCHL_row_extre_0.2'
#     target_df = create_data(info_path, sector_df, mode='bkt')
