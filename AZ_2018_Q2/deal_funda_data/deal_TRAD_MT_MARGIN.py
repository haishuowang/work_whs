import pandas as pd
import numpy as np
import os
from functools import reduce
import open_lib_c.shared_tools.back_test as bt
import time


def pkl_to_csv():
    root_path = '/mnt/mfs/DAT_EQT/EM_Tab14/adj_data/TRAD_MT_MARGIN'

    save_path = '/mnt/mfs/dat_whs/csv_data/TRAD_MT_MARGIN'
    save_pkl_path = '/mnt/mfs/dat_whs/tmp_figure/TRAD_MT_MARGIN'
    bt.AZ_Path_create(save_pkl_path)
    col_list = ['RZRQYE', 'RZMRE', 'RZYE', 'RQMCL', 'RQYE', 'RQYL', 'RQCHL', 'RZCHE']
    for col in col_list:
        data = pd.read_pickle(os.path.join(root_path, col + '.pkl'))
        data.columns = bt.AZ_add_stock_suffix(data.columns)

        data.to_csv(os.path.join(save_path, col + '.csv'))
        data.to_pickle(os.path.join(save_pkl_path, col + '.pkl'))


def extreme_data(zscore_df, limit=2):
    zscore_df_copy = zscore_df.copy()
    zscore_df_copy[(zscore_df <= limit) & (zscore_df >= -limit)] = 0
    zscore_df_copy[zscore_df > limit] = 1
    zscore_df_copy[zscore_df < -limit] = -1
    return zscore_df_copy


def pnd_roll_mean_row_extre_set(tab_name, data, n, limit_list):
    data_roll_mean = bt.AZ_Rolling_mean(data, n)
    data_roll_mean.replace(0, np.nan, inplace=True)
    data_pnd_roll_mean_row_extre = bt.AZ_Row_zscore(data_roll_mean)
    data_pnd_roll_mean_row_extre.to_pickle(os.path.join(base_save_path,
                                                        '{}_p{}d_roll_mean_row_zs.pkl'.format(tab_name, n)))
    for limit in limit_list:
        target_df = extreme_data(data_pnd_roll_mean_row_extre, limit=limit)
        target_df.to_pickle()


def pnd_roll_mean_col_extre(tab_name, data, n, limit_list):
    data_pnd_col_extre = bt.AZ_Col_zscore(data, n)
    data_pnd_col_extre.to_pickle(os.path.join(base_save_path, '{}_p{}d__col_zs.pkl'.format(tab_name, n)))
    for limit in limit_list:
        extreme_data(data_pnd_col_extre, limit=limit)


def create_data():
    # 融资融券数据
    root_path = '/mnt/mfs/DAT_EQT/EM_Tab14/adj_data/TRAD_MT_MARGIN'
    name_list = ['RZRQYE', 'RZMRE', 'RZYE', 'RQMCL', 'RQYE', 'RQYL', 'RQCHL', 'RZCHE']
    for tab_name in name_list:
        data = pd.read_pickle(os.path.join(root_path, tab_name + '.pkl'))
        data.columns = bt.AZ_add_stock_suffix(data.columns)
        # 均值
        rolling_mean_list = [3, 10, 20, 60]
        limit_list = [1, 2]
        for n in rolling_mean_list:
            pnd_roll_mean_row_extre_set(tab_name, data, n, limit_list)
            pnd_roll_mean_col_extre(tab_name, data, n, limit_list)


if __name__ == '__main__':
    base_save_path = '/mnt/mfs/dat_whs/data/base_data'
    # funda_data()
