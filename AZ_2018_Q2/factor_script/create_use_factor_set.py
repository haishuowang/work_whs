import os
import pandas as pd
import loc_lib.shared_tools.back_test as bt


sector_name = 'market_top_500'
use_factor = os.listdir('/mnt/mfs/dat_whs/data/factor_data/' + sector_name)


def moment_fun(root_path, sector_df):

    aadj_r = bt.AZ_Load_csv(root_path + 'EM_Funda/DERIVED_14/aadj_r.csv')
    ma30 = bt.AZ_Rolling_mean(aadj_r, 30)
    ma300 = bt.AZ_Rolling_mean(aadj_r, 300)


def vol_fun(root_path, sector_df):
    aadj_r = bt.AZ_Load_csv(root_path + 'EM_Funda/DERIVED_14/aadj_r.csv')


if __name__ == '__main__':
    pass