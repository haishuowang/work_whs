import pandas as pd
import os

# target_file_list = ['market_top_300to800plus_industry_10_15_True_20181117_2314_hold_5__7',
#                     'market_top_300plus_True_20181115_1919_hold_5__7']

target_file_list = ['market_top_300to800plus_True_20181124_1156_hold_20__11']

from_path = '/mnt/mfs/dat_whs/alpha_data'
to_path = '/media/hdd1/DAT_PreCalc/PreCalc_whs'
for file_name in target_file_list:
    bashCommand = f"cp {from_path}/{file_name}.pkl {to_path}"
    os.system(bashCommand)
