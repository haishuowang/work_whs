import pandas as pd
import os


target_file_list = [
    # 'market_top_300to800plus_industry_45_50_True_20181205_1859_hold_5__11',
    # 'market_top_800plus_industry_45_50_True_20181206_2040_hold_5__11',
    # 'market_top_800plus_industry_55_True_20181206_2253_hold_5__11'
    # 'market_top_300to800plus_industry_45_50_True_20181208_1305_hold_20__11'
    # 'market_top_300to800plus_industry_10_15_True_20181203_2013_hold_20__13',
    # 'market_top_800plus_True_20181202_1830_hold_20__7',
    # 'market_top_300to800plus_True_20181228_1404_hold_5__16',
    # 'market_top_300to800plus_industry_10_15_True_20181230_0155_hold_5__16',
    # 'market_top_300plus_industry_10_15_True_20190102_1034_hold_20__16'
    # 'market_top_300to800plus_industry_10_15_True_20190105_0015_hold_5__19',
    # 'market_top_300to800plus_industry_10_15_True_20190105_0357_hold_20__16',
    # 'market_top_300to800plus_industry_10_15_True_20190106_0028_hold_20__19',
    # 'market_top_300to800plus_True_20190103_1202_hold_20__16',
    # 'market_top_300to800plus_industry_20_25_30_35_True_20190106_1249_hold_20__16'
    # 'market_top_300to800plus_industry_10_15_True_20190105_0357_hold_20__16'
    'market_top_300plus_industry_20_25_30_35_True_20190214_2009_hold_5__20'
]

# target_file_list = ['market_top_300to800plus_industry_55_True_20181202_1714_hold_20__7',
#                     'market_top_300to800plus_industry_45_50_True_20181202_1423_hold_20__7',
#                     'market_top_800plus_industry_45_50_True_20181130_0412_hold_5__7']

from_path = '/mnt/mfs/dat_whs/alpha_data'
to_path = '/media/hdd1/DAT_PreCalc/PreCalc_whs/config_file'
for file_name in target_file_list:
    bashCommand = f"cp {from_path}/{file_name}.pkl {to_path}"
    os.system(bashCommand)
