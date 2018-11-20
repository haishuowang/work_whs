import pandas as pd
import os

# target_file_list = ['market_top_300to800plus_industry_10_15_True_20181117_2314_hold_5__7',
#                     'market_top_300plus_True_20181115_1919_hold_5__7']

target_file_list = ['bulletin_num_df.csv', 'news_num_df.csv']


from_path = '/mnt/mfs/dat_whs/EM_Funda/my_data_test'
to_path = '/mnt/mfs/DAT_PUBLIC/dat_whs'
for file_name in target_file_list:
    bashCommand = f"cp {from_path}/{file_name} {to_path}"
    os.system(bashCommand)
