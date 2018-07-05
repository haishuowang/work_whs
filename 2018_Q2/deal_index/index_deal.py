import pandas as pd
import os
import open_lib.shared_tools.back_test as bt


index_list = ['000016', '000300', '000905']

index_pct_path = '/mnt/mfs/DAT_EQT/EM_Tab09/raw_data/INDEX_TD_DAILYSYS/split_data/SECURITYCODE,TRADEDATE,CHGpkl'
index_save_path = '/mnt/mfs/DAT_EQT/EM_Tab09/adj_data/INDEX_TD_DAILYSYS'
bt.AZ_Path_create(index_save_path)
data = pd.read_pickle(index_pct_path)
data_sort = data.sort_values('TRADEDATE')
# index_list = sorted(list(set(data_sort['SECURITYCODE'].values)))
for index_code in index_list:
    part_data = data_sort[data_sort['SECURITYCODE'] == index_code]
    target_df = part_data.pivot('TRADEDATE', 'SECURITYCODE', 'CHG')
    target_df.to_pickle(os.path.join(index_save_path, index_code + '.pkl'))
