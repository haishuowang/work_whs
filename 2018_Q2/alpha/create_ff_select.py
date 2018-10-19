import pandas as pd
import os

root_path = '/media/hdd1/DAT_PreCalc/PreCalc_whs'
config_info_file_list = [x for x in os.listdir(root_path) if 'pkl' in x]

all_factor_set = set()
for config_info_file in config_info_file_list:
    config_info_dict = pd.read_pickle(os.path.join(root_path, config_info_file))
    factor_info = config_info_dict['factor_info']
    all_factor_set = all_factor_set | set(factor_info[['name1', 'name2', 'name3']].values.ravel())

all_factor_list = list(sorted(all_factor_set))

use_factor_list = [x for x in all_factor_set if x.startswith('R_')]

data = pd.read_csv('/mnt/mfs/release/stable_release/open_lib/shared_utils/etlkit/const/FundNameChange_Official.csv')

use_data_list = []
for data_file in data['RenamedColumn'].values:
    for facator_name in use_factor_list:
        if data_file in facator_name:
            use_data_list += [data_file]


with open('/mnt/mfs/alpha_whs/ff.select', 'r') as f:
    a = f.read()
old_data_list = a.split('\n')

use_data_list = sorted(set(use_data_list) - set(old_data_list))

with open('/mnt/mfs/alpha_whs/ff.select', 'a') as f:
    f.write('\n' + '\n'.join(use_data_list))

