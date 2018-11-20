import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *


# def test(x):
#     print('a1', x)
#     time.sleep(1 / 2)
#
#
# root_path_1 = '/mnt/mfs/dat_whs/EM_Funda/my_data'
# root_path_2 = '/mnt/mfs/dat_whs/EM_Funda/my_data_test'
# file_list = sorted([x for x in os.listdir(root_path_1) if x.startswith('stock_code_df')])
# for file_name in file_list:
#     print(file_name)
#     data_1 = pd.read_csv(f'{root_path_1}/stock_code_df_tab1_1', sep='|', index_col=0)
#     data_2 = pd.read_csv(f'{root_path_2}/stock_code_df_tab1_1', sep='|', index_col=0)
#     xinx = sorted(list(set(data_1.index) & set(data_2.index)))
#     xnms = sorted(list(set(data_1.columns) & set(data_2.columns)))
#     data1 = data_1.reindex(index=xinx, columns=xnms)
#     data2 = data_2.reindex(index=xinx, columns=xnms)
#     c = data_1 != data_2
#     print(c.sum().sum())

# safe_list = ['market_top_300plus',
#                 'market_top_300plus_industry_10_15',
#                 'market_top_300plus_industry_20_25_30_35',
#                 'market_top_300plus_industry_40',
#                 'market_top_300plus_industry_45_50',
#                 'market_top_300plus_industry_55',
#
#                 'market_top_300to800plus',
#                 'market_top_300to800plus_industry_10_15',
#                 'market_top_300to800plus_industry_20_25_30_35',
#                 'market_top_300to800plus_industry_40',
#                 'market_top_300to800plus_industry_45_50',
#                 'market_top_300to800plus_industry_55',
#
#                 'market_top_800plus',
#                 'market_top_800plus_industry_10_15',
#                 'market_top_800plus_industry_20_25_30_35',
#                 'market_top_800plus_industry_40',
#                 'market_top_800plus_industry_45_50',
#                 'market_top_800plus_industry_55'
#                 ]
# delete_path = '/mnt/mfs/dat_whs/data/new_factor_data'
# delete_list = os.listdir(delete_path)
# for diretory_name in delete_list:
#     if diretory_name not in safe_list:
#         bashCommand = f"rm -fr {delete_path}/{diretory_name}"
#         os.system(bashCommand)

def get_date_index(root_path):
    data = pd.read_csv(f'{root_path}/intra_daily_vwap.csv', index_col=0, sep='|', parse_dates=True)
    date_index = data.index
    return date_index


root_path = '/mnt/mfs/dat_whs/EM_Funda/my_data_test'
# all_file = [x for x in os.listdir(root_path) if x.startswith('intra')]
all_file = ['intra_dn_vol.csv']
date_index = get_date_index(root_path)
for file_name in all_file:
    print(file_name)
    data = pd.read_csv(f'{root_path}/{file_name}', index_col=0, sep='|', parse_dates=True)
    print(data.index[-1])
    data.index = date_index
    data.to_csv(f'{root_path}/{file_name}', sep='|')
