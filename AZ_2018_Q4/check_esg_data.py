from work_whs.loc_lib.pre_load import *


config_file_list = ['market_top_300plus_True_20181102_1327_hold_5__7',
                    'market_top_300to800plus_industry_10_15_True_20181103_1342_hold_5__7',
                    'market_top_300to800plus_industry_45_50_True_20181103_2326_hold_5__7',
                    'market_top_800plus_True_20181104_0237_hold_5__7']

data_load_path = '/mnt/mfs/dat_whs/data/new_factor_data'

for config_file in config_file_list:
    print('_______________________________________________')
    print(config_file)
    data = pd.read_pickle(f'/media/hdd1/DAT_PreCalc/PreCalc_whs/{config_file}.pkl')
    sector_name = config_file.split('True')[0][:-1]
    factor_info = data['factor_info']
    for factor_name in sorted(list(set(factor_info.name1))):
        print(factor_name)
        # factor_data = pd.read_pickle(f'{data_load_path}/{sector_name}/{factor_name}.pkl')
        # print(factor_data.abs().sum(1).tail())

