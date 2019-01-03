import pandas as pd
import os
from datetime import datetime


def move_config_file():
    target_file_list = ['market_top_300to800plus_industry_55_True_20181202_1714_hold_20__7',
                        'market_top_300to800plus_industry_45_50_True_20181202_1423_hold_20__7',
                        'market_top_800plus_industry_45_50_True_20181130_0412_hold_5__7']

    from_path = '/mnt/mfs/dat_whs/alpha_data'
    to_path = '/media/hdd1/DAT_PreCalc/PreCalc_whs/config_file'
    for file_name in target_file_list:
        bashCommand = f"cp {from_path}/{file_name}.pkl {to_path}"
        os.system(bashCommand)


def delete_expire_file(target_path, target_date):
    file_list = os.listdir(target_path)
    for file_name in file_list:
        move_timestamp = os.path.getmtime(os.path.join(target_path, file_name))
        move_datetime = datetime.fromtimestamp(move_timestamp)
        if target_date > move_datetime:
            print(file_name)
            # os.remove(os.path.join(target_path, file_name))

# if __name__ == '__main__':
#     target_path_list = ['/mnt/mfs/dat_whs/result/result',
#                         '/mnt/mfs/dat_whs/result/log'
#                         '/mnt/mfs/dat_whs/result/para'
#                         ]
#     for target_path in target_path_list:
#         target_date = pd.to_datetime('20181101')
#         delete_expire_file(target_path, target_date)

