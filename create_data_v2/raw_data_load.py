import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *

import work_whs.create_data_v2.raw_data_info as rdi

data_info = rdi.data_info


def get_data(root_path, fun, part_path, *args):
    print(f'{root_path}/{part_path}')
    return fun(f'{root_path}/{part_path}', *args)


def get_data_fun(root_path, data_name_list):
    base_data = dict()
    for data_name in data_name_list:
        base_data[data_name] = get_data(root_path, *data_info[data_name])
    return base_data


