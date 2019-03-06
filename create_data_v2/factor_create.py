import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *
import work_whs.create_data_v2.raw_data_info as rdi

root_path = '/mnt/mfs/DAT_EQT'

if __name__ == '__main__':
    data_name_list = [
        'aadj_r',
        'index300',
        'index500',
        'index800',
    ]

    data_dict = rdi.get_data_fun(root_path, data_name_list)
    data_dict['aadj_r_300'] = data_dict['aadj_r'].sub(data_dict['index300'], axis=0)
    data_dict['aadj_r_500'] = data_dict['aadj_r'].sub(data_dict['index500'], axis=0)
    data_dict['aadj_r_800'] = data_dict['aadj_r'].sub(data_dict['index800'], axis=0)


