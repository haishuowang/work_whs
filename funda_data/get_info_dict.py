import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *

root_path = '/mnt/mfs/dat_whs/data/factor_to_fun'
info_dict = OrderedDict()
file_name_list = sorted(os.listdir(root_path))


def add_info_data(info_dict, root_path, file_name):
    info_data = pd.read_pickle(f'{root_path}/{file_name}')
    info_data['raw_data_path'] = [str(x) for x in info_data['raw_data_path']]
    info_dict.update(dict({file_name: info_data}))
    return info_dict


for file_name in file_name_list:
    if 'fun' not in file_name:
        info_dict = add_info_data(info_dict, root_path, file_name)

