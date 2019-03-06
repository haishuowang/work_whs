import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *


def load_index_data(target_path, weight_list):
    index_name_list = ['000300', '000905']
    data = bt.AZ_Load_csv(target_path)
    target_df = data[index_name_list].mul(weight_list).sum(1)
    return target_df * 0.01


def AZ_Load_csv(target_path, parse_dates=True, sep='|'):
    target_df = pd.read_table(target_path, sep=sep, index_col=0, low_memory=False, parse_dates=parse_dates)
    return target_df


class LoadData:
    @staticmethod
    def stock_alpha(return_df, index_df):
        aadj_r_alpha = return_df.sub(index_df, axis=0)
        return aadj_r_alpha


def get_data(root_path, fun, part_path, *args):
    print(f'{root_path}/{part_path}')
    return fun(f'{root_path}/{part_path}', *args)


def get_data_fun(root_path, data_name_list):
    base_data = dict()
    for data_name in data_name_list:
        base_data[data_name] = get_data(root_path, *data_info[data_name])
    return base_data


loaddata = LoadData()
data_info = dict()
data_info['close'] = [bt.AZ_Load_csv, 'EM_Funda/DERIVED_14/aadj_p.csv']
data_info['high'] = [bt.AZ_Load_csv, 'EM_Funda/DERIVED_14/aadj_p_HIGH.csv']
data_info['low'] = [bt.AZ_Load_csv, 'EM_Funda/DERIVED_14/aadj_p_LOW.csv']
data_info['open'] = [bt.AZ_Load_csv, 'EM_Funda/DERIVED_14/aadj_p_OPEN.csv']
data_info['aadj_r'] = [bt.AZ_Load_csv, 'EM_Funda/DERIVED_14/aadj_r.csv']

data_info['index300'] = [load_index_data, 'EM_Funda/INDEX_TD_DAILYSYS/CHG.csv', [1, 0]]
data_info['index500'] = [load_index_data, 'EM_Funda/INDEX_TD_DAILYSYS/CHG.csv', [0, 1]]
data_info['index800'] = [load_index_data, 'EM_Funda/INDEX_TD_DAILYSYS/CHG.csv', [0.5, 0.5]]

if __name__ == '__main__':
    root_path = '/mnt/mfs/DAT_EQT'
    data_name_list = [
        'aadj_r',
        'index300',
        'index500',
        'index800',
    ]

    data_dict = get_data_fun(root_path, data_name_list)
    data_dict['aadj_r_300'] = data_dict['aadj_r'].sub(data_dict['index300'], axis=0)
    data_dict['aadj_r_500'] = data_dict['aadj_r'].sub(data_dict['index500'], axis=0)
    data_dict['aadj_r_800'] = data_dict['aadj_r'].sub(data_dict['index800'], axis=0)
