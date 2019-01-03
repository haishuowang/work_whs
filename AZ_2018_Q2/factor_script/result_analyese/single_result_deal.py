import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *


def get_result_data(root_path, file_name):
    data = pd.read_csv(f'{root_path}/{file_name}', sep='|', index_col=0, header=None)
    print(data)
    data.columns = ['name_1', 'fun_name', 'sector_name', 'in_condition', 'out_condition', 'ic', 'sp_d', 'sp_m', 'sp_u',
                    'pot_in', 'fit_ratio', 'leve_ratio', 'sp_in', 'sp_q_out']
    data_sort = data.sort_values(by='sp_in')
    return data_sort


root_path = '/mnt/mfs/dat_whs/result/result'
# def main():
file_name_list = [x for x in os.listdir(root_path) if 'single_test' in x and
                  os.path.getsize(os.path.join(root_path, x)) != 0]
print(file_name_list)
for file_name in file_name_list:
    result_df = get_result_data(root_path, file_name)

# if __name__ == '__main__':
