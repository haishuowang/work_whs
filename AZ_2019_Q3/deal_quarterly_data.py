import sys

sys.path.append('/mnt/mfs')
import string
from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load.plt import plot_send_result

root_path = '/mnt/mfs/DAT_EQT'
table_name = 'LICO_FN_GENERALFINA'
file_name_list = os.listdir(f'{root_path}/EM_Funda/{table_name}')


class IndClass:
    def __init__(self, root_path):
        self.df_set = {}
        for i in range(10):
            self.df_set[str(i).zfill(2)] = bt.AZ_Load_csv(f'{root_path}/EM_Funda/LICO_IM_INCHG/'
                                                          f'ZhongZheng_Level1_{str(i).zfill(2)}.csv')


ind_set = IndClass(root_path).df_set

for file_name in file_name_list[:1]:
    file_name = 'ROA1_First.csv'
    data = bt.AZ_Load_csv(f'{root_path}/EM_Funda/{table_name}/{file_name}', parse_dates=False)

    filer_data = data[data.index.str.endswith('Q4')]
    print(filer_data)
    # for i in range(10):
    #     ind_set[str(i).zfill(2)]
