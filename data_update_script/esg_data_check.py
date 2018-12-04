import sys
sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *

root_path = '/mnt/mfs/dat_whs/tmp'
# date_path_list = os.listdir(root_path)

date_list = ['201811211648', '201811231800']

data_path_1 = f'{root_path}/{date_list[0]}'
data_path_2 = f'{root_path}/{date_list[1]}'

sector_list = sorted(os.listdir(data_path_1))
for sector_name in sector_list:
    sector_path_1 = f'{data_path_1}/{sector_name}'
    sector_path_2 = f'{data_path_2}/{sector_name}'
    index_list = sorted(os.listdir(sector_path_1))
    for index_name in index_list:
        print(index_name)
        index_df_1 = pd.read_pickle(f'{sector_path_1}/{index_name}').dropna(how='all', axis=1).dropna(how='all', axis=0)
        index_df_2 = pd.read_pickle(f'{sector_path_2}/{index_name}').dropna(how='all', axis=1).dropna(how='all', axis=0)
        xinx = sorted(list(set(index_df_1.index) & set(index_df_2.index)))
        xnms = sorted(list(set(index_df_1.columns) | set(index_df_2.columns)))
        a = index_df_1.reindex(index=xinx, columns=xnms).replace(np.nan, 0)
        b = index_df_2.reindex(index=xinx, columns=xnms).replace(np.nan, 0)

        c = a != b
        d = c.sum().sum()
        print(d)
        if d != 0:
            exit(-1)
