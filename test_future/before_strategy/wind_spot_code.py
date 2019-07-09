import sys
sys.path.append('/mnt/mfs/')

from work_whs.loc_lib.pre_load import *

wind_spot_path = '/mnt/mfs/dat_whs/spot_data'
save_root_path = '/mnt/mfs/dat_whs/tmp/wind_spot_code'
path_name_list = os.listdir(wind_spot_path)

for path_name in path_name_list:
    spot_path = f'{wind_spot_path}/{path_name}'
    file_name_list = [x[:-4] for x in os.listdir(spot_path)]
    for file_name in file_name_list:
        file_path = f'{spot_path}/{file_name}.xls'
        data = pd.read_excel(file_path, index_col=0, nrows=1, skiprows=[0, 1, 3, 4, 6, 7, 8])
        save_path = f'{save_root_path}/{path_name}'
        bt.AZ_Path_create(save_path)
        data.T.to_csv(f'{save_path}/{file_name}.csv')
        print(file_path)
