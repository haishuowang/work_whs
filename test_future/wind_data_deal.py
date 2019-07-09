import sys

sys.path.append('/mnf/mfs')
from work_whs.loc_lib.pre_load import *

fut_name = 'J'
load_root_path = '/mnt/mfs/dat_whs/spot_data/煤焦钢矿_螺纹线材'
# spot_list = ['国内外煤炭库存', '国内外煤炭运价', '电力']
spot_list = [x[:-4] for x in os.listdir(load_root_path)]
for spot_name in spot_list:
    data = pd.read_excel(f'{load_root_path}/{spot_name}.xls',
                         index_col=0, encoding='gbk', skiprows=[0, 1, 3, 4, 5, 6, 7, 8, 9], parse_dates=True)
    bt.AZ_Path_create(f'/mnt/mfs/DAT_FUT/spot/{fut_name}')
    data.to_csv(f'/mnt/mfs/DAT_FUT/spot/{fut_name}/{spot_name}.csv', sep='|')
