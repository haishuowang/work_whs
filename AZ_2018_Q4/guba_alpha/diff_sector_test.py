import sys
sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *

guba_data = bt.AZ_Load_csv('/media/hdd1/DAT_EQT/EM_Funda/dat_whs/bar_num_df.csv')

sector_data_df = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/LICO_IM_INCHG/ZhongZheng_Level2_0604.csv')


