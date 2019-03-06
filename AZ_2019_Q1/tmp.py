import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load.bkt import get_main_model

target_path = '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_F3/T500P'
file_name_list = os.path.join(target_path)
print(len(file_name_list))
# data = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_F3/T300P/REMRATIO.VAGR02.031')
