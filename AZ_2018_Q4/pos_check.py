import sys
sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *

root_path = '/mnt/mfs/AAPOS'

pos_file_list = sorted([x for x in os.listdir(root_path) if x.startswith('WHS')])
return_df = bt.AZ_Load_csv(f'/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv')
for pos_file in pos_file_list:
    pos_df = bt.AZ_Load_csv(f'{root_path}/{pos_file}')



