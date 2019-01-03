import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *

root_path = '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_F1'
file_name_list = os.listdir(root_path)
for file_name in file_name_list:
    print('___________________________________________________________________')
    print(file_name)
    data = pd.read_csv(f'{root_path}/{file_name}', sep='|', index_col=0)
    a = list(data.iloc[-1].replace(np.inf, np.nan).values)
    print(a)
    print(np.nansum(a))
