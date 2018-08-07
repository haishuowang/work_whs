import pandas as pd
import numpy as np
import os
from itertools import combinations


# zhongxin_level1 = pd.read_pickle('/mnt/mfs/DAT_EQT/STK_Groups3/ZhongXing_Level1.pkl')
# for key in zhongxin_level1.keys():
#     tmp_data = zhongxin_level1[key]
#     tmp_data.to_pickle('/mnt/mfs/dat_whs/data/sector_data/' + key + '.pkl')
root_path = '/mnt/mfs/DAT_EQT/EM_Tab14/adj_data/TRAD_MT_MARGIN'
name_list = ['RZRQYE', 'RZMRE', 'RZYE', 'RQMCL', 'RQYE', 'RQYL', 'RQCHL', 'RZCHE']
for tab_name_1, tab_name_2 in list(combinations(name_list, 2))[2:3]:
    data_1 = pd.read_pickle(os.path.join(root_path, tab_name_1 + '.pkl'))
    data_2 = pd.read_pickle(os.path.join(root_path, tab_name_2 + '.pkl'))

    data_df = data_1.div(data_2, fill_value=0)
    data_df.replace(np.inf, 0, inplace=True)
