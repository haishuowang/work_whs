import pandas as pd
import numpy as np
import os
import re

root_path = '/mnt/mfs/DAT_EQT/DailyPX/EqtIndex/000300.SH'


file_list = [x for x in os.listdir(root_path) if len(x) > 20]
# print(file_list)
target_df = pd.DataFrame()
for file_name in sorted(file_list):
    load_path = os.path.join(root_path, file_name)
    date = pd.to_datetime(re.findall('\d+', file_name))
    tmp_data = pd.read_csv(load_path, index_col=0)
    weight = tmp_data[['Weight']].T
    weight.index = [date]
    target_df = target_df.append(weight)
target_df.to_pickle(os.path.join(root_path, ))