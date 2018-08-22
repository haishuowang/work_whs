import os
import pandas as pd
import datetime
from multiprocessing import Pool, Lock

all_file_list = ['R_' + x for x in os.listdir('/mnt/mfs/DAT_EQT/EM_Funda/LICO_FN_SIGQUAFINA') if 'first' in x.lower()]
all_file_list.remove('R_UpSampleDate_First.csv')

check_file_list = os.listdir('/mnt/mfs/DAT_EQT/EM_Funda/daily')

in_list = []
out_list = []
for file_name in all_file_list:
    if file_name in check_file_list:
        in_list.append(file_name)
    else:
        out_list.append(file_name)
