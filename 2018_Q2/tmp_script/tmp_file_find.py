import pandas as pd
import time
from multiprocessing import Pool
import os
import loc_lib.shared_tools.back_test as bt

pool = Pool()
all_file_list = [x for x in os.listdir('/mnt/mfs/DAT_EQT/EM_Funda/daily') if 'first' in x.lower()]

list_1 = []
list_2 = []
list_3 = []

a = time.time()
for file_name in all_file_list:
    load_path = os.path.join('/mnt/mfs/DAT_EQT/EM_Funda/daily', file_name)
    data = bt.AZ_Load_csv(load_path)
    data_num = data.iloc[-750:].count(axis=1).mean()
    print(file_name, data_num)
    if data_num <= 500:
        list_1 += [file_name]
    elif 500 < data_num <= 1500:
        list_2 += [file_name]
    elif 1500 < data_num:
        list_3 += [file_name]
    else:
        print('error')
b = time.time()
print(b-a)
