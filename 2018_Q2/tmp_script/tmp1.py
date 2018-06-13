import feather
import os
import pandas as pd
import datetime
# from itertools import product, permutations, combinations
# import sys
# sys.path.append('../loc_lib')
# file = open('back_test_import.txt', 'r')
# exec(compile(file.read(), '', 'exec'))
# load_path = r'/media/hdd1/whs_data/adj_data/index_universe'
# # save_path = r'/media/hdd1/whs_data/adj_data/index_universe_f'
# # os.mkdir(save_path)
#
# start_time = datetime.datetime.now()
# file_list = os.listdir(load_path)
# for file_name in file_list[:31]:
#     print(file_name[:-4])
#     data = pd.read_pickle(os.path.join(load_path, file_name))
#     # feather.write_dataframe(data, os.path.join(save_path, file_name[:-4]+'.feather'))
# end_time = datetime.datetime.now()
# print(len(file_list), (end_time - start_time).seconds)
# start_time = datetime.datetime.now()
# file_list = os.listdir(save_path)
# for file_name in file_list:
#     print(file_name[:-4])
#     DATA = feather.read_dataframe(os.path.join(save_path, file_name))
# end_time = datetime.datetime.now()
# print(len(file_list), (end_time - start_time).seconds)
