import os
from shutil import copy

file_name_list = ['WHS018AUG01', 'WHS018JUL01', 'WHS018JUN01']
for file_name in file_name_list:
    from_path = '/mnt/mfs/work_whs/AZ_2018_Q2/alpha/{}.py'.format(file_name)
    to_path = '/mnt/mfs/alpha_whs/{}.py'.format(file_name)
    copy(from_path, to_path)
