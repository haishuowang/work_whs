import os

root_path = '/mnt/mfs/DAT_FUT/intraday/fut_1mbar'
fut_name_list = os.listdir(root_path)
for fut_name in fut_name_list:
    fut_path = f'{root_path}/{fut_name}'
    movie_name = os.listdir(fut_path)
    for temp in movie_name:
        new_name = temp[:-4] + '.CFE'
        print(temp, new_name)
        os.rename(f'{fut_path}/{temp}', f'{fut_path}/{new_name}')