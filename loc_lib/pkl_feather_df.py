import feather
import pandas as pd
import os
from datetime import datetime
import gc


def path_create(target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)


def pkl_feather_df(load_path, save_path=None):
    if save_path is None:
        save_path = load_path+'_f'
    path_create(save_path)
    file_list = os.listdir(load_path)
    print(len(file_list))
    for file_name in sorted(file_list):
        print(file_name)
        data = pd.read_pickle(os.path.join(load_path, file_name))
        data = data.astype(float)
        data.index = pd.DataFrame(data.index.astype(str))
        feather.write_dataframe(data, os.path.join(save_path, file_name.split('.')[0] + '.ftr'))


def feather_pkl_test(path_1, path_2):
    start_time = datetime.now()
    file_list_1 = os.listdir(path_1)
    for i in range(200):
        for file_1 in file_list_1[:1]:
            data = feather.read_dataframe(os.path.join(path_1, file_1))
            # del data
    # gc.collect()
    end_time = datetime.now()
    print((end_time - start_time).seconds + 0.1 ** 6 * (end_time - start_time).microseconds)

    start_time = datetime.now()
    file_list_2 = os.listdir(path_2)
    for i in range(200):
        for file_2 in file_list_2[:1]:
            data = pd.read_pickle(os.path.join(path_2, file_2))
    #         del data
    # gc.collect()
    end_time = datetime.now()
    print((end_time - start_time).seconds + 0.1 ** 6 * (end_time - start_time).microseconds)


def pkl_time_tran(load_path):
    file_list = os.listdir(load_path)
    for file_name in file_list:
        print(file_name)
        data = pd.read_pickle(os.path.join(load_path, file_name))
        # data = data.astype(float)
        # if type(data.index[0]) is tuple:
        #     data.index = pd.to_datetime([x[0] for x in data.index])
        # else:
        print(data.index[0], type(data.index[0]))
        data.to_pickle(os.path.join(load_path, file_name))


def adj_factor(load_path):
    save_path = load_path + '_f'
    file_list = os.listdir(load_path)
    path_create(save_path)
    for file_name in file_list:
        print(file_name)
        data = pd.read_pickle(os.path.join(load_path, file_name))
        if len(data.columns) != 3543:
            data = data[[x for x in data.columns if type(x) == str]]
        data.to_pickle(os.path.join(save_path, file_name))


if __name__ == '__main__':
    load_path = '/media/hdd0/whs/data/adj_data/index_universe'
    # pkl_feather_df(load_path)
    # load_path = '/media/hdd0/whs/data/adj_data/fnd_pct'
    # pkl_feather_df(load_path)
    adj_factor(load_path)
