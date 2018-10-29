import pandas as pd
import numpy as np
from multiprocessing import Pool
import os
from collections import OrderedDict
import gc


def path_create(target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)


def deal_fun(year_name, load_path, save_path):
    year_load_path = os.path.join(load_path, year_name)
    stock_list = os.listdir(year_load_path)
    for stock_name in stock_list:
        stock_data = pd.read_csv(os.path.join(year_load_path, stock_name), header=None)
        stock_data.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
        for date, part_data in stock_data.groupby('Date'):
            year, month, day = date.split('/')
            date_path = os.path.join(save_path, year, year + month, year + month + day)
            path_create(date_path)
            part_data.to_csv(os.path.join(date_path, stock_name), index=False)


if __name__ == '__main__':
    load_path = r'/media/haishuowang/DATA/stock_minute_data_raw'
    save_path = r'/home/haishuowang/Downloads/stock_minute_data_file'
    year_list = os.listdir(load_path)
    pool = Pool(8)
    for year_name in year_list:
        pool.apply_async(deal_fun, args=(year_name, load_path, save_path))
    pool.close()
    pool.join()
