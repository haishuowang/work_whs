# 服务器上运行的程序，消耗大量内存
import pandas as pd
import numpy as np
from multiprocessing import Pool
import os
from collections import OrderedDict
import gc


def path_create(target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)


def path_create(target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        return True
    elif len(os.listdir(target_path)) != 6:
        return True
    else:
        return False


def save_data(part_data, path, column, stock_name):
    save_column_path = os.path.join(path, column)
    path_create(save_column_path)
    part_data[['Date', 'Time', column]].to_csv(os.path.join(save_column_path, stock_name), index=False)
    for x in locals().keys():
        del locals()[x]
    gc.collect()


def deal_day_data(date, day_data, day_save_path):
    stock_name = sorted(list(day_data.keys()))
    index = day_data[stock_name[0]].index
    result = path_create(day_save_path)
    if result:
        for column in ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']:
            exec('{}_df = pd.DataFrame()'.format(column))
            exec('{}_df = pd.DataFrame([[None] * len(stock_name)] * 240)'.format(column))
            exec('{}_df.columns = stock_name'.format(column))
            exec('{}_df.index = index'.format(column))

        for stock in stock_name:
            # print(stock)
            stock_daily_data = day_data[stock]
            for column in ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']:
                exec('{}_df[stock] = stock_daily_data[column]'.format(column))
            del stock_daily_data
            gc.collect()

        for column in ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']:
            exec('{}_df.to_csv(\'{}\')'.format(column, os.path.join(day_save_path, column + '.csv')))
            exec('del {}_df'.format(column))
            gc.collect()
        print('day:{} finished'.format(date))
    else:
        del day_data
        gc.collect()
        print('day:{} finished'.format(date))
        pass
    gc.collect()


def deal_fun(year_name, load_path, save_path):
    load_year_path = os.path.join(load_path, year_name)
    stock_list = os.listdir(load_year_path)
    year_dict_data = OrderedDict()

    for stock_name in sorted(stock_list):
        # print(stock_name)
        stock_path = os.path.join(load_year_path, stock_name)
        stock_data = pd.read_csv(stock_path, header=None)
        stock_data.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
        stock_data.index = stock_data['Time'].values
        for date, part_data in stock_data.groupby(by='Date'):
            year, month, day = date.split('/')
            date = int(date.replace('/', ''))
            save_day_path = os.path.join(save_path, year, year + month, year + month + day)
            path_create(save_day_path)
            try:
                year_dict_data[date].update({stock_name.split('.')[0]: part_data})
            except Exception as err:
                year_dict_data[date] = {stock_name.split('.')[0]: part_data}
    # pool = Pool(6)
    for date in sorted(list(year_dict_data.keys())):
        save_day_path = os.path.join(save_path, str(date)[:4], str(date)[:6], str(date))
        # pool.apply_async(deal_day_data, args=(date, year_dict_data[date], save_day_path))
        deal_day_data(date, year_dict_data[date], save_day_path)
        year_dict_data.pop(date)
    del year_dict_data
    gc.collect()
    # pool.close()
    # pool.join()
    for x in locals().keys():
        del locals()[x]
    gc.collect()


if __name__ == '__main__':

    load_path = r'/media/hdd0/Data/data/EQT_1MBar'
    save_path = r'/media/hdd0/Data/data/EQT_1MBar_Data'
    year_list = os.listdir(load_path)
    year_list = ['Stk_1F_2017']
    for year_name in year_list[:1]:
        deal_fun(year_name, load_path, save_path)

