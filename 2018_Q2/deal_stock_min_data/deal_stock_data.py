# 本地处理文件的程序 需要被拆分号的文件
import pandas as pd
import numpy as np
from multiprocessing import Pool
import os
from collections import OrderedDict
import gc


#
# root_path = r'/media/haishuowang/DATA/stock_minute_data_map'
# save_path = r'/media/haishuowang/DATA/equity_minute_data'
#
#
def path_create(target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        return True
    elif len(os.listdir(target_path)) != 6:
        return True
    else:
        return False


#
#
# def save_data_fun(day_data, save_day_path):
#     for column in ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']:
#         exec('{}_df = pd.DataFrame()'.format(column))
#
#     for stock_name in day_data.keys():
#         stock_data = day_data[stock_name]
#         stock_data.set_index('Time')
#         for column in ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']:
#             exec('{0}_df = pd.concat([{0}_df, stock_data[\'{0}\']], axis=1)'.format(column))
#
#     for column in ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']:
#         exec('{}_df.to_csv(\'{}\')'.format(column, os.path.join(save_day_path, column + '.csv')))
#
#
# year_list = os.listdir(root_path)
# year_list = [2001, 2002, 2003, 2004, 2005, 2006, 2007]
# for year in year_list[:1]:
#     year = str(year)
#     # year = str(2000)
#     year_path = os.path.join(root_path, year)
#     month_list = os.listdir(year_path)
#     for month in month_list:l
#         month = str(month)
#         month_path = os.path.join(year_path, str(month))
#         day_list = os.listdir(month_path)
#         for day in day_list:
#             day = str(day)
#             print(day)
#             day_path = os.path.join(month_path, day)
#             day_data = pd.read_pickle(os.path.join(day_path, day + '_map'))
#             save_day_path = os.path.join(save_path, year, month, day)
#             path_create(save_day_path)
#             save_data_fun(day_data, save_day_path)

def deal_day_data(index, day_path, day_save_path):
    stock_list = os.listdir(day_path)
    date = day_path[-8:]
    stock_name = [x[:-4] for x in stock_list]
    result = path_create(day_save_path)
    if result:
        for column in ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']:
            exec('{}_df = pd.DataFrame()'.format(column))
            exec('{}_df = pd.DataFrame([[None] * len(stock_name)] * 240)'.format(column))
            exec('{}_df.columns = stock_name'.format(column))
            exec('{}_df.index = index'.format(column))

        for stock in stock_list:
            # print(stock)
            stock_daily_path = os.path.join(day_path, stock)
            stock_daily_data = pd.read_csv(stock_daily_path, index_col=1)
            for column in ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']:
                exec('{}_df[stock[:-4]] = stock_daily_data[column]'.format(column))
                # exec('print(stock_daily_data[\'{}\'])'.format(column))

        for column in ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']:
            exec('{}_df.to_csv(\'{}\')'.format(column, os.path.join(day_save_path, column + '.csv')))
            exec('del {}_df'.format(column))
            gc.collect()
        for x in locals().keys():
            del locals()[x]
        gc.collect()
        print('day:{} finished'.format(date))
    else:
        for x in locals().keys():
            del locals()[x]
        gc.collect()
        print('day:{} finished'.format(date))
        pass


def deal_year_data(index, year, year_path, save_path):
    month_list = os.listdir(year_path)
    for month in sorted(month_list):
        month_path = os.path.join(year_path, month)
        day_list = os.listdir(month_path)
        for day in sorted(day_list):
            day_path = os.path.join(month_path, day)
            day_save_path = os.path.join(save_path, year, month, day)
            deal_day_data(index, day_path, day_save_path)


if __name__ == '__main__':
    index = pd.read_csv(r'/home/haishuowang/Downloads/stock_minute_data_file/1999/199907/19990726/SH000001.csv',
                        index_col=1).index
    load_path = r'/home/haishuowang/Downloads/stock_minute_data_file'
    save_path = r'/home/haishuowang/Downloads/stock_minute_data'
    year_list = os.listdir(load_path)

    pool = Pool(4)

    for year in sorted(year_list):
        print(year)
        year_path = os.path.join(load_path, year)
        # deal_year_data(year, year_path, save_path)
        pool.apply_async(deal_year_data, args=(index, year, year_path, save_path))
    pool.close()
    pool.join()

    # day_path = r'/home/haishuowang/Downloads/stock_minute_data_file/2003/200301/20030109'
    # day_save_path = r'/home/haishuowang/Downloads/stock_minute_data/2003/200301/20030109'
    # deal_day_data(index, day_path, day_save_path)
