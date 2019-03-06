import pandas as pd
import os
from multiprocessing import Pool

# root_path = r'/media/hdd0/Data/data/EQT_1MBar_Data/stock_minute_data'
root_path = '/media/hdd0/data/adj_data/equity/intraday/eqt_5mbar'

# root_path = '/media/hdd0/data/adj_data/equity/intraday/eqt_1mbar'


def sort_stock(year):
    year_path = os.path.join(root_path, year)
    month_list = os.listdir(year_path)
    for month in sorted(month_list):
        month_path = os.path.join(year_path, month)
        day_list = os.listdir(month_path)
        for day in sorted(day_list):
            print(day)
            day_path = os.path.join(month_path, day)
            ind_list = os.listdir(day_path)
            for ind in ind_list:
                ind_path = os.path.join(day_path, ind)
                ind_data = pd.read_csv(ind_path, index_col=0)
                ind_data = ind_data[sorted(ind_data.columns)]
                ind_data.index.name = 'Time'
                ind_data.to_csv(ind_path)


def find_error(year):
    year_path = os.path.join(root_path, year)
    month_list = os.listdir(year_path)
    for month in sorted(month_list):
        month_path = os.path.join(year_path, month)
        day_list = os.listdir(month_path)
        for day in sorted(day_list):
            # print(day)
            day_path = os.path.join(month_path, day)
            ind_list = os.listdir(day_path)
            for ind in ind_list:
                ind_path = os.path.join(day_path, ind)
                ind_data = pd.read_csv(ind_path)
                if len(ind_data) != 48:
                    print(day, ind)


year_list = os.listdir(root_path)
pool = Pool(6)
for year in sorted(year_list):
    pool.apply_async(find_error, args=(year,))
    # find_error(year)
pool.close()
pool.join()
