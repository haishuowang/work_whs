import pandas as pd
import os
from multiprocessing import Pool
import gc


def path_create(target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)


def resample_bar(year):
    year_path = os.path.join(root_path, year)
    month_list = os.listdir(year_path)
    for month in sorted(month_list):
        print(month)
        month_path = os.path.join(year_path, month)
        day_list = os.listdir(month_path)
        for day in sorted(day_list):
            print(day)
            day_path = os.path.join(month_path, day)
            drop_index = pd.to_datetime([day + ' ' + '09:30', day + ' ' + '13:00'])
            Close_path = os.path.join(day_path, 'Close.csv')
            Volume_path = os.path.join(day_path, 'Volume.csv')
            Close = pd.read_csv(Close_path, index_col=0)
            Close.index = pd.to_datetime([day + ' ' + x for x in Close.index])
            Close_5min_bar = Close.resample('5T').first().dropna(how='all')
            Close_5min_bar = Close_5min_bar.drop(index=drop_index)
            Volume_5min_bar = pd.DataFrame(index=Close_5min_bar.index, columns=Close_5min_bar.columns)
            Volume = pd.read_csv(Volume_path, index_col=0)
            Volume.index = pd.to_datetime([day + ' ' + x for x in Volume.index])
            for i in range(len(Close_5min_bar.index)):
                i_time = Close_5min_bar.index[i]

                if i == 0:
                    Volume_5min_bar.loc[i_time] = Volume.loc[:i_time].sum()
                else:
                    Volume_5min_bar.loc[i_time] = Volume.loc[Close_5min_bar.index[i-1]:i_time].sum()
            day_save_path = os.path.join(save_path, day[:4], day[:6],day)
            path_create(day_save_path)
            Close_5min_bar.to_csv(os.path.join(day_save_path, 'Close.csv'))
            Volume_5min_bar.to_csv(os.path.join(day_save_path, 'Volume.csv'))
            del Close, Volume, Close_5min_bar, Volume_5min_bar
        gc.collect()


global root_path, save_path
# root_path = '/home/haishuowang/Downloads/stock_minute_data'
root_path = '/media/hdd0/data/adj_data/equity/intraday/eqt_1mbar'
# save_path = '/home/haishuowang/Downloads/stock_5minute_data'
save_path = '/media/hdd0/data/adj_data/equity/intraday/eqt_5mbar'

year_list = os.listdir(root_path)
pool = Pool(12)
year_list = [x for x in year_list if len(x) == 4]
for year in sorted(year_list)[1:]:
    print(year)
    pool.apply_async(resample_bar, args=(year,))
    # resample_bar(year)
pool.close()
pool.join()




