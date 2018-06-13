# 生成历史VWAP
import pandas as pd
import os
import gc
from multiprocessing import Pool


def vwap_close_15_minute(day_path):
    index_list = ['09:30', '09:45', '10:00', '10:15', '10:30', '10:45', '11:00', '11:15', '11:30',
                  '13:15', '13:30', '13:45', '14:00', '14:15', '14:30', '14:45', '15:00']
    save_path = os.path.join(day_path, 'vwap.csv')
    close_data = pd.read_csv(os.path.join(day_path, 'Close.csv'), index_col=0)
    volume_data = pd.read_csv(os.path.join(day_path, 'Volume.csv'), index_col=0)

    daily_df = pd.DataFrame([[None] * len(close_data.columns)] * 16)
    daily_df.columns = close_data.columns
    daily_df.index = index_list[1:]
    for i in range(16):
        if i == 0:
            use_close = close_data.iloc[
                (index_list[i + 1] >= close_data.index) & (close_data.index > index_list[i])].fillna(method='bfill')
        else:
            use_close = close_data.iloc[
                (index_list[i + 1] >= close_data.index) & (close_data.index > index_list[i])].fillna(method='ffill')
        use_volume = volume_data.iloc[
            (index_list[i + 1] >= volume_data.index) & (close_data.index > index_list[i])].fillna(0)
        tick_vwap = (use_close * use_volume).sum() / use_volume.sum()
        for stock in tick_vwap.index:
            if tick_vwap[stock] != tick_vwap[stock]:
                tick_vwap[stock] = close_data.loc[index_list[i + 1], stock]
        daily_df.loc[index_list[i + 1]] = tick_vwap.round(4)
    daily_df.to_csv(save_path)
    print('day:{} finished'.format(day_path.split('/')[-1]))
    for x in locals().keys():
        del locals()[x]
    gc.collect()


if __name__ == '__main__':
    root_path = r'/media/hdd0/Data/data/EQT_1MBar_Data/stock_minute_data'
    year_list = os.listdir(root_path)
    for year in sorted(year_list):
        year_path = os.path.join(root_path, year)
        month_list = os.listdir(year_path)
        for month in sorted(month_list):
            month_path = os.path.join(year_path, month)
            day_list = os.listdir(month_path)
            pool = Pool(4)
            for day in sorted(day_list):
                # print(day)
                day_path = os.path.join(month_path, day)
                pool.apply_async(vwap_close_15_minute, args=(day_path,))
            pool.close()
            pool.join()
