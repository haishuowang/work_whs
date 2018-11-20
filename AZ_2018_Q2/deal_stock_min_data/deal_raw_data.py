import pandas as pd
import numpy as np
import os
import sys

sys.path.append('/mnt/mfs/work_whs')
sys.path.append('/mnt/mfs/work_whs/AZ_2018_Q2')
import loc_lib.shared_tools.back_test as bt
import collections


def data_load(file_name, target_dict):
    data = pd.read_csv(os.path.join('/mnt/mfs/DAT_PUBLIC/Stk_1F_2018_0928/Stk_1F_2018', file_name), header=None,
                       index_col=1)
    data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
    data.index.name = 'Time'
    date_list = np.array(sorted(set(data['Date'].values)))
    date_list_part = date_list[date_list > '2018/08/15']
    for date in date_list_part:
        part_data = data[data['Date'] == date]
        date_key = pd.to_datetime(date).strftime('%Y%m%d')
        part_Close = part_data['Close']
        part_Close.name = file_name[:-4]
        part_High = part_data['High']
        part_High.name = file_name[:-4]
        part_Low = part_data['Low']
        part_Low.name = file_name[:-4]
        part_Open = part_data['Open']
        part_Open.name = file_name[:-4]
        part_Turnover = part_data['Turnover']
        part_Turnover.name = file_name[:-4]
        part_Volume = part_data['Close']
        part_Volume.name = file_name[:-4]
        if date_key not in target_dict.keys():
            part_dict = collections.OrderedDict()
            part_dict['Close'] = part_Close
            part_dict['High'] = part_High
            part_dict['Low'] = part_Low
            part_dict['Open'] = part_Open
            part_dict['Turnover'] = part_Turnover
            part_dict['Volume'] = part_Volume
            target_dict[date_key] = part_dict
        else:
            part_dict = target_dict[date_key]
            part_dict['Close'] = pd.concat([part_dict['Close'], part_Close], axis=1)
            part_dict['High'] = pd.concat([part_dict['High'], part_High], axis=1)
            part_dict['Low'] = pd.concat([part_dict['Low'], part_Low], axis=1)
            part_dict['Open'] = pd.concat([part_dict['Open'], part_Open], axis=1)
            part_dict['Turnover'] = pd.concat([part_dict['Turnover'], part_Turnover], axis=1)
            part_dict['Volume'] = pd.concat([part_dict['Volume'], part_Volume], axis=1)
            target_dict[date_key] = part_dict

        # Close = pd.concat([Close, part_Close], axis=1)
        # High = pd.concat([High, part_High], axis=1)
        # Low = pd.concat([Low, part_Low], axis=1)
        # Open = pd.concat([Open, part_Open], axis=1)
        # Turnover = pd.concat([Turnover, part_Turnover], axis=1)
        # Volume = pd.concat([Volume, part_Volume], axis=1)
    return target_dict


def save_date(date_key, target_dict, root_save_path):
    part_save_path = os.path.join(root_save_path, date_key[:4], date_key[:6], date_key)
    bt.AZ_Path_create(part_save_path)
    target_dict[date_key]['Close'].to_csv(os.path.join(part_save_path, 'Close.csv'))
    target_dict[date_key]['High'].to_csv(os.path.join(part_save_path, 'High.csv'))
    target_dict[date_key]['Low'].to_csv(os.path.join(part_save_path, 'Low.csv'))
    target_dict[date_key]['Open'].to_csv(os.path.join(part_save_path, 'Open.csv'))
    target_dict[date_key]['Turnover'].to_csv(os.path.join(part_save_path, 'Turnover.csv'))
    target_dict[date_key]['Volume'].to_csv(os.path.join(part_save_path, 'Volume.csv'))


def create_raw_data():
    file_name_list = sorted(os.listdir('/mnt/mfs/DAT_PUBLIC/Stk_1F_2018_0928/Stk_1F_2018'))
    root_save_path = '/mnt/mfs/DAT_PUBLIC/intraday_test/eqt_1mbar'

    # Close = pd.DataFrame()
    # High = pd.DataFrame()
    # Low = pd.DataFrame()
    # Open = pd.DataFrame()
    # Turnover = pd.DataFrame()
    # Volume = pd.DataFrame()

    target_dict = collections.OrderedDict()
    for file_name in file_name_list:
        print(f'Load {file_name}!')
        target_dict = data_load(file_name, target_dict)

    date_list = np.array(sorted(list(target_dict.keys())))
    date_list_adj = date_list[date_list > '20180815']
    for date_key in date_list_adj:
        print(f'Deal {date_key}!')
        save_date(date_key, target_dict, root_save_path)


def clear_raw_data():
    root_save_path = '/mnt/mfs/DAT_PUBLIC/intraday_test/eqt_1mbar'
    year_list = ['2018']
    for year in sorted(year_list):
        year_path = os.path.join(root_save_path, year)
        month_list = os.listdir(year_path)
        for month in sorted(month_list):
            month_path = os.path.join(year_path, month)
            day_list = os.listdir(month_path)
            for day in sorted(day_list):
                print(day)
                # part_save_path = os.path.join(root_save_path, day[:4], day[:6], day)
                day_path = os.path.join(month_path, day)

                close = pd.read_csv(os.path.join(day_path, 'Close.csv'), index_col=0).astype(float)
                EQT_list = sorted([x for x in close.columns if (x[2] == '0' or x[2] == '3') and x.startswith('SZ')
                                   or (x[2] == '6' and x.startswith('SH'))])
                if len(EQT_list) > 100:
                    close = close[EQT_list]
                    high = pd.read_csv(os.path.join(day_path, 'High.csv'), index_col=0).astype(float)[EQT_list]
                    low = pd.read_csv(os.path.join(day_path, 'Low.csv'), index_col=0).astype(float)[EQT_list]
                    open = pd.read_csv(os.path.join(day_path, 'Open.csv'), index_col=0).astype(float)[EQT_list]
                    turnover = pd.read_csv(os.path.join(day_path, 'Turnover.csv'), index_col=0).astype(float)[EQT_list]
                    volume = pd.read_csv(os.path.join(day_path, 'Volume.csv'), index_col=0).astype(float)[EQT_list]

                    xnms = [x[2:] + '.' + x[:2] for x in EQT_list]
                    print(xnms[0])
                    high.columns = xnms
                    open.columns = xnms
                    low.columns = xnms
                    close.columns = xnms
                    turnover.columns = xnms
                    volume.columns = xnms
                    print(close)
                    high.to_csv(os.path.join(day_path, 'High.csv'))
                    open.to_csv(os.path.join(day_path, 'Open.csv'))
                    low.to_csv(os.path.join(day_path, 'Low.csv'))
                    close.to_csv(os.path.join(day_path, 'Close.csv'))
                    turnover.to_csv(os.path.join(day_path, 'Turnover.csv'))
                    volume.to_csv(os.path.join(day_path, 'Volume.csv'))


if __name__ == '__main__':
    # create_raw_data()
    # clear_raw_data()
    pass
