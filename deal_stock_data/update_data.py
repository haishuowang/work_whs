# -*- coding:utf-8 -*-

__author__ = 'weijie'

from EmQuantAPI import *
from datetime import timedelta, datetime
import time
import copy
import numpy as np
import pandas as pd


# class DataUpdate(object):
#
#     def __init__(self, root_save_path='/mnt/mfs/dat_whs/intra_data', time_period=3, fresh_num=3):
#         self.root_save_path = root_save_path
#         # 下载数据周期
#         self.time_period = time_period
#         # 更新数据点个数
#         self.fresh_num = fresh_num
#         # 下载数据
#         self.download_list = ['Close', 'High', 'Low', 'Volume', 'Amount', 'Vwap']
#
#         self.Close_save_path = os.path.join(root_save_path, 'intra_Close.csv')
#         self.High_save_path = os.path.join(root_save_path, 'intra_High.csv')
#         self.Low_save_path = os.path.join(root_save_path, 'intra_Low.csv')
#         self.Volume_save_path = os.path.join(root_save_path, 'intra_Volume.csv')
#         self.Amount_save_path = os.path.join(root_save_path, 'intra_Amount.csv')
#         self.Vwap_save_path = os.path.join(root_save_path, 'intra_Vwap.csv')
#
#         # 获取当天可更新股票
#         # stock_codes = c.sector("001004", now_date.strftime('%Y%m%d')).Codes[:10]
#         self.today_date = datetime.now()
#         # 交易日期
#         self.am_open_time = datetime(*self.today_date.timetuple()[:3], 9, 30)
#         self.am_close_time = datetime(*self.today_date.timetuple()[:3], 11, 30)
#
#         self.pm_open_time = datetime(*self.today_date.timetuple()[:3], 13, 00)
#         self.pm_close_time = datetime(*self.today_date.timetuple()[:3], 15, 00)
#
#     def e_cstCallBack(quantdata):
#         stock_name = list(quantdata.Data.keys())[0]
#         save_data.add(stock_name)
#         print(stock_name)
#         data_array = np.array(quantdata.Data[stock_name])
#         tmp_df = pd.DataFrame(data_array.reshape([6, int(len(data_array) / 6)]).T,
#                               columns=['Time', 'Now', 'High', 'Low', 'Volume', 'Amount'])
#
#         Close_df.loc[end_time, stock_name] = tmp_df['Now'].iloc[-1]
#         High_df.loc[end_time, stock_name] = tmp_df['High'].iloc[-1]
#         Low_df.loc[end_time, stock_name] = tmp_df['Low'].iloc[-1]
#         Volume_df.loc[end_time, stock_name] = tmp_df['Volume'].iloc[-1]
#         Amount_df.loc[end_time, stock_name] = tmp_df['Amount'].iloc[-1]
#
#     def b_cstCallBack(quantdata):
#         stock_name = list(quantdata.Data.keys())[0]
#         save_data.add(stock_name)
#         print(stock_name)
#         data_array = np.array(quantdata.Data[stock_name])
#         tmp_df = pd.DataFrame(data_array.reshape([6, int(len(data_array) / 6)]).T,
#                               columns=['Time', 'Now', 'High', 'Low', 'Volume', 'Amount'])
#
#         Close_df.loc[begin_time, stock_name] = tmp_df['Now'].iloc[-1]
#         High_df.loc[begin_time, stock_name] = tmp_df['High'].iloc[-1]
#         Low_df.loc[begin_time, stock_name] = tmp_df['Low'].iloc[-1]
#         Volume_df.loc[end_time, stock_name] = tmp_df['Volume'].iloc[-1]
#         Amount_df.loc[end_time, stock_name] = tmp_df['Amount'].iloc[-1]
#
#     def find_last_time(self, now_date):
#         mul_factor = int(now_date.minute / self.time_period)
#         if now_date <= (self.am_open_time + timedelta(minutes=self.time_period)):
#             end_date = self.am_open_time + timedelta(minutes=self.time_period)
#             next_date = self.am_open_time + timedelta(minutes=self.time_period)
#
#         elif (self.am_close_time >= now_date) and (now_date > self.am_open_time + timedelta(minutes=self.time_period)):
#             end_date = datetime(*self.today_date.timetuple()[:3], now_date.hour) + \
#                        timedelta(minutes=mul_factor * self.time_period)
#             next_date = datetime(*self.today_date.timetuple()[:3], now_date.hour) + \
#                         timedelta(minutes=(mul_factor + 1) * time_period)
#
#         elif (pm_open_time + timedelta(minutes=time_period)) >= now_date > am_close_time:
#             end_date = am_close_time
#             next_date = pm_open_time + timedelta(minutes=time_period)
#
#         elif (pm_close_time >= now_date) and (now_date > (pm_open_time + timedelta(minutes=time_period))):
#             end_date = datetime(today_year, today_month, today_day, now_date.hour) + \
#                        timedelta(minutes=mul_factor * self.time_period)
#             next_date = datetime(today_year, today_month, today_day, now_date.hour) + \
#                         timedelta(minutes=(mul_factor + 1) * self.time_period)
#
#         else:
#             end_date = pm_close_time
#             next_date = pm_close_time
#
#         return end_date, next_date
#
#     def get_history_update_time(self):
#         # 读取更新数据
#         data = pd.read_csv(self.Close_save_path, index_col=0)
#
#         data_index = pd.to_datetime(data.index)
#         end_date_list = data_index[data_index > am_open_time]
#         # 获取今天更新过的所有数据点
#         end_time_list = [x.strftime('%H%M%S') for x in end_date_list]
#         begin_time_list = [(x - timedelta(minutes=time_period)).strftime('%H%M%S') for x in end_date_list]
#         return begin_time_list, end_time_list
#
#     @staticmethod
#     def sleep_fun(use_date, use_time):
#         delta_time = (use_date - datetime.now()).seconds + 10
#         print('now_time:{}, waiting {} seconds update {}'
#               .format(datetime.now().strftime('%H%M%S'), delta_time, use_time))
#         time.sleep(delta_time)
#
#     def update_intra_data(self):
#         end_date = self.am_open_time
#         # now_date = am_open_time + timedelta(seconds=610)
#         while end_date != self.pm_close_time:
#             now_date = datetime.now()
#
#             end_date, next_date = find_last_time(now_date, time_period, am_open_time, am_close_time, pm_open_time,
#                                                  pm_close_time)
#             begin_time_list, end_time_list = self.get_history_update_time()
#             print(now_date, end_date, next_date)
#             end_time = end_date.strftime('%H%M%S')
#             next_time = next_date.strftime('%H%M%S')
#             if end_time not in end_time_list and now_date > end_date:
#                 begin_date = end_date - timedelta(minutes=time_period)
#                 begin_time = begin_date.strftime('%H%M%S')
#                 begin_time_list += [begin_time]
#                 end_time_list += [end_time]
#                 for i in range(min(len(begin_time_list), fresh_num)):
#                     begin_time = begin_time_list[-i - 1]
#                     end_time = end_time_list[-i - 1]
#                     update_intraday_data(begin_time, end_time, download_list)
#                     print('success update {}'.format(end_time))
#
#                 sleep_fun(next_date, next_time)
#             else:
#                 sleep_fun(end_date, end_time)
#
#         print('today is over!')


def e_cstCallBack(quantdata):
    stock_name = list(quantdata.Data.keys())[0]
    save_data.add(stock_name)
    print(stock_name)
    data_array = np.array(quantdata.Data[stock_name])
    tmp_df = pd.DataFrame(data_array.reshape([6, int(len(data_array) / 6)]).T,
                          columns=['Time', 'Now', 'High', 'Low', 'Volume', 'Amount'])

    Close_df.loc[end_time, stock_name] = tmp_df['Now'].iloc[-1]
    High_df.loc[end_time, stock_name] = tmp_df['High'].iloc[-1]
    Low_df.loc[end_time, stock_name] = tmp_df['Low'].iloc[-1]
    Volume_df.loc[end_time, stock_name] = tmp_df['Volume'].iloc[-1]
    Amount_df.loc[end_time, stock_name] = tmp_df['Amount'].iloc[-1]


def b_cstCallBack(quantdata):
    stock_name = list(quantdata.Data.keys())[0]
    save_data.add(stock_name)
    print(stock_name)
    data_array = np.array(quantdata.Data[stock_name])
    tmp_df = pd.DataFrame(data_array.reshape([6, int(len(data_array) / 6)]).T,
                          columns=['Time', 'Now', 'High', 'Low', 'Volume', 'Amount'])

    Close_df.loc[begin_time, stock_name] = tmp_df['Now'].iloc[-1]
    High_df.loc[begin_time, stock_name] = tmp_df['High'].iloc[-1]
    Low_df.loc[begin_time, stock_name] = tmp_df['Low'].iloc[-1]
    Volume_df.loc[end_time, stock_name] = tmp_df['Volume'].iloc[-1]
    Amount_df.loc[end_time, stock_name] = tmp_df['Amount'].iloc[-1]


def load_intra_data(download_list):
    for x in download_list:
        path_condition = os.path.exists(os.path.join(root_save_path, 'intra_{}.csv'.format(x)))
        if path_condition:
            exec('all_{0}_df = pd.read_csv({0}_save_path, index_col=0)'.format(x))
            exec('all_{0}_df.index = pd.to_datetime(all_{0}_df.index)'.format(x))
        else:
            exec('all_{0}_df = pd.DataFrame()'.format(x))
    result = []
    for x in download_list:
        exec('result += [all_{0}_df]'.format(x))
    return result


def get_data(end_time, stock_codes, download_list, cstCallBack_fun):
    global Close_df, High_df, Low_df, Volume_df, Amount_df, Vwap_df, save_data
    save_data = set()

    Close_df = pd.DataFrame(columns=stock_codes)
    High_df = pd.DataFrame(columns=stock_codes)
    Low_df = pd.DataFrame(columns=stock_codes)
    Volume_df = pd.DataFrame(columns=stock_codes)
    Amount_df = pd.DataFrame(columns=stock_codes)

    print(','.join(download_list))
    # 数据下载
    data = c.cst(','.join(stock_codes), 'Time, Now, High, Low, Volume, Amount', end_time, end_time, "", cstCallBack_fun)
    # 等待时间
    time.sleep(5)

    a = time.time()
    before_set = set()
    # 等待数据全部下载完毕
    while True:
        time.sleep(2)
        if set(stock_codes) - save_data == before_set:
            break
        before_set = set(stock_codes) - save_data
        b = time.time()
        if b - a > 60 * 2:
            print('load time error')
            break
    print(High_df)
    return Close_df, High_df, Low_df, Volume_df, Amount_df


def update_intraday_data(begin_time, end_time, download_list):
    loginResult = c.start("ForceLogin=1")
    if loginResult.ErrorCode != 0:
        print("login in fail")
        exit()
    # 获取当天可更新股票
    stock_codes = c.sector("001004", now_date.strftime('%Y%m%d')).Codes
    b_Close_df, b_High_df, b_Low_df, b_Volume_df, b_Amount_df = \
        get_data(begin_time, stock_codes, download_list, b_cstCallBack)
    e_Close_df, e_High_df, e_Low_df, e_Volume_df, e_Amount_df = \
        get_data(end_time, stock_codes, download_list, e_cstCallBack)
    c.stop()
    Close_df = e_Close_df
    High_df = e_High_df
    Low_df = e_Low_df
    Volume_df = e_Volume_df - b_Volume_df
    Amount_df = e_Amount_df - b_Amount_df
    Vwap_df = (Amount_df / Volume_df).round(4)

    all_Close_df, all_High_df, all_Low_df, all_Volume_df, all_Amount_df, all_Vwap_df = load_intra_data(download_list)
    index_time = datetime(today_year, today_month, today_day, int(end_time[:2]), int(end_time[2:4]))
    Close_df.index = [index_time]
    High_df.index = [index_time]
    Low_df.index = [index_time]
    Volume_df.index = [index_time]
    Amount_df.index = [index_time]
    Vwap_df.index = [index_time]

    if index_time in all_Close_df.index:
        all_Close_df.loc[index_time] = Close_df.loc[index_time]
        all_Close_df.to_csv(Close_save_path)

        all_High_df.loc[index_time] = High_df.loc[index_time]
        all_High_df.to_csv(High_save_path)

        all_Low_df.loc[index_time] = Low_df.loc[index_time]
        all_Low_df.to_csv(Low_save_path)

        all_Volume_df.loc[index_time] = Volume_df.loc[index_time]
        all_Volume_df.to_csv(Volume_save_path)

        all_Amount_df.loc[index_time] = Amount_df.loc[index_time]
        all_Amount_df.to_csv(Amount_save_path)

        all_Vwap_df.loc[index_time] = Vwap_df.loc[index_time]
        all_Vwap_df.to_csv(Vwap_save_path)
    else:
        all_Close_df = all_Close_df.append(Close_df)
        all_Close_df.to_csv(Close_save_path)

        all_High_df = all_High_df.append(High_df)
        all_High_df.to_csv(High_save_path)

        all_Low_df = all_Low_df.append(Low_df)
        all_Low_df.to_csv(Low_save_path)

        all_Volume_df = all_Volume_df.append(Volume_df)
        all_Volume_df.to_csv(Volume_save_path)

        all_Amount_df = all_Amount_df.append(Amount_df)
        all_Amount_df.to_csv(Amount_save_path)

        all_Vwap_df = all_Vwap_df.append(Vwap_df)
        all_Vwap_df.to_csv(Vwap_save_path)


def find_last_time(now_date, time_period, am_open_time, am_close_time, pm_open_time, pm_close_time):
    """
    给定当前日期,返回上个执行日期和下个执行日期
    :param now_date: 当前日期
    :param time_period: 时间周期
    :param am_open_time: 上午开盘时间
    :param am_close_time: 上午收盘时间
    :param pm_open_time: 下午开盘时间
    :param pm_close_time: 下午收盘时间
    :return: 上个执行日期和下个执行日期
    """
    if now_date <= (am_open_time + timedelta(minutes=time_period)):
        end_date = am_open_time + timedelta(minutes=time_period)
        next_date = am_open_time + timedelta(minutes=time_period)

    elif (am_close_time >= now_date) and (now_date > am_open_time + timedelta(minutes=time_period)):
        end_date = datetime(today_year, today_month, today_day, now_date.hour) + \
                   timedelta(minutes=int(now_date.minute / time_period) * time_period)
        next_date = datetime(today_year, today_month, today_day, now_date.hour) + \
                    timedelta(minutes=(int(now_date.minute / time_period) + 1) * time_period)

    elif (pm_open_time + timedelta(minutes=time_period)) >= now_date > am_close_time:
        end_date = am_close_time
        next_date = pm_open_time + timedelta(minutes=time_period)

    elif (pm_close_time >= now_date) and (now_date > (pm_open_time + timedelta(minutes=time_period))):
        end_date = datetime(today_year, today_month, today_day, now_date.hour) + \
                   timedelta(minutes=int(now_date.minute / time_period) * time_period)
        next_date = datetime(today_year, today_month, today_day, now_date.hour) + \
                    timedelta(minutes=(int(now_date.minute / time_period) + 1) * time_period)

    else:
        end_date = pm_close_time
        next_date = pm_close_time

    return end_date, next_date


def sleep_fun(use_date, use_time):
    delta_time = (use_date - datetime.now()).seconds + 10
    print('now_time:{}, waiting {} seconds update {}'
          .format(datetime.now().strftime('%H%M%S'), delta_time, use_time))
    time.sleep(delta_time)


if __name__ == '__main__':

    root_save_path = '/mnt/mfs/dat_whs/intra_data'
    # 下载数据周期
    time_period = 10
    # 更新数据点个数
    fresh_num = 3
    download_list = ['Close', 'High', 'Low', 'Volume', 'Amount', 'Vwap']

    Close_save_path = os.path.join(root_save_path, 'intra_Close.csv')
    High_save_path = os.path.join(root_save_path, 'intra_High.csv')
    Low_save_path = os.path.join(root_save_path, 'intra_Low.csv')
    Volume_save_path = os.path.join(root_save_path, 'intra_Volume.csv')
    Amount_save_path = os.path.join(root_save_path, 'intra_Amount.csv')
    Vwap_save_path = os.path.join(root_save_path, 'intra_Vwap.csv')

    now_date = datetime.now()
    # 获取当天可更新股票
    # stock_codes = c.sector("001004", now_date.strftime('%Y%m%d')).Codes[:10]

    today_year = now_date.year
    today_month = now_date.month
    today_day = now_date.day
    # 交易日期
    am_open_time = datetime(today_year, today_month, today_day, 9, 30)
    am_close_time = datetime(today_year, today_month, today_day, 11, 30)

    pm_open_time = datetime(today_year, today_month, today_day, 13, 00)
    pm_close_time = datetime(today_year, today_month, today_day, 15, 00)

    # begin_time_list = []
    # end_time_list = []

    # 初始时间为开盘时间
    end_date = am_open_time
    # now_date = am_open_time + timedelta(seconds=610)
    while end_date != pm_close_time:
        now_date = datetime.now()
        data = pd.read_csv('/mnt/mfs/dat_whs/intra_data/intra_Close.csv', index_col=0)

        # 读取更新数据
        data_index = pd.to_datetime(data.index)
        end_date_list = data_index[data_index > am_open_time]
        # 获取今天更新过的所有数据点
        end_time_list = [x.strftime('%H%M%S') for x in end_date_list]
        begin_time_list = [(x - timedelta(minutes=time_period)).strftime('%H%M%S') for x in end_date_list]

        end_date, next_date = find_last_time(now_date, time_period, am_open_time, am_close_time, pm_open_time,
                                             pm_close_time)

        print(now_date, end_date, next_date)
        end_time = end_date.strftime('%H%M%S')
        next_time = next_date.strftime('%H%M%S')
        if end_time not in end_time_list and now_date > end_date:
            begin_date = end_date - timedelta(minutes=time_period)
            begin_time = begin_date.strftime('%H%M%S')
            begin_time_list += [begin_time]
            end_time_list += [end_time]
            for i in range(min(len(begin_time_list), fresh_num)):
                begin_time = begin_time_list[-i - 1]
                end_time = end_time_list[-i - 1]

                update_intraday_data(begin_time, end_time, download_list)

                print('success update {}'.format(end_time))

            sleep_fun(next_date, next_time)
        else:
            sleep_fun(end_date, end_time)

    print('today is over!')
