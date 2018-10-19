import pandas as pd
from datetime import timedelta, datetime
import os
import time


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


def AZ_split_stock(stock_list):
    """
    在stock_list中寻找A股代码
    :param stock_list:
    :return:
    """
    eqa = [x for x in stock_list if (x.startswith('0') or x.startswith('3')) and x.endswith('SZ')
           or x.startswith('6') and x.endswith('SH')]
    return eqa


def fun(x):
    # if len(x) > 1:
    #     print(x)
    return x.iloc[-1]


def AZ_window_cut_1min(begin_time, end_time, data):
    target_df = data[(data[0] >= begin_time) & (data[0] < end_time)]
    return target_df


def AZ_unstack_1min(index, columns, values, data):
    target_df = data.groupby([index, columns])[values].apply(fun).unstack()
    target_df = target_df[AZ_split_stock(target_df.columns)]
    return target_df


def get_data(begin_time, end_time):
    root_load_path = r'/media/hdd1/data_raw'
    save_date = datetime.now().strftime('%Y%m%d')
    close_1min_data = pd.read_table(os.path.join(root_load_path, '{}_Close_1min.txt'.format(save_date)),
                                    header=None, sep='|')
    value_1min_data = pd.read_table(os.path.join(root_load_path, '{}_TotalValueTrade_1min.txt'.format(save_date)),
                                    header=None, sep='|')
    volume_1min_data = pd.read_table(os.path.join(root_load_path, '{}_TotalVolumeTrade_1min.txt'.format(save_date)),
                                     header=None, sep='|')

    close_1min_cut_data = AZ_window_cut_1min(begin_time, end_time, close_1min_data)
    value_1min_cut_data = AZ_window_cut_1min(begin_time, end_time, value_1min_data)
    volume_1min_cut_data = AZ_window_cut_1min(begin_time, end_time, volume_1min_data)

    if len(close_1min_cut_data) != 0:
        close_1min_df = AZ_unstack_1min(0, 1, 2, close_1min_cut_data)
        value_1min_df = AZ_unstack_1min(0, 1, 2, value_1min_cut_data)
        volume_1min_df = AZ_unstack_1min(0, 1, 2, volume_1min_cut_data)

        part_value_clear_df = value_1min_df.sum()
        part_volume_clear_df = volume_1min_df.sum()
        part_vwap_clear_df = (part_value_clear_df / part_volume_clear_df).round(4)
        part_close_clear_df = close_1min_df.fillna(method='ffill').iloc[-1]
    else:
        part_value_clear_df = pd.Series(index=close_1min_cut_data.columns)
        part_volume_clear_df = pd.Series(index=close_1min_cut_data.columns)
        part_vwap_clear_df = pd.Series(index=close_1min_cut_data.columns)
        part_close_clear_df = pd.Series(index=close_1min_cut_data.columns)
    return part_close_clear_df, part_value_clear_df, part_volume_clear_df, part_vwap_clear_df


def update_intraday_data(tmp_begin_date, tmp_end_date, time_period):
    tmp_end_time = tmp_end_date.strftime('%Y%m%d%H%M')
    tmp_begin_time = tmp_begin_date.strftime('%Y%m%d%H%M')

    begin_time_int = int(tmp_begin_time) * 100000
    end_time_int = int(tmp_end_time) * 100000
    root_save_path = '/mnt/mfs/DAT_EQT/intraday/'
    part_close_clear_df, part_value_clear_df, part_volume_clear_df, part_vwap_clear_df = \
        get_data(begin_time_int, end_time_int)

    close_path = os.path.join(root_save_path, 'Close_{}min_{}.csv'.format(time_period, tmp_begin_date.year))
    value_path = os.path.join(root_save_path, 'Value_{}min_{}.csv'.format(time_period, tmp_begin_date.year))
    volume_path = os.path.join(root_save_path, 'Volume_{}min_{}.csv'.format(time_period, tmp_begin_date.year))
    vwap_path = os.path.join(root_save_path, 'Vwap_{}min_{}.csv'.format(time_period, tmp_begin_date.year))

    if os.path.exists(close_path):
        close_clear_data = pd.read_csv(close_path, sep='|', index_col=0, parse_dates=True)
        if tmp_end_date in close_clear_data.index:
            close_clear_data.loc[tmp_end_date] = part_close_clear_df[close_clear_data.columns].values
        else:
            part_close_clear_data = pd.DataFrame(part_close_clear_df).T
            part_close_clear_data.index = [tmp_end_date]

            close_clear_data = close_clear_data.append(part_close_clear_data)
    else:
        close_clear_data = pd.DataFrame(part_close_clear_df).T
        close_clear_data.index = [tmp_end_date]

    if os.path.exists(value_path):
        value_clear_data = pd.read_csv(value_path, sep='|', index_col=0, parse_dates=True)
        if tmp_end_date in close_clear_data.index:
            value_clear_data.loc[tmp_end_date] = part_value_clear_df[value_clear_data.columns].values
        else:
            part_value_clear_data = pd.DataFrame(part_value_clear_df).T
            part_value_clear_data.index = [tmp_end_date]

            value_clear_data = value_clear_data.append(part_value_clear_data)
    else:
        value_clear_data = pd.DataFrame(part_value_clear_df).T
        value_clear_data.index = [tmp_end_date]

    if os.path.exists(volume_path):
        volume_clear_data = pd.read_csv(volume_path, sep='|', index_col=0, parse_dates=True)
        if tmp_end_date in close_clear_data.index:
            volume_clear_data.loc[tmp_end_date] = part_volume_clear_df[volume_clear_data.columns].values
        else:
            part_volume_clear_data = pd.DataFrame(part_volume_clear_df).T
            part_volume_clear_data.index = [tmp_end_date]
            volume_clear_data = volume_clear_data.append(part_volume_clear_data)
    else:
        volume_clear_data = pd.DataFrame(part_volume_clear_df).T
        volume_clear_data.index = [tmp_end_date]

    if os.path.exists(vwap_path):
        vwap_clear_data = pd.read_csv(vwap_path, sep='|', index_col=0, parse_dates=True)
        if tmp_end_date in close_clear_data.index:
            vwap_clear_data.loc[tmp_end_date] = part_vwap_clear_df[vwap_clear_data.columns].values
        else:
            part_vwap_clear_data = pd.DataFrame(part_vwap_clear_df).T
            part_vwap_clear_data.index = [tmp_end_date]

            vwap_clear_data = vwap_clear_data.append(part_vwap_clear_data)
    else:
        vwap_clear_data = pd.DataFrame(part_vwap_clear_df).T
        vwap_clear_data.index = [tmp_end_date]

    close_clear_data.sort_index().to_csv(close_path, sep='|')
    value_clear_data.sort_index().to_csv(value_path, sep='|')
    volume_clear_data.sort_index().to_csv(volume_path, sep='|')
    vwap_clear_data.sort_index().to_csv(vwap_path, sep='|')


def sleep_fun(use_date, now_date):
    if use_date > now_date:
        delta_time = (use_date - now_date).seconds + 10
        now_date = now_date + timedelta(seconds=delta_time)
        print('now_time:{}, waiting {} seconds update {}'
              .format(datetime.now().strftime('%H%M%S'), delta_time, use_date.strftime('%H%M%S')))
        return now_date
    else:
        return now_date


if __name__ == '__main__':
    # 下载数据周期
    time_period = 10
    # 更新数据点个数
    fresh_num = 3

    now_date = datetime.now()
    today_year = now_date.year
    today_month = now_date.month
    today_day = now_date.day

    am_open_time = datetime(today_year, today_month, today_day, 9, 30)
    am_close_time = datetime(today_year, today_month, today_day, 11, 30)

    pm_open_time = datetime(today_year, today_month, today_day, 13, 00)
    pm_close_time = datetime(today_year, today_month, today_day, 15, 00)

    end_date = am_open_time

    now_date = datetime(today_year, today_month, today_day, 9, 31)
    while end_date != pm_close_time:
        end_date, next_date = find_last_time(now_date, time_period, am_open_time, am_close_time, pm_open_time,
                                             pm_close_time)

        print(now_date, end_date)
        if now_date > end_date:
            print('**********************************************')
            print(end_date, next_date)
            for i in range(1):
                tmp_end_date = (end_date - timedelta(minutes=time_period) * (0 - i))
                tmp_begin_date = (end_date - timedelta(minutes=time_period) * (1 - i))
                print(tmp_begin_date, tmp_end_date)
                if tmp_begin_date >= am_open_time:
                    update_intraday_data(tmp_begin_date, tmp_end_date, time_period)
                else:
                    pass
            print('success update {}'.format(end_date.strftime('%H%M%S')))
            now_date = sleep_fun(next_date, now_date)
        else:
            now_date = sleep_fun(end_date, now_date)
            pass
    print('TODAY END!!!')
