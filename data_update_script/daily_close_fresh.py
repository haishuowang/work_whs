import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *
import shutil


def AZ_clear_columns(stock_list):
    return [x[2:] + '.' + x[:2] for x in stock_list]


def AZ_split_stock(stock_list):
    """
    在stock_list中寻找A股代码
    :param stock_list:
    :return:
    """
    eqa = [x for x in stock_list if (x.startswith('0') or x.startswith('3')) and x.endswith('SZ')
           or x.startswith('6') and x.endswith('SH')]
    eqa = sorted(eqa)
    return eqa


def AZ_split_stock_raw(stock_list):
    eqa = [x for x in stock_list if (x.startswith('0') or x.startswith('3')) and x.endswith('SZ')
           or x.startswith('6') and x.endswith('SH')]
    eqa = sorted(eqa)
    return eqa


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
    index_list = pd.to_datetime([str(x)[:12] for x in target_df.index]) + timedelta(minutes=1)
    target_df.index = [x.strftime('%H:%M') for x in index_list]
    return target_df


def get_data(begin_time, end_time):
    root_load_path = r'/mnt/mfs/DAT_EQT/intraday/eqt_1mbar'
    save_date = datetime.now().strftime('%Y%m%d')
    Close_1min_data = pd.read_table(os.path.join(root_load_path, '{}_Close_1min.txt'.format(save_date)),
                                    header=None, sep='|')
    High_1min_data = pd.read_table(os.path.join(root_load_path, '{}_High_1min.txt'.format(save_date)),
                                   header=None, sep='|')
    Low_1min_data = pd.read_table(os.path.join(root_load_path, '{}_Low_1min.txt'.format(save_date)),
                                  header=None, sep='|')
    Open_1min_data = pd.read_table(os.path.join(root_load_path, '{}_Open_1min.txt'.format(save_date)),
                                   header=None, sep='|')
    value_1min_data = pd.read_table(os.path.join(root_load_path, '{}_TotalValueTrade_1min.txt'.format(save_date)),
                                    header=None, sep='|')
    volume_1min_data = pd.read_table(os.path.join(root_load_path, '{}_TotalVolumeTrade_1min.txt'.format(save_date)),
                                     header=None, sep='|')

    Close_1min_cut_data = AZ_window_cut_1min(begin_time, end_time, Close_1min_data)
    High_1min_cut_data = AZ_window_cut_1min(begin_time, end_time, High_1min_data)
    Low_1min_cut_data = AZ_window_cut_1min(begin_time, end_time, Low_1min_data)
    Open_1min_cut_data = AZ_window_cut_1min(begin_time, end_time, Open_1min_data)
    value_1min_cut_data = AZ_window_cut_1min(begin_time, end_time, value_1min_data)
    volume_1min_cut_data = AZ_window_cut_1min(begin_time, end_time, volume_1min_data)

    Close_1min_df = AZ_unstack_1min(0, 1, 2, Close_1min_cut_data)
    High_1min_df = AZ_unstack_1min(0, 1, 2, High_1min_cut_data)
    Low_1min_df = AZ_unstack_1min(0, 1, 2, Low_1min_cut_data)
    Open_1min_df = AZ_unstack_1min(0, 1, 2, Open_1min_cut_data)
    value_1min_df = AZ_unstack_1min(0, 1, 2, value_1min_cut_data)
    volume_1min_df = AZ_unstack_1min(0, 1, 2, volume_1min_cut_data)
    return Close_1min_df, High_1min_df, Low_1min_df, Open_1min_df, value_1min_df, volume_1min_df


def get_data_fun(value, begin_time, end_time):
    root_load_path = r'/mnt/mfs/DAT_PUBLIC/data_raw'
    save_date = datetime.now().strftime('%Y%m%d')
    target_1min_data = pd.read_table(os.path.join(root_load_path, f'{save_date}_{value}_1min.txt'),
                                     header=None, sep='|')
    target_1min_cut_data = AZ_window_cut_1min(begin_time, end_time, target_1min_data)
    target_1min_df = AZ_unstack_1min(0, 1, 2, target_1min_cut_data)
    return target_1min_df.fillna(method='ffill')


def min_update_save_fun(target_1min_df, target_path):
    def columns_deal(columns):
        return [x[-2:] + x[:6] for x in columns.astype(str)]

    target_1min_df.columns = columns_deal(target_1min_df.columns)
    if os.path.exists(target_path):
        target_clear_data = pd.read_csv(target_path, index_col=0)
        target_clear_data = target_clear_data.combine_first(target_1min_df)
    else:
        target_clear_data = target_1min_df
    target_clear_data.to_csv(target_path, index_label='Time')
    print(datetime.now())
    return target_clear_data


def min_fresh_save_fun(target_1min_df, target_path):
    data = bt.AZ_Load_csv(target_path)
    target_1min_df_last = target_1min_df.iloc[[-1]]
    target_1min_df_last.index = [pd.to_datetime(datetime.now().strftime('%Y%m%d'))]
    data_new = data.combine_first(pd.DataFrame(target_1min_df_last))
    data_new.to_csv(target_path, sep='|')


def update_intraday_data(tmp_begin_date, tmp_end_date, time_period):
    tmp_end_time = tmp_end_date.strftime('%Y%m%d%H%M')
    tmp_begin_time = tmp_begin_date.strftime('%Y%m%d%H%M')
    begin_time_int = int(tmp_begin_time) * 100000
    end_time_int = int(tmp_end_time) * 100000
    today_date = tmp_begin_time[:8]
    root_save_path = f'/mnt/mfs/DAT_EQT/intraday/daily_close'
    bt.AZ_Path_create(root_save_path)
    file_dict = dict({'Close_intra': 'Close',
                      })
    for target_file in file_dict.keys():
        target_1min_df = get_data_fun(file_dict[target_file], begin_time_int, end_time_int)
        target_path = os.path.join(root_save_path, f'{target_file}.csv')
        if len(target_1min_df) != time_period:
            send_email.send_email('data loss', ['whs@yingpei.com'], [], f'[Data Loss]')
        min_fresh_save_fun(target_1min_df, target_path)


def sleep_fun(use_date):
    if use_date > datetime.now():
        delta_time = (use_date - datetime.now()).seconds + 10
        print('now_time:{}, waiting {} seconds update {}'
              .format(datetime.now().strftime('%H%M%S'), delta_time, use_date.strftime('%H%M%S')))
        time.sleep(delta_time)
    else:
        pass


if __name__ == '__main__':
    # 下载数据周期
    time_period = 5
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

    shutil.copy('/media/hdd1/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/NEW.csv',
                '/mnt/mfs/DAT_EQT/intraday/daily_close/Close_intra.csv')
    while end_date != pm_close_time:
        now_date = datetime.now()
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
            sleep_fun(next_date)
        else:
            sleep_fun(end_date)
            pass
    print('TODAY END!!!')
