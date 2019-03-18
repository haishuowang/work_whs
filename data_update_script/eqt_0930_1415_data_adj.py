import gc
import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *


def find_target_file(begin_time, end_time, time_type, endswith=None):
    def time_judge(begin_time, end_time, target_time):
        if end_time >= datetime.fromtimestamp(target_time) >= begin_time:
            return True
        else:
            return False

    result_root_path = '/mnt/mfs/DAT_EQT/intraday/daily_0930_1415'
    raw_file_name_list = [x for x in os.listdir(result_root_path)
                          if os.path.getsize(os.path.join(result_root_path, x)) != 0]

    if endswith is not None:
        raw_file_name_list = [x for x in os.listdir(result_root_path)
                              if x[:-4].endswith(endswith)]

    if time_type == 'm':
        result_file_name_list = [x[:-4] for x in raw_file_name_list if
                                 time_judge(begin_time, end_time, os.path.getmtime(os.path.join(result_root_path, x)))]
        return sorted(result_file_name_list)
    elif time_type == 'c':
        result_file_name_list = [x[:-4] for x in raw_file_name_list if
                                 time_judge(begin_time, end_time, os.path.getctime(os.path.join(result_root_path, x)))]

        return sorted(result_file_name_list)

    elif time_type == 'a':
        result_file_name_list = [x[:-4] for x in raw_file_name_list if
                                 time_judge(begin_time, end_time, os.path.getatime(os.path.join(result_root_path, x)))]

        return sorted(result_file_name_list)

    else:
        return []


def adj_price(price_df):
    factor_1 = bt.AZ_Load_csv('/media/hdd1/DAT_EQT/EM_Funda/TRAD_SK_FACTOR1/TAFACTOR.csv')
    factor_1 = factor_1.reindex(index=price_df.index)
    price_df = price_df.reindex(columns=factor_1.columns)
    return price_df / factor_1


def save_fun(file_name):
    save_path = f'/mnt/mfs/DAT_EQT/intraday/daily_0930_1415/{file_name}.csv'
    data = bt.AZ_Load_csv(save_path)
    data.to_csv(f'/mnt/mfs/DAT_EQT/intraday/daily_0930_1415/{file_name}.csv', sep='|')
    data_adj = adj_price(data)
    data_adj.to_csv(f'/mnt/mfs/DAT_EQT/intraday/daily_0930_1415/adj_{file_name}.csv', sep='|')


def main_fun():
    end_time = datetime.now()
    begin_time = datetime(end_time.year, end_time.month, end_time.day)
    a = find_target_file(begin_time, end_time, 'm', endswith=None)
    print(a)
    file_name_list = ['turnover', 'close', 'high', 'low', 'open']
    target_name_list = ['adj_turnover', 'adj_close', 'adj_high', 'adj_low', 'adj_open']
    while True:
        if len(list(set(file_name_list) - set(a))) == 0:
            if len(list(set(target_name_list) - set(a))) != 0:
                for i in range(len(file_name_list)):
                    file_name = file_name_list[i]
                    save_fun(file_name)
                print('clearing')
                break
            else:
                print('cleared')
                break
        else:
            print('raw data not update')
            pass


if __name__ == '__main__':
    # end_time = datetime.now()
    # begin_time = datetime(end_time.year, end_time.month, end_time.day)
    # a = find_target_file(begin_time, end_time, 'm', endswith=None)
    # print(a)
    a = time.time()
    main_fun()
    b = time.time()
    print(b-a)
