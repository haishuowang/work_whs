__author__ = 'jerry', 'rzj'

from EmQuantAPI import *

import os
import datetime as datetime
import pandas as pd


def get_QuantAPI_data(API_data):
    new_dates = [x.replace('/', '-') for x in API_data.Dates]
    data = pd.DataFrame(index=new_dates)
    for k in API_data.Data.keys():
        data[k] = API_data.Data[k][0]
    data = data.sort_index()
    return data.rename_axis('Date')


def RZ_ExistDir(now_path, file_name=''):
    if file_name != '':
        now_path = now_path + '/' + file_name
    return os.path.exists(now_path)


def get_instockData(now_key, tmp_code, start_date, end_date):
    QuantAPI_data = c.edb(f"{tmp_code}",
                          f"IsLatest=0,StartDate={start_date},EndDate={end_date}")
    data = get_QuantAPI_data(QuantAPI_data)
    if len(data) == 0:
        return None
    data.columns = [now_key]
    return data


def AZ_TempFilePath(path):
    i = 0
    path_split = os.path.split(path)
    path_format = path_split[0] + '/.' + path_split[1] + '.temp{}'
    new_path = path_format.format(i)
    while RZ_ExistDir(new_path):
        i += 1
        new_path = path_format.format(i)
    return new_path


def SaveSafe(new_data, now_path, **kwargs):
    write_path = AZ_TempFilePath(now_path)
    new_data.to_csv(write_path, sep='|', **kwargs)
    os.rename(write_path, now_path)


def update_instockData(new_data, folder_path, ins_name, file_name):
    now_path = folder_path + ins_name + '/' + file_name + '.csv'
    if os.path.exists(now_path):
        old_data = pd.read_csv(now_path, sep='|', index_col=0)
        old_index = old_data.index.values
        new_index = new_data.index.values
        old_index = old_index[old_index < new_index[0]]
        new_data = old_data.reindex(index=old_index).append(new_data, sort=False)
    SaveSafe(new_data, now_path)


def main():
    save_path = '/mnt/mfs/DAT_FUT/spot/'
    instocklist = {'AG': ['EMI00069027', 'COMEX_AG'],
                   'ZN': ['EMI00069031', 'LME_ZN'],
                   # 'CU' : 'EMI00069038', 'SHFE_CU']
                   'PB': ['EMI00069047', 'SHFE_PB'],
                   'J_tianjinggang': ['EMI00018808', 'Tianjinggang_J'],
                   'J_lianyungang': ['EMI00018809', 'Lianyungang_J'],
                   'J_rizhaogang': ['EMI00018810', 'Rizhaogang_J'],
                   'RU': ['EMM00167660', 'SHFE_RU']
                   }

    today_date = pd.to_datetime(datetime.datetime.today().date())
    start_date = str((today_date - datetime.timedelta(4)).date())
    end_date = str(today_date.date())

    print('Commodity Instock data, update begins......\n')

    get_data_ls = list()
    c.start("ForceLogin=0")
    for now_key, now_code in instocklist.items():
        now_data = get_instockData(now_key, now_code[0], start_date, end_date)
        if now_data is None:
            print(now_key, ' ', now_code[0], ' no data collected\n')
            continue
        get_data_ls.append([now_key, now_code, now_data])
    c.stop()

    for now_key, now_code, now_data in get_data_ls:
        update_instockData(now_data, save_path, now_key, now_code[1])

    print('All updated processed, program ends.\n')


if __name__ == '__main__':
    # 12 14点
    main()
