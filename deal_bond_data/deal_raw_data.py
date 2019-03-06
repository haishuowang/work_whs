import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *
import collections


def data_load(root_path, file_name, target_dict):
    data = pd.read_csv(os.path.join(root_path, file_name), header=None,
                       index_col=1)
    data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
    data.index.name = 'Time'
    date_list = np.array(sorted(set(data['Date'].values)))
    # date_list_part = date_list[date_list > '2018/08/15']
    for date in date_list:
        print(date)
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
        part_Volume = part_data['Volume']
        part_Volume.name = file_name[:-4]
        if date_key not in target_dict.keys():
            part_dict = collections.OrderedDict()
            part_dict['Close'] = pd.DataFrame(part_Close)
            part_dict['High'] = pd.DataFrame(part_High)
            part_dict['Low'] = pd.DataFrame(part_Low)
            part_dict['Open'] = pd.DataFrame(part_Open)
            part_dict['Turnover'] = pd.DataFrame(part_Turnover)
            part_dict['Volume'] = pd.DataFrame(part_Volume)
            target_dict[date_key] = part_dict
        else:
            part_dict = target_dict[date_key]
            part_dict['Close'] = pd.concat([part_dict['Close'], part_Close], axis=1, sort=True)
            part_dict['High'] = pd.concat([part_dict['High'], part_High], axis=1, sort=True)
            part_dict['Low'] = pd.concat([part_dict['Low'], part_Low], axis=1, sort=True)
            part_dict['Open'] = pd.concat([part_dict['Open'], part_Open], axis=1, sort=True)
            part_dict['Turnover'] = pd.concat([part_dict['Turnover'], part_Turnover], axis=1, sort=True)
            part_dict['Volume'] = pd.concat([part_dict['Volume'], part_Volume], axis=1, sort=True)
            target_dict[date_key] = part_dict
    return target_dict


def clear_columns(df):
    df.columns = bt.AZ_clear_columns(df.columns)
    return df


def save_date(date_key, target_dict, root_save_path):
    part_save_path = os.path.join(root_save_path, date_key)
    bt.AZ_Path_create(part_save_path)

    clear_columns(target_dict[date_key]['Close']).to_csv(os.path.join(part_save_path, 'Close.csv'),
                                                         sep='|', index_label='Time')
    clear_columns(target_dict[date_key]['High']).to_csv(os.path.join(part_save_path, 'High.csv'),
                                                        sep='|', index_label='Time')
    clear_columns(target_dict[date_key]['Low']).to_csv(os.path.join(part_save_path, 'Low.csv'),
                                                       sep='|', index_label='Time')
    clear_columns(target_dict[date_key]['Open']).to_csv(os.path.join(part_save_path, 'Open.csv'),
                                                        sep='|', index_label='Time')
    clear_columns(target_dict[date_key]['Turnover']).to_csv(os.path.join(part_save_path, 'Turnover.csv'),
                                                            sep='|', index_label='Time')
    clear_columns(target_dict[date_key]['Volume']).to_csv(os.path.join(part_save_path, 'Volume.csv'),
                                                          sep='|', index_label='Time')


def create_raw_data(root_load_path, root_save_path):
    file_name_list = sorted(os.listdir(root_load_path))
    # root_save_path = '/mnt/mfs/DAT_PUBLIC/intraday_test/bond_1mbar'

    target_dict = collections.OrderedDict()
    for file_name in file_name_list:
        print(f'Load {file_name}!')
        target_dict = data_load(root_load_path, file_name, target_dict)

    date_list = np.array(sorted(list(target_dict.keys())))

    for date_key in date_list:
        print(f'Deal {date_key}!')
        save_date(date_key, target_dict, root_save_path)


def clear_raw_data():
    root_save_path = '/mnt/mfs/DAT_PUBLIC/intraday_test/eqt_1mbar'
    # year_list = ['2018']
    year_list = os.listdir(root_save_path)
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
                    # print(xnms[0])
                    high.columns = xnms
                    open.columns = xnms
                    low.columns = xnms
                    close.columns = xnms
                    turnover.columns = xnms
                    volume.columns = xnms
                    # print(close[sorted(xnms)])
                    high.to_csv(os.path.join(day_path, 'High.csv'))
                    open.to_csv(os.path.join(day_path, 'Open.csv'))
                    low.to_csv(os.path.join(day_path, 'Low.csv'))
                    close.to_csv(os.path.join(day_path, 'Close.csv'))
                    turnover.to_csv(os.path.join(day_path, 'Turnover.csv'))
                    volume.to_csv(os.path.join(day_path, 'Volume.csv'))


if __name__ == '__main__':
    root_load_path_list = [
        # '/mnt/mfs/DAT_PUBLIC/2017可转债',
        # '/mnt/mfs/DAT_PUBLIC/2018可转债',
        '/mnt/mfs/DAT_PUBLIC/2019可转债'
    ]
    # root_save_path = '/mnt/mfs/DAT_PUBLIC/intraday_test/bond_1mbar'
    root_save_path = '/mnt/mfs/DAT_EQT/intraday/bond_1mbar'
    pool = Pool(3)
    for root_load_path in root_load_path_list:
        args = (root_load_path, root_save_path)
        # create_raw_data(*args)
        pool.apply_async(create_raw_data, args)
    pool.close()
    pool.join()

