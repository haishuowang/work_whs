import gc
import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *


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


def part_info_table(root_path, today, begin_time, end_time):
    # try:
    day_path = f'{root_path}/{today[:4]}/{today[:6]}/{today}'
    turnover = pd.read_csv(os.path.join(day_path, 'Turnover.csv'), index_col=0).astype(float)
    turnover_d = turnover.loc[begin_time: end_time].sum()
    turnover_d.index = bt.AZ_clear_columns(turnover_d.index)
    turnover_d = turnover_d[AZ_split_stock(turnover_d.index)]
    turnover_d.name = today

    volume = pd.read_csv(f'{day_path}/Volume.csv', index_col=0).astype(float)
    volume_d = volume.loc[begin_time: end_time].sum()
    volume_d.index = bt.AZ_clear_columns(volume_d.index)
    volume_d = volume_d[AZ_split_stock(volume_d.index)]
    volume_d.name = today

    close = pd.read_csv(f'{day_path}/Close.csv', index_col=0).astype(float).fillna(method='ffill')
    close_d = close.loc[begin_time: end_time].iloc[-1]
    close_d.index = bt.AZ_clear_columns(close_d.index)
    close_d = close_d[AZ_split_stock(close_d.index)]
    close_d.name = today

    high = pd.read_csv(f'{day_path}/High.csv', index_col=0).astype(float)
    high_d = high.loc[begin_time: end_time].max()
    high_d.index = bt.AZ_clear_columns(high_d.index)
    high_d = high_d[AZ_split_stock(high_d.index)]
    high_d.name = today

    low = pd.read_csv(f'{day_path}/Low.csv', index_col=0).astype(float)
    low_d = low.loc[begin_time: end_time].min()
    low_d.index = bt.AZ_clear_columns(low_d.index)
    low_d = low_d[AZ_split_stock(low_d.index)]
    low_d.name = today
    print(f'{day_path}/Open.csv')
    open = pd.read_csv(f'{day_path}/Open.csv', index_col=0).astype(float)
    open_d = open.loc[begin_time: end_time].iloc[0]
    open_d.index = bt.AZ_clear_columns(open_d.index)
    open_d = open_d[AZ_split_stock(open_d.index)]
    open_d.name = today
    print(today, 'deal')
    return turnover_d, volume_d, close_d, high_d, low_d, open_d
    # except Exception as error:
    #     print(today, error)
    #     send_email.send_email(today+error, ['whs@yingpei.com'], [], 'Wonderfully')
    #     return None, None, None


def adj_price(price_df):
    factor_1 = bt.AZ_Load_csv('/media/hdd1/DAT_EQT/EM_Funda/TRAD_SK_FACTOR1/TAFACTOR.csv')
    factor_1 = factor_1.reindex(index=price_df.index)
    price_df = price_df.reindex(columns=factor_1.columns)
    return price_df / factor_1


def save_fun(file_result, file_name):
    save_path = f'/mnt/mfs/DAT_EQT/intraday/daily_0930_1415/{file_name}.csv'
    if os.path.exists(save_path):
        data = bt.AZ_Load_csv(save_path)
        data = data[AZ_split_stock(data.columns)]
        data = data.combine_first(file_result)
    else:
        data = file_result
    data.to_csv(f'/mnt/mfs/DAT_EQT/intraday/daily_0930_1415/{file_name}.csv', sep='|')
    data_adj = adj_price(data)
    data_adj.to_csv(f'/mnt/mfs/DAT_EQT/intraday/daily_0930_1415/adj_{file_name}.csv', sep='|')
    # data_adj.to_csv(f'/mnt/mfs/dat_whs/tmp/adj_{file_name}.csv', sep='|')


def main_fun():
    today = datetime.now().strftime('%Y%m%d')
    # today = '20190314'
    root_path = '/mnt/mfs/DAT_EQT/intraday/eqt_1mbar'
    begin_time, end_time = '09:30', '14:15'
    a = time.time()
    file_result_list = part_info_table(root_path, today, begin_time, end_time)
    file_name_list = ['turnover', 'volume', 'close', 'high', 'low', 'open']
    for i in range(len(file_name_list)):
        file_name = file_name_list[i]
        file_result = pd.DataFrame(file_result_list[i]).T
        save_fun(file_result, file_name)
    b = time.time()
    print(b - a)


if __name__ == '__main__':
    main_fun()
