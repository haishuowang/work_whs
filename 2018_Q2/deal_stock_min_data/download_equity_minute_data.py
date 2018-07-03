# 更新每天股票分钟数据
# 接口不稳定 当出现10002003错误代码时 停止下载30s后重新下载出错的股票
# 接口相当不稳定，程序开启时有可能崩溃，要多开几次
import pandas as pd
from EmQuantAPI import *
from datetime import datetime, timedelta
import time


def path_create(target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)


def vwap_close_15_minute(day_path):
    """

    :param day_path:
    :return:
    """
    index_list = ['09:30', '09:45', '10:00', '10:15', '10:30', '10:45', '11:00', '11:15', '11:30',
                  '13:15', '13:30', '13:45', '14:00', '14:15', '14:30', '14:45', '15:00']
    close_data = pd.read_csv(os.path.join(day_path, 'Close.csv'), index_col=0)
    volume_data = pd.read_csv(os.path.join(day_path, 'Volume.csv'), index_col=0)
    save_path = os.path.join(day_path, 'vwap.csv')

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
    print('day:{} vwap finished'.format(day_path.split('/')[-1]))


def download_minute_data(update_time, save_root_path):
    """

    :param update_time:
    :param save_root_path:
    :return:
    """
    save_path = os.path.join(save_root_path, update_time[:4], update_time[:6], update_time)
    stock_code = c.sector("001004", update_time).Codes
    stock_code_c = [x.split('.')[1] + x.split('.')[0] for x in stock_code]
    index_list = []
    factor_list = ['High', 'Open', 'Low', 'Close', 'Volume', 'Turnover']

    path_create(save_path)
    error_code = []
    for i in range(6):
        exec('df_{} = pd.DataFrame([[None] * len(stock_code)] * 240)'.format(i))
        exec('df_{}.columns = stock_code_c'.format(i))
    i_code = 0
    while i_code < len(stock_code):
        stock = stock_code[i_code]
        print('{} downloading...'.format(stock))
        stock_data = c.cmc(stock, "High,Open,Low,Close,Volume,Amount", update_time, update_time,
                           "IsHistory=1,Period=1")
        if stock_data.Data:
            if not index_list:
                index_list = [x[-8:-3] for x in stock_data.Dates]
            for i in range(6):
                exec('df_{0}[stock.split(\'.\')[1] + stock.split(\'.\')[0]] = stock_data.Data[{0}]'.format(i))
            i_code += 1
        elif stock_data.ErrorCode == 10002003:
            time.sleep(30)
            pass
        else:
            error_code.append(stock)
            i_code += 1
            pass
    for i in range(6):
        exec('df_{}.index = index_list'.format(i))
        exec('df_{0}.dropna(1).to_csv(os.path.join(save_path, factor_list[{0}]+\'.csv\'))'.format(i))


if __name__ == '__main__':
    save_root_path = r'/media/hdd0/data/adj_data/equity/intraday/eqt_1mbar'
    update_time = (datetime.today() - timedelta(1)).strftime('%Y%m%d')
    # update_time = '20180511'
    if pd.to_datetime(update_time).weekday() <= 4:
        print(pd.to_datetime(update_time).weekday(), 4)
        loginResult = c.start("ForceLogin=1")
        download_minute_data(update_time, save_root_path)
        day_path = os.path.join(save_root_path, update_time[:4], update_time[:6], update_time)
        vwap_close_15_minute(day_path)
        print('{} download success'.format(update_time))
        logoutResult = c.stop()
    else:
        print('error_date!!!')
    exit(0)
