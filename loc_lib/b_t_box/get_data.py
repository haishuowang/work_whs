import pandas as pd
from datetime import timedelta
import os
import open_lib.shared_paths.paths as olspp
import gc

global root_path
# root_path = olspp.adj_eqt_1mbar_path
root_path = '/media/hdd0/data/adj_data/equity/intraday/eqt_1mbar'


def stock_select(all_stock):
    new_stock = []
    for stock in all_stock:
        if stock.startswith('SH'):
            if stock[2] == '6':
                new_stock.append(stock)
        else:
            if stock[2] == '0' or stock[2:5] == '300':
                new_stock.append(stock)
    return new_stock


def eqt_1mbar_daily_data(pre_date):
    price_data_path = os.path.join(root_path, pre_date[:4], pre_date[:6], pre_date, 'Close.csv')
    volume_data_path = os.path.join(root_path, pre_date[:4], pre_date[:6], pre_date, 'Volume.csv')
    vwap_data_path = os.path.join(root_path, pre_date[:4], pre_date[:6], pre_date, 'vwap.csv')
    # print(price_data_path)
    try:

        price_data = pd.read_csv(price_data_path, index_col=0)
        volume_data = pd.read_csv(volume_data_path, index_col=0)
        vwap_data = pd.read_csv(vwap_data_path, index_col=0)
        # print(price_data)
        # l_index = pd.to_datetime([pre_date + ' ' + x for x in price_data.index])
        # s_index = pd.to_datetime([pre_date + ' ' + x for x in vwap_data.index])

        price_data.index = pd.to_datetime([pre_date + ' ' + x for x in price_data.index])
        volume_data.index = pd.to_datetime([pre_date + ' ' + x for x in volume_data.index])
        vwap_data.index = pd.to_datetime([pre_date + ' ' + x for x in vwap_data.index])

        new_stock = stock_select(price_data.columns)

        price.update({pre_date: price_data[new_stock]})
        volume.update({pre_date: volume_data[new_stock]})
        vwap.update({pre_date: vwap_data[new_stock]})

        del price_data, volume_data, vwap_data
        gc.collect()
    except IOError:
        pass


def eqt_1mbar_data(begin_date, end_date):
    global price, volume, vwap
    price = {}
    volume = {}
    vwap = {}
    pre_date = begin_date
    while pre_date <= end_date:
        print(pre_date)
        eqt_1mbar_daily_data(pre_date)
        pre_date = (pd.to_datetime(pre_date) + timedelta(1)).strftime('%Y%m%d')
    return price, volume, vwap


def eqt_5mbar_daily_data(pre_date, eqt_5mbar_path):
    price_data_path = os.path.join(eqt_5mbar_path, pre_date[:4], pre_date[:6], pre_date, 'Close.csv')
    volume_data_path = os.path.join(eqt_5mbar_path, pre_date[:4], pre_date[:6], pre_date, 'Volume.csv')
    vwap_data_path = os.path.join(root_path, pre_date[:4], pre_date[:6], pre_date, 'vwap.csv')
    try:
        price_data = pd.read_csv(price_data_path, index_col=0)
        volume_data = pd.read_csv(volume_data_path, index_col=0)
        vwap_data = pd.read_csv(vwap_data_path, index_col=0)

        new_stock = stock_select(price_data.columns)

        price.update({pre_date: price_data[new_stock]})
        volume.update({pre_date: volume_data[new_stock]})
        vwap.update({pre_date: vwap_data[new_stock]})

        del price_data, volume_data, vwap_data
        gc.collect()
    except IOError:
        pass


def eqt_5mbar_data(begin_date, end_date):
    eqt_5mbar_path = '/media/hdd0/data/adj_data/equity/intraday/eqt_5mbar'
    global price, volume, vwap
    price = {}
    volume = {}
    vwap = {}
    pre_date = begin_date
    while pre_date <= end_date:
        print(pre_date)
        eqt_5mbar_daily_data(pre_date, eqt_5mbar_path)
        pre_date = (pd.to_datetime(pre_date) + timedelta(1)).strftime('%Y%m%d')
    return price, volume, vwap


def universe_eqt_5m(universe, begin_date, end_date):
    eqt_5mbar_path = '/media/hdd0/data/adj_data/equity/intraday/eqt_5mbar'

    price = {}
    volume = {}
    vwap = {}
    pre_date = begin_date
    while pre_date <= end_date:
        pre_date = (pd.to_datetime(pre_date) + timedelta(1)).strftime('%Y%m%d')


# def eqt_daily_data(universe, begein_date, end_date):
#     daily_path = '/media/hdd0/data/raw_data/equity/extraday/taobao'
#     stock_file = os.listdir(daily_path)
#     all_stock_data = pd.DataFrame
#     for stock in universe.columns:
#         stock_data = pd.read_csv(os.path.join(daily_path, stock.lower + '.csv'))
#         stock_data = pd.read_csv(os.path.join(daily_path, stock.lower() + '.csv'),
#                                  index_col='交易日期', encoding='gbk')[['股票代码', '新浪行业', '新浪概念', '新浪地域',
#                                                                     '开盘价', '最高价', '最低价', '收盘价', '后复权价',
#                                                                     '前复权价', '涨跌幅', '成交量', '成交额', '换手率',
#                                                                     '流通市值', '总市值', '是否涨停', '是否跌停', ]]
#         stock_


# if __name__ == '__main__':
#     begin_date = '20010726'
#     end_date = '20010926'
#     # price, volume, vwap = eqt_1mbar_data(begin_date, end_date)
#     price, volume, vwap = eqt_5mbar_data(begin_date, end_date)
