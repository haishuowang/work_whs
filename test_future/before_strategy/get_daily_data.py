import get_data as gd
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

instruments = ['IF', 'IC', 'IH', 'TF', 'T', 'cu', 'zn', 'al', 'pb', 'au', 'rb', 'hc', 'ru', 'bu', 'ni', 'sn', 'CF',
               'FG', 'MA', 'OI', 'RM', 'SR', 'TA', 'WH', 'TC', 'a', 'c', 'cs', 'i', 'j', 'jd', 'jm', 'l', 'm', 'p',
               'pp', 'v', 'y']


# instruments = ['cu', 'rb', 'i', 'ru', 'SR', 'm', 'au',  'p', 'ni', 'bu', 'TA']


def get_daily_data(instrument, begin_date='begin', end_date='end'):
    if instrument != 'ZC':
        path = r'\\192.168.126.38\DFCData\CleanData\ExtradayData\Wind\Future\WindData_Futures\AdjustedDailyData' \
               r'\OpenInterest\{}.csv'.format(
            instrument)
        # path = r'C:\Users\haishuo.wang\Desktop\some_data\OpenInterest\{}.csv'.format(instrument)
        data = pd.read_csv(path, index_col=0)
        data = data.dropna(axis=0, how='all')
        data.index = pd.to_datetime(data.index.astype(str))
        if begin_date == 'begin':
            begin_data = data.index[0]
        else:
            begin_data = pd.to_datetime(begin_date)

        if end_date == 'end':
            end_date = data.index[-1]
        else:
            end_date = pd.to_datetime(end_date)
        data = data.ix[(data.index >= begin_data) & (data.index <= end_date)]
        contracts = data['mostactive1']
    else:
        path1 = r'\\192.168.126.38\DFCData\CleanData\ExtradayData\Wind\Future\WindData_Futures' \
                r'\AdjustedDailyData\OpenInterest\TC.csv'
        path2 = r'\\192.168.126.38\DFCData\CleanData\ExtradayData\Wind\Future\WindData_Futures' \
                r'\AdjustedDailyData\OpenInterest\ZC.csv'
        data1 = pd.read_csv(path1, index_col=0)
        data1.index = pd.to_datetime(data1.index.astype(str))
        data2 = pd.read_csv(path2, index_col=0)
        data2.index = pd.to_datetime(data2.index.astype(str))
        cut_date = pd.to_datetime('20151218')
        data = pd.concat([data1[data1.index <= cut_date], data2.ix[data2.index > cut_date]], axis=0)
        contracts = data['mostactive1']
    return data, contracts


def get_recent_contract_data(instrument, beginData='begin', endData='end'):
    path = r'C:\Users\haishuo.wang\Desktop\WHS\roll_resent_contract\{}.csv'.format(instrument)
    data = pd.read_csv(path, index_col=0)
    data = data.dropna(axis=0, how='all')
    data.index = pd.to_datetime(data.index.astype(str))

    if beginData == 'begin':
        beginData = data.index[0]
    else:
        beginData = pd.to_datetime(beginData)

    if endData == 'end':
        endData = data.index[-1]
    else:
        endData = pd.to_datetime(endData)
    data = data.ix[(data.index >= beginData) & (data.index <= endData)]
    contracts = data['Contract']
    return data, contracts

# if __name__ == '__main__':
#     instrument = 'rb'
#     data, contract = get_daily_data(instrument, beginData='begin', endData='end')
#     print(data['ClosePrice'])
