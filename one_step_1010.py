# ____author_____:yolanda
import pandas as pd
import numpy as np
import os
from collections import OrderedDict
from datetime import datetime, timedelta
import time as t
import pymysql
import pickle
from EmQuantAPI import *



def Loadcsv(path1, filename):
    '加载csv'
    df = pd.read_csv(path1 + '%s' % filename, sep='|', index_col=0, low_memory=False, parse_dates=True)
    return df


def mysql_extract(down_list):
    connect = pymysql.Connect(
        host='192.168.16.10',
        port=3306,
        user='whs',
        passwd='kj23#12!^3weghWhjqQ2rjj197',
        db='choice_fndb',
        charset="utf8"
    )
    cursor = connect.cursor()
    print("Connection success!")

    print("开始提取数据......")
    sql = "select  a.INFOCODE,b.SHOWTIME,a.EITIME,a.RELATECODE " \
          "from INFO_AN_NEWSRELATIONSEC a LEFT JOIN " \
          "INFO_AN_NEWSBASIC b on(a.INFOCODE = b.INFOCODE) " \
          "where b.SOURCE not like '%东方财富%' and b.SOURCE " \
          "NOT LIKE 'FX%' and b.SOURCE NOT LIKE '%期货%' and " \
          "b.SOURCE NOT LIKE '%黄金%' and b.SOURCE NOT LIKE " \
          "'%FOF%'"

    cursor.execute(sql)
    df = np.array(cursor.fetchall())
    cursor.close()
    df = pd.DataFrame(df, columns=list(down_dict.values()))
    return df


def GetTradeDates(startdate, enddate):
    """
    :param startdate:20050101
    :param enddate: 当日
    :return: 交易日列表
    """
    date = Loadcsv("/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/", "aadj_p.csv").index.tolist()

    date = np.array([x.strftime('%Y%m%d') for x in date])
    return date


def Nondii_extract():
    """
    从NON_DII中提取所需数据并转换成dataframe（已经从数据库中提取的表），并且只保留了A股数据
    """
    down_list = down_dict.values()
    data = mysql_extract(down_list)
    code_list = []
    for i in list(data['RELATECODE']):
        if i[:2] in ['60', '00'] and len(i) == 6:
            code_list.append(i)

    data = data[data['RELATECODE'].isin(code_list)]
    data.sort_values(by='SHOWTIME', inplace=True)
    data1 = data.reset_index(drop=True)

    return data1


def AZ_add_stock_suffix(stock_list):
    """
    :param stock_list: 待变换的代码列表
    :return: 转换之后的代码列表
    """
    return list(map(lambda x: x + '.SH' if x.startswith('6') else x + '.SZ', stock_list))


def Handle_info(df, update_date):
    """
    控制7点前的信息为当天的信息，之后为明天的信息
    :param df: 所有数据
    :return: 加上sector_time和final_time的数据
    """
    df['SHOWTIME'] = pd.to_datetime(df['SHOWTIME'], format='%Y-%m-%d %H:%M:%S')
    df['EITIME'] = pd.to_datetime(df['EITIME'], format='%Y-%m-%d %H:%M:%S')

    showtime = df['SHOWTIME']
    eitime = df['EITIME']

    stock_list = AZ_add_stock_suffix(df['RELATECODE'].tolist())

    df['CODE'] = stock_list

    boundary_time = pd.to_datetime('20160601000000', format='%Y%m%d%H%M%S', errors='ignore')
    time_sector1 = list(showtime[showtime < boundary_time] + timedelta(hours=1))
    time_sector2 = eitime.iloc[len(time_sector1):]
    time_sector = time_sector1 + list(time_sector2)

    df['sector_time'] = time_sector

    final_time = []
    for i in range(len(time_sector)):
        if time_sector[i].hour < 7:
            final_time.append(datetime.strftime(time_sector[i], "%Y%m%d"))
        else:
            final_time.append(datetime.strftime(time_sector[i] + timedelta(days=1), "%Y%m%d"))

    df['final_time'] = final_time

    df = df.sort_values(by='final_time')

    df = df[df['final_time'] < update_date]

    return df


def Gen_dii(df, code_list):
    """
    产生DII数据
    :param df:
    :param code_list:
    :return:
    """
    data = np.array([0] * len(code_list))
    df['final_time'] = df['final_time'].astype("int")
    day_list = set(df['final_time'].tolist())

    for day in day_list:
        df1 = df[(df.final_time == day)]['CODE'].tolist()
        df1_list = []
        for value in set(df1):
            df1_list.append([value, df1.count(value)])
        df2 = [[x, 0] for x in code_list if x not in df1]
        df1_list += df2
        df1_list = np.array(sorted(df1_list, key=lambda x: x[0]))

        df1_list = df1_list[:, 1].astype("int")

        data = np.row_stack((data, df1_list))
        print(data.shape)
    data = np.delete(data, 0, 0)
    data = pd.DataFrame(data, columns=code_list)  # 存成dateframe格式，将股票代码作为列名

    data['TRADE_DAT'] = day_list
    data.sort_values(by='TRADE_DAT', inplace=True)

    return data


def Handle_tradedt(df, date):
    """
    :param df:所有日期的数据
    :param date:交易日列表
    :return: 非交易日的延后到后面第一个交易日,再合并
    """
    df = df[df['TRADE_DAT'] < int("%s%02d%02d" % (datetime.now().year, datetime.now().month, datetime.now().day))]
    final_time = df['TRADE_DAT'].astype(str)
    li = []
    for d in final_time:
        if d in list(date):
            li.append(d)
        else:
            li.append((date[date > d])[0])
    df['trade_time'] = li
    df1 = df.groupby(df['trade_time']).sum()
    df1.drop(['TRADE_DAT', 'trade_time'], axis=1, inplace=True)
    return df1


if __name__ == '__main__':
    down_dict = OrderedDict({'新闻编码': 'INFOCODE', '发布时间': 'SHOWTIME',
                             '入库时间': 'EITIME', '相关代码': 'RELATECODE'})
    path = '/mnt/mfs/dat_whs'

    startdate = datetime.strptime("20050101", "%Y%m%d")
    enddate = datetime.now()

    start = t.time()  # 记录程序运行时间
    date = GetTradeDates(startdate, enddate)

    df = Nondii_extract()
    #
    data = Handle_info(df, date[-1])
    #
    code_list = sorted(set(df['CODE'].tolist()))

    data = Gen_dii(data, code_list)
    #
    data = Handle_tradedt(data, date)
    #
    # 存储起来
    with open(path + "/" + "DII_INFO_AN_NEWS_7h.pk", "wb") as f1:  # 存储
        pickle.dump(data, f1, pickle.HIGHEST_PROTOCOL)

    end = t.time()
    TIME = end - start
    print("costed time:%.2fs" % TIME)
