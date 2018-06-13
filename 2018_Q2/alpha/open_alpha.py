import sys

sys.path.append('../..')
import loc_lib.b_t_box.get_data as gd
import loc_lib.b_t_box.choose_stock as cs
import pandas as pd
import time
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import gc
import numpy as np
from multiprocessing import Pool
import os
import copy


def pcf(df):
    pcf_df = (df - df.shift(1)) / df.shift(1)
    pcf_df[(-0.2 > pcf_df) | (pcf_df > 0.2)] = 0
    return pcf_df


def Sharpe(pnl):
    pnl = pd.DataFrame(pnl, columns=['pnl'])
    pnl_diff = (pnl - pnl.shift(1)).dropna()
    return ((16 * pnl_diff.mean()) / pnl_diff.std())[0]


def signal_position_Open(choose_price, n_open=6, limit=0.6, hold_time=8):
    """
    计算开盘30分钟的sharpe
    :param choose_price:
    :param n_open:
    :param limit:
    :param hold_time:
    :return:
    """
    position = pd.DataFrame(0, index=choose_price.index, columns=choose_price.columns)
    position.iloc[n_open:n_open + hold_time * 3] = 1
    pcf = (choose_price.iloc[:n_open] - choose_price.iloc[:n_open].shift(1)) / choose_price.iloc[:n_open].shift(1)
    signal = pcf.mean() / pcf.std()
    signal[(signal > -limit) & (signal < limit)] = 0
    signal[signal > limit] = -1
    signal[signal < -limit] = 1
    position = signal * position
    return position


def signal_position_Open_bbangs(choose_price, n_open=6, limit=0.6, hold_time=8):
    """

    :param choose_price:
    :param n_open:
    :param limit:
    :param hold_time:
    :return:
    """
    position = pd.DataFrame(0, index=choose_price.index, columns=choose_price.columns)
    position.iloc[n_open:n_open + hold_time * 3] = 1
    choose_price.iloc[:n_open][-1]-choose_price.iloc[:n_open]
    signal[(signal > -limit) & (signal < limit)] = 0
    signal[signal > limit] = -1
    signal[signal < -limit] = 1
    position = signal * position
    return position


def trade_fun(position, choose_vwap, log_time=0):
    cost = 0.0005
    position_lag = position.shift(log_time)
    position_trade = position_lag.loc[choose_vwap.index]
    position_trade.iloc[-2:] = 0
    vwap_diff = choose_vwap.rolling(window=2).apply(lambda x: (x[-1] - x[0]) / x[0])
    day_pnl = sum((position_trade.shift(2) * vwap_diff).sum())
    day_trade = sum(position_trade.diff().abs().sum())
    day_cost = day_trade * cost
    day_pnl = day_pnl - day_cost
    print(day_pnl, day_cost)
    return day_pnl, day_trade


def back_test_fun(vol_num, hold_time, n_open, limit, stock_num):
    start = time.time()
    cost = 0.0005
    date_list = sorted(price.keys())
    pnl_list = [0.] * len(date_list)
    trade_list = [0] * len(date_list)
    for i_date in range(vol_num, len(date_list)):
        date = date_list[i_date]
        # print(date)

        date = date_list[i_date]
        pre_price_data = pd.DataFrame()
        pre_turnover_data = pd.DataFrame()

        part_price = copy.deepcopy(price[date])
        part_vwap = copy.deepcopy(vwap[date])

        part_vwap.index = pd.to_datetime([date + ' ' + x for x in part_vwap.index])
        part_price.index = pd.to_datetime(part_price.index)
        today_stock = part_price.columns

        for i_vol in range(vol_num):
            back_date = date_list[i_date - 1 - i_vol]
            tmp_price_data = copy.deepcopy(price[back_date])
            tmp_turn_data = copy.deepcopy(volume[back_date])
            pre_price_data = pd.concat([pre_price_data, tmp_price_data], axis=0)
            pre_turnover_data = pd.concat([pre_turnover_data, tmp_turn_data], axis=0)

        # 根据波动率、交易量与股价选股 返回一个股票list
        universe = cs.high_vol_stock(pre_price_data, pre_turnover_data, p_num=stock_num, t_num=600, v_num=700)

        # 筛选出的股票与当天开盘的股票求交集
        choose_stock = sorted(list(set(universe) & set(today_stock)))
        # 筛选股票的price和vwap
        choose_price = part_price[choose_stock]
        choose_vwap = part_vwap[choose_stock]
        # 每天信号和仓位生成
        position = signal_position_Open(choose_price, n_open, limit, hold_time)
        # 每天的交易函数
        pnl_list[i_date], trade_list[i_date] = trade_fun(position, choose_vwap)

    end = time.time()

    asset = np.cumsum(pnl_list)
    sharpe = Sharpe(asset)
    print('_______________________')
    print('Processing Cost:{} second'.format(end - start))
    print('vol_num={} hold_time={} n_open={} limit={} stock_num={} asset={} trade_cost={}'
          .format(vol_num, hold_time, n_open, limit, stock_num, asset[-1], sum(trade_list) * cost))
    print('_______________________')


if __name__ == '__main__':
    begin_date = '20100101'
    end_date = '20170101'
    data_load_path = '/media/hdd0/data/adj_data/equity/intraday/special'
    price = pd.read_pickle(os.path.join(data_load_path, 'close_5m_2010-2017.pkl'))
    volume = pd.read_pickle(os.path.join(data_load_path, 'volume_5m_2010-2017.pkl'))
    vwap = pd.read_pickle(os.path.join(data_load_path, 'vwap_5m_2010-2017.pkl'))

    vol_num = 2
    limit = 0.6
    hold_time = 8
    n_open = 6
    stock_num = 500

    back_test_fun(vol_num, hold_time, n_open, limit, stock_num)
    # for i_date in range(vol_num, len(date_list))[:1]:
    #     date = date_list[i_date]
    #     pre_price_data = pd.DataFrame()
    #     pre_turnover_data = pd.DataFrame()
    #
    #     part_price = copy.deepcopy(price[date])
    #     part_vwap = copy.deepcopy(vwap[date])
    #
    #     part_vwap.index = pd.to_datetime([date + ' ' + x for x in part_vwap.index])
    #     part_price.index = pd.to_datetime(part_price.index)
    #     today_stock = part_price.columns
    #     for i_vol in range(vol_num):
    #         back_date = date_list[i_date - 1 - i_vol]
    #         tmp_price_data = copy.deepcopy(price[back_date])
    #         tmp_turn_data = copy.deepcopy(volume[back_date])
    #         pre_price_data = pd.concat([pre_price_data, tmp_price_data], axis=0)
    #         pre_turnover_data = pd.concat([pre_turnover_data, tmp_turn_data], axis=0)
    #     # 根据波动率、交易量与股价选股 返回一个股票list
    #     universe = cs.high_vol_stock(pre_price_data, pre_turnover_data, p_num=stock_num, t_num=800, v_num=700)
    #     # 筛选出的股票与当天开盘的股票求交集
    #     choose_stock = sorted(list(set(universe) & set(today_stock)))
    #     # 筛选股票的price和vwap
    #     choose_price = part_price[choose_stock]
    #     choose_vwap = part_vwap[choose_stock]
    #     # 每天信号和仓位生成
    #     signal_position_Open(choose_price, n_open=6, limit=6)
