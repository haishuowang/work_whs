import pandas as pd
import numpy as np
import os
from collections import OrderedDict
import loc_lib.shared_tools.back_test as bt


root_path = '/mnt/mfs/dat_whs'
stock_data_path = '/mnt/mfs/DAT_EQT'


def AZ_split_stock(stock_list):
    """
    在stock_list中寻找A股代码
    :param stock_list:
    :return:
    """
    eqa = [x for x in stock_list if (x.startswith('0') or x.startswith('3')) and x.endswith('SZ')
           or x.startswith('6') and x.endswith('SH')]
    return eqa


# 获取剔除新股的矩阵
def get_new_stock_info(xnms, xinx):
    new_stock_data = bt.AZ_Load_csv(os.path.join(stock_data_path, 'EM_Tab01/CDSY_SECUCODE/LISTSTATE.csv'))
    new_stock_data.fillna(method='ffill', inplace=True)
    # 获取交易日信息
    return_df = bt.AZ_Load_csv(os.path.join(stock_data_path, 'EM_Funda/DERIVED_14/aadj_r.csv')).astype(float)
    trade_time = return_df.index
    new_stock_data = new_stock_data.reindex(index=trade_time).fillna(method='ffill')
    target_df = new_stock_data.shift(40).notnull().astype(int)
    target_df = target_df.reindex(columns=xnms, index=xinx)
    return target_df


# 获取剔除st股票的矩阵
def get_st_stock_info(xnms, xinx):
    data = bt.AZ_Load_csv(os.path.join(stock_data_path, 'EM_Tab01/CDSY_CHANGEINFO/CHANGEA.csv'))
    data = data.reindex(columns=xnms, index=xinx)
    data.fillna(method='ffill', inplace=True)

    data = data.astype(str)
    target_df = data.applymap(lambda x: 0 if 'ST' in x or 'PT' in x else 1)
    return target_df


# 生成每天的position
def pos_daily_fun(df, n=5):
    return df.rolling(window=n, min_periods=1).sum()


# 读取 sector(行业 最大市值等)
def load_sector_data(begin_date, end_date, sector_name):

    market_top_n = bt.AZ_Load_csv(os.path.join(stock_data_path, 'EM_Funda/DERIVED_10/' + sector_name + '.csv'))
    market_top_n = market_top_n.shift(1)[(market_top_n.index >= begin_date) & (market_top_n.index < end_date)]
    market_top_n = market_top_n[market_top_n.index >= begin_date]
    market_top_n.dropna(how='all', axis='columns', inplace=True)
    xnms = market_top_n.columns
    xinx = market_top_n.index

    new_stock_df = get_new_stock_info(xnms, xinx).shift(1)
    st_stock_df = get_st_stock_info(xnms, xinx).shift(2)
    sector_df = market_top_n * new_stock_df * st_stock_df
    sector_df.replace(0, np.nan, inplace=True)
    return sector_df


# 读取 因涨跌停以及停牌等 不能变动仓位的日期信息
# 涨停不可买入　跌停不可卖出
def load_locked_data(xnms, xinx):
    raw_suspendday_df = bt.AZ_Load_csv(os.path.join(stock_data_path, 'EM_Funda/TRAD_TD_SUSPENDDAY/SUSPENDREASON.csv'))
    suspendday_df = raw_suspendday_df.isnull()
    suspendday_df = suspendday_df.reindex(columns=xnms, index=xinx, fill_value=True)
    suspendday_df.replace(0, np.nan, inplace=True)

    return_df = bt.AZ_Load_csv(os.path.join(stock_data_path, 'EM_Funda/DERIVED_14/aadj_r.csv')).astype(float)
    limit_buy_df = (return_df < 0.095).astype(int)
    limit_buy_df = limit_buy_df.reindex(columns=xnms, index=xinx, fill_value=1)
    limit_buy_df.replace(0, np.nan, inplace=True)

    limit_sell_df = (return_df > -0.095).astype(int)
    limit_sell_df = limit_sell_df.reindex(columns=xnms, index=xinx, fill_value=1)
    limit_sell_df.replace(0, np.nan, inplace=True)
    return suspendday_df, limit_buy_df, limit_sell_df


# 涨跌停都不可交易
def load_locked_data_both(xnms, xinx):
    raw_suspendday_df = bt.AZ_Load_csv(os.path.join(stock_data_path, 'EM_Funda/TRAD_TD_SUSPENDDAY/SUSPENDREASON.csv'))
    suspendday_df = raw_suspendday_df.isnull()
    suspendday_df = suspendday_df.reindex(columns=xnms, index=xinx, fill_value=True)
    suspendday_df.replace(0, np.nan, inplace=True)

    return_df = bt.AZ_Load_csv(os.path.join(stock_data_path, 'EM_Funda/DERIVED_14/aadj_r.csv')).astype(float)
    limit_buy_sell_df = (return_df.abs() < 0.095).astype(int)
    limit_buy_sell_df = limit_buy_sell_df.reindex(columns=xnms, index=xinx, fill_value=1)
    limit_buy_sell_df.replace(0, np.nan, inplace=True)
    return suspendday_df, limit_buy_sell_df


# 读取 上证50,沪深300等数据
def load_index_data(xinx, index_name):
    # root_index_path = os.path.join(root_path, 'data/index_data')
    # target_df = pd.read_pickle(os.path.join(root_index_path, index_name + '.pkl'))
    # target_df = target_df[target_df.columns[0]].reindex(index=xinx)

    data = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Tab09/INDEX_TD_DAILYSYS/CHG.csv')
    target_df = data[index_name].reindex(index=xinx)
    return target_df * 0.01


# 读取return
def load_pct(xnms, xinx, return_file_name=None):
    if return_file_name is None:
        return_file_name = 'pct_f1d'
    load_path = os.path.join(root_path, 'data/return_data/{}.pkl'.format(return_file_name))
    # load_path = os.path.join(root_path, 'data/return_data/open_1_hour_vwap.pkl')
    target_df = pd.read_pickle(load_path)
    target_df = target_df.reindex(columns=xnms, index=xinx)
    return target_df


# 读取部分factor
def load_part_factor(sector_name, xnms, xinx, file_list):
    factor_set = OrderedDict()
    for file_name in file_list:
        load_path = os.path.join(root_path, 'data/new_factor_data/' + sector_name)
        target_df = pd.read_pickle(os.path.join(load_path, file_name + '.pkl'))
        # target_df = target_df[(target_df.index >= begin_date) & (target_df.index < end_date)]
        target_df.fillna(0, inplace=True)
        factor_set[file_name] = target_df.reindex(columns=xnms, index=xinx)
    return factor_set


def create_log_save_path(target_path):
    top_path = os.path.split(target_path)[0]
    if not os.path.exists(top_path):
        os.mkdir(top_path)
    if not os.path.exists(target_path):
        os.mknod(target_path)


def deal_mix_factor(mix_factor, sector_df, suspendday_df, limit_buy_sell_df, hold_time, lag, if_only_long):
    if if_only_long:
        mix_factor = mix_factor[mix_factor > 0]
    mix_factor.replace(np.nan, 0, inplace=True)
    # 下单日期pos
    order_df = mix_factor.replace(np.nan, 0).shift(lag-1)

    # 排除入场场涨跌停的影响
    order_df = order_df * sector_df * limit_buy_sell_df * suspendday_df

    daily_pos = pos_daily_fun(order_df.shift(1), n=hold_time)
    # 排除出场涨跌停的影响
    daily_pos = daily_pos * suspendday_df * limit_buy_sell_df
    daily_pos.fillna(method='ffill', inplace=True)
    # 获得最终仓位信息
    return daily_pos


def deal_mix_factor_both(mix_factor, sector_df, suspendday_df, limit_buy_df, limit_sell_df, hold_time, lag, if_only_long):
    if if_only_long:
        mix_factor = mix_factor[mix_factor > 0]
    mix_factor.replace(np.nan, 0, inplace=True)
    # 下单日期pos
    order_df = mix_factor.replace(np.nan, 0).shift(lag-1)
    # 排除下单时停牌对策略的影响
    order_df = order_df * suspendday_df
    # 排除下单时涨停对策略的影响
    buy_order_df = (order_df.where(order_df > 0, other=0) * limit_buy_df).replace(np.nan, 0)
    # 排除下单时跌停对策略的影响
    sell_order_df = (order_df.where(order_df < 0, other=0) * limit_sell_df).replace(np.nan, 0)
    # 获取下单信息
    order_df = (buy_order_df + sell_order_df)
    # sector筛选
    order_df = order_df * sector_df

    daily_pos = pos_daily_fun(order_df.shift(1), n=hold_time)

    # 排除出场时停牌对策略的影响
    daily_pos = daily_pos * suspendday_df
    daily_pos.fillna(method='ffill', inplace=True)
    # 排除出场时涨跌停对策略的影响
    buy_pos_df = daily_pos.where(daily_pos > 0, other=0) * limit_buy_df
    buy_pos_df.fillna(method='ffill', inplace=True)

    sell_pos_df = daily_pos.where(daily_pos < 0, other=0) * limit_sell_df
    sell_pos_df.fillna(method='ffill', inplace=True)

    # 获得最终仓位信息
    daily_pos = buy_pos_df + sell_pos_df
    daily_pos.fillna(method='ffill', inplace=True)
    return daily_pos
