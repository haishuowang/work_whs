import pandas as pd
import numpy as np
import os
from collections import OrderedDict

root_path = '/media/hdd1/dat_whs'


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
def get_new_stock_info(begin_date, end_date):
    new_stock_data = pd.read_pickle('/mnt/mfs/DAT_EQT/EM_Tab01/CDSY_SECUCODE/LISTSTATE.pkl')
    new_stock_data.fillna(method='ffill', inplace=True)
    target_df = new_stock_data.shift(40)[(new_stock_data.index >= begin_date) &
                                         (new_stock_data.index < end_date)].notnull().astype(int)
    eqa = AZ_split_stock(target_df.columns)
    target_df = target_df[eqa]
    return target_df


# 获取剔除st股票的矩阵
def get_st_stock_info(begin_date, end_date):
    data = pd.read_pickle('/mnt/mfs/DAT_EQT/EM_Tab01/CDSY_CHANGEINFO/CHANGEA.pkl')
    data.fillna(method='ffill', inplace=True)
    data = data[(data.index >= begin_date) & (data.index < end_date)]
    data = data.astype(str)
    target_df = data.applymap(lambda x: 0 if 'ST' in x or 'PT' in x else 1)
    # target_df.to_pickle('/mnt/mfs/dat_whs/data/error_stock_info/st_pt_stock.pkl')
    return target_df


# 生成每天的position
def pos_daily_fun(df, n=5):
    return df.rolling(window=n, min_periods=1).sum()


# 读取 sector(行业 最大市值等)
def load_sector_data(begin_date, end_date, sector_name):
    market_top_n = pd.read_pickle(os.path.join(root_path, 'data/sector_data/' + sector_name + '.pkl'))
    market_top_n = market_top_n.shift(1)[(market_top_n.index >= begin_date) & (market_top_n.index < end_date)]
    market_top_n = market_top_n[market_top_n.index >= begin_date]
    market_top_n.dropna(how='all', axis='columns', inplace=True)
    xnms = market_top_n.columns
    xinx = market_top_n.index
    new_stock_df = get_new_stock_info(begin_date, end_date)
    new_stock_df = new_stock_df.reindex(columns=xnms, index=xinx, fill_value=1)
    st_stock_df = get_st_stock_info(begin_date, end_date)
    st_stock_df = st_stock_df.reindex(columns=xnms, index=xinx, fill_value=1)
    market_top_n = market_top_n * new_stock_df * st_stock_df
    return market_top_n


# 读取 因涨跌停以及停牌等 不能变动仓位的日期信息
def load_locked_data(xnms, xinx, if_limit_updn=True):
    raw_suspendday_df = pd.read_pickle('/mnt/mfs/DAT_EQT/EM_Funda/TRAD_TD_SUSPENDDAY/SUSPENDREASON.pkl')
    suspendday_df = raw_suspendday_df.isnull()
    locked_df = suspendday_df.reindex(columns=xnms, index=xinx, fill_value=True)

    if if_limit_updn:
        return_df = pd.read_pickle('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.pkl').astype(float)
        limit_updn_df = (return_df.abs() < 0.095)
        limit_updn_df = limit_updn_df.reindex(columns=xnms, index=xinx, fill_value=1)
        locked_df = limit_updn_df * suspendday_df

    locked_df = locked_df.reindex(columns=xnms, index=xinx, fill_value=1)
    return locked_df


# 读取 上证50,沪深300等数据
def load_index_data(xinx, index_name):
    root_index_path = os.path.join(root_path, 'data/index_data')
    target_df = pd.read_pickle(os.path.join(root_index_path, index_name + '.pkl'))
    target_df = target_df.reindex(index=xinx)
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
        load_path = os.path.join(root_path, 'data/factor_data/' + sector_name)
        target_df = pd.read_pickle(os.path.join(load_path, file_name + '.pkl'))
        # target_df = target_df[(target_df.index >= begin_date) & (target_df.index < end_date)]
        target_df.fillna(0, inplace=True)
        factor_set[file_name] = target_df.reindex(columns=xnms, index=xinx)
    return factor_set


# 处理停牌，跌涨停，sector筛选对仓位的影响
def deal_mix_factor(mix_factor, sector_df, locked_df, hold_time, lag, if_only_long):
    mix_factor = mix_factor.replace(np.nan, 0).shift(lag)
    if if_only_long:
        mix_factor = mix_factor[mix_factor > 0]
    daily_pos = pos_daily_fun(mix_factor, n=hold_time)

    # 排除涨跌停和停牌对策略的影响
    if locked_df is not None:
        daily_pos = daily_pos * locked_df
        daily_pos.fillna(method='ffill', inplace=True)

    # sector筛选
    if sector_df is not None:
        daily_pos = daily_pos * sector_df
    return daily_pos


def create_log_save_path(target_path):
    top_path = os.path.split(target_path)[0]
    if not os.path.exists(top_path):
        os.mkdir(top_path)
    if not os.path.exists(target_path):
        os.mknod(target_path)


def deal_mix_factor_c(mix_factor, sector_df, locked_df, hold_time, lag, if_only_long):
    mix_factor.replace(np.nan, 0, inplace=True)
    if if_only_long:
        mix_factor = mix_factor[mix_factor > 0]
    order_df = mix_factor.replace(np.nan, 0).shift(1)

    # 排除涨跌停和停牌对策略的影响
    if locked_df is not None:
        order_df = order_df * locked_df
        order_df.fillna(method='ffill', inplace=True)

    # sector筛选
    if sector_df is not None:
        order_df = order_df * sector_df

    daily_pos = pos_daily_fun(order_df.shift(lag-1), n=hold_time)
    return daily_pos
