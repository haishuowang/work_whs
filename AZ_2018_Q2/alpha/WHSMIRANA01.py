import pandas as pd
import numpy as np
import sys
sys.path.append("/mnt/mfs/LIB_ROOT")
import open_lib_c.shared_paths.path as pt
from itertools import product, permutations, combinations
import os
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')


def mul_fun(a, b):
    return a.mul(b, fill_value=0)


def sub_fun(a, b):
    return a.sub(b, fill_value=0)


def add_fun(a, b):
    return a.add(b, fill_value=0)


# 构建每天的position
def position_daily_fun(df, n=5):
    return df.rolling(window=n, min_periods=1).sum()


def create_fun_set_2(fun_set):
    mix_fun_set = {}
    for fun_1, fun_2 in product(fun_set, repeat=2):
        exe_str_1 = """def {0}_{1}_fun(a, b, c):
            mix_1 = {0}_fun(a, b)
            mix_2 = {1}_fun(mix_1, c)
            return mix_2
        """.format(fun_1.__name__.split('_')[0], fun_2.__name__.split('_')[0])
        exec(compile(exe_str_1, '', 'exec'))
        exec('mix_fun_set[\'{0}_{1}_fun\'] = {0}_{1}_fun'
             .format(fun_1.__name__.split('_')[0], fun_2.__name__.split('_')[0]))
    return mix_fun_set


def AZ_Load_csv(target_path):
    target_df = pd.read_table(target_path, sep='|', index_col=0)
    target_df.index = pd.to_datetime(target_df.index)
    return target_df


# 获取剔除新股的矩阵
def get_new_stock_info(xnms, xinx):
    new_stock_data = AZ_Load_csv(root_path.EM_Tab01.CDSY_SECUCODE / 'LISTSTATE.csv')
    new_stock_data.fillna(method='ffill', inplace=True)
    # 获取交易日信息
    return_df = AZ_Load_csv(root_path.EM_Funda.DERIVED_14 / 'aadj_r.csv').astype(float)
    trade_time = return_df.index
    new_stock_data = new_stock_data.reindex(index=trade_time).fillna(method='ffill')
    target_df = new_stock_data.shift(40).notnull().astype(int)
    target_df = target_df.reindex(columns=xnms, index=xinx)
    return target_df


# 获取剔除st股票的矩阵
def get_st_stock_info(xnms, xinx):
    data = AZ_Load_csv(root_path.EM_Tab01.CDSY_CHANGEINFO / 'CHANGEA.csv')
    data = data.reindex(columns=xnms, index=xinx)
    data.fillna(method='ffill', inplace=True)

    data = data.astype(str)
    target_df = data.applymap(lambda x: 0 if 'ST' in x or 'PT' in x else 1)
    return target_df


# 生成每天的position
def pos_daily_fun(df, n=5):
    return df.rolling(window=n, min_periods=1).sum()


# 读取 sector(行业 最大市值等)
def load_sector_data(sector_name):
    market_top_n = AZ_Load_csv(root_path.EM_Funda.DERIVED_10 / (sector_name + '.csv'))
    # market_top_n = market_top_n.shift(1)
    market_top_n.dropna(how='all', axis='columns', inplace=True)
    xnms = market_top_n.columns
    xinx = market_top_n.index

    new_stock_df = get_new_stock_info(xnms, xinx)
    st_stock_df = get_st_stock_info(xnms, xinx)
    sector_df = market_top_n * new_stock_df * st_stock_df
    sector_df.replace(0, np.nan, inplace=True)
    return sector_df


# 读取 因涨跌停以及停牌等 不能变动仓位的日期信息
def load_locked_data(xnms, xinx):
    raw_suspendday_df = AZ_Load_csv(root_path.EM_Funda.TRAD_TD_SUSPENDDAY / 'SUSPENDREASON.csv')
    suspendday_df = raw_suspendday_df.isnull()
    suspendday_df = suspendday_df.reindex(columns=xnms, index=xinx, fill_value=True)
    suspendday_df.replace(0, np.nan, inplace=True)

    return_df = AZ_Load_csv(root_path.EM_Funda.DERIVED_14 / 'aadj_r.csv').astype(float)
    limit_buy_df = (return_df < 0.095).astype(int)
    limit_buy_df = limit_buy_df.reindex(columns=xnms, index=xinx, fill_value=1)
    limit_buy_df.replace(0, np.nan, inplace=True)

    limit_sell_df = (return_df > -0.095).astype(int)
    limit_sell_df = limit_sell_df.reindex(columns=xnms, index=xinx, fill_value=1)
    limit_sell_df.replace(0, np.nan, inplace=True)
    return suspendday_df, limit_buy_df, limit_sell_df


def load_part_factor(sector_name, xnms, xinx, file_list):
    factor_set = OrderedDict()
    for file_name in file_list:
        target_df = pd.read_pickle(os.path.join(root_factor_path, file_name + '.pkl'))
        target_df.fillna(0, inplace=True)
        factor_set[file_name] = target_df.reindex(columns=xnms, index=xinx)
    return factor_set


def deal_mix_factor_c(mix_factor, sector_df, suspendday_df, limit_buy_df, limit_sell_df, hold_time, lag, if_only_long):
    if if_only_long:
        mix_factor = mix_factor[mix_factor > 0]
    mix_factor.replace(np.nan, 0, inplace=True)
    # 下单日期pos
    order_df = mix_factor.replace(np.nan, 0).shift(lag - 1)
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


if __name__ == '__main__':
    mode = 'pro'
    config_info_path = '/mnt/mfs/alpha_whs/config01.pkl'
    root_path = pt._BinFiles(mode)
    if mode == 'pro':
        root_factor_path = '/media/hdd1/DAT_PreCalc/PreCalc_whs/market_top_500'
    elif mode == 'bkt':
        root_factor_path = '/mnt/mfs/dat_whs/data/new_factor_data/market_top_500'
    else:
        print('mode error !')
        exit()
    return_df = AZ_Load_csv(root_path.EM_Funda.DERIVED_14 / 'aadj_r.csv').astype(float)
    xnms = return_df.columns
    xinx = return_df.index

    config_info = pd.read_pickle(config_info_path)
    factor_info = config_info['factor_info']
    if_hedge = True
    if_only_long = False
    sector_name = 'market_top_500'
    lag = 1
    hold_time = 11
    sector_df = load_sector_data(sector_name)

    suspendday_df, limit_buy_df, limit_sell_df = load_locked_data(xnms, xinx)

    fun_set = [mul_fun, sub_fun, add_fun]
    mix_fun_set = create_fun_set_2(fun_set)

    sum_factor_df = pd.DataFrame()
    for i in range(len(factor_info)):
        fun_name, name1, name2, name3, buy_sell = factor_info.iloc[i]
        fun = mix_fun_set[fun_name]
        try:
            factor_set = load_part_factor(sector_name, xnms, xinx, [name1, name2, name3])
            choose_1 = factor_set[name1]
            choose_2 = factor_set[name2]
            choose_3 = factor_set[name3]
        except:
            print('{} or {} or {} NOT EXIST!'.format(name1, name2, name3))
            continue
        mix_factor = fun(choose_1, choose_2, choose_3)
        if buy_sell == 1:
            sum_factor_df = sum_factor_df.add(mix_factor, fill_value=0)
        elif buy_sell == -1:
            sum_factor_df = sum_factor_df.add(-mix_factor, fill_value=0)
        else:
            print('error!')

    daily_pos = deal_mix_factor_c(sum_factor_df, sector_df, suspendday_df, limit_buy_df, limit_sell_df, hold_time,
                                  lag, if_only_long).round(14)

    daily_pos['IF01'] = daily_pos.sum(axis=1)
    daily_pos.to_csv('/mnt/mfs/AAPOS/WHSMIRANA01.pos', sep='|', index_label='Date')
