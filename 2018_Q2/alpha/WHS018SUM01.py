import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from itertools import product, permutations, combinations
import os
from collections import OrderedDict
import warnings
import sys
sys.path.append("/mnt/mfs/LIB_ROOT")
import open_lib.shared_paths.path as pt
from open_lib.shared_tools import send_email
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
def get_new_stock_info(xnms, xinx, root_path):
    new_stock_data = AZ_Load_csv(root_path.EM_Tab01.CDSY_SECUCODE/'LISTSTATE.csv')
    new_stock_data.fillna(method='ffill', inplace=True)
    # 获取交易日信息
    return_df = AZ_Load_csv(root_path.EM_Funda.DERIVED_14/'aadj_r.csv').astype(float)
    trade_time = return_df.index
    new_stock_data = new_stock_data.reindex(index=trade_time).fillna(method='ffill')
    target_df = new_stock_data.shift(40).notnull().astype(int)
    target_df = target_df.reindex(columns=xnms, index=xinx)
    return target_df


# 获取剔除st股票的矩阵
def get_st_stock_info(xnms, xinx, root_path):
    data = AZ_Load_csv(root_path.EM_Tab01.CDSY_CHANGEINFO/'CHANGEA.csv')
    data = data.reindex(columns=xnms, index=xinx)
    data.fillna(method='ffill', inplace=True)

    data = data.astype(str)
    target_df = data.applymap(lambda x: 0 if 'ST' in x or 'PT' in x else 1)
    return target_df


# 生成每天的position
def pos_daily_fun(df, n=5):
    return df.rolling(window=n, min_periods=1).sum()


# 读取 sector(行业 最大市值等)
def load_sector_data(sector_name, root_path):
    market_top_n = AZ_Load_csv(root_path.EM_Funda.DERIVED_10 / (sector_name + '.csv'))
    market_top_n.dropna(how='all', axis='columns', inplace=True)
    xnms = market_top_n.columns
    xinx = market_top_n.index

    new_stock_df = get_new_stock_info(xnms, xinx, root_path)
    st_stock_df = get_st_stock_info(xnms, xinx, root_path)
    sector_df = market_top_n * new_stock_df * st_stock_df
    sector_df.replace(0, np.nan, inplace=True)
    return sector_df


# 涨跌停都不可交易
def load_locked_data(xnms, xinx, root_path):
    raw_suspendday_df = AZ_Load_csv(root_path.EM_Funda.TRAD_TD_SUSPENDDAY / 'SUSPENDREASON.csv')
    suspendday_df = raw_suspendday_df.isnull()
    suspendday_df = suspendday_df.reindex(columns=xnms, index=xinx, fill_value=True)
    suspendday_df.replace(0, np.nan, inplace=True)

    return_df = AZ_Load_csv(root_path.EM_Funda.DERIVED_14/'aadj_r.csv').astype(float)
    limit_buy_sell_df = (return_df.abs() < 0.095).astype(int)
    limit_buy_sell_df = limit_buy_sell_df.reindex(columns=xnms, index=xinx, fill_value=1)
    limit_buy_sell_df.replace(0, np.nan, inplace=True)
    return suspendday_df, limit_buy_sell_df


# 读取部分factor
def load_part_factor(sector_name, xnms, xinx, root_factor_path, file_list):
    factor_set = OrderedDict()
    for file_name in file_list:
        target_df = pd.read_pickle(os.path.join(root_factor_path, file_name + '.pkl'))
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
    order_df = mix_factor.replace(np.nan, 0)

    limit_buy_sell_df_c = limit_buy_sell_df.shift(-1)
    limit_buy_sell_df_c.iloc[-1] = 1

    suspendday_df_c = suspendday_df.shift(-1)
    suspendday_df_c.iloc[-1] = 1
    # 排除入场场涨跌停的影响
    order_df = order_df * sector_df * limit_buy_sell_df_c * suspendday_df_c

    daily_pos = pos_daily_fun(order_df, n=hold_time)
    # 排除出场涨跌停的影响
    daily_pos = daily_pos * limit_buy_sell_df * suspendday_df
    daily_pos.fillna(method='ffill', inplace=True)
    # 获得最终仓位信息
    return daily_pos


if __name__ == '__main__':
    a = time.time()
    mode = 'pro'
    root_path = pt._BinFiles(mode)
    sector_name = 'market_top_2000'
    alpha_name_list = ['018JUN', '018JUL', '018AUG']
    config_info_path = [f'/mnt/mfs/alpha_whs/{alpha_name_list[0]}.pkl',
                        f'/mnt/mfs/alpha_whs/{alpha_name_list[1]}.pkl',
                        f'/mnt/mfs/alpha_whs/{alpha_name_list[2]}.pkl']

    if mode == 'pro':
        root_factor_path = '/media/hdd1/DAT_PreCalc/PreCalc_whs/tmp/{}'.format(sector_name)
    elif mode == 'bkt':
        root_factor_path = '/mnt/mfs/dat_whs/data/new_factor_data/{}'.format(sector_name)
    else:
        print('mode error !')
        exit()

    end_date = datetime.now()
    begin_date = pd.to_datetime('20120101')

    return_df = AZ_Load_csv(root_path.EM_Funda.DERIVED_14 / 'aadj_r.csv').astype(float)
    return_df = return_df[(return_df.index > begin_date)]

    xnms = return_df.columns
    xinx = return_df.index

    config_info1 = pd.read_pickle(config_info_path[0])
    config_info2 = pd.read_pickle(config_info_path[1])
    config_info3 = pd.read_pickle(config_info_path[2])

    factor_info1 = config_info1['factor_info']
    factor_info2 = config_info2['factor_info']
    factor_info3 = config_info3['factor_info']

    if_hedge = True
    if_only_long = False
    lag = 2
    hold_time = 20

    sector_df = load_sector_data(sector_name, root_path)
    suspendday_df, limit_buy_sell_df = load_locked_data(xnms, xinx, root_path)

    fun_set = [mul_fun, sub_fun, add_fun]
    mix_fun_set = create_fun_set_2(fun_set)

    factor_info = factor_info1.append(factor_info2.append(factor_info3))
    print(len(factor_info))
    sum_factor_df = pd.DataFrame()
    for i in range(len(factor_info)):
        fun_name, name1, name2, name3, buy_sell = factor_info.iloc[i]
        fun = mix_fun_set[fun_name]
        try:
            factor_set = load_part_factor(sector_name, xnms, xinx, root_factor_path, [name1, name2, name3])
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
    daily_pos = deal_mix_factor(sum_factor_df, sector_df, suspendday_df, limit_buy_sell_df, hold_time, lag,
                                if_only_long).round(14)

    daily_pos['IC01'] = -daily_pos.sum(axis=1)
    daily_pos.round(10).to_csv(f'/mnt/mfs/work_whs/WHS018SUM01.pos', sep='|', index_label='Date')
    b = time.time()
    print('alpha WHS018SUM01 cost time: {}'.format(b - a))
