import pandas as pd
import numpy as np
import os
from itertools import product, permutations, combinations
from multiprocessing import Pool, Lock, cpu_count
import time
from collections import OrderedDict
import random
from datetime import datetime
import sys

# sys.path.append('/mnt/mfs/work_whs')
# sys.path.append('/mnt/mfs/work_whs/AZ_2018_Q2')
sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *
# 读取数据的函数 以及
from work_whs.AZ_2018_Q2.factor_script.main_file import main_file_return_hedge as mf
import talib as ta


# product 笛卡尔积　　（有放回抽样排列）
# permutations 排列　　（不放回抽样排列）
# combinations 组合,没有重复　　（不放回抽样组合）
# combinations_with_replacement 组合,有重复　　（有放回抽样组合）


# def mul_fun(a, b):
#     return a.mul(b)

def create_sector(root_path, name_list, sector_name, begin_date):
    market_top_n = bt.AZ_Load_csv(os.path.join(root_path, 'EM_Funda/DERIVED_10/' + sector_name + '.csv'))
    market_top_n = market_top_n[(market_top_n.index >= begin_date)]

    sum_df = pd.DataFrame()
    for n in name_list:
        tmp_df = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/LICO_IM_INCHG/Global_Level1_{}.csv'.format(n))
        tmp_df = tmp_df[(tmp_df.index >= begin_date)]
        sum_df = sum_df.add(tmp_df, fill_value=0)

    if sum_df[sum_df > 1].sum().sum() != 0:
        print('error', name_list)
    else:
        market_top_n_sector = market_top_n.mul(sum_df)
        market_top_n_sector.dropna(how='all', axis='columns', inplace=True)
        market_top_n_sector.to_csv('/mnt/mfs/dat_whs/data/sector_data/{}_industry_{}.csv'
                                   .format(sector_name, '_'.join([str(x) for x in name_list])), sep='|')


def mul_fun(a, b):
    a_l = a.where(a > 0, 0)
    a_s = a.where(a < 0, 0)

    b_l = b.where(b > 0, 0)
    b_s = b.where(b < 0, 0)

    pos_l = a_l.mul(b_l)
    pos_s = a_s.mul(b_s)

    pos = pos_l.sub(pos_s)
    return pos


def sub_fun(a, b):
    return a.sub(b)


def add_fun(a, b):
    return a.add(b)


def AZ_Cut_window(df, begin_date, end_date=None, column=None):
    if column is None:
        if end_date is None:
            return df[df.index > begin_date]
        else:
            return df[(df.index > begin_date) & (df.index < end_date)]
    else:
        if end_date is None:
            return df[df[column] > begin_date]
        else:
            return df[(df[column] > begin_date) & (df[column] < end_date)]


def AZ_Leverage_ratio(asset_df):
    """
    返回250天的return/(负的 一个月的return)
    :param asset_df:
    :return:
    """
    asset_20 = asset_df - asset_df.shift(20)
    asset_250 = asset_df - asset_df.shift(250)
    if asset_250.mean() > 0:
        return round(asset_250.mean() / (-asset_20.min()), 2)
    else:
        return round(asset_250.mean() / (-asset_20.max()), 2)


def pos_daily_fun(df, n=5):
    return df.rolling(window=n, min_periods=1).sum()


def AZ_Pot(pos_df_daily, last_asset):
    trade_times = pos_df_daily.diff().abs().sum().sum()
    if trade_times == 0:
        return 0
    else:
        pot = last_asset / trade_times * 10000
        return round(pot, 2)


def create_fun_set_2_(fun_set):
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


def out_sample_perf_c(pnl_df_out, way=1):
    # 根据sharpe大小,统计样本外的表现
    # if cut_point_list is None:
    #     cut_point_list = [0.30]
    # if way == 1:
    #     rolling_sharpe, cut_sharpe = \
    #         bt.AZ_Rolling_sharpe(pnl_df_out, roll_year=0.5, year_len=250, cut_point_list=cut_point_list, output=True)
    # else:
    #     rolling_sharpe, cut_sharpe = \
    #         bt.AZ_Rolling_sharpe(-pnl_df_out, roll_year=0.5, year_len=250, cut_point_list=cut_point_list, output=True)
    if way == 1:
        sharpe_out = bt.AZ_Sharpe_y(pnl_df_out)
    else:
        sharpe_out = bt.AZ_Sharpe_y(-pnl_df_out)
    out_condition = sharpe_out > 0.8
    return out_condition, round(sharpe_out * way, 2)


def filter_all(cut_date, pos_df_daily, pct_n, if_return_pnl=False, if_only_long=False):
    pnl_df = (pos_df_daily * pct_n).sum(axis=1)
    pnl_df = pnl_df.replace(np.nan, 0)

    # pnl_df = pd.Series(pnl_df)
    # 样本内表现
    return_in = pct_n[pct_n.index < cut_date]

    pnl_df_in = pnl_df[pnl_df.index < cut_date]
    asset_df_in = pnl_df_in.cumsum()
    last_asset_in = asset_df_in.iloc[-1]
    pos_df_daily_in = pos_df_daily[pos_df_daily.index < cut_date]
    pot_in = AZ_Pot(pos_df_daily_in, last_asset_in)

    leve_ratio = AZ_Leverage_ratio(asset_df_in)
    if leve_ratio < 0:
        leve_ratio = 100
    sharpe_q_in_df = bt.AZ_Rolling_sharpe(pnl_df_in, roll_year=1, year_len=250, min_periods=1,
                                          cut_point_list=[0.3, 0.5, 0.7], output=False)
    sp_in = bt.AZ_Sharpe_y(pnl_df_in)
    fit_ratio = bt.AZ_fit_ratio(pos_df_daily_in, return_in)
    ic = round(bt.AZ_Normal_IC(pos_df_daily_in, pct_n, min_valids=None, lag=0).mean(), 6)
    sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d = sharpe_q_in_df.values
    in_condition_u = sharpe_q_in_df_u > 0.9 and leve_ratio > 1
    in_condition_d = sharpe_q_in_df_d < -0.9 and leve_ratio > 1
    # 分双边和只做多
    if if_only_long:
        in_condition = in_condition_u
    else:
        in_condition = in_condition_u | in_condition_d

    if sharpe_q_in_df_m > 0:
        way = 1
    else:
        way = -1

    # 样本外表现
    pnl_df_out = pnl_df[pnl_df.index >= cut_date]
    out_condition, sharpe_q_out = out_sample_perf_c(pnl_df_out, way=way)
    if if_return_pnl:
        return in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
               fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df
    else:
        return in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
               fit_ratio, leve_ratio, sp_in, sharpe_q_out


def filter_time_para_fun(time_para_dict, pos_df_daily, adj_return, if_return_pnl=False, if_only_long=False):
    pnl_df = (pos_df_daily * adj_return).sum(axis=1)

    pnl_df = pnl_df.replace(np.nan, 0)
    result_dict = OrderedDict()
    for time_key in time_para_dict.keys():
        begin_para, cut_para, end_para_1, end_para_2, end_para_3, end_para_4 = time_para_dict[time_key]

        # 样本内索引
        sample_in_index = (adj_return.index >= begin_para) & (adj_return.index < cut_para)
        # 样本外索引
        sample_out_index_1 = (adj_return.index >= cut_para) & (adj_return.index < end_para_1)
        sample_out_index_2 = (adj_return.index >= cut_para) & (adj_return.index < end_para_2)
        sample_out_index_3 = (adj_return.index >= cut_para) & (adj_return.index < end_para_3)
        sample_out_index_4 = (adj_return.index >= cut_para) & (adj_return.index < end_para_4)
        # 样本内表现
        pos_df_daily_in = pos_df_daily[sample_in_index]
        if len(pos_df_daily_in.abs().sum(axis=1).replace(0, np.nan).dropna()) / len(pos_df_daily_in) < 0.1:
            continue
        adj_return_in = adj_return[sample_in_index]
        pnl_df_in = pnl_df[sample_in_index]

        asset_df_in = pnl_df_in.cumsum()
        last_asset_in = asset_df_in.iloc[-1]

        pot_in = AZ_Pot(pos_df_daily_in, last_asset_in)

        leve_ratio = AZ_Leverage_ratio(asset_df_in)

        if leve_ratio < 0:
            leve_ratio = 100
        sharpe_q_in_df = bt.AZ_Rolling_sharpe(pnl_df_in, roll_year=1, year_len=250, min_periods=1,
                                              cut_point_list=[0.3, 0.5, 0.7], output=False)
        sharpe_q_in_df = round(sharpe_q_in_df, 4)
        sp_in = bt.AZ_Sharpe_y(pnl_df_in)
        fit_ratio = bt.AZ_fit_ratio(pos_df_daily_in, adj_return_in)

        ic = round(bt.AZ_Normal_IC(pos_df_daily_in, adj_return_in, min_valids=None, lag=0).mean(), 6)
        sp_in_u, sp_in_m, sp_in_d = sharpe_q_in_df.values

        in_condition_u = sp_in_u > 0.9 and leve_ratio > 1
        in_condition_d = sp_in_d < -0.9 and leve_ratio > 1
        # 分双边和只做多
        if if_only_long:
            in_condition = in_condition_u
        else:
            in_condition = in_condition_u | in_condition_d

        if sp_in_m > 0:
            way = 1
        else:
            way = -1

        # 样本外表现
        pnl_df_out_1 = pnl_df[sample_out_index_1]
        pnl_df_out_2 = pnl_df[sample_out_index_2]
        pnl_df_out_3 = pnl_df[sample_out_index_3]
        pnl_df_out_4 = pnl_df[sample_out_index_4]

        out_condition_1, sp_out_1 = out_sample_perf_c(pnl_df_out_1, way=way)
        out_condition_2, sp_out_2 = out_sample_perf_c(pnl_df_out_2, way=way)
        out_condition_3, sp_out_3 = out_sample_perf_c(pnl_df_out_3, way=way)
        out_condition_4, sp_out_4 = out_sample_perf_c(pnl_df_out_4, way=way)
        if if_return_pnl:
            result_dict[time_key] = [in_condition, out_condition_1, out_condition_2, out_condition_3, out_condition_4,
                                     ic, sp_in_u, sp_in_m, sp_in_d, pot_in, fit_ratio, leve_ratio,
                                     sp_in, sp_out_1, sp_out_2, sp_out_3, sp_out_4, pnl_df]
        else:
            result_dict[time_key] = [in_condition, out_condition_1, out_condition_2, out_condition_3, out_condition_4,
                                     ic, sp_in_u, sp_in_m, sp_in_d, pot_in, fit_ratio, leve_ratio,
                                     sp_in, sp_out_1, sp_out_2, sp_out_3, sp_out_4]
    return result_dict


def ma_ls_fun(df, n_short=5, n_long=100):
    ma_long = df.rolling(window=n_long, min_periods=0).mean()
    ma_short = df.rolling(window=n_short, min_periods=0).mean()
    ma_dif = ma_short - ma_long
    ma_dif[ma_dif == 0] = 0
    ma_dif[ma_dif > 0] = 1
    ma_dif[ma_dif < 0] = -1
    return ma_dif


def BBANDS(Close, timeperiod, limit_up_down):
    def BBANDS_raw(Close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0):
        upperband, middleband, lowerband = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for i in Close.columns:
            upperband[i], middleband[i], lowerband[i] = ta.BBANDS(Close[i], timeperiod, nbdevup, nbdevdn, matype)
        upperband.replace(np.nan, 0, inplace=True)
        middleband.replace(np.nan, 0, inplace=True)
        lowerband.replace(np.nan, 0, inplace=True)
        return upperband, middleband, lowerband

    up_line, mid_line, down_line = BBANDS_raw(Close, timeperiod, nbdevup=limit_up_down,
                                              nbdevdn=limit_up_down, matype=0)
    target_df = Close.copy()
    target_df.iloc[:, :] = 0
    target_df[(Close <= up_line) & (Close >= down_line)] = 0
    target_df[Close > up_line] = 1
    target_df[Close < down_line] = -1
    return target_df


def main_fun(sector_name, hold_time, if_only_long, time_para_dict):
    root_path = '/mnt/mfs/DAT_EQT'
    if_save = False
    if_new_program = True

    begin_date = pd.to_datetime('20100101')
    cut_date = pd.to_datetime('20160401')
    end_date = datetime.now()
    lag = 2
    return_file = ''

    if_hedge = True
    # if_only_long = False

    if sector_name.startswith('market_top_300plus') \
            or sector_name.startswith('index_000300'):
        if_weight = 1
        ic_weight = 0

    elif sector_name.startswith('market_top_300to800plus') \
            or sector_name.startswith('index_000905'):
        if_weight = 0
        ic_weight = 1

    else:
        if_weight = 0.5
        ic_weight = 0.5

    main = mf.FactorTest(root_path, if_save, if_new_program, begin_date, cut_date, end_date, time_para_dict,
                         sector_name, hold_time, lag, return_file, if_hedge, if_only_long, if_weight, ic_weight)
    main.load_index_data('000300')


def load_index_data(index_name):
    data = bt.AZ_Load_csv(os.path.join('/mnt/mfs/DAT_EQT', 'EM_Funda/INDEX_TD_DAILYSYS/CHG.csv'))
    target_df = data[index_name]
    return target_df * 0.01


if __name__ == '__main__':
    begin_date = pd.to_datetime('20100101')
    end_date = datetime.now()

    hedge_df = load_index_data('000905')
    hedge_df = hedge_df[(hedge_df.index < end_date) & (hedge_df.index > begin_date)]
    hedge_price = hedge_df.cumsum()
    hedge_price = pd.DataFrame(hedge_price, columns=['000300'])
    signals_BBANDS = BBANDS(hedge_price, 5, 1)
    signals_ma = ma_ls_fun(hedge_price, n_short=5, n_long=100)
    market_up = signals_ma[signals_ma == 1].dropna()

    Mon_return = hedge_df[market_up.index[market_up.index.weekday == 0]].sum()
    Tue_return = hedge_df[market_up.index[market_up.index.weekday == 1]].sum()
    Wed_return = hedge_df[market_up.index[market_up.index.weekday == 2]].sum()
    Thu_return = hedge_df[market_up.index[market_up.index.weekday == 3]].sum()
    Fri_return = hedge_df[market_up.index[market_up.index.weekday == 4]].sum()

    print(f'Mon_return = {Mon_return}')
    print(f'Tue_return = {Tue_return}')
    print(f'Wed_return = {Wed_return}')
    print(f'Thu_return = {Thu_return}')
    print(f'Fri_return = {Fri_return}')
