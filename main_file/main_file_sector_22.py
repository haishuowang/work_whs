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

sys.path.append('/mnt/mfs/work_whs')
sys.path.append('/mnt/mfs/work_whs/AZ_2018_Q2')
import loc_lib.shared_tools.back_test as bt
from loc_lib.shared_tools import send_email
# 读取数据的函数 以及
from work_whs.main_file import main_file_return_hedge as mf


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


def create_fun_set_2():
    fun_set = [add_fun, sub_fun, mul_fun]
    mix_fun_set = []
    for fun_1, fun_2 in product(fun_set, repeat=2):
        exe_str_1 = """def {0}_{1}_fun(a, b, c):
            mix_1 = {0}_fun(a, b)
            mix_2 = {1}_fun(mix_1, c)
            return mix_2
        """.format(fun_1.__name__.split('_')[0], fun_2.__name__.split('_')[0])
        exec(compile(exe_str_1, '', 'exec'))
        exec('mix_fun_set += [{0}_{1}_fun]'.format(fun_1.__name__.split('_')[0], fun_2.__name__.split('_')[0]))
    return mix_fun_set


def create_fun_set_2_crt():
    fun_2 = mul_fun
    mix_fun_set = []
    for fun_1 in [add_fun, sub_fun, mul_fun]:
        exe_str_1 = """def {0}_{1}_fun(a, b, c):
                mix_1 = {0}_fun(a, b)
                mix_2 = {1}_fun(mix_1, c)
                return mix_2
            """.format(fun_1.__name__.split('_')[0], fun_2.__name__.split('_')[0])
        exec(compile(exe_str_1, '', 'exec'))
        exec('mix_fun_set += [{0}_{1}_fun]'.format(fun_1.__name__.split('_')[0], fun_2.__name__.split('_')[0]))
    return mix_fun_set


def create_fun_set_2_crt_():
    fun_2 = mul_fun
    mix_fun_set = dict()
    for fun_1 in [add_fun, sub_fun, mul_fun]:
        exe_str_1 = """def {0}_{1}_fun(a, b, c):
                mix_1 = {0}_fun(a, b)
                mix_2 = {1}_fun(mix_1, c)
                return mix_2
            """.format(fun_1.__name__.split('_')[0], fun_2.__name__.split('_')[0])
        exec(compile(exe_str_1, '', 'exec'))
        exec('mix_fun_set[\'{0}_{1}_fun\'] = {0}_{1}_fun'
             .format(fun_1.__name__.split('_')[0], fun_2.__name__.split('_')[0]))
    return mix_fun_set


class FactorTestSector(mf.FactorTest):
    def __init__(self, *args):
        super(FactorTestSector, self).__init__(*args)

    def load_jerry_factor(self, file_name):
        factor_path = '/mnt/mfs/temp/dat_jerry/signal'
        raw_df = bt.AZ_Load_csv(f'{factor_path}/{file_name}')
        a = list(set(raw_df.iloc[-1, :100].dropna().values))
        tmp_df = raw_df.reindex(index=self.xinx, columns=self.xnms)
        if len(a) > 5:
            target_df = self.row_extre(tmp_df, self.sector_df, 0.3)
        else:
            target_df = tmp_df
        if self.if_only_long:
            target_df = target_df[target_df > 0]
        return target_df

    def load_intra_factor(self, file_name):
        factor_path = '/mnt/mfs/dat_whs/EM_Funda/dat_whs'
        raw_df = bt.AZ_Load_csv(f'{factor_path}/{file_name}')
        tmp_df = raw_df.reindex(index=self.xinx, columns=self.xnms)
        target_df = self.row_extre(tmp_df, self.sector_df, 0.3)
        return target_df

    def part_test_index_3_(self, key, name_1, name_2, name_3, log_save_file, result_save_file, total_para_num):
        lock = Lock()
        start_time = time.time()
        load_time_1 = time.time()
        # load因子,同时根据stock_universe筛选数据.
        factor_1 = self.load_intra_factor(name_1)
        factor_2 = self.load_intra_factor(name_2)
        factor_3 = self.load_intra_factor(name_3)

        load_time_2 = time.time()
        # 加载花费数据时间
        load_delta = round(load_time_2 - load_time_1, 2)
        # 生成混合函数集
        fun_mix_2_set = create_fun_set_2()
        #################
        # 更换filter函数 #
        #################
        filter_name = filter_time_para_fun.__name__

        for fun in fun_mix_2_set:
            mix_factor = fun(factor_1, factor_2, factor_3)
            if len(mix_factor.abs().sum(axis=1).replace(0, np.nan).dropna()) / len(mix_factor) < 0.5:
                continue

            daily_pos = self.deal_mix_factor(mix_factor).shift(2)
            # 返回样本内筛选结果

            result_dict = filter_time_para_fun(self.time_para_dict, daily_pos, self.return_choose,
                                               if_only_long=self.if_only_long)
            for time_key in result_dict.keys():
                in_condition, *filter_result = result_dict[time_key]
                # result 存储
                if in_condition:
                    if self.if_save:
                        with lock:
                            f = open(result_save_file, 'a')
                            write_list = [time_key, key, fun.__name__, name_1, name_2, name_3, filter_name,
                                          self.sector_name, in_condition] + filter_result
                            f.write('|'.join([str(x) for x in write_list]) + '\n')
                    print([time_key, in_condition, fun.__name__, name_1, name_2, name_3] + filter_result)
        end_time = time.time()
        # 参数存储
        if self.if_save:
            with lock:
                f = open(log_save_file, 'a')
                write_list = [key, name_1, name_2, name_3, filter_name, self.sector_name,
                              round(end_time - start_time, 4),
                              load_delta]
                f.write('|'.join([str(x) for x in write_list]) + '\n')

        print('{}%, {}, {}, {}, {}, cost {} seconds, load_cost {} seconds'
              .format(round(key / total_para_num * 100, 4), key, name_1, name_2, name_3,
                      round(end_time - start_time, 2), load_delta))

    def test_index_3_(self, list_1, list_2, list_3, pool_num=20, suffix_name='', old_file_name=''):
        print(1)
        para_ready_df, log_save_file, result_save_file, total_para_num = \
            self.save_load_control_(list_1, list_2, list_3, suffix_name, old_file_name)
        print(2)
        # self.check_factor(list_3, result_save_file)

        a_time = time.time()
        pool = Pool(pool_num)
        for key in list(para_ready_df.index):
            name_1, name_2, name_3 = para_ready_df.loc[key]
            args_list = (key, name_1, name_2, name_3, log_save_file, result_save_file, total_para_num)
            # self.part_test_index_3_(*args_list)
            pool.apply_async(self.part_test_index_3_, args=args_list)
        pool.close()
        pool.join()

        b_time = time.time()
        print('Success!Processing end, Cost {} seconds'.format(round(b_time - a_time, 2)))

    def single_test(self, fun_name, name1, name2, name3):
        fun_set = [add_fun, sub_fun, mul_fun]
        fun_mix_2_set = create_fun_set_2_(fun_set)
        fun = fun_mix_2_set[fun_name]

        factor_1 = self.load_jerry_factor(name1)
        factor_2 = self.load_jerry_factor(name2)
        factor_3 = self.load_jerry_factor(name3)
        mix_factor = fun(factor_1, factor_2, factor_3)
        daily_pos = self.deal_mix_factor(mix_factor).shift(2)
        in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
        fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df = \
            filter_all(self.cut_date, daily_pos, self.return_choose, if_return_pnl=True, if_only_long=False)
        return mix_factor, in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, \
               pot_in, fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df


def get_file_name(sector_name):
    tmp_file_list = os.listdir(f'/media/hdd2/dat_whs/data/single_factor_pnl/{sector_name}')
    target_file_list = [x.split('|') for x in tmp_file_list if '|' in x]
    target_file_list = [x for x in target_file_list if x[0].startswith('REM')]
    data = pd.DataFrame(target_file_list, columns=['factor_name', 'sector_name', 'hold_time', 'if_only_long'])
    return data


def get_pnl_table(part_df, sector_name):
    file_name_df = part_df.apply(lambda x: '|'.join(x), axis=1)
    all_pnl_df = pd.DataFrame()
    for file_name in file_name_df:
        # print(file_name)
        pnl_df = pd.read_csv(f'/media/hdd2/dat_whs/data/single_factor_pnl/{sector_name}/{file_name}',
                             index_col=0, parse_dates=True)
        pnl_df.columns = [file_name]
        all_pnl_df = pd.concat([all_pnl_df, pnl_df], axis=1)
    all_pnl_df = all_pnl_df.loc[pd.to_datetime('20130101'):]
    return all_pnl_df


def corr_matrix(corr_df):
    corr_df[(corr_df > 0.8) | (corr_df < -0.8)] = 1
    corr_df[(corr_df < 0.8) & (corr_df > -0.8)] = 0
    all_factor_set = set(corr_df.index)

    while True:
        corr_num_sort = corr_df.sum().sort_values()
        # print(corr_num_sort)
        b = corr_num_sort.iloc[-1]

        factor_name = corr_num_sort.index[-1]
        if b > 1:
            corr_df = corr_df.drop(factor_name, axis=1).drop(factor_name, axis=0)
            # print(corr_df)
        else:
            break
    corr_num_sort = corr_df.sum().sort_values()
    use_factor_set = set(corr_df.index)
    print(len(corr_df.index))
    # delete_set = all_factor_set - use_factor_set
    return use_factor_set


def main_fun(sector_name, hold_time, if_only_long, time_para_dict):
    root_path = '/mnt/mfs/DAT_EQT'
    if_save = True
    if_new_program = True

    begin_date = pd.to_datetime('20100101')
    cut_date = pd.to_datetime('20160401')
    end_date = pd.to_datetime('20180901')
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

    main = FactorTestSector(root_path, if_save, if_new_program, begin_date, cut_date, end_date, time_para_dict,
                            sector_name, hold_time, lag, return_file, if_hedge, if_only_long, if_weight, ic_weight)
    pool_num = 20
    suffix_name = os.path.basename(__file__).split('.')[0].split('_')[-1]

    factor_list = os.listdir('/mnt/mfs/dat_whs/EM_Funda/dat_whs')

    main.test_index_3_(factor_list, factor_list, factor_list, pool_num, suffix_name=suffix_name)


time_para_dict = OrderedDict()

time_para_dict['time_para_1'] = [pd.to_datetime('20100101'), pd.to_datetime('20150101'),
                                 pd.to_datetime('20150401'), pd.to_datetime('20150701'),
                                 pd.to_datetime('20151001'), pd.to_datetime('20160101')]

time_para_dict['time_para_2'] = [pd.to_datetime('20110101'), pd.to_datetime('20160101'),
                                 pd.to_datetime('20160401'), pd.to_datetime('20160701'),
                                 pd.to_datetime('20161001'), pd.to_datetime('20170101')]

time_para_dict['time_para_3'] = [pd.to_datetime('20120601'), pd.to_datetime('20170601'),
                                 pd.to_datetime('20170901'), pd.to_datetime('20171201'),
                                 pd.to_datetime('20180301'), pd.to_datetime('20180601')]

time_para_dict['time_para_4'] = [pd.to_datetime('20130601'), pd.to_datetime('20180601'),
                                 pd.to_datetime('20181101'), pd.to_datetime('20181101'),
                                 pd.to_datetime('20181101'), pd.to_datetime('20181101')]

time_para_dict['time_para_5'] = [pd.to_datetime('20130701'), pd.to_datetime('20180701'),
                                 pd.to_datetime('20181101'), pd.to_datetime('20181101'),
                                 pd.to_datetime('20181101'), pd.to_datetime('20181101')]

time_para_dict['time_para_6'] = [pd.to_datetime('20130101'), pd.to_datetime('20180329'),
                                 pd.to_datetime('20190329'), pd.to_datetime('20190329'),
                                 pd.to_datetime('20190329'), pd.to_datetime('20190329')]

if __name__ == '__main__':
    sector_name_list = [
        'index_000300',
        'index_000905',
        'market_top_300plus',
        'market_top_300plus_industry_10_15',
        'market_top_300plus_industry_20_25_30_35',
        'market_top_300plus_industry_40',
        'market_top_300plus_industry_45_50',
        'market_top_300plus_industry_55',

        'market_top_300to800plus',
        'market_top_300to800plus_industry_10_15',
        'market_top_300to800plus_industry_20_25_30_35',
        'market_top_300to800plus_industry_40',
        'market_top_300to800plus_industry_45_50',
        'market_top_300to800plus_industry_55'
    ]

    hold_time_list = [1, 5, 10]
    for if_only_long in [False, True]:
        for hold_time in hold_time_list:
            for sector_name in sector_name_list:
                main_fun(sector_name, hold_time, if_only_long, time_para_dict)
