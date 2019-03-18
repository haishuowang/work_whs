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

    # def load_return_data(self):
    #     return_choose = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/intraday/vwap_return/aadj_r0940_1000.csv')
    #     return_choose = return_choose[(return_choose.index >= self.begin_date)
    #                                   & (return_choose.index < self.end_date)]
    #     return return_choose

    # def load_index_data(self, index_name):
    #     data = bt.AZ_Load_csv(f'/mnt/mfs/DAT_EQT/intraday/vwap_return/index_{index_name}_0940_1000.csv')
    #     target_df = data.iloc[:, 0].reindex(index=self.xinx)
    #     return target_df * 0.01

    def load_tech_factor(self, file_name):
        load_path = os.path.join('/mnt/mfs/dat_whs/data/new_factor_data/' + self.sector_name)
        target_df = pd.read_pickle(os.path.join(load_path, file_name + '.pkl')) \
            .reindex(index=self.xinx, columns=self.xnms)
        if self.if_only_long:
            target_df = target_df[target_df > 0]
        return target_df

    def load_daily_factor(self, file_name):
        load_path = '/mnt/mfs/DAT_EQT/EM_Funda/daily/'
        tmp_df = bt.AZ_Load_csv(os.path.join(load_path, file_name + '.csv')) \
            .reindex(index=self.xinx, columns=self.xnms)

        target_df = self.row_extre(tmp_df, self.sector_df, 0.3)
        if self.if_only_long:
            target_df = target_df[target_df > 0]
        return target_df

    def load_whs_factor(self, file_name):
        load_path = '/mnt/mfs/DAT_EQT/EM_Funda/dat_whs/'
        tmp_df = bt.AZ_Load_csv(os.path.join(load_path, file_name + '.csv')) \
            .reindex(index=self.xinx, columns=self.xnms)
        tmp_df = tmp_df.fillna(method='ffill')
        target_df = self.row_extre(tmp_df, self.sector_df, 0.3)
        if self.if_only_long:
            target_df = target_df[target_df > 0]
        return target_df

    def load_remy_factor(self, file_name):
        load_path = '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_F1'
        raw_df = bt.AZ_Load_csv(f'{load_path}/{file_name}')
        a = list(set(raw_df.iloc[-1, :100].dropna().values))
        tmp_df = raw_df.reindex(index=self.xinx, columns=self.xnms)
        if len(a) > 5:
            target_df = self.row_extre(tmp_df, self.sector_df, 0.3)
        else:
            target_df = tmp_df
            pass
        if self.if_only_long:
            target_df = target_df[target_df > 0]
        return target_df

    def load_jerry_factor(self, file_name):
        factor_path = '/mnt/mfs/temp/dat_jerry/signal'
        raw_df = bt.AZ_Load_csv(f'{factor_path}/{file_name}')
        a = list(set(raw_df.iloc[-1, :100].dropna().values))
        tmp_df = raw_df.reindex(index=self.xinx, columns=self.xnms)
        if len(a) > 5:
            target_df = self.row_extre(tmp_df, self.sector_df, 0.3)
        else:
            target_df = tmp_df
            pass
        if self.if_only_long:
            target_df = target_df[target_df > 0]
        return target_df

    def save_load_control_single(self, factor_list, suffix_name, file_name):
        # 参数存储与加载的路径控制
        result_save_path = '/mnt/mfs/dat_whs/result'
        if self.if_new_program:
            now_time = datetime.now().strftime('%Y%m%d_%H%M')
            if self.if_only_long:
                file_name = '{}_{}_{}_hold_{}_{}_{}_long.txt' \
                    .format(self.sector_name, self.if_hedge, now_time, self.hold_time, self.return_file, suffix_name)
            else:
                file_name = '{}_{}_{}_hold_{}_{}_{}.txt' \
                    .format(self.sector_name, self.if_hedge, now_time, self.hold_time, self.return_file, suffix_name)

            log_save_file = os.path.join(result_save_path, 'log', file_name)
            result_save_file = os.path.join(result_save_path, 'result', file_name)
            para_save_file = os.path.join(result_save_path, 'para', file_name)
            para_dict = dict()
            para_ready_df = pd.DataFrame(factor_list)
            total_para_num = len(para_ready_df)
            if self.if_save:
                self.create_log_save_path(log_save_file)
                self.create_log_save_path(result_save_file)
                self.create_log_save_path(para_save_file)
                para_dict['para_ready_df'] = para_ready_df
                para_dict['factor_list'] = factor_list
                pd.to_pickle(para_dict, para_save_file)

        else:
            log_save_file = os.path.join(result_save_path, 'log', file_name)
            result_save_file = os.path.join(result_save_path, 'result', file_name)
            para_save_file = os.path.join(result_save_path, 'para', file_name)
            para_tested_df = pd.read_table(log_save_file, sep='|', header=None, index_col=0)
            para_all_df = pd.read_pickle(para_save_file)
            total_para_num = len(para_all_df)
            para_ready_df = para_all_df.loc[sorted(list(set(para_all_df.index) - set(para_tested_df.index)))]
        print(file_name)
        print(f'para_num:{len(para_ready_df)}')
        return para_ready_df, log_save_file, result_save_file, total_para_num

    def part_test_index(self, key, name_1, log_save_file, result_save_file,
                        total_para_num, my_factor_dict, filter_data):
        lock = Lock()
        start_time = time.time()
        load_time_1 = time.time()
        # load因子,同时根据stock_universe筛选数据.
        # factor_1 = getattr(self, my_factor_dict[name_1])(name_1)
        factor_1 = getattr(self, my_factor_dict[name_1])(name_1)
        f_1_filter = factor_1 * filter_data
        # factor_1 = self.load_tech_factor(name_1)

        load_time_2 = time.time()
        # 加载花费数据时间
        load_delta = round(load_time_2 - load_time_1, 2)
        #################
        # 更换filter函数 #
        #################

        filter_name = filter_all.__name__
        if len(f_1_filter.abs().sum(axis=1).replace(0, np.nan).dropna()) / len(f_1_filter) < 0.5:
            return -1
        daily_pos_raw = self.deal_mix_factor(factor_1).shift(2)
        daily_pos = self.deal_mix_factor(f_1_filter).shift(2)
        # 返回样本内筛选结果
        *result_list_raw, pnl_df_raw = filter_all(self.cut_date, daily_pos_raw, self.return_choose,
                                                  if_return_pnl=True, if_only_long=self.if_only_long)

        *result_list, pnl_df = filter_all(self.cut_date, daily_pos, self.return_choose,
                                          if_return_pnl=True, if_only_long=self.if_only_long)

        pnl_save_path = f'/media/hdd1/dat_whs/data/single_factor_pnl/{self.sector_name}'
        bt.AZ_Path_create(pnl_save_path)
        pnl_df.to_csv(f'{pnl_save_path}/{name_1}|{self.sector_name}|{self.hold_time}|{if_only_long}')
        if abs(bt.AZ_Sharpe_y(pnl_df)) > 0.7:
            plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df),
                             f'{name_1}|{self.sector_name}|{self.hold_time}|{self.if_only_long}',
                             text='|'.join([str(x) for x in result_list]))

        in_condition, *filter_result = result_list
        # result 存储
        if self.if_save:
            with lock:
                f = open(result_save_file, 'a')
                write_list = [key, name_1, filter_name,
                              self.sector_name, in_condition] + filter_result
                f.write('|'.join([str(x) for x in write_list]) + '\n')
        print([in_condition, name_1] + filter_result)

        end_time = time.time()
        # 参数存储
        if self.if_save:
            with lock:
                f = open(log_save_file, 'a')
                write_list = [key, name_1, filter_name, self.sector_name,
                              round(end_time - start_time, 4),
                              load_delta]
                f.write('|'.join([str(x) for x in write_list]) + '\n')

        print('{}%, {}, cost {} seconds, load_cost {} seconds'
              .format(round(key / total_para_num * 100, 4), key, name_1, round(end_time - start_time, 2), load_delta))

    def test_index(self, my_factor_dict, pool_num=20, suffix_name='', old_file_name='', filter_data=1):
        factor_list = list(my_factor_dict.keys())
        # print(factor_list)
        para_ready_df, log_save_file, result_save_file, total_para_num = \
            self.save_load_control_single(factor_list, suffix_name, old_file_name)
        # self.check_factor(factor_list, result_save_file)
        a_time = time.time()
        pool = Pool(pool_num)
        for key in list(para_ready_df.index):
            name_1 = para_ready_df.loc[key][0]
            args_list = (key, name_1, log_save_file, result_save_file, total_para_num, my_factor_dict, filter_data)
            # self.part_test_index(*args_list)
            pool.apply_async(self.part_test_index, args=args_list)
        pool.close()
        pool.join()

        b_time = time.time()
        print('Success!Processing end, Cost {} seconds'.format(round(b_time - a_time, 2)))

    def single_test(self, name_1):
        factor_1 = getattr(self, my_factor_dict[name_1])(name_1)
        daily_pos = self.deal_mix_factor(factor_1).shift(2)
        in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
        fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df = filter_all(self.cut_date, daily_pos, self.return_choose,
                                                                        if_return_pnl=True,
                                                                        if_only_long=self.if_only_long)
        result_list = [in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d,
                       pot_in, fit_ratio, leve_ratio, sp_in, sharpe_q_out]
        plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df),
                         f'{name_1}|{self.sector_name}|{self.hold_time}|{self.if_only_long}',
                         text='|'.join([str(x) for x in result_list]))

        return in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
               fit_ratio, leve_ratio, sp_in, sharpe_q_out

    def single_test_c(self, name_list, buy_sell_way_list):
        mix_factor = pd.DataFrame()
        for i in range(len(name_list)):
            tmp_name = name_list[i]
            # result_list = self.single_test(tmp_name)
            # print(tmp_name, result_list)
            # print(1)

            buy_sell_way = buy_sell_way_list[i]
            tmp_factor = getattr(self, my_factor_dict[tmp_name])(tmp_name)
            mix_factor = mix_factor.add(tmp_factor * buy_sell_way, fill_value=0)
        daily_pos = self.deal_mix_factor(mix_factor).shift(2)
        in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
        fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df = \
            filter_all(self.cut_date, daily_pos, self.return_choose, if_return_pnl=True, if_only_long=False)
        print(in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d,
              pot_in, fit_ratio, leve_ratio, sp_in, sharpe_q_out)
        return mix_factor, in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, \
               pot_in, fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df

    def single_test_real(self, name_list, buy_sell_way_list):
        mix_factor = pd.DataFrame()
        for i in range(len(name_list)):
            tmp_name = name_list[i]
            # result_list = self.single_test(tmp_name)
            # print(tmp_name, result_list)
            # print(1)

            buy_sell_way = buy_sell_way_list[i]
            tmp_factor = getattr(self, my_factor_dict[tmp_name])(tmp_name)
            part_daily_pos = self.deal_mix_factor(tmp_factor).shift(2)
            mix_factor = mix_factor.add(part_daily_pos * buy_sell_way, fill_value=0)

        daily_pos = mix_factor / len(buy_sell_way_list)
        in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
        fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df = \
            filter_all(self.cut_date, daily_pos, self.return_choose, if_return_pnl=True, if_only_long=False)
        print(in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d,
              pot_in, fit_ratio, leve_ratio, sp_in, sharpe_q_out)
        return mix_factor, in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, \
               pot_in, fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df

    def mul_test_c(self, name_list, buy_sell_way_list):
        mix_factor = None
        for i in range(len(name_list)):
            tmp_name = name_list[i]
            buy_sell_way = buy_sell_way_list[i]
            tmp_factor = getattr(self, my_factor_dict[tmp_name])(tmp_name)
            if mix_factor is None:
                mix_factor = tmp_factor
            else:
                mix_factor = mul_fun(mix_factor, tmp_factor * buy_sell_way)
        daily_pos = self.deal_mix_factor(mix_factor).shift(2)
        in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
        fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df = \
            filter_all(self.cut_date, daily_pos, self.return_choose, if_return_pnl=True, if_only_long=False)
        print(in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d,
              pot_in, fit_ratio, leve_ratio, sp_in, sharpe_q_out)
        return mix_factor, in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, \
               pot_in, fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df


def main_fun(sector_name, hold_time, if_only_long, time_para_dict):
    root_path = '/mnt/mfs/DAT_EQT'
    if_save = False
    if_new_program = True

    begin_date = pd.to_datetime('20130101')
    cut_date = pd.to_datetime('20160401')
    # end_date = pd.to_datetime('20181201')
    end_date = datetime.now()
    lag = 2
    return_file = ''

    if_hedge = True
    # if_only_long = False

    if sector_name.startswith('market_top_300plus'):
        if_weight = 1
        ic_weight = 0

    elif sector_name.startswith('market_top_300to800plus'):
        if_weight = 0
        ic_weight = 1

    else:
        if_weight = 0.5
        ic_weight = 0.5

    main = FactorTestSector(root_path, if_save, if_new_program, begin_date, cut_date, end_date, time_para_dict,
                            sector_name, hold_time, lag, return_file, if_hedge, if_only_long, if_weight, ic_weight)

    pool_num = 28

    main.test_index(my_factor_dict, pool_num, suffix_name='tech_test')
    print(1)


def main_fun_test(sector_name, hold_time, if_only_long, time_para_dict, data):
    root_path = '/mnt/mfs/DAT_EQT'
    if_save = False
    if_new_program = True

    begin_date = pd.to_datetime('20130101')
    cut_date = pd.to_datetime('20160401')
    # end_date = pd.to_datetime('20181201')
    end_date = datetime.now()
    lag = 2
    return_file = ''

    if_hedge = True
    # if_only_long = False

    if sector_name.startswith('market_top_300plus'):
        if_weight = 1
        ic_weight = 0

    elif sector_name.startswith('market_top_300to800plus'):
        if_weight = 0
        ic_weight = 1

    else:
        if_weight = 0.5
        ic_weight = 0.5

    main = FactorTestSector(root_path, if_save, if_new_program, begin_date, cut_date, end_date, time_para_dict,
                            sector_name, hold_time, lag, return_file, if_hedge, if_only_long, if_weight, ic_weight)

    pool_num = 28

    main.test_index(my_factor_dict, pool_num, suffix_name='tech_test', filter_data=data)
    print(1)


data_1_raw = bt.AZ_Load_csv('/mnt/mfs/temp/dat_jerry/Shareholder_Meeting_Announcement_T800.csv').replace(0, np.nan)
data_1 = data_1_raw.fillna(method='ffill', limit=20)

data_2_raw = bt.AZ_Load_csv('/mnt/mfs/temp/dat_jerry/Shareholder_Meeting_Date_T800.csv').replace(0, np.nan)
data_2 = data_2_raw.fillna(method='ffill', limit=20)

data_3 = bt.AZ_Load_csv('/mnt/mfs/temp/dat_jerry/Shareholder_Meeting_between_T800.csv')

data_4 = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_PWY/goodnews_data/goodnews.csv')

my_factor_dict = dict({
    # 'WILLR_200_40': 'load_tech_factor',
    # 'WILLR_200_30': 'load_tech_factor',
    # 'WILLR_200_20': 'load_tech_factor',
    # 'WILLR_140_40': 'load_tech_factor',
    # 'WILLR_140_30': 'load_tech_factor',
    # 'WILLR_140_20': 'load_tech_factor',
    # 'WILLR_100_40': 'load_tech_factor',
    # 'WILLR_100_30': 'load_tech_factor',
    # 'WILLR_100_20': 'load_tech_factor',
    # 'WILLR_40_40': 'load_tech_factor',
    # 'WILLR_40_30': 'load_tech_factor',
    # 'WILLR_40_20': 'load_tech_factor',
    # 'WILLR_20_40': 'load_tech_factor',
    # 'WILLR_20_30': 'load_tech_factor',
    # 'WILLR_20_20': 'load_tech_factor',
    # 'WILLR_10_40': 'load_tech_factor',
    # 'WILLR_10_30': 'load_tech_factor',
    # 'WILLR_10_20': 'load_tech_factor',
    'BBANDS_10_2': 'load_tech_factor',
    'BBANDS_10_1.5': 'load_tech_factor',
    'BBANDS_10_1': 'load_tech_factor',
    'BBANDS_200_2': 'load_tech_factor',
    'BBANDS_200_1.5': 'load_tech_factor',
    'BBANDS_200_1': 'load_tech_factor',
    'BBANDS_140_2': 'load_tech_factor',
    'BBANDS_140_1.5': 'load_tech_factor',
    'BBANDS_140_1': 'load_tech_factor',
    'BBANDS_100_2': 'load_tech_factor',
    'BBANDS_100_1.5': 'load_tech_factor',
    'BBANDS_100_1': 'load_tech_factor',
    'BBANDS_40_2': 'load_tech_factor',
    'BBANDS_40_1.5': 'load_tech_factor',
    'BBANDS_40_1': 'load_tech_factor',
    'BBANDS_20_2': 'load_tech_factor',
    'BBANDS_20_1.5': 'load_tech_factor',
    'BBANDS_20_1': 'load_tech_factor',
    'MA_LINE_160_60': 'load_tech_factor',
    'MA_LINE_120_60': 'load_tech_factor',
    'MA_LINE_100_40': 'load_tech_factor',
    'MA_LINE_60_20': 'load_tech_factor',
    'MA_LINE_10_5': 'load_tech_factor',
    'MACD_20_60_18': 'load_tech_factor',
    'MACD_12_26_9': 'load_tech_factor',
    'RSI_200_30': 'load_tech_factor',
    'RSI_140_30': 'load_tech_factor',
    'RSI_100_30': 'load_tech_factor',
    'RSI_40_30': 'load_tech_factor',
    'RSI_200_10': 'load_tech_factor',
    'RSI_140_10': 'load_tech_factor',
    'RSI_100_10': 'load_tech_factor',
    'RSI_40_10': 'load_tech_factor',
    'ATR_200_0.2': 'load_tech_factor',
    'ATR_140_0.2': 'load_tech_factor',
    'ATR_100_0.2': 'load_tech_factor',
    'ATR_40_0.2': 'load_tech_factor',
    'ADOSC_60_160_0': 'load_tech_factor',
    'ADOSC_60_120_0': 'load_tech_factor',
    'ADOSC_40_100_0': 'load_tech_factor',
    'ADOSC_20_60_0': 'load_tech_factor',
    'MFI_200_70_30': 'load_tech_factor',
    'MFI_140_70_30': 'load_tech_factor',
    'MFI_100_70_30': 'load_tech_factor',
    'MFI_40_70_30': 'load_tech_factor',
    'CMO_200_0': 'load_tech_factor',
    'CMO_140_0': 'load_tech_factor',
    'CMO_100_0': 'load_tech_factor',
    'CMO_40_0': 'load_tech_factor',
    'AROON_200_80': 'load_tech_factor',
    'AROON_140_80': 'load_tech_factor',
    'AROON_100_80': 'load_tech_factor',
    'AROON_40_80': 'load_tech_factor',
    'ADX_200_20_10': 'load_tech_factor',
    'ADX_140_20_10': 'load_tech_factor',
    'ADX_100_20_10': 'load_tech_factor',
    'ADX_40_20_10': 'load_tech_factor',
    'CCI_p150d_limit_12': 'load_tech_factor',
    'CCI_p120d_limit_12': 'load_tech_factor',
    'CCI_p60d_limit_12': 'load_tech_factor',
    'CCI_p20d_limit_12': 'load_tech_factor',
    'MACD_40_160': 'load_tech_factor',
    'MACD_40_200': 'load_tech_factor',
    'MACD_20_200': 'load_tech_factor',
    'MACD_20_100': 'load_tech_factor',
    'MACD_10_30': 'load_tech_factor',
    'bias_turn_p120d': 'load_tech_factor',
    'bias_turn_p60d': 'load_tech_factor',
    'bias_turn_p20d': 'load_tech_factor',
    'turn_p150d_0.18': 'load_tech_factor',
    'turn_p30d_0.24': 'load_tech_factor',
    'turn_p120d_0.2': 'load_tech_factor',
    'turn_p60d_0.2': 'load_tech_factor',
    'turn_p20d_0.2': 'load_tech_factor',
    'log_price_0.2': 'load_tech_factor',
    'wgt_return_p120d_0.2': 'load_tech_factor',
    'wgt_return_p60d_0.2': 'load_tech_factor',
    'wgt_return_p20d_0.2': 'load_tech_factor',
    'return_p90d_0.2': 'load_tech_factor',
    'return_p30d_0.2': 'load_tech_factor',
    'return_p120d_0.2': 'load_tech_factor',
    'return_p60d_0.2': 'load_tech_factor',
    'return_p20d_0.2': 'load_tech_factor',
    'volume_moment_p20120d': 'load_tech_factor',
    'volume_moment_p1040d': 'load_tech_factor',
    'volume_moment_p530d': 'load_tech_factor',
    'moment_p50300d': 'load_tech_factor',
    'moment_p30200d': 'load_tech_factor',
    'moment_p40200d': 'load_tech_factor',
    'moment_p20200d': 'load_tech_factor',
    'moment_p20100d': 'load_tech_factor',
    'moment_p10100d': 'load_tech_factor',
    'moment_p1060d': 'load_tech_factor',
    'moment_p510d': 'load_tech_factor',
    'continue_ud_p200d': 'load_tech_factor',
    'evol_p200d': 'load_tech_factor',
    'vol_count_down_p200d': 'load_tech_factor',
    'vol_p200d': 'load_tech_factor',
    'continue_ud_p100d': 'load_tech_factor',
    'evol_p100d': 'load_tech_factor',
    'vol_count_down_p100d': 'load_tech_factor',
    'vol_p100d': 'load_tech_factor',
    'continue_ud_p90d': 'load_tech_factor',
    'evol_p90d': 'load_tech_factor',
    'vol_count_down_p90d': 'load_tech_factor',
    'vol_p90d': 'load_tech_factor',
    'continue_ud_p50d': 'load_tech_factor',
    'evol_p50d': 'load_tech_factor',
    'vol_count_down_p50d': 'load_tech_factor',
    'vol_p50d': 'load_tech_factor',
    'continue_ud_p30d': 'load_tech_factor',
    'evol_p30d': 'load_tech_factor',
    'vol_count_down_p30d': 'load_tech_factor',
    'vol_p30d': 'load_tech_factor',
    'continue_ud_p120d': 'load_tech_factor',
    'evol_p120d': 'load_tech_factor',
    'vol_count_down_p120d': 'load_tech_factor',
    'vol_p120d': 'load_tech_factor',
    'continue_ud_p60d': 'load_tech_factor',
    'evol_p60d': 'load_tech_factor',
    'vol_count_down_p60d': 'load_tech_factor',
    'vol_p60d': 'load_tech_factor',
    'continue_ud_p20d': 'load_tech_factor',
    'evol_p20d': 'load_tech_factor',
    'vol_count_down_p20d': 'load_tech_factor',
    'vol_p20d': 'load_tech_factor',
    'continue_ud_p10d': 'load_tech_factor',
    'evol_p10d': 'load_tech_factor',
    'vol_count_down_p10d': 'load_tech_factor',
    'vol_p10d': 'load_tech_factor',
    'volume_count_down_p120d': 'load_tech_factor',
    'volume_count_down_p60d': 'load_tech_factor',
    'volume_count_down_p20d': 'load_tech_factor',
    'volume_count_down_p10d': 'load_tech_factor',
    'price_p120d_hl': 'load_tech_factor',
    'price_p60d_hl': 'load_tech_factor',
    'price_p20d_hl': 'load_tech_factor',
    'price_p10d_hl': 'load_tech_factor',
    'aadj_r_p120d_col_extre_0.2': 'load_tech_factor',
    'aadj_r_p60d_col_extre_0.2': 'load_tech_factor',
    'aadj_r_p20d_col_extre_0.2': 'load_tech_factor',
    'aadj_r_p10d_col_extre_0.2': 'load_tech_factor',
    'aadj_r_p345d_continue_ud': 'load_tech_factor',
    'aadj_r_p345d_continue_ud_pct': 'load_tech_factor',
    'aadj_r_row_extre_0.2': 'load_tech_factor',
    'TVOL_p90d_col_extre_0.2': 'load_tech_factor',
    'TVOL_p30d_col_extre_0.2': 'load_tech_factor',
    'TVOL_p120d_col_extre_0.2': 'load_tech_factor',
    'TVOL_p60d_col_extre_0.2': 'load_tech_factor',
    'TVOL_p20d_col_extre_0.2': 'load_tech_factor',
    'TVOL_p10d_col_extre_0.2': 'load_tech_factor',
    'TVOL_p345d_continue_ud': 'load_tech_factor',
    'TVOL_row_extre_0.2': 'load_tech_factor',

    'BBANDS_alpha_100_1.5_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_100_1.5_0_1': 'load_tech_factor',
    'BBANDS_alpha_100_1.5_1_0': 'load_tech_factor',
    'BBANDS_alpha_100_1_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_100_1_0_1': 'load_tech_factor',
    'BBANDS_alpha_100_1_1_0': 'load_tech_factor',
    'BBANDS_alpha_100_2_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_100_2_0_1': 'load_tech_factor',
    'BBANDS_alpha_100_2_1_0': 'load_tech_factor',
    'BBANDS_alpha_10_1.5_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_10_1.5_0_1': 'load_tech_factor',
    'BBANDS_alpha_10_1.5_1_0': 'load_tech_factor',
    'BBANDS_alpha_10_1_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_10_1_0_1': 'load_tech_factor',
    'BBANDS_alpha_10_1_1_0': 'load_tech_factor',
    'BBANDS_alpha_10_2_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_10_2_0_1': 'load_tech_factor',
    'BBANDS_alpha_10_2_1_0': 'load_tech_factor',
    'BBANDS_alpha_140_1.5_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_140_1.5_0_1': 'load_tech_factor',
    'BBANDS_alpha_140_1.5_1_0': 'load_tech_factor',
    'BBANDS_alpha_140_1_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_140_1_0_1': 'load_tech_factor',
    'BBANDS_alpha_140_1_1_0': 'load_tech_factor',
    'BBANDS_alpha_140_2_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_140_2_0_1': 'load_tech_factor',
    'BBANDS_alpha_140_2_1_0': 'load_tech_factor',
    'BBANDS_alpha_200_1.5_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_200_1.5_0_1': 'load_tech_factor',
    'BBANDS_alpha_200_1.5_1_0': 'load_tech_factor',
    'BBANDS_alpha_200_1_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_200_1_0_1': 'load_tech_factor',
    'BBANDS_alpha_200_1_1_0': 'load_tech_factor',
    'BBANDS_alpha_200_2_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_200_2_0_1': 'load_tech_factor',
    'BBANDS_alpha_200_2_1_0': 'load_tech_factor',
    'BBANDS_alpha_20_1.5_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_20_1.5_0_1': 'load_tech_factor',
    'BBANDS_alpha_20_1.5_1_0': 'load_tech_factor',
    'BBANDS_alpha_20_1_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_20_1_0_1': 'load_tech_factor',
    'BBANDS_alpha_20_1_1_0': 'load_tech_factor',
    'BBANDS_alpha_20_2_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_20_2_0_1': 'load_tech_factor',
    'BBANDS_alpha_20_2_1_0': 'load_tech_factor',
    'BBANDS_alpha_40_1.5_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_40_1.5_0_1': 'load_tech_factor',
    'BBANDS_alpha_40_1.5_1_0': 'load_tech_factor',
    'BBANDS_alpha_40_1_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_40_1_0_1': 'load_tech_factor',
    'BBANDS_alpha_40_1_1_0': 'load_tech_factor',
    'BBANDS_alpha_40_2_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_40_2_0_1': 'load_tech_factor',
    'BBANDS_alpha_40_2_1_0': 'load_tech_factor',
    'MACD_alpha_12_26_9_0.5_0.5': 'load_tech_factor',
    'MACD_alpha_12_26_9_0_1': 'load_tech_factor',
    'MACD_alpha_12_26_9_1_0': 'load_tech_factor',
    'MACD_alpha_20_60_18_0.5_0.5': 'load_tech_factor',
    'MACD_alpha_20_60_18_0_1': 'load_tech_factor',
    'MACD_alpha_20_60_18_1_0': 'load_tech_factor',
    'MA_LINE_alpha_100_40_0.5_0.5': 'load_tech_factor',
    'MA_LINE_alpha_100_40_0_1': 'load_tech_factor',
    'MA_LINE_alpha_100_40_1_0': 'load_tech_factor',
    'MA_LINE_alpha_10_5_0.5_0.5': 'load_tech_factor',
    'MA_LINE_alpha_10_5_0_1': 'load_tech_factor',
    'MA_LINE_alpha_10_5_1_0': 'load_tech_factor',
    'MA_LINE_alpha_120_60_0.5_0.5': 'load_tech_factor',
    'MA_LINE_alpha_120_60_0_1': 'load_tech_factor',
    'MA_LINE_alpha_120_60_1_0': 'load_tech_factor',
    'MA_LINE_alpha_160_60_0.5_0.5': 'load_tech_factor',
    'MA_LINE_alpha_160_60_0_1': 'load_tech_factor',
    'MA_LINE_alpha_160_60_1_0': 'load_tech_factor',
    'MA_LINE_alpha_60_20_0.5_0.5': 'load_tech_factor',
    'MA_LINE_alpha_60_20_0_1': 'load_tech_factor',
    'MA_LINE_alpha_60_20_1_0': 'load_tech_factor',
    'RSI_alpha100_1.5_0.5_0.5': 'load_tech_factor',
    'RSI_alpha100_1.5_0_1': 'load_tech_factor',
    'RSI_alpha100_1.5_1_0': 'load_tech_factor',
    'RSI_alpha100_1_0.5_0.5': 'load_tech_factor',
    'RSI_alpha100_1_0_1': 'load_tech_factor',
    'RSI_alpha100_1_1_0': 'load_tech_factor',
    'RSI_alpha100_2_0.5_0.5': 'load_tech_factor',
    'RSI_alpha100_2_0_1': 'load_tech_factor',
    'RSI_alpha100_2_1_0': 'load_tech_factor',
    'RSI_alpha10_1.5_0.5_0.5': 'load_tech_factor',
    'RSI_alpha10_1.5_0_1': 'load_tech_factor',
    'RSI_alpha10_1.5_1_0': 'load_tech_factor',
    'RSI_alpha10_1_0.5_0.5': 'load_tech_factor',
    'RSI_alpha10_1_0_1': 'load_tech_factor',
    'RSI_alpha10_1_1_0': 'load_tech_factor',
    'RSI_alpha10_2_0.5_0.5': 'load_tech_factor',
    'RSI_alpha10_2_0_1': 'load_tech_factor',
    'RSI_alpha10_2_1_0': 'load_tech_factor',
    'RSI_alpha140_1.5_0.5_0.5': 'load_tech_factor',
    'RSI_alpha140_1.5_0_1': 'load_tech_factor',
    'RSI_alpha140_1.5_1_0': 'load_tech_factor',
    'RSI_alpha140_1_0.5_0.5': 'load_tech_factor',
    'RSI_alpha140_1_0_1': 'load_tech_factor',
    'RSI_alpha140_1_1_0': 'load_tech_factor',
    'RSI_alpha140_2_0.5_0.5': 'load_tech_factor',
    'RSI_alpha140_2_0_1': 'load_tech_factor',
    'RSI_alpha140_2_1_0': 'load_tech_factor',
    'RSI_alpha200_1.5_0.5_0.5': 'load_tech_factor',
    'RSI_alpha200_1.5_0_1': 'load_tech_factor',
    'RSI_alpha200_1.5_1_0': 'load_tech_factor',
    'RSI_alpha200_1_0.5_0.5': 'load_tech_factor',
    'RSI_alpha200_1_0_1': 'load_tech_factor',
    'RSI_alpha200_1_1_0': 'load_tech_factor',
    'RSI_alpha200_2_0.5_0.5': 'load_tech_factor',
    'RSI_alpha200_2_0_1': 'load_tech_factor',
    'RSI_alpha200_2_1_0': 'load_tech_factor',
    'RSI_alpha20_1.5_0.5_0.5': 'load_tech_factor',
    'RSI_alpha20_1.5_0_1': 'load_tech_factor',
    'RSI_alpha20_1.5_1_0': 'load_tech_factor',
    'RSI_alpha20_1_0.5_0.5': 'load_tech_factor',
    'RSI_alpha20_1_0_1': 'load_tech_factor',
    'RSI_alpha20_1_1_0': 'load_tech_factor',
    'RSI_alpha20_2_0.5_0.5': 'load_tech_factor',
    'RSI_alpha20_2_0_1': 'load_tech_factor',
    'RSI_alpha20_2_1_0': 'load_tech_factor',
    'RSI_alpha40_1.5_0.5_0.5': 'load_tech_factor',
    'RSI_alpha40_1.5_0_1': 'load_tech_factor',
    'RSI_alpha40_1.5_1_0': 'load_tech_factor',
    'RSI_alpha40_1_0.5_0.5': 'load_tech_factor',
    'RSI_alpha40_1_0_1': 'load_tech_factor',
    'RSI_alpha40_1_1_0': 'load_tech_factor',
    'RSI_alpha40_2_0.5_0.5': 'load_tech_factor',
    'RSI_alpha40_2_0_1': 'load_tech_factor',
    'RSI_alpha40_2_1_0': 'load_tech_factor',
    'pnd_continue_pct_ud_alpha345_0.5_0.5': 'load_tech_factor',
    'pnd_continue_pct_ud_alpha345_0_1': 'load_tech_factor',
    'pnd_continue_pct_ud_alpha345_1_0': 'load_tech_factor',
    'pnd_continue_ud_alpha345_0.5_0.5': 'load_tech_factor',
    'pnd_continue_ud_alpha345_0_1': 'load_tech_factor',
    'pnd_continue_ud_alpha345_1_0': 'load_tech_factor',
})

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
        'market_top_300to800plus_industry_55',
    ]
    data_list = [
        data_1, data_2, data_3, data_4,
    ]
    hold_time_list = [5, 20]
    for if_only_long in [False, True]:
        for hold_time in hold_time_list:
            for sector_name in sector_name_list:
                for data in data_list:
                    # main_fun(sector_name, hold_time, if_only_long, time_para_dict=[])
                    main_fun_test(sector_name, hold_time, if_only_long, [], data)
