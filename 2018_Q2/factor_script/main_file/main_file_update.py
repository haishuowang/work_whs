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
sys.path.append('/mnt/mfs/work_whs/2018_Q2')
import loc_lib.shared_tools.back_test as bt
from loc_lib.shared_tools import send_email
# 读取数据的函数 以及
from factor_script import main_file_return_hedge as mf


# product 笛卡尔积　　（有放回抽样排列）
# permutations 排列　　（不放回抽样排列）
# combinations 组合,没有重复　　（不放回抽样组合）
# combinations_with_replacement 组合,有重复　　（有放回抽样组合）


# def mul_fun(a, b):
#     return a.mul(b)


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


def create_fun_set_2(fun_set):
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


class FactorTestCRT(mf.FactorTest):
    def __init__(self, *args):
        super(FactorTestCRT, self).__init__(*args)

    def load_change_factor(self, file_name):
        load_path = '/mnt/mfs/DAT_EQT/EM_Funda/daily/'
        raw_df = bt.AZ_Load_csv(os.path.join(load_path, file_name + '.csv')) \
            .reindex(index=self.xinx, columns=self.xnms)
        QTTM_df = bt.AZ_Load_csv(os.path.join(load_path, '_'.join(file_name.split('_')[:-1]) + '_QTTM.csv')) \
            .reindex(index=self.xinx, columns=self.xnms)
        QTTM_df_ma = bt.AZ_Rolling_mean(QTTM_df.abs().replace(0, np.nan), 60)
        tmp_df = raw_df / QTTM_df_ma
        # target_df = bt.AZ_Row_zscore(tmp_df)
        target_df = self.row_extre(tmp_df, self.sector_df, 0.2)
        return target_df

    def load_ratio_factor(self, file_name):
        load_path = '/mnt/mfs/DAT_EQT/EM_Funda/daily/'
        tmp_df = bt.AZ_Load_csv(os.path.join(load_path, file_name + '.csv')) \
            .reindex(index=self.xinx, columns=self.xnms)
        # target_df = bt.AZ_Row_zscore(tmp_df)
        target_df = self.row_extre(tmp_df, self.sector_df, 0.2)
        return target_df

    def load_tech_factor(self, file_name):
        load_path = os.path.join('/mnt/mfs/dat_whs/data/new_factor_data/' + self.sector_name)
        target_df = pd.read_pickle(os.path.join(load_path, file_name + '.pkl')) \
            .reindex(index=self.xinx, columns=self.xnms)
        return target_df

    @staticmethod
    def create_all_para_(change_list, ratio_list, tech_list):
        target_list = list(product(change_list, ratio_list, tech_list))
        return target_list

    def save_load_control_(self, change_list, ratio_list, tech_list, suffix_name, file_name):
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
            para_ready_df = pd.DataFrame(list(self.create_all_para_(change_list, ratio_list, tech_list)))
            total_para_num = len(para_ready_df)
            if self.if_save:
                self.create_log_save_path(log_save_file)
                self.create_log_save_path(result_save_file)
                self.create_log_save_path(para_save_file)
                para_dict['para_ready_df'] = para_ready_df
                para_dict['change_list'] = change_list
                para_dict['ratio_list'] = ratio_list
                para_dict['tech_list'] = tech_list
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

    def part_test_index_3_(self, key, name_1, name_2, name_3, log_save_file, result_save_file, total_para_num):
        lock = Lock()
        start_time = time.time()
        load_time_1 = time.time()
        # load因子,同时根据stock_universe筛选数据.
        change_factor = self.load_change_factor(name_1)
        ratio_factor = self.load_ratio_factor(name_2)
        tech_factor = self.load_tech_factor(name_3)
        load_time_2 = time.time()
        # 加载花费数据时间
        load_delta = round(load_time_2 - load_time_1, 2)
        # 生成混合函数集
        fun_mix_2_set = create_fun_set_2_crt()
        #################
        # 更换filter函数 #
        #################
        filter_name = filter_time_para_fun.__name__

        for fun in fun_mix_2_set:
            mix_factor = fun(change_factor, ratio_factor, tech_factor)
            if len(mix_factor.abs().sum(axis=1).replace(0, np.nan).dropna()) / len(mix_factor) < 0.1:
                continue

            daily_pos = self.deal_mix_factor(mix_factor).shift(2)
            # 返回样本内筛选结果

            result_dict = filter_time_para_fun(self.time_para_dict, daily_pos, self.return_choose, if_only_long=False)
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

    def test_index_3_(self, change_list, ratio_list, tech_list, pool_num=20, suffix_name='', file_name=''):

        para_ready_df, log_save_file, result_save_file, total_para_num = \
            self.save_load_control_(change_list, ratio_list, tech_list, suffix_name, file_name)
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
        change_factor = self.load_change_factor(name1)
        ratio_factor = self.load_ratio_factor(name2)
        tech_factor = self.load_tech_factor(name3)
        mix_factor = fun(change_factor, ratio_factor, tech_factor)
        daily_pos = self.deal_mix_factor(mix_factor).shift(2)
        in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
        fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df = \
            filter_all(self.cut_date, daily_pos, self.return_choose, if_return_pnl=True, if_only_long=False)
        # result_dict = filter_all(self.cut_date, daily_pos, self.return_choose)
        return mix_factor, in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, \
               pot_in, fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df


if __name__ == '__main__':
    root_path = '/mnt/mfs/DAT_EQT'
    if_save = True
    if_new_program = True

    begin_date = pd.to_datetime('20100101')
    cut_date = pd.to_datetime('20160401')
    end_date = pd.to_datetime('20180901')

    sector_name = 'market_top_2000'
    index_name = '000905'
    return_file = 'pct_p1d'
    hold_time = 20
    lag = 2
    return_file = ''

    if_hedge = True
    if_only_long = False
    time_para_dict = OrderedDict()

    time_para_dict['time_para_1'] = [pd.to_datetime('20110101'), pd.to_datetime('20150101'),
                                     pd.to_datetime('20150401'), pd.to_datetime('20150701'),
                                     pd.to_datetime('20151001'), pd.to_datetime('20160101')]

    time_para_dict['time_para_2'] = [pd.to_datetime('20120101'), pd.to_datetime('20160101'),
                                     pd.to_datetime('20160401'), pd.to_datetime('20160701'),
                                     pd.to_datetime('20161001'), pd.to_datetime('20170101')]

    time_para_dict['time_para_3'] = [pd.to_datetime('20130601'), pd.to_datetime('20170601'),
                                     pd.to_datetime('20170901'), pd.to_datetime('20171201'),
                                     pd.to_datetime('20180301'), pd.to_datetime('20180601')]

    time_para_dict['time_para_4'] = [pd.to_datetime('20140601'), pd.to_datetime('20180601'),
                                     pd.to_datetime('20180901'), pd.to_datetime('20180901'),
                                     pd.to_datetime('20180901'), pd.to_datetime('20180901')]

    time_para_dict['time_para_5'] = [pd.to_datetime('20140701'), pd.to_datetime('20180701'),
                                     pd.to_datetime('20180901'), pd.to_datetime('20180901'),
                                     pd.to_datetime('20180901'), pd.to_datetime('20180901')]

    time_para_dict['time_para_6'] = [pd.to_datetime('20140801'), pd.to_datetime('20180801'),
                                     pd.to_datetime('20180901'), pd.to_datetime('20180901'),
                                     pd.to_datetime('20180901'), pd.to_datetime('20180901')]
    main = FactorTestCRT(root_path, if_save, if_new_program, begin_date, cut_date, end_date, time_para_dict,
                         sector_name, hold_time, lag, return_file, if_hedge, if_only_long)

    change_list = ['R_ACCOUNTPAY_QYOY',
                   'R_ACCOUNTREC_QYOY',
                   'R_ASSETDEVALUELOSS_s_QYOY',
                   'R_Cashflow_s_YOY_First',
                   'R_CFO_s_YOY_First',
                   'R_CostSales_QYOY',
                   'R_EBITDA2_QYOY',
                   'R_EPSDiluted_YOY_First',
                   'R_ESTATEINVEST_QYOY',
                   'R_FairVal_TotProfit_QYOY',
                   'R_FINANCEEXP_s_QYOY',
                   'R_GrossProfit_TTM_QYOY',
                   'R_GSCF_sales_QYOY',
                   'R_IntDebt_Mcap_QYOY',
                   'R_INVESTINCOME_s_QYOY',
                   'R_LTDebt_WorkCap_QYOY',
                   'R_LOANREC_s_QYOY',
                   'R_NetAssets_s_YOY_First',
                   'R_NetInc_s_QYOY',
                   'R_NETPROFIT_s_QYOY',
                   'R_OPCF_TTM_QYOY',
                   'R_OperCost_sales_QYOY',
                   'R_OperProfit_YOY_First',
                   'R_OPEX_sales_QYOY',
                   'R_ROE1_QYOY',
                   'R_SUMLIAB_QYOY',
                   'R_TangAssets_IntDebt_QYOY',
                   'R_WorkCapital_QYOY',
                   'R_OTHERLASSET_QYOY',
                   'R_NetIncRecur_QYOY.csv'
                   ]

    ratio_list = ['R_DebtAssets_QTTM',
                  'R_EBITDA_IntDebt_QTTM',
                  'R_LTDebt_WorkCap_QTTM',
                  'R_OPCF_TotDebt_QTTM',
                  'R_OPEX_sales_TTM_First',
                  'R_SalesGrossMGN_QTTM',
                  'R_CurrentAssetsTurnover_QTTM',
                  'R_TangAssets_TotLiab_QTTM',
                  'R_NetROA_TTM_First',
                  'R_EBIT_sales_QTTM'
                  ]

    tech_list = ['CCI_p150d_limit_12',
                 'MACD_40_160',
                 'bias_turn_p60d',
                 'vol_p50d',
                 'vol_p100d',
                 'vol_p200d',
                 'evol_p30d',
                 'evol_p90d',
                 'moment_p30200d',
                 'moment_p50300d',
                 'turn_p30d_0.24',
                 'turn_p150d_0.18',
                 'TVOL_p30d_col_extre_0.2',
                 'TVOL_p90d_col_extre_0.2',
                 'TVOL_row_extre_0.2',
                 'aadj_r_p20d_col_extre_0.2',
                 'aadj_r_p345d_continue_ud_pct',
                 'aadj_r_p345d_continue_ud',
                 'volume_moment_p1040d',
                 'volume_moment_p20120d',
                 'return_p30d_0.2',
                 'return_p90d_0.2'
                 ]

    pool_num = 20
    main.test_index_3_(change_list, ratio_list, tech_list, pool_num)
