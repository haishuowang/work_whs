import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
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
from factor_script.script_filter_fun import pos_daily_fun, out_sample_perf, filter_all, filter_time_para_fun

import work_whs.AZ_2018_Q2.factor_script.result_analyese.base_result_deal_import_v1 as base_rd


# product 笛卡尔积　　（有放回抽样排列）
# permutations 排列　　（不放回抽样排列）
# combinations 组合,没有重复　　（不放回抽样组合）
# combinations_with_replacement 组合,有重复　　（有放回抽样组合）

def plot_send_result(pnl_df, sharpe_ratio, subject):
    figure_save_path = os.path.join('/mnt/mfs/dat_whs', 'tmp_figure')
    plt.figure(figsize=[16, 8])
    plt.plot(pnl_df.index, pnl_df.cumsum(), label='sharpe_ratio={}'.format(sharpe_ratio))
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(figure_save_path, '{}.png'.format(subject)))
    text = ''
    to = ['whs@yingpei.com']
    filepath = [os.path.join(figure_save_path, '{}.png'.format(subject))]
    send_email.send_email(text, to, filepath, subject)


def mul_fun(a, b):
    return a.mul(b)


def mul_fun_c(a, b):
    a_l = a[a > 0]
    a_s = a[a < 0]

    b_l = b[b > 0]
    b_s = b[b < 0]

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


class FactorTest:
    def __init__(self, root_path, if_save, if_new_program, begin_date, cut_date, end_date, time_para_dict, sector_name,
                 hold_time, lag, return_file, if_hedge, if_only_long, if_weight=0.5, ic_weight=0.5,
                 para_adj_set_list=None):
        self.root_path = root_path
        self.if_save = if_save
        self.if_new_program = if_new_program
        self.begin_date = begin_date
        self.cut_date = cut_date
        self.end_date = end_date
        self.time_para_dict = time_para_dict
        self.sector_name = sector_name
        self.hold_time = hold_time
        self.lag = lag
        self.return_file = return_file
        self.if_hedge = if_hedge
        self.if_only_long = if_only_long
        self.if_weight = if_weight
        self.ic_weight = ic_weight
        if para_adj_set_list is None:
            self.para_adj_set_list = [
                {'pot_in_num': 50, 'leve_ratio_num': 2, 'sp_in': 1.5, 'ic_num': 0.0, 'fit_ratio': 2},
                {'pot_in_num': 40, 'leve_ratio_num': 2, 'sp_in': 1.5, 'ic_num': 0.0, 'fit_ratio': 2},
                {'pot_in_num': 50, 'leve_ratio_num': 2, 'sp_in': 1, 'ic_num': 0.0, 'fit_ratio': 1},
                {'pot_in_num': 50, 'leve_ratio_num': 1, 'sp_in': 1, 'ic_num': 0.0, 'fit_ratio': 2},
                {'pot_in_num': 50, 'leve_ratio_num': 1, 'sp_in': 1, 'ic_num': 0.0, 'fit_ratio': 1},
                {'pot_in_num': 40, 'leve_ratio_num': 1, 'sp_in': 1, 'ic_num': 0.0, 'fit_ratio': 1}]

        return_choose = self.load_return_data()
        self.xinx = return_choose.index
        sector_df = self.load_sector_data()
        self.xnms = sector_df.columns

        return_choose = return_choose.reindex(columns=self.xnms)
        self.sector_df = sector_df.reindex(index=self.xinx)
        print('Loaded sector DataFrame!')
        if if_hedge:
            if ic_weight + if_weight != 1:
                exit(-1)
        else:
            if_weight = 0
            ic_weight = 0

        index_df_1 = self.load_index_data('000300').fillna(0)
        # index_weight_1 = self.load_index_weight_data('000300')
        index_df_2 = self.load_index_data('000905').fillna(0)
        # index_weight_2 = self.load_index_weight_data('000905')
        #
        # weight_df = if_weight * index_weight_1 + ic_weight * index_weight_2
        hedge_df = if_weight * index_df_1 + ic_weight * index_df_2

        self.return_choose = return_choose.sub(hedge_df, axis=0)
        print('Loaded return DataFrame!')

        suspendday_df, limit_buy_sell_df = self.load_locked_data()
        limit_buy_sell_df_c = limit_buy_sell_df.shift(-1)
        limit_buy_sell_df_c.iloc[-1] = 1

        suspendday_df_c = suspendday_df.shift(-1)
        suspendday_df_c.iloc[-1] = 1
        self.suspendday_df_c = suspendday_df_c
        self.limit_buy_sell_df_c = limit_buy_sell_df_c
        print('Loaded suspendday_df and limit_buy_sell DataFrame!')

    def reindex_fun(self, df):
        return df.reindex(index=self.xinx, columns=self.xnms)

    @staticmethod
    def create_log_save_path(target_path):
        top_path = os.path.split(target_path)[0]
        if not os.path.exists(top_path):
            os.mkdir(top_path)
        if not os.path.exists(target_path):
            os.mknod(target_path)

    @staticmethod
    def row_extre(raw_df, sector_df, percent):
        raw_df = raw_df * sector_df
        target_df = raw_df.rank(axis=1, pct=True)
        target_df[target_df >= 1 - percent] = 1
        target_df[target_df <= percent] = -1
        target_df[(target_df > percent) & (target_df < 1 - percent)] = 0
        return target_df

    @staticmethod
    def pos_daily_fun(df, n=5):
        return df.rolling(window=n, min_periods=1).sum()

    def check_factor(self, name_list, file_name):
        load_path = os.path.join('/mnt/mfs/dat_whs/data/new_factor_data/' + self.sector_name)
        exist_factor = set([x[:-4] for x in os.listdir(load_path)])
        print()
        use_factor = set(name_list)
        a = use_factor - exist_factor
        if len(a) != 0:
            print('factor not enough!')
            print(a)
            print(len(a))
            send_email.send_email(f'{file_name} factor not enough!', ['whs@yingpei.com'], [], 'Factor Test Warning!')

    @staticmethod
    def create_all_para(tech_name_list, funda_name_list):

        target_list_1 = []
        for tech_name in tech_name_list:
            for value in combinations(funda_name_list, 2):
                target_list_1 += [[tech_name] + list(value)]

        target_list_2 = []
        for funda_name in funda_name_list:
            for value in combinations(tech_name_list, 2):
                target_list_2 += [[funda_name] + list(value)]

        target_list = target_list_1 + target_list_2
        return target_list

    # 获取剔除新股的矩阵
    def get_new_stock_info(self, xnms, xinx):
        new_stock_data = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Tab01/CDSY_SECUCODE/LISTSTATE.csv'))
        new_stock_data.fillna(method='ffill', inplace=True)
        # 获取交易日信息
        return_df = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Funda/DERIVED_14/aadj_r.csv')).astype(float)
        trade_time = return_df.index
        new_stock_data = new_stock_data.reindex(index=trade_time).fillna(method='ffill')
        target_df = new_stock_data.shift(40).notnull().astype(int)
        target_df = target_df.reindex(columns=xnms, index=xinx)
        return target_df

    # 获取剔除st股票的矩阵
    def get_st_stock_info(self, xnms, xinx):
        data = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Tab01/CDSY_CHANGEINFO/CHANGEA.csv'))
        data = data.reindex(columns=xnms, index=xinx)
        data.fillna(method='ffill', inplace=True)

        data = data.astype(str)
        target_df = data.applymap(lambda x: 0 if 'ST' in x or 'PT' in x else 1)
        return target_df

    def load_return_data(self):
        return_choose = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Funda/DERIVED_14/aadj_r.csv'))
        return_choose = return_choose[(return_choose.index >= self.begin_date) & (return_choose.index < self.end_date)]
        return return_choose

    # 获取sector data
    def load_sector_data(self):
        market_top_n = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Funda/DERIVED_10/' + self.sector_name + '.csv'))
        market_top_n = market_top_n.reindex(index=self.xinx)
        market_top_n.dropna(how='all', axis='columns', inplace=True)
        xnms = market_top_n.columns
        xinx = market_top_n.index

        new_stock_df = self.get_new_stock_info(xnms, xinx)
        st_stock_df = self.get_st_stock_info(xnms, xinx)
        sector_df = market_top_n * new_stock_df * st_stock_df
        sector_df.replace(0, np.nan, inplace=True)
        return sector_df

    def load_index_weight_data(self, index_name):
        index_info = bt.AZ_Load_csv(self.root_path + f'/EM_Funda/IDEX_YS_WEIGHT_A/SECURITYNAME_{index_name}.csv')
        index_info = self.reindex_fun(index_info)
        index_mask = (index_info.notnull() * 1).replace(0, np.nan)

        mkt_cap = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Funda/LICO_YS_STOCKVALUE/AmarketCapExStri.csv'))
        mkt_roll = mkt_cap.rolling(250, min_periods=0).mean()
        mkt_roll = self.reindex_fun(mkt_roll)

        mkt_roll_qrt = np.sqrt(mkt_roll)
        mkt_roll_qrt_index = mkt_roll_qrt * index_mask
        index_weight = mkt_roll_qrt_index.div(mkt_roll_qrt_index.sum(axis=1), axis=0)
        return index_weight

    # 涨跌停都不可交易
    def load_locked_data(self):
        raw_suspendday_df = bt.AZ_Load_csv(
            os.path.join(self.root_path, 'EM_Funda/TRAD_TD_SUSPENDDAY/SUSPENDREASON.csv'))
        suspendday_df = raw_suspendday_df.isnull().astype(int)
        suspendday_df = suspendday_df.reindex(columns=self.xnms, index=self.xinx, fill_value=True)
        suspendday_df.replace(0, np.nan, inplace=True)

        return_df = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Funda/DERIVED_14/aadj_r.csv')).astype(float)
        limit_buy_sell_df = (return_df.abs() < 0.095).astype(int)
        limit_buy_sell_df = limit_buy_sell_df.reindex(columns=self.xnms, index=self.xinx, fill_value=1)
        limit_buy_sell_df.replace(0, np.nan, inplace=True)
        return suspendday_df, limit_buy_sell_df

    # 获取index data
    def load_index_data(self, index_name):
        data = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Funda/INDEX_TD_DAILYSYS/CHG.csv'))
        target_df = data[index_name].reindex(index=self.xinx)
        return target_df * 0.01

    # 读取部分factor
    def load_part_factor(self, sector_name, xnms, xinx, file_list):
        factor_set = OrderedDict()
        for file_name in file_list:
            load_path = os.path.join('/mnt/mfs/dat_whs/data/new_factor_data/' + sector_name)
            target_df = pd.read_pickle(os.path.join(load_path, file_name + '.pkl'))
            factor_set[file_name] = target_df.reindex(columns=xnms, index=xinx).fillna(0)
        return factor_set

    # 读取factor
    def load_factor(self, file_name):
        factor_set = OrderedDict()
        load_path = os.path.join('/mnt/mfs/dat_whs/data/new_factor_data/' + self.sector_name)
        target_df = pd.read_pickle(os.path.join(load_path, file_name + '.pkl'))
        factor_set[file_name] = target_df.reindex(columns=self.xnms, index=self.xinx).fillna(0)
        return factor_set

    def deal_mix_factor(self, mix_factor):
        if self.if_only_long:
            mix_factor = mix_factor[mix_factor > 0]
        # 下单日期pos
        order_df = mix_factor.replace(np.nan, 0)
        # 排除入场场涨跌停的影响
        order_df = order_df * self.sector_df * self.limit_buy_sell_df_c * self.suspendday_df_c
        order_df = order_df.div(order_df.abs().sum(axis=1).replace(0, np.nan), axis=0)
        order_df[order_df > 0.05] = 0.05
        order_df[order_df < -0.05] = -0.05
        daily_pos = pos_daily_fun(order_df, n=self.hold_time)
        daily_pos.fillna(0, inplace=True)
        # 排除出场涨跌停的影响
        daily_pos = daily_pos * self.limit_buy_sell_df_c * self.suspendday_df_c
        daily_pos.fillna(method='ffill', inplace=True)
        return daily_pos

    def save_load_control(self, tech_name_list, funda_name_list, suffix_name, file_name):
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
            para_ready_df = pd.DataFrame(list(self.create_all_para(tech_name_list, funda_name_list)))
            total_para_num = len(para_ready_df)
            if self.if_save:
                self.create_log_save_path(log_save_file)
                self.create_log_save_path(result_save_file)
                self.create_log_save_path(para_save_file)
                para_dict['para_ready_df'] = para_ready_df
                para_dict['tech_name_list'] = tech_name_list
                para_dict['funda_name_list'] = funda_name_list
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

    ##################################################################################
    ##################################################################################
    ##################################################################################
    @staticmethod
    def load_result_data(result_file_name):
        data = pd.read_csv('/mnt/mfs/dat_whs/result/result/{}.txt'.format(result_file_name),
                           sep='|', header=None, error_bad_lines=False)

        data.columns = ['time_para', 'key', 'fun_name', 'name1', 'name2', 'name3', 'filter_fun_name', 'sector_name',
                        'con_in', 'con_out_1', 'con_out_2', 'con_out_3', 'con_out_4', 'ic', 'sp_u', 'sp_m', 'sp_d',
                        'pot_in', 'fit_ratio', 'leve_ratio', 'sp_in', 'sp_out_1', 'sp_out_2', 'sp_out_3', 'sp_out_4']

        return data

    @staticmethod
    def survive_ratio(data, pot_in_num, leve_ratio_num, sp_in, ic_num, fit_ratio):
        data_1 = data[data['time_para'] == 'time_para_1']
        data_2 = data[data['time_para'] == 'time_para_2']
        data_3 = data[data['time_para'] == 'time_para_3']
        data_4 = data[data['time_para'] == 'time_para_4']
        data_5 = data[data['time_para'] == 'time_para_5']
        data_6 = data[data['time_para'] == 'time_para_6']

        a_1 = data_1[(data_1['ic'].abs() > ic_num) &
                     (data_1['pot_in'].abs() > pot_in_num) &
                     (data_1['leve_ratio'].abs() > leve_ratio_num) &
                     (data_1['sp_in'].abs() > sp_in) &
                     (data_1['fit_ratio'].abs() > fit_ratio)]
        a_2 = data_2[(data_2['ic'].abs() > ic_num) &
                     (data_2['pot_in'].abs() > pot_in_num) &
                     (data_2['leve_ratio'].abs() > leve_ratio_num) &
                     (data_2['sp_in'].abs() > sp_in) &
                     (data_2['fit_ratio'].abs() > fit_ratio)]
        a_3 = data_3[(data_3['ic'].abs() > ic_num) &
                     (data_3['pot_in'].abs() > pot_in_num) &
                     (data_3['leve_ratio'].abs() > leve_ratio_num) &
                     (data_3['sp_in'].abs() > sp_in) &
                     (data_3['fit_ratio'].abs() > fit_ratio)]
        a_4 = data_4[(data_4['ic'].abs() > ic_num) &
                     (data_4['pot_in'].abs() > pot_in_num) &
                     (data_4['leve_ratio'].abs() > leve_ratio_num) &
                     (data_4['sp_in'].abs() > sp_in) &
                     (data_4['fit_ratio'].abs() > fit_ratio)]
        a_5 = data_5[(data_5['ic'].abs() > ic_num) &
                     (data_5['pot_in'].abs() > pot_in_num) &
                     (data_5['leve_ratio'].abs() > leve_ratio_num) &
                     (data_5['sp_in'].abs() > sp_in) &
                     (data_5['fit_ratio'].abs() > fit_ratio)]
        a_6 = data_6[(data_6['ic'].abs() > ic_num) &
                     (data_6['pot_in'].abs() > pot_in_num) &
                     (data_6['leve_ratio'].abs() > leve_ratio_num) &
                     (data_6['sp_in'].abs() > sp_in) &
                     (data_6['fit_ratio'].abs() > fit_ratio)]
        return a_1, a_2, a_3, a_4, a_5, a_6

    def survive_ratio_test(self, data, para_adj_set_list):
        for para_adj_set in para_adj_set_list:
            a_1, a_2, a_3, a_4, a_5, a_6 = self.survive_ratio(data, **para_adj_set)
            for con_out_name in ['con_out_4', 'con_out_3']:
                sr_1 = a_1[con_out_name].sum() / len(a_1)
                sr_2 = a_2[con_out_name].sum() / len(a_2)
                sr_3 = a_3[con_out_name].sum() / len(a_3)
                sr_4 = a_4[con_out_name].sum() / len(a_4)
                sr_5 = a_5[con_out_name].sum() / len(a_5)
                sr_6 = a_6[con_out_name].sum() / len(a_6)
                print(sr_1, sr_2, sr_3, sr_4, sr_5, sr_6)
                print(len(a_1), len(a_2), len(a_3), len(a_4), len(a_5), len(a_6))
                sr_list_in = np.array([sr_1, sr_2, sr_3])
                sr_list_out = np.array([sr_4, sr_5, sr_6])
                cond_1 = sum(sr_list_in > 0.5) >= 2  # and sum(sr_list_in > 0.2) == 3
                # cond_2 = (len(a_1) > 20) and (len(a_2) > 20) and (len(a_3) > 20)

                cond_3_1 = sum(sr_list_out > 0.55) >= 1
                cond_3_2 = sum(sr_list_out > 0.3) >= 2
                cond_3_3 = sum(sr_list_out > 0.1) >= 3

                cond_3 = cond_3_1 and cond_3_2 and cond_3_3
                cond_4 = (len(a_4) > 20) and (len(a_5) > 20) and (len(a_6) > 20)
                print(cond_1, cond_3, cond_4)
                if cond_1 and cond_3 and cond_4:
                    return para_adj_set
        return None

    def bkt_fun(self, pnl_save_path, a_n, i):
        x, key, fun_name, name1, name2, name3, filter_fun_name, sector_name, \
        con_in, con_out_1, con_out_2, con_out_3, con_out_4, ic, \
        sp_u, sp_m, sp_d, pot_in, fit_ratio, leve_ratio, \
        sp_in, sp_out_1, sp_out_2, sp_out_3, sp_out_4 = a_n.loc[i]

        mix_factor, con_in_c, con_out_c, ic_c, sp_u_c, sp_m_c, sp_d_c, pot_in_c, fit_ratio_c, leve_ratio_c, \
        sp_in_c, sp_out_c, pnl_df_c = self.single_test(fun_name, name1, name2, name3)
        plot_send_result(pnl_df_c, bt.AZ_Sharpe_y(pnl_df_c), '{}, key={}'.format(i, key))

        print('***************************************************')
        print('now {}\'s is running, key={}, {}, {}, {}, {}'.format(i, key, fun_name, name1, name2, name3))
        print(con_in_c, con_out_c, ic_c, sp_u_c, sp_m_c, sp_d_c, pot_in_c, fit_ratio_c, leve_ratio_c, sp_out_c)
        print(con_in, con_out_1, ic, sp_u, sp_m, sp_d, pot_in, fit_ratio, leve_ratio, sp_out_1)

        if sp_m > 0:
            if not os.path.exists(os.path.join(pnl_save_path, '{}|{}|{}.csv'.format(x, key, fun_name))):
                pnl_df_c.to_pickle(os.path.join(pnl_save_path, '{}|{}|{}.csv'.format(x, key, fun_name)))

            else:
                pnl_df_c.to_pickle(os.path.join(pnl_save_path, '{}|{}|{}.csv'.format(x, key, fun_name)))
                print('file exist!')
            return mix_factor
        else:
            if not os.path.exists(os.path.join(pnl_save_path, '{}|{}|{}.csv'.format(x, key, fun_name))):
                (-pnl_df_c).to_pickle(os.path.join(pnl_save_path, '{}|{}|{}.csv'.format(x, key, fun_name)))
            else:
                (-pnl_df_c).to_pickle(os.path.join(pnl_save_path, '{}|{}|{}.csv'.format(x, key, fun_name)))
                print('file exist!')
            return -mix_factor

    def pos_sum_c(self, data, time_para, result_file_name, pot_in_num, leve_ratio_num, sp_in, ic_num, fit_ratio):
        time_para_dict['time_para_1'] = [pd.to_datetime('20110101'), pd.to_datetime('20150101'),
                                         pd.to_datetime('20150701')]

        time_para_dict['time_para_2'] = [pd.to_datetime('20120101'), pd.to_datetime('20160101'),
                                         pd.to_datetime('20160701')]

        time_para_dict['time_para_3'] = [pd.to_datetime('20130601'), pd.to_datetime('20170601'),
                                         pd.to_datetime('20171201')]

        time_para_dict['time_para_4'] = [pd.to_datetime('20140601'), pd.to_datetime('20180601'),
                                         pd.to_datetime('20181001')]

        time_para_dict['time_para_5'] = [pd.to_datetime('20140701'), pd.to_datetime('20180701'),
                                         pd.to_datetime('20181001')]

        time_para_dict['time_para_6'] = [pd.to_datetime('20140801'), pd.to_datetime('20180801'),
                                         pd.to_datetime('20181001')]

        data_n = data[data['time_para'] == time_para]
        begin_date, cut_date, end_date = time_para_dict[time_para]
        a_n = data_n[(data_n['ic'].abs() > ic_num) &
                     (data_n['pot_in'].abs() > pot_in_num) &
                     (data_n['leve_ratio'].abs() > leve_ratio_num) &
                     (data_n['sp_in'].abs() > sp_in) &
                     (data_n['fit_ratio'].abs() > fit_ratio)]

        sum_factor_df = pd.DataFrame()
        pnl_save_path = '/mnt/mfs/dat_whs/data/mix_factor_pnl/' + result_file_name
        bt.AZ_Path_create(pnl_save_path)

        result_list = []
        pool = Pool(20)
        for i in a_n.index:
            # bkt_fun(pnl_save_path, a_n, i)
            result_list.append(pool.apply_async(bkt_fun, args=(pnl_save_path, a_n, i,)))
        pool.close()
        pool.join()

        for res in result_list:
            sum_factor_df = sum_factor_df.add(res.get(), fill_value=0)

        sum_pos_df = self.deal_mix_factor(sum_factor_df).shift(2)
        in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
        fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df = filter_all(cut_date, sum_pos_df, main_model.return_choose,
                                                                        if_return_pnl=True, if_only_long=False)
        print(in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in,
              fit_ratio, leve_ratio, sp_in, sharpe_q_out)
        plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df), 'mix_factor')
        return sum_pos_df, pnl_df

    def result_deal(self, result_file_name):
        # #############################################################################
        # 结果分析
        data = self.load_result_data(result_file_name)
        # #############################################################################
        # 切除一部分因子
        filter_cond = data[['name1', 'name2', 'name3']] \
            .apply(lambda x: not (('R_COMPANYCODE_First_row_extre_0.3' in set(x)) or
                                  ('return_p20d_0.2' in set(x)) or
                                  ('price_p120d_hl' in set(x)) or
                                  ('return_p60d_0.2' in set(x)) or
                                  ('wgt_return_p120d_0.2' in set(x)) or
                                  ('wgt_return_p20d_0.2' in set(x)) or
                                  ('log_price_0.2' in set(x)) or
                                  ('TVOL_row_extre_0.2' in set(x)) or
                                  ('TVOL_row_extre_0.2' in set(x)) or
                                  ('turn_p30d_0.24' in set(x))
                                  # ('RSI_140_30' in set(x)) or
                                  # ('CMO_200_0' in set(x)) or
                                  # ('CMO_40_0' in set(x))
                                  # ('ATR_40_0.2' in set(x))
                                  # ('ADX_200_40_20' in set(x))
                                  # ('ATR_140_0.2' in set(x))
                                  ), axis=1)
        data = data[filter_cond]
        # #############################################################################
        # 结果分析
        survive_result = self.survive_ratio_test(data, self.para_adj_set_list)
        if survive_result is None:
            print(f'{result_file_name} not satisfaction!!!!!!!!')
            exit(1)
        else:
            pass
        #############################################################################
        # 回测函数
        time_para = 'time_para_5'
        print(time_para)

        begin_date, cut_date, _, end_date, _ = time_para_dict[time_para]

        sum_pos_df, pnl_df = self.pos_sum_c(data, time_para, result_file_name, **survive_result)


if __name__ == '__main__':
    root_path = '/mnt/mfs/DAT_EQT'
    if_save = False
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

    main = FactorTest(root_path, if_save, if_new_program, begin_date, cut_date, end_date, time_para_dict, sector_name,
                      hold_time, lag, return_file, if_hedge, if_only_long)
    #
    # tech_name_list = ['CCI_p120d_limit_12',
    #                   'MACD_20_100']
    # funda_name_list = ['R_SalesGrossMGN_s_First_row_extre_0.3',
    #                    'R_MgtExp_sales_s_First_row_extre_0.3']
    # pool_num = 20
    # main.test_index_3(tech_name_list, funda_name_list, pool_num)
