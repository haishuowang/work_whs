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
from open_lib.shared_tools import send_email
# 读取数据的函数 以及
# from factor_script.script_filter_fun import pos_daily_fun


# product 笛卡尔积　　（有放回抽样排列）
# permutations 排列　　（不放回抽样排列）
# combinations 组合,没有重复　　（不放回抽样组合）
# combinations_with_replacement 组合,有重复　　（有放回抽样组合）


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


def filter_time_para_fun(time_para_dict, pos_df_daily, adj_return, index_df, if_hedge=True, hedge_ratio=1,
                         if_return_pnl=False, if_only_long=False):
    if if_hedge:
        hedge_df = hedge_ratio * index_df.mul(pos_df_daily.sum(axis=1), axis=0)
        pnl_df = -hedge_df.sub((pos_df_daily * adj_return).sum(axis=1), axis=0)
    else:
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


class FactorTest:
    def __init__(self, root_path, if_save, if_new_program, begin_date, cut_date, end_date, time_para_dict, sector_name,
                 index_name, hold_time, lag, return_file, if_hedge, if_only_long):
        self.root_path = root_path
        self.if_save = if_save
        self.if_new_program = if_new_program
        self.begin_date = begin_date
        self.cut_date = cut_date
        self.end_date = end_date
        self.time_para_dict = time_para_dict
        self.sector_name = sector_name
        self.index_name = index_name
        self.hold_time = hold_time
        self.lag = lag
        self.return_file = return_file
        self.if_hedge = if_hedge
        self.if_only_long = if_only_long

        self.sector_df = self.load_sector_data()
        print('Loaded sector DataFrame!')
        self.xnms = self.sector_df.columns
        self.xinx = self.sector_df.index

        return_choose = bt.AZ_Load_csv(os.path.join(root_path, 'EM_Funda/DERIVED_14/aadj_r.csv'))
        self.return_choose = return_choose.reindex(index=self.xinx, columns=self.xnms)
        print('Loaded return DataFrame!')

        suspendday_df, limit_buy_sell_df = self.load_locked_data()
        limit_buy_sell_df_c = limit_buy_sell_df.shift(-1)
        limit_buy_sell_df_c.iloc[-1] = 1

        suspendday_df_c = suspendday_df.shift(-1)
        suspendday_df_c.iloc[-1] = 1
        self.suspendday_df_c = suspendday_df_c
        self.limit_buy_sell_df_c = limit_buy_sell_df_c
        print('Loaded suspendday_df and limit_buy_sell DataFrame!')
        self.index_df = self.load_index_data()
        print('Loaded index DataFrame!')

    @staticmethod
    def pos_daily_fun(df, n=5):
        return df.rolling(window=n, min_periods=1).sum()

    def check_factor(self, tech_name_list, funda_name_list, file_name):
        load_path = os.path.join('/mnt/mfs/dat_whs/data/new_factor_data/' + self.sector_name)
        exist_factor = set([x[:-4] for x in os.listdir(load_path)])
        print()
        use_factor = set(tech_name_list + funda_name_list)
        a = use_factor - exist_factor
        if len(a) != 0:
            print('factor not enough!')
            print(a)
            print(len(a))
            send_email.send_email(f'{file_name} factor not enough!', ['whs@yingpei.com'], [], 'Factor Test Warning!')

    @staticmethod
    def row_extre(raw_df, sector_df, percent):
        raw_df = raw_df * sector_df
        target_df = raw_df.rank(axis=1, pct=True)
        target_df[target_df >= 1 - percent] = 1
        target_df[target_df <= percent] = -1
        target_df[(target_df > percent) & (target_df < 1 - percent)] = 0
        return target_df

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

    @staticmethod
    def create_log_save_path(target_path):
        top_path = os.path.split(target_path)[0]
        if not os.path.exists(top_path):
            os.mkdir(top_path)
        if not os.path.exists(target_path):
            os.mknod(target_path)

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

    # 获取sector data
    def load_sector_data(self):

        market_top_n = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Funda/DERIVED_10/' + self.sector_name + '.csv'))
        market_top_n = market_top_n[(market_top_n.index >= self.begin_date) & (market_top_n.index < self.end_date)]
        market_top_n.dropna(how='all', axis='columns', inplace=True)
        xnms = market_top_n.columns
        xinx = market_top_n.index

        new_stock_df = self.get_new_stock_info(xnms, xinx)
        st_stock_df = self.get_st_stock_info(xnms, xinx)
        sector_df = market_top_n * new_stock_df * st_stock_df
        sector_df.replace(0, np.nan, inplace=True)
        return sector_df

    # 涨跌停都不可交易
    def load_locked_data(self):
        raw_suspendday_df = bt.AZ_Load_csv(
            os.path.join(self.root_path, 'EM_Funda/TRAD_TD_SUSPENDDAY/SUSPENDREASON.csv'))
        suspendday_df = raw_suspendday_df.isnull()
        suspendday_df = suspendday_df.reindex(columns=self.xnms, index=self.xinx, fill_value=True)
        suspendday_df.replace(0, np.nan, inplace=True)

        return_df = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Funda/DERIVED_14/aadj_r.csv')).astype(float)
        limit_buy_sell_df = (return_df.abs() < 0.095).astype(int)
        limit_buy_sell_df = limit_buy_sell_df.reindex(columns=self.xnms, index=self.xinx, fill_value=1)
        limit_buy_sell_df.replace(0, np.nan, inplace=True)
        return suspendday_df, limit_buy_sell_df

    # 获取index data
    def load_index_data(self):
        data = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Tab09/INDEX_TD_DAILYSYS/CHG.csv'))
        target_df = data[self.index_name].reindex(index=self.xinx)
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
    def load_factor(self, file_list):
        factor_set = OrderedDict()
        for file_name in file_list:
            load_path = os.path.join('/mnt/mfs/dat_whs/data/new_factor_data/' + sector_name)
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
        daily_pos = pos_daily_fun(order_df, n=self.hold_time)
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
            if self.if_save:
                self.create_log_save_path(log_save_file)
                self.create_log_save_path(result_save_file)
                self.create_log_save_path(para_save_file)
                para_dict['para_ready_df'] = para_ready_df
                para_dict['tech_name_list'] = tech_name_list
                para_dict['funda_name_list'] = funda_name_list
                pd.to_pickle(para_dict, para_save_file)
                total_para_num = len(para_ready_df)
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

    def part_test_index_3(self, key, name_1, name_2, name_3, log_save_file, result_save_file, total_para_num):
        lock = Lock()
        start_time = time.time()
        load_time_1 = time.time()
        # load因子,同时根据stock_universe筛选数据.
        factor_set = self.load_part_factor(self.sector_name, self.xnms, self.xinx, [name_1, name_2, name_3])
        load_time_2 = time.time()
        # 加载花费数据时间
        load_delta = round(load_time_2 - load_time_1, 2)
        # 生成混合函数集
        fun_set = [sub_fun, add_fun, mul_fun]
        fun_mix_2_set = create_fun_set_2(fun_set)
        #################
        # 更换filter函数 #
        #################
        filter_name = filter_time_para_fun.__name__

        for fun in fun_mix_2_set:
            mix_factor = fun(factor_set[name_1], factor_set[name_2], factor_set[name_3])
            if len(mix_factor.abs().sum(axis=1).replace(0, np.nan).dropna()) / len(mix_factor) < 0.1:
                continue

            daily_pos = self.deal_mix_factor(mix_factor).shift(2)
            # 返回样本内筛选结果
            result_dict = filter_time_para_fun(self.time_para_dict, daily_pos, self.return_choose, self.index_df,
                                               if_hedge=self.if_hedge, hedge_ratio=1, if_return_pnl=False,
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

    def test_index_3(self, tech_name_list, funda_name_list, pool_num=20, suffix_name='', file_name=''):

        para_ready_df, log_save_file, result_save_file, total_para_num = \
            self.save_load_control(tech_name_list, funda_name_list, suffix_name, file_name)
        self.check_factor(tech_name_list, funda_name_list, result_save_file)
        a_time = time.time()
        pool = Pool(pool_num)
        for key in list(para_ready_df.index):
            name_1, name_2, name_3 = para_ready_df.loc[key]
            args_list = (key, name_1, name_2, name_3, log_save_file, result_save_file, total_para_num)
            # self.part_test_index_3(*args_list)
            pool.apply_async(self.part_test_index_3, args=args_list)
        pool.close()
        pool.join()

        b_time = time.time()
        print('Success!Processing end, Cost {} seconds'.format(round(b_time - a_time, 2)))


if __name__ == '__main__':
    root_path = '/mnt/mfs/DAT_EQT'
    if_save = True
    if_new_program = True

    begin_date = pd.to_datetime('20100101')
    cut_date = pd.to_datetime('20160401')
    end_date = pd.to_datetime('20180901')

    sector_name = 'market_top_1000'
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
    main = FactorTest(root_path, if_save, if_new_program, begin_date, cut_date, end_date, time_para_dict, sector_name,
                      index_name, hold_time, lag, return_file, if_hedge, if_only_long)

    # alpha3
    # tech_name_list = ['CCI_p120d_limit_12',
    #                   'MACD_20_100',
    #                   'MACD_40_200',
    #                   'log_price_0.2',
    #                   'bias_turn_p20d',
    #                   'bias_turn_p120d',
    #                   'vol_p20d',
    #                   'vol_p60d',
    #                   'evol_p20d',
    #                   'moment_p20100d',
    #                   'turn_p20d_0.2',
    #                   'turn_p120d_0.2',
    #                   'vol_count_down_p60d',
    #                   'TVOL_p20d_col_extre_0.2',
    #                   'TVOL_p120d_col_extre_0.2',
    #                   'price_p20d_hl',
    #                   'price_p120d_hl',
    #                   'aadj_r_p345d_continue_ud_pct',
    #                   'volume_moment_p530d',
    #                   'return_p60d_0.2',
    #                   ]
    #
    # funda_name_list = ['R_SalesGrossMGN_s_First_row_extre_0.3',
    #                    'R_MgtExp_sales_s_First_row_extre_0.3',
    #                    'R_EBITDA_QTTM_and_MCAP_0.3',
    #                    'R_NetAssets_s_POP_First_row_extre_0.3',
    #                    'R_EBITDA_QYOY_and_MCAP_0.3',
    #                    'R_NetCashflowPS_s_First_row_extre_0.3',
    #                    'R_TotLiab_s_YOY_First_row_extre_0.3',
    #                    'R_FairValChg_TotProfit_s_First_row_extre_0.3',
    #                    'R_NonOperProft_TotProfit_s_First_row_extre_0.3',
    #                    'R_EBITDA_QTTM_and_R_SUMASSET_First_0.3',
    #                    'R_Revenue_s_POP_First_row_extre_0.3',
    #                    'R_NetMargin_s_YOY_First_row_extre_0.3',
    #                    'R_OperCost_sales_s_First_row_extre_0.3',
    #                    'R_IntDebt_Y3YGR_and_R_SUMASSET_First_0.3',
    #                    'R_EBIT2_Y3YGR_and_MCAP_0.3',
    #                    'R_OperProfit_sales_s_First_row_extre_0.3']

    tech_name_list = ['CCI_p120d_limit_12',
                      'MACD_20_100']
    funda_name_list = ['R_SalesGrossMGN_s_First_row_extre_0.3',
                       'R_MgtExp_sales_s_First_row_extre_0.3']
    pool_num = 20
    main.test_index_3(tech_name_list, funda_name_list, pool_num)
