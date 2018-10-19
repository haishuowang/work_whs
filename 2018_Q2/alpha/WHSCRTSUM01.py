import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import product, permutations, combinations
from sklearn.cluster import KMeans
import random
from collections import OrderedDict
from datetime import datetime
import time


class BackTest:
    @staticmethod
    def AZ_Load_csv(target_path, index_time_type=True):
        target_df = pd.read_table(target_path, sep='|', index_col=0, low_memory=False).round(8)
        if index_time_type:
            target_df.index = pd.to_datetime(target_df.index)
        return target_df

    @staticmethod
    def AZ_Rolling(df, n, min_periods=1):
        return df.rolling(window=n, min_periods=min_periods)

    @staticmethod
    def AZ_Rolling_mean(df, n, min_periods=1):
        target = df.rolling(window=n, min_periods=min_periods).mean()
        target.iloc[:n - 1] = np.nan
        return target

    @staticmethod
    def AZ_Path_create(target_path):
        """
        添加新路径
        :param target_path:
        :return:
        """
        if not os.path.exists(target_path):
            os.makedirs(target_path)

    @staticmethod
    def AZ_Sharpe_y(pnl_df):
        return round((np.sqrt(250) * pnl_df.mean()) / pnl_df.std(), 4)

    @staticmethod
    def AZ_Rolling_sharpe(pnl_df, roll_year=1, year_len=250, min_periods=1, cut_point_list=None, output=False):
        if cut_point_list is None:
            cut_point_list = [0.05, 0.33, 0.5, 0.66, 0.95]
        rolling_sharpe = pnl_df.rolling(int(roll_year * year_len), min_periods=min_periods) \
            .apply(lambda x: np.sqrt(year_len) * x.mean() / x.std(), raw=True)
        rolling_sharpe.iloc[:int(roll_year * year_len) - 1] = np.nan
        cut_sharpe = rolling_sharpe.quantile(cut_point_list)
        if output:
            return rolling_sharpe, cut_sharpe.round(4)
        else:
            return cut_sharpe.round(4)

    @staticmethod
    def AZ_Normal_IC(signal, pct_n, min_valids=None, lag=0):
        signal = signal.shift(lag)
        signal = signal.replace(0, np.nan)
        corr_df = signal.corrwith(pct_n, axis=1).dropna()

        if min_valids is not None:
            signal_valid = signal.count(axis=1)
            signal_valid[signal_valid < min_valids] = np.nan
            signal_valid[signal_valid >= min_valids] = 1
            corr_signal = corr_df * signal_valid
        else:
            corr_signal = corr_df
        return round(corr_signal, 6)

    @staticmethod
    def AZ_turnover(pos_df):
        diff_sum = pos_df.diff().abs().sum().sum()
        pos_sum = pos_df.abs().sum().sum()
        if pos_sum == 0:
            return .0
        return diff_sum / float(pos_sum)

    @staticmethod
    def AZ_annual_return(pos_df, return_df):
        temp_pnl = (pos_df * return_df).sum().sum()
        temp_pos = pos_df.abs().sum().sum()
        if temp_pos == 0:
            return .0
        else:
            return temp_pnl * 250.0 / temp_pos

    def AZ_fit_ratio(self, pos_df, return_df):
        '''
        传入仓位 和 每日收益
        :param pos_df:
        :param return_df:
        :return: 时间截面上的夏普 * sqrt（abs（年化）/换手率）， 当换手率为0时，返回0
        '''
        sharp_ratio = self.AZ_Sharpe_y((pos_df * return_df).sum(axis=1))
        ann_return = self.AZ_annual_return(pos_df, return_df)
        turnover = self.AZ_turnover(pos_df)
        if turnover == 0:
            return .0
        else:
            return round(sharp_ratio * np.sqrt(abs(ann_return) / turnover), 2)


bt = BackTest()


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
    if way == 1:
        sharpe_out = bt.AZ_Sharpe_y(pnl_df_out)
    else:
        sharpe_out = bt.AZ_Sharpe_y(-pnl_df_out)
    out_condition = sharpe_out > 0.8
    return out_condition, round(sharpe_out * way, 2)


def filter_all(cut_date, pos_df_daily, pct_n, index_df, if_hedge=True, hedge_ratio=1,
               if_return_pnl=False, if_only_long=False):
    if if_hedge:
        hedge_df = hedge_ratio * index_df.mul(pos_df_daily.sum(axis=1), axis=0)
        pnl_df = -hedge_df.sub((pos_df_daily * pct_n).sum(axis=1), axis=0)
    else:
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

        return_choose = bt.AZ_Load_csv(os.path.join(root_path, 'EM_Funda/DERIVED_14/aadj_r.csv'))
        return_choose = return_choose.reindex(columns=self.xnms)
        self.return_choose = return_choose[(return_choose.index >= begin_date) & (return_choose.index <= begin_date)]
        self.xinx = self.sector_df.index
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
        market_top_n = market_top_n[(market_top_n.index >= self.begin_date) & (market_top_n.index <= self.end_date)]
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


class FactorTestCRT(FactorTest):
    def __init__(self, *args):
        super(FactorTestCRT, self).__init__(*args)

    def load_change_factor(self, file_name):
        load_path = f'{self.root_path}/EM_Funda/daily/'
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
        load_path = f'{self.root_path}/EM_Funda/daily/'
        tmp_df = bt.AZ_Load_csv(os.path.join(load_path, file_name + '.csv')) \
            .reindex(index=self.xinx, columns=self.xnms)
        target_df = self.row_extre(tmp_df, self.sector_df, 0.2)
        return target_df

    def load_tech_factor(self, file_name):
        load_path = os.path.join('/media/hdd1/DAT_PreCalc/PreCalc_whs/' + self.sector_name)
        # load_path = os.path.join('/mnt/mfs/dat_whs/data/new_factor_data/' + self.sector_name)
        target_df = pd.read_pickle(os.path.join(load_path, file_name + '.pkl')) \
            .reindex(index=self.xinx, columns=self.xnms)
        return target_df

    @staticmethod
    def row_extre(raw_df, sector_df, percent):
        raw_df = raw_df * sector_df
        target_df = raw_df.rank(axis=1, pct=True)
        target_df[target_df >= 1 - percent] = 1
        target_df[target_df <= percent] = -1
        target_df[(target_df > percent) & (target_df < 1 - percent)] = 0
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

    def single_test(self, fun_name, name1, name2, name3):
        mix_fun_set = create_fun_set_2_crt_()
        fun = mix_fun_set[fun_name]

        choose_1 = self.load_change_factor(name1)
        choose_2 = self.load_ratio_factor(name2)
        choose_3 = self.load_tech_factor(name3)
        mix_factor = fun(choose_1, choose_2, choose_3)
        return mix_factor


def main():
    config_set = pd.read_pickle(f'/mnt/mfs/alpha_whs/CRTJUN01.pkl')
    config_data = config_set['factor_info']

    time_para_dict = dict()

    time_para_dict['time_para_1'] = [pd.to_datetime('20110101'), pd.to_datetime('20150101'),
                                     pd.to_datetime('20150701')]

    time_para_dict['time_para_2'] = [pd.to_datetime('20120101'), pd.to_datetime('20160101'),
                                     pd.to_datetime('20160701')]

    time_para_dict['time_para_3'] = [pd.to_datetime('20130601'), pd.to_datetime('20170601'),
                                     pd.to_datetime('20171201')]

    time_para_dict['time_para_4'] = [pd.to_datetime('20140601'), pd.to_datetime('20180601'),
                                     pd.to_datetime('20180901')]

    time_para_dict['time_para_5'] = [pd.to_datetime('20140701'), pd.to_datetime('20180701'),
                                     pd.to_datetime('20180901')]

    time_para_dict['time_para_6'] = [pd.to_datetime('20140801'), pd.to_datetime('20180801'),
                                     pd.to_datetime('20180901')]

    begin_date = pd.to_datetime('20140601')
    cut_date = pd.to_datetime('20180601')
    end_date = datetime.now()
    sum_factor_df = pd.DataFrame()

    root_path = '/media/hdd1/DAT_EQT'
    # root_path = '/mnt/mfs/DAT_EQT'
    if_save = False
    if_new_program = True

    sector_name = 'market_top_2000'
    index_name = '000905'
    hold_time = 20
    lag = 2
    return_file = ''

    if_hedge = True
    if_only_long = False
    time_para_dict = dict()

    main = FactorTestCRT(root_path, if_save, if_new_program, begin_date, cut_date, end_date, time_para_dict,
                         sector_name, index_name, hold_time, lag, return_file, if_hedge, if_only_long)
    print(len(config_data.index))
    for i in config_data.index:
        fun_name, name1, name2, name3, buy_sell = config_data.loc[i]
        mix_factor = main.single_test(fun_name, name1, name2, name3)

        if buy_sell > 0:
            sum_factor_df = sum_factor_df.add(mix_factor, fill_value=0)

        else:
            sum_factor_df = sum_factor_df.add(-mix_factor, fill_value=0)

    sum_pos_df = main.deal_mix_factor(sum_factor_df)
    sum_pos_df['IC01'] = -sum_pos_df.sum(axis=1)
    sum_pos_df.round(10).to_csv(f'/mnt/mfs/AAPOS/WHSCRTSUM01.pos', sep='|', index_label='Date')


if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    print(round(t2 - t1, 4))
