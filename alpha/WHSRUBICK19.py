import numpy as np
import pandas as pd
import os
import sys
from itertools import product, permutations, combinations
from datetime import datetime
import time
import matplotlib.pyplot as plt
from collections import OrderedDict
import sys

sys.path.append("/mnt/mfs/LIB_ROOT")
import open_lib.shared_paths.path as pt
from open_lib.shared_tools import send_email


def plot_send_result(pnl_df, sharpe_ratio, subject, text=''):
    figure_save_path = os.path.join('/mnt/mfs/dat_whs', 'tmp_figure')
    plt.figure(figsize=[16, 8])
    plt.plot(pnl_df.index, pnl_df.cumsum(), label='sharpe_ratio={}'.format(sharpe_ratio))
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(figure_save_path, '{}.png'.format(subject)))
    plt.close()
    to = ['whs@yingpei.com']
    filepath = [os.path.join(figure_save_path, '{}.png'.format(subject))]
    send_email.send_email(text, to, filepath, subject)


class BackTest:
    @staticmethod
    def AZ_Load_csv(target_path, index_time_type=True):
        if index_time_type:
            target_df = pd.read_table(target_path, sep='|', index_col=0, low_memory=False, parse_dates=True)
        else:
            target_df = pd.read_table(target_path, sep='|', index_col=0, low_memory=False)
        return target_df

    @staticmethod
    def AZ_Catch_error(func):
        def _deco(*args, **kwargs):
            try:
                ret = func(*args, **kwargs)
            except:
                ret = sys.exc_info()
                print(ret[0], ":", ret[1])
            return ret

        return _deco

    @staticmethod
    def AZ_Time_cost(func):
        t1 = time.time()

        def _deco(*args, **kwargs):
            ret = func(*args, **kwargs)
            return ret

        t2 = time.time()
        print(f'cost_time: {t2-t1}')
        return _deco

    @staticmethod
    def AZ_Sharpe_y(pnl_df):
        return round((np.sqrt(250) * pnl_df.mean()) / pnl_df.std(), 4)

    @staticmethod
    def AZ_MaxDrawdown(asset_df):
        return asset_df - np.maximum.accumulate(asset_df)

    def AZ_Col_zscore(self, df, n, cap=None, min_periods=1):
        df_mean = self.AZ_Rolling_mean(df, n, min_periods=min_periods)
        df_std = df.rolling(window=n, min_periods=min_periods).std()
        target = (df - df_mean) / df_std
        if cap is not None:
            target[target > cap] = cap
            target[target < -cap] = -cap
        return target

    @staticmethod
    def AZ_Row_zscore(df, cap=None):
        df_mean = df.mean(axis=1)
        df_std = df.std(axis=1)
        target = df.sub(df_mean, axis=0).div(df_std, axis=0)
        if cap is not None:
            target[target > cap] = cap
            target[target < -cap] = -cap
        return target

    @staticmethod
    def AZ_Rolling(df, n, min_periods=1):
        return df.rolling(window=n, min_periods=min_periods)

    @staticmethod
    def AZ_Rolling_mean(df, n, min_periods=1):
        target = df.rolling(window=n, min_periods=min_periods).mean()
        target.iloc[:n - 1] = np.nan
        return target

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
    def AZ_Pot(pos_df, asset_last):
        """
        计算 pnl/turover*10000的值,衡量cost的影响
        :param pos_df: 仓位信息
        :param asset_last: 最后一天的收益
        :return:
        """
        trade_times = pos_df.diff().abs().sum().sum()
        if trade_times == 0:
            return 0
        else:
            pot = asset_last / trade_times * 10000
            return round(pot, 2)

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

    def AZ_Normal_IR(self, signal, pct_n, min_valids=None, lag=0):
        corr_signal = self.AZ_Normal_IC(signal, pct_n, min_valids, lag)
        ic_mean = corr_signal.mean()
        ic_std = corr_signal.std()
        ir = ic_mean / ic_std
        return ir, corr_signal

    @staticmethod
    def AZ_Leverage_ratio(asset_df):
        """
        返回250天的return/(负的 一个月的return)
        :param asset_df:
        :return:
        """
        asset_20 = asset_df - asset_df.shift(20)
        asset_250 = asset_df - asset_df.shift(250)
        if asset_250.mean() > 0:
            return asset_250.mean() / (-asset_20.min())
        else:
            return asset_250.mean() / (-asset_20.max())

    @staticmethod
    def AZ_Locked_date_deal(position_df, locked_df):
        """
        处理回测中停牌,涨停等 仓位需要锁死的情况
        :param position_df:仓位信息
        :param locked_df:停牌 涨跌停等不能交易信息(能交易记为1, 不能记为nan)
        :return:
        """

        position_df_adj = (position_df * locked_df).dropna(how='all', axis=0) \
            .fillna(method='ffill')
        return position_df_adj

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
    def AZ_split_stock(stock_list):
        """
        在stock_list中寻找A股代码
        :param stock_list:
        :return:
        """
        eqa = [x for x in stock_list if (x.startswith('0') or x.startswith('3')) and x.endwith('SZ')
               or x.startswith('6') and x.endwith('SH')]
        return eqa

    @staticmethod
    def AZ_add_stock_suffix(stock_list):
        """
        whs
        给stock_list只有数字的 A股代码 添加后缀
        如 000001 运行后 000001.SZ
        :param stock_list:
        :return:　　
        """
        return list(map(lambda x: x + '.SH' if x.startswith('6') else x + '.SZ', stock_list))

    @staticmethod
    def AZ_Delete_file(target_path, except_list=None):
        if except_list is None:
            except_list = []
        assert type(except_list) == list
        file_list = os.listdir(target_path)
        file_list = list(set(file_list) - set(except_list))
        for file_name in sorted(file_list):
            os.remove(os.path.join(target_path, file_name))

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
        """
        传入仓位 和 每日收益
        :param pos_df:
        :param return_df:
        :return: 时间截面上的夏普 * sqrt（abs（年化）/换手率）， 当换手率为0时，返回0
        """
        sharp_ratio = self.AZ_Sharpe_y((pos_df * return_df).sum(axis=1))
        ann_return = self.AZ_annual_return(pos_df, return_df)
        turnover = self.AZ_turnover(pos_df)
        if turnover == 0:
            return .0
        else:
            return round(sharp_ratio * np.sqrt(abs(ann_return) / turnover), 2)

    def AZ_fit_ratio_rolling(self, pos_df, pnl_df, roll_year=1, year_len=250, min_periods=1, cut_point_list=None,
                             output=False):
        if cut_point_list is None:
            cut_point_list = [0.05, 0.33, 0.5, 0.66, 0.95]
        rolling_sharpe, cut_sharpe = self.AZ_Rolling_sharpe(pnl_df, roll_year=roll_year, year_len=year_len,
                                                            min_periods=min_periods, cut_point_list=cut_point_list,
                                                            output=True)
        rolling_return = pnl_df.rolling(int(roll_year * year_len), min_periods=min_periods).apply(
            lambda x: 250.0 * x.sum().sum())

        rolling_diff_pos = pos_df.diff().abs().sum(axis=1).rolling(int(roll_year * year_len),
                                                                   min_periods=min_periods).apply(
            lambda x: x.sum().sum())
        rolling_return.iloc[:int(roll_year * year_len) - 1] = np.nan
        rolling_diff_pos.iloc[:int(roll_year * year_len) - 1] = np.nan
        rolling_fit_ratio = rolling_sharpe * np.sqrt(abs(rolling_return) / rolling_diff_pos)
        rolling_fit_ratio = rolling_fit_ratio.replace(np.inf, np.nan)
        rolling_fit_ratio = rolling_fit_ratio.replace(-np.inf, np.nan)
        cut_fit = rolling_fit_ratio.quantile(cut_point_list)
        return cut_fit.round(4)

    @staticmethod
    def AZ_VAR(pos_df, return_df, confidence_level, backward_len=500, forwward_len=250):
        tradeDayList = pos_df.index[:-forwward_len]
        col01 = return_df.columns[0]
        varList = []
        cut_point_list = [0.05, 0.33, 0.5, 0.66, 0.95]
        if len(tradeDayList) == 0:
            print('数据量太少')
        else:
            for tradeDay in tradeDayList:
                tempPos = pos_df.loc[tradeDay, :]
                dayIndex = list(return_df.loc[:tradeDay, col01].index[-backward_len:]) + list(
                    return_df.loc[tradeDay:, col01].index[:forwward_len])
                return_df_c = return_df[list(tempPos.index)]
                historyReturn = list(return_df_c.mul(tempPos, axis=1).loc[dayIndex[0]:dayIndex[-1], :].sum(axis=1))
                historyReturn.sort()
                varList.append(historyReturn[int(len(historyReturn) * confidence_level)])
        var = pd.DataFrame({'var': varList}, index=tradeDayList)
        var = var.dropna()
        var_fit = var.quantile(cut_point_list)
        return list(var_fit['var'])


bt = BackTest()


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
        # print('Loaded sector DataFrame!')
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
        # print('Loaded return DataFrame!')

        suspendday_df, limit_buy_sell_df = self.load_locked_data()
        limit_buy_sell_df_c = limit_buy_sell_df.shift(-1)
        limit_buy_sell_df_c.iloc[-1] = 1

        suspendday_df_c = suspendday_df.shift(-1)
        suspendday_df_c.iloc[-1] = 1
        self.suspendday_df_c = suspendday_df_c
        self.limit_buy_sell_df_c = limit_buy_sell_df_c
        # print('Loaded suspendday_df and limit_buy_sell DataFrame!')

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


class FactorTestSector(FactorTest):
    def __init__(self, *args):
        super(FactorTestSector, self).__init__(*args)

    def load_tech_factor(self, file_name):
        # load_path = os.path.join('/mnt/mfs/dat_whs/data/new_factor_data/' + self.sector_name)
        load_path = os.path.join('/media/hdd1/DAT_PreCalc/PreCalc_whs/' + self.sector_name)
        target_df = pd.read_pickle(os.path.join(load_path, file_name + '.pkl')) \
            .reindex(index=self.xinx, columns=self.xnms)
        if self.if_only_long:
            target_df = target_df[target_df > 0]
        return target_df

    def load_daily_factor(self, file_name):
        load_path = f'{self.root_path}/EM_Funda/daily/'
        tmp_df = bt.AZ_Load_csv(os.path.join(load_path, file_name + '.csv')) \
            .reindex(index=self.xinx, columns=self.xnms)

        target_df = self.row_extre(tmp_df, self.sector_df, 0.3)
        if self.if_only_long:
            target_df = target_df[target_df > 0]
        return target_df

    def load_whs_factor(self, file_name):
        load_path = f'{self.root_path}/EM_Funda/dat_whs/'
        tmp_df = bt.AZ_Load_csv(os.path.join(load_path, file_name + '.csv')) \
            .reindex(index=self.xinx, columns=self.xnms)

        target_df = self.row_extre(tmp_df, self.sector_df, 0.3)
        if self.if_only_long:
            target_df = target_df[target_df > 0]
        return target_df

    def load_remy_factor(self, file_name):
        load_path = f'{self.root_path}/EM_Funda/DERIVED_F1'
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

    def single_test(self, name_1):
        factor_1 = getattr(self, my_factor_dict[name_1])(name_1)
        daily_pos = self.deal_mix_factor(factor_1).shift(2)
        in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
        fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df = filter_all(self.cut_date, daily_pos, self.return_choose,
                                                                        if_return_pnl=True,
                                                                        if_only_long=self.if_only_long)
        if bt.AZ_Sharpe_y(pnl_df) > 0:
            return 1
        else:
            return -1

    def single_test_c(self, name_list):
        mix_factor = pd.DataFrame()
        for i in range(len(name_list)):
            tmp_name = name_list[i]
            buy_sell_way = self.single_test(tmp_name)
            tmp_factor = getattr(self, my_factor_dict[tmp_name])(tmp_name)
            mix_factor = mix_factor.add(tmp_factor * buy_sell_way, fill_value=0)
        # daily_pos = self.deal_mix_factor(mix_factor).shift(2)
        # in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
        # fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df = \
        #     filter_all(self.cut_date, daily_pos, self.return_choose, if_return_pnl=True, if_only_long=False)
        # print(in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d,
        #       pot_in, fit_ratio, leve_ratio, sp_in, sharpe_q_out)
        return mix_factor


def load_index_data(index_name, xinx):
    data = bt.AZ_Load_csv(os.path.join('/mnt/mfs/DAT_EQT', 'EM_Tab09/INDEX_TD_DAILYSYS/CHG.csv'))
    target_df = data[index_name].reindex(index=xinx)
    return target_df * 0.01


def get_corr_matrix(cut_date=None):
    pos_file_list = [x for x in os.listdir('/mnt/mfs/AAPOS') if x.startswith('WHS')]
    return_df = bt.AZ_Load_csv('/media/hdd1/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv').astype(float)

    index_df_1 = load_index_data('000300', return_df.index).fillna(0)
    index_df_2 = load_index_data('000905', return_df.index).fillna(0)

    sum_pnl_df = pd.DataFrame()
    for pos_file_name in pos_file_list:
        pos_df = bt.AZ_Load_csv('/mnt/mfs/AAPOS/{}'.format(pos_file_name))

        cond_1 = 'IF01' in pos_df.columns
        cond_2 = 'IC01' in pos_df.columns
        if cond_1 and cond_2:
            hedge_df = 0.5 * index_df_1 + 0.5 * index_df_2
            return_df_c = return_df.sub(hedge_df, axis=0)
        elif cond_1:
            hedge_df = index_df_1
            return_df_c = return_df.sub(hedge_df, axis=0)
        elif cond_2:
            hedge_df = index_df_2
            return_df_c = return_df.sub(hedge_df, axis=0)
        else:
            print('alpha hedge error')
            continue
        pnl_df = (pos_df.shift(2) * return_df_c).sum(axis=1)
        pnl_df.name = pos_file_name
        sum_pnl_df = pd.concat([sum_pnl_df, pnl_df], axis=1)
        # plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df), 'mix_factor')
    if cut_date is not None:
        sum_pnl_df = sum_pnl_df[sum_pnl_df.index > cut_date]
    return sum_pnl_df


def get_all_pnl_corr(pnl_df, col_name):
    all_pnl_df = pd.read_csv('/mnt/mfs/AATST/corr_tst_pnls', sep='|', index_col=0, parse_dates=True)
    all_pnl_df_c = pd.concat([all_pnl_df, pnl_df], axis=1)
    a = all_pnl_df_c.iloc[-600:].corr()[col_name]
    return a[a > 0.71]


def corr_test_fun(pnl_df, alpha_name):
    sum_pnl_df = get_corr_matrix(cut_date=None)
    sum_pnl_df_c = pd.concat([sum_pnl_df, pnl_df], axis=1)
    corr_self = sum_pnl_df_c.corr()[[alpha_name]]
    other_corr = get_all_pnl_corr(pnl_df, alpha_name)
    print(other_corr)
    self_corr = corr_self[corr_self > 0.7].dropna(axis=0)
    print(self_corr)
    if len(self_corr) >= 2 or len(other_corr) >= 2:
        print('FAIL!')
        send_email.send_email('FAIL!\n' + self_corr.to_html(),
                              ['whs@yingpei.com'],
                              [],
                              '[RESULT DEAL]' + alpha_name)
    else:
        print('SUCCESS!')
        send_email.send_email('SUCCESS!\n' + self_corr.to_html(),
                              ['whs@yingpei.com'],
                              [],
                              '[RESULT DEAL]' + alpha_name)
    print('______________________________________')
    return 0


def config_test():
    factor_str = 'REMFF.13|REMWB.08|bias_turn_p20d|ab_sale_mng_exp'
    info_str = 'market_top_300to800plus|5|True'

    factor_name_list = factor_str.split('|')
    alpha_name = 'WHSRUBICK19'
    sector_name, hold_time, if_only_long = info_str.split('|')
    hold_time = int(hold_time)
    if if_only_long == 'True':
        if_only_long = True
    else:
        if_only_long = False

    cut_date = '20180601'
    begin_date = pd.to_datetime('20130101')
    end_date = datetime.now()

    root_path = '/media/hdd1/DAT_EQT'
    # root_path = '/mnt/mfs/DAT_EQT'
    if_save = False
    if_new_program = True

    lag = 2
    return_file = ''

    if_hedge = True

    if sector_name.startswith('market_top_300plus'):
        if_weight = 1
        ic_weight = 0

    elif sector_name.startswith('market_top_300to800plus'):
        if_weight = 0
        ic_weight = 1

    else:
        if_weight = 0.5
        ic_weight = 0.5

    time_para_dict = dict()

    main = FactorTestSector(root_path, if_save, if_new_program, begin_date, cut_date, end_date, time_para_dict,
                            sector_name, hold_time, lag, return_file, if_hedge, if_only_long, if_weight, ic_weight)
    mix_factor = main.single_test_c(factor_name_list)

    sum_pos_df_new = main.deal_mix_factor(mix_factor)
    if if_weight != 0:
        sum_pos_df_new['IF01'] = -if_weight * sum_pos_df_new.sum(axis=1)
    if ic_weight != 0:
        sum_pos_df_new['IC01'] = -ic_weight * sum_pos_df_new.sum(axis=1)

    pnl_df = (sum_pos_df_new.shift(2) * main.return_choose).sum(axis=1)
    pnl_df.name = alpha_name

    corr_test_fun(pnl_df, alpha_name)

    plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df), alpha_name)

    sum_pos_df_new.round(10).fillna(0).to_csv(f'/mnt/mfs/AAPOS/{alpha_name}.pos', sep='|', index_label='Date')
    return sum_pos_df_new


my_factor_dict = dict({
    'RZCHE_p120d_col_extre_0.2': 'load_tech_factor',
    'RZCHE_p60d_col_extre_0.2': 'load_tech_factor',
    'RZCHE_p20d_col_extre_0.2': 'load_tech_factor',
    'RZCHE_p10d_col_extre_0.2': 'load_tech_factor',
    'RZCHE_p345d_continue_ud': 'load_tech_factor',
    'RZCHE_row_extre_0.2': 'load_tech_factor',
    'RQCHL_p120d_col_extre_0.2': 'load_tech_factor',
    'RQCHL_p60d_col_extre_0.2': 'load_tech_factor',
    'RQCHL_p20d_col_extre_0.2': 'load_tech_factor',
    'RQCHL_p10d_col_extre_0.2': 'load_tech_factor',
    'RQCHL_p345d_continue_ud': 'load_tech_factor',
    'RQCHL_row_extre_0.2': 'load_tech_factor',
    'RQYL_p120d_col_extre_0.2': 'load_tech_factor',
    'RQYL_p60d_col_extre_0.2': 'load_tech_factor',
    'RQYL_p20d_col_extre_0.2': 'load_tech_factor',
    'RQYL_p10d_col_extre_0.2': 'load_tech_factor',
    'RQYL_p345d_continue_ud': 'load_tech_factor',
    'RQYL_row_extre_0.2': 'load_tech_factor',
    'RQYE_p120d_col_extre_0.2': 'load_tech_factor',
    'RQYE_p60d_col_extre_0.2': 'load_tech_factor',
    'RQYE_p20d_col_extre_0.2': 'load_tech_factor',
    'RQYE_p10d_col_extre_0.2': 'load_tech_factor',
    'RQYE_p345d_continue_ud': 'load_tech_factor',
    'RQYE_row_extre_0.2': 'load_tech_factor',
    'RQMCL_p120d_col_extre_0.2': 'load_tech_factor',
    'RQMCL_p60d_col_extre_0.2': 'load_tech_factor',
    'RQMCL_p20d_col_extre_0.2': 'load_tech_factor',
    'RQMCL_p10d_col_extre_0.2': 'load_tech_factor',
    'RQMCL_p345d_continue_ud': 'load_tech_factor',
    'RQMCL_row_extre_0.2': 'load_tech_factor',
    'RZYE_p120d_col_extre_0.2': 'load_tech_factor',
    'RZYE_p60d_col_extre_0.2': 'load_tech_factor',
    'RZYE_p20d_col_extre_0.2': 'load_tech_factor',
    'RZYE_p10d_col_extre_0.2': 'load_tech_factor',
    'RZYE_p345d_continue_ud': 'load_tech_factor',
    'RZYE_row_extre_0.2': 'load_tech_factor',
    'RZMRE_p120d_col_extre_0.2': 'load_tech_factor',
    'RZMRE_p60d_col_extre_0.2': 'load_tech_factor',
    'RZMRE_p20d_col_extre_0.2': 'load_tech_factor',
    'RZMRE_p10d_col_extre_0.2': 'load_tech_factor',
    'RZMRE_p345d_continue_ud': 'load_tech_factor',
    'RZMRE_row_extre_0.2': 'load_tech_factor',
    'RZRQYE_p120d_col_extre_0.2': 'load_tech_factor',
    'RZRQYE_p60d_col_extre_0.2': 'load_tech_factor',
    'RZRQYE_p20d_col_extre_0.2': 'load_tech_factor',
    'RZRQYE_p10d_col_extre_0.2': 'load_tech_factor',
    'RZRQYE_p345d_continue_ud': 'load_tech_factor',
    'RZRQYE_row_extre_0.2': 'load_tech_factor',
    'WILLR_200_40': 'load_tech_factor',
    'WILLR_200_30': 'load_tech_factor',
    'WILLR_200_20': 'load_tech_factor',
    'WILLR_140_40': 'load_tech_factor',
    'WILLR_140_30': 'load_tech_factor',
    'WILLR_140_20': 'load_tech_factor',
    'WILLR_100_40': 'load_tech_factor',
    'WILLR_100_30': 'load_tech_factor',
    'WILLR_100_20': 'load_tech_factor',
    'WILLR_40_40': 'load_tech_factor',
    'WILLR_40_30': 'load_tech_factor',
    'WILLR_40_20': 'load_tech_factor',
    'WILLR_20_40': 'load_tech_factor',
    'WILLR_20_30': 'load_tech_factor',
    'WILLR_20_20': 'load_tech_factor',
    'WILLR_10_40': 'load_tech_factor',
    'WILLR_10_30': 'load_tech_factor',
    'WILLR_10_20': 'load_tech_factor',
    'BBANDS_10_2': 'load_tech_factor',
    'BBANDS_10_1.5': 'load_tech_factor',
    'BBANDS_10_1': 'load_tech_factor',
    'MACD_20_60_18': 'load_tech_factor',
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
    'MACD_12_26_9': 'load_tech_factor',
    'intra_up_vwap_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_up_vol_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_up_div_dn_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_up_div_daily_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_up_15_bar_vwap_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_up_15_bar_vol_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_up_15_bar_div_dn_15_bar_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_up_15_bar_div_daily_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_dn_vwap_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_dn_vol_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_dn_div_daily_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_dn_15_bar_vwap_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_dn_15_bar_vol_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_dn_15_bar_div_daily_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_up_vwap_row_extre_0.3': 'load_tech_factor',
    'intra_up_vol_row_extre_0.3': 'load_tech_factor',
    'intra_up_div_dn_row_extre_0.3': 'load_tech_factor',
    'intra_up_div_daily_row_extre_0.3': 'load_tech_factor',
    'intra_up_15_bar_vwap_row_extre_0.3': 'load_tech_factor',
    'intra_up_15_bar_vol_row_extre_0.3': 'load_tech_factor',
    'intra_up_15_bar_div_dn_15_bar_row_extre_0.3': 'load_tech_factor',
    'intra_up_15_bar_div_daily_row_extre_0.3': 'load_tech_factor',
    'intra_dn_vwap_row_extre_0.3': 'load_tech_factor',
    'intra_dn_vol_row_extre_0.3': 'load_tech_factor',
    'intra_dn_div_daily_row_extre_0.3': 'load_tech_factor',
    'intra_dn_15_bar_vwap_row_extre_0.3': 'load_tech_factor',
    'intra_dn_15_bar_vol_row_extre_0.3': 'load_tech_factor',
    'intra_dn_15_bar_div_daily_row_extre_0.3': 'load_tech_factor',
    'tab5_15_row_extre_0.3': 'load_tech_factor',
    'tab5_14_row_extre_0.3': 'load_tech_factor',
    'tab5_13_row_extre_0.3': 'load_tech_factor',
    'tab4_5_row_extre_0.3': 'load_tech_factor',
    'tab4_2_row_extre_0.3': 'load_tech_factor',
    'tab4_1_row_extre_0.3': 'load_tech_factor',
    'tab2_11_row_extre_0.3': 'load_tech_factor',
    'tab2_9_row_extre_0.3': 'load_tech_factor',
    'tab2_8_row_extre_0.3': 'load_tech_factor',
    'tab2_7_row_extre_0.3': 'load_tech_factor',
    'tab2_4_row_extre_0.3': 'load_tech_factor',
    'tab2_1_row_extre_0.3': 'load_tech_factor',
    'tab1_9_row_extre_0.3': 'load_tech_factor',
    'tab1_8_row_extre_0.3': 'load_tech_factor',
    'tab1_7_row_extre_0.3': 'load_tech_factor',
    'tab1_5_row_extre_0.3': 'load_tech_factor',
    'tab1_2_row_extre_0.3': 'load_tech_factor',
    'tab1_1_row_extre_0.3': 'load_tech_factor',
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
    'TotRev_and_mcap_intdebt_QYOY_Y3YGR_0.3': 'load_tech_factor',
    'TotRev_and_asset_QYOY_Y3YGR_0.3': 'load_tech_factor',
    'TotRev_and_mcap_QYOY_Y3YGR_0.3': 'load_tech_factor',
    'TotRev_and_mcap_intdebt_Y3YGR_Y5YGR_0.3': 'load_tech_factor',
    'TotRev_and_asset_Y3YGR_Y5YGR_0.3': 'load_tech_factor',
    'TotRev_and_mcap_Y3YGR_Y5YGR_0.3': 'load_tech_factor',
    'NetProfit_and_mcap_intdebt_QYOY_Y3YGR_0.3': 'load_tech_factor',
    'NetProfit_and_asset_QYOY_Y3YGR_0.3': 'load_tech_factor',
    'NetProfit_and_mcap_QYOY_Y3YGR_0.3': 'load_tech_factor',
    'NetProfit_and_mcap_intdebt_Y3YGR_Y5YGR_0.3': 'load_tech_factor',
    'NetProfit_and_asset_Y3YGR_Y5YGR_0.3': 'load_tech_factor',
    'NetProfit_and_mcap_Y3YGR_Y5YGR_0.3': 'load_tech_factor',
    'EBIT_and_mcap_intdebt_QYOY_Y3YGR_0.3': 'load_tech_factor',
    'EBIT_and_asset_QYOY_Y3YGR_0.3': 'load_tech_factor',
    'EBIT_and_mcap_QYOY_Y3YGR_0.3': 'load_tech_factor',
    'EBIT_and_mcap_intdebt_Y3YGR_Y5YGR_0.3': 'load_tech_factor',
    'EBIT_and_asset_Y3YGR_Y5YGR_0.3': 'load_tech_factor',
    'EBIT_and_mcap_Y3YGR_Y5YGR_0.3': 'load_tech_factor',
    'OPCF_and_mcap_intdebt_QYOY_Y3YGR_0.3': 'load_tech_factor',
    'OPCF_and_asset_QYOY_Y3YGR_0.3': 'load_tech_factor',
    'OPCF_and_mcap_QYOY_Y3YGR_0.3': 'load_tech_factor',
    'OPCF_and_mcap_intdebt_Y3YGR_Y5YGR_0.3': 'load_tech_factor',
    'OPCF_and_asset_Y3YGR_Y5YGR_0.3': 'load_tech_factor',
    'OPCF_and_mcap_Y3YGR_Y5YGR_0.3': 'load_tech_factor',
    'R_OTHERLASSET_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_WorkCapital_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_TangAssets_IntDebt_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_SUMLIAB_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_ROE1_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_OPEX_sales_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_OperProfit_YOY_First_and_QTTM_0.3': 'load_tech_factor',
    'R_OperCost_sales_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_OPCF_TTM_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_NETPROFIT_s_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_NetInc_s_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_NetAssets_s_YOY_First_and_QTTM_0.3': 'load_tech_factor',
    'R_LOANREC_s_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_LTDebt_WorkCap_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_INVESTINCOME_s_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_IntDebt_Mcap_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_GSCF_sales_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_GrossProfit_TTM_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_FINANCEEXP_s_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_FairVal_TotProfit_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_ESTATEINVEST_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_EPSDiluted_YOY_First_and_QTTM_0.3': 'load_tech_factor',
    'R_EBITDA2_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_CostSales_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_CFO_s_YOY_First_and_QTTM_0.3': 'load_tech_factor',
    'R_Cashflow_s_YOY_First_and_QTTM_0.3': 'load_tech_factor',
    'R_ASSETDEVALUELOSS_s_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_ACCOUNTREC_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_ACCOUNTPAY_QYOY_and_QTTM_0.3': 'load_tech_factor',
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
    'PBLast_p120d_col_extre_0.2': 'load_tech_factor',
    'PBLast_p60d_col_extre_0.2': 'load_tech_factor',
    'PBLast_p20d_col_extre_0.2': 'load_tech_factor',
    'PBLast_p10d_col_extre_0.2': 'load_tech_factor',
    'PBLast_p345d_continue_ud': 'load_tech_factor',
    'PBLast_row_extre_0.2': 'load_tech_factor',
    'PS_TTM_p120d_col_extre_0.2': 'load_tech_factor',
    'PS_TTM_p60d_col_extre_0.2': 'load_tech_factor',
    'PS_TTM_p20d_col_extre_0.2': 'load_tech_factor',
    'PS_TTM_p10d_col_extre_0.2': 'load_tech_factor',
    'PS_TTM_p345d_continue_ud': 'load_tech_factor',
    'PS_TTM_row_extre_0.2': 'load_tech_factor',
    'PE_TTM_p120d_col_extre_0.2': 'load_tech_factor',
    'PE_TTM_p60d_col_extre_0.2': 'load_tech_factor',
    'PE_TTM_p20d_col_extre_0.2': 'load_tech_factor',
    'PE_TTM_p10d_col_extre_0.2': 'load_tech_factor',
    'PE_TTM_p345d_continue_ud': 'load_tech_factor',
    'PE_TTM_row_extre_0.2': 'load_tech_factor',
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

    'R_ACCOUNTPAY_QYOY': 'load_daily_factor',
    'R_ACCOUNTREC_QYOY': 'load_daily_factor',
    'R_ASSETDEVALUELOSS_s_QYOY': 'load_daily_factor',
    'R_AssetDepSales_s_First': 'load_daily_factor',
    'R_BusinessCycle_First': 'load_daily_factor',
    'R_CFOPS_s_First': 'load_daily_factor',
    'R_CFO_TotRev_s_First': 'load_daily_factor',
    'R_CFO_s_YOY_First': 'load_daily_factor',
    'R_Cashflow_s_YOY_First': 'load_daily_factor',
    'R_CostSales_QYOY': 'load_daily_factor',
    'R_CostSales_s_First': 'load_daily_factor',
    'R_CurrentAssetsTurnover_QTTM': 'load_daily_factor',
    'R_DaysReceivable_First': 'load_daily_factor',
    'R_DebtAssets_QTTM': 'load_daily_factor',
    'R_DebtEqt_First': 'load_daily_factor',
    'R_EBITDA2_QYOY': 'load_daily_factor',
    'R_EBITDA_IntDebt_QTTM': 'load_daily_factor',
    'R_EBITDA_sales_TTM_First': 'load_daily_factor',
    'R_EBIT_sales_QTTM': 'load_daily_factor',
    'R_EPS_s_First': 'load_daily_factor',
    'R_EPS_s_YOY_First': 'load_daily_factor',
    'R_ESTATEINVEST_QYOY': 'load_daily_factor',
    'R_FCFTot_Y3YGR': 'load_daily_factor',
    'R_FINANCEEXP_s_QYOY': 'load_daily_factor',
    'R_FairValChgPnL_s_First': 'load_daily_factor',
    'R_FairValChg_TotProfit_s_First': 'load_daily_factor',
    'R_FairVal_TotProfit_QYOY': 'load_daily_factor',
    'R_FairVal_TotProfit_TTM_First': 'load_daily_factor',
    'R_FinExp_sales_s_First': 'load_daily_factor',
    'R_GSCF_sales_s_First': 'load_daily_factor',
    'R_GrossProfit_TTM_QYOY': 'load_daily_factor',
    'R_INVESTINCOME_s_QYOY': 'load_daily_factor',
    'R_LTDebt_WorkCap_QTTM': 'load_daily_factor',
    'R_MgtExp_sales_s_First': 'load_daily_factor',
    'R_NETPROFIT_s_QYOY': 'load_daily_factor',
    'R_NOTICEDATE_First': 'load_daily_factor',
    'R_NetAssets_s_POP_First': 'load_daily_factor',
    'R_NetAssets_s_YOY_First': 'load_daily_factor',
    'R_NetCashflowPS_s_First': 'load_daily_factor',
    'R_NetIncRecur_QYOY': 'load_daily_factor',
    'R_NetIncRecur_s_First': 'load_daily_factor',
    'R_NetInc_TotProfit_s_First': 'load_daily_factor',
    'R_NetInc_s_First': 'load_daily_factor',
    'R_NetInc_s_QYOY': 'load_daily_factor',
    'R_NetMargin_s_YOY_First': 'load_daily_factor',
    'R_NetProfit_sales_s_First': 'load_daily_factor',
    'R_NetROA_TTM_First': 'load_daily_factor',
    'R_NetROA_s_First': 'load_daily_factor',
    'R_NonOperProft_TotProfit_s_First': 'load_daily_factor',
    'R_OPCF_NetInc_s_First': 'load_daily_factor',
    'R_OPCF_TTM_QYOY': 'load_daily_factor',
    'R_OPCF_TotDebt_QTTM': 'load_daily_factor',
    'R_OPCF_sales_s_First': 'load_daily_factor',
    'R_OPEX_sales_TTM_First': 'load_daily_factor',
    'R_OPEX_sales_s_First': 'load_daily_factor',
    'R_OTHERLASSET_QYOY': 'load_daily_factor',
    'R_OperCost_sales_s_First': 'load_daily_factor',
    'R_OperProfit_YOY_First': 'load_daily_factor',
    'R_OperProfit_s_POP_First': 'load_daily_factor',
    'R_OperProfit_s_YOY_First': 'load_daily_factor',
    'R_OperProfit_sales_s_First': 'load_daily_factor',
    'R_ParentProfit_s_POP_First': 'load_daily_factor',
    'R_ParentProfit_s_YOY_First': 'load_daily_factor',
    'R_ROENetIncRecur_s_First': 'load_daily_factor',
    'R_ROE_s_First': 'load_daily_factor',
    'R_RecurNetProft_NetProfit_s_First': 'load_daily_factor',
    'R_RevenuePS_s_First': 'load_daily_factor',
    'R_RevenueTotPS_s_First': 'load_daily_factor',
    'R_Revenue_s_POP_First': 'load_daily_factor',
    'R_Revenue_s_YOY_First': 'load_daily_factor',
    'R_SUMLIAB_QYOY': 'load_daily_factor',
    'R_SUMLIAB_Y3YGR': 'load_daily_factor',
    'R_SalesCost_s_First': 'load_daily_factor',
    'R_SalesGrossMGN_QTTM': 'load_daily_factor',
    'R_SalesGrossMGN_s_First': 'load_daily_factor',
    'R_SalesNetMGN_s_First': 'load_daily_factor',
    'R_TangAssets_TotLiab_QTTM': 'load_daily_factor',
    'R_Tax_TotProfit_QTTM': 'load_daily_factor',
    'R_Tax_TotProfit_s_First': 'load_daily_factor',
    'R_TotAssets_s_YOY_First': 'load_daily_factor',
    'R_TotLiab_s_YOY_First': 'load_daily_factor',
    'R_TotRev_TTM_Y3YGR': 'load_daily_factor',
    'R_TotRev_s_POP_First': 'load_daily_factor',
    'R_TotRev_s_YOY_First': 'load_daily_factor',
    'R_WorkCapital_QYOY': 'load_daily_factor',

    'bar_num_7_df': 'load_whs_factor',
    'bar_num_12_df': 'load_whs_factor',
    'repurchase': 'load_whs_factor',
    'dividend': 'load_whs_factor',
    'repurchase_news_title': 'load_whs_factor',
    'repurchase_news_summary': 'load_whs_factor',
    'dividend_news_title': 'load_whs_factor',
    'dividend_news_summary': 'load_whs_factor',
    'staff_changes_news_title': 'load_whs_factor',
    'staff_changes_news_summary': 'load_whs_factor',
    'funds_news_title': 'load_whs_factor',
    'funds_news_summary': 'load_whs_factor',
    'meeting_decide_news_title': 'load_whs_factor',
    'meeting_decide_news_summary': 'load_whs_factor',
    'restricted_shares_news_title': 'load_whs_factor',
    'restricted_shares_news_summary': 'load_whs_factor',
    'son_company_news_title': 'load_whs_factor',
    'son_company_news_summary': 'load_whs_factor',
    'suspend_news_title': 'load_whs_factor',
    'suspend_news_summary': 'load_whs_factor',
    'shares_news_title': 'load_whs_factor',
    '': 'load_whs_factor',
    'shares_news_summary': 'load_whs_factor',
    'ab_inventory': 'load_whs_factor',
    'ab_rec': 'load_whs_factor',
    'ab_others_rec': 'load_whs_factor',
    'ab_ab_pre_rec': 'load_whs_factor',
    'ab_sale_mng_exp': 'load_whs_factor',
    'ab_grossprofit': 'load_whs_factor',
    'lsgg_num_df_5': 'load_whs_factor',
    'lsgg_num_df_20': 'load_whs_factor',
    'lsgg_num_df_60': 'load_whs_factor',
    'bulletin_num_df': 'load_whs_factor',
    'bulletin_num_df_5': 'load_whs_factor',
    'bulletin_num_df_20': 'load_whs_factor',
    'bulletin_num_df_60': 'load_whs_factor',
    'news_num_df_5': 'load_whs_factor',
    'news_num_df_20': 'load_whs_factor',
    'news_num_df_60': 'load_whs_factor',
    'staff_changes': 'load_whs_factor',
    'funds': 'load_whs_factor',
    'meeting_decide': 'load_whs_factor',
    'restricted_shares': 'load_whs_factor',
    'son_company': 'load_whs_factor',
    'suspend': 'load_whs_factor',
    'shares': 'load_whs_factor',
    'buy_key_title__word': 'load_whs_factor',
    'sell_key_title_word': 'load_whs_factor',
    'buy_summary_key_word': 'load_whs_factor',
    'sell_summary_key_word': 'load_whs_factor',

})
my_factor_dict_2 = dict({
    'REMTK.40': 'load_remy_factor',
    'REMTK.39': 'load_remy_factor',
    'REMTK.38': 'load_remy_factor',
    'REMTK.37': 'load_remy_factor',
    'REMTK.36': 'load_remy_factor',
    'REMTK.35': 'load_remy_factor',
    'REMTK.34': 'load_remy_factor',
    'REMTK.33': 'load_remy_factor',
    'REMTK.32': 'load_remy_factor',
    'REMTK.31': 'load_remy_factor',
    'REMFF.40': 'load_remy_factor',
    'REMFF.39': 'load_remy_factor',
    'REMFF.38': 'load_remy_factor',
    'REMFF.37': 'load_remy_factor',
    'REMFF.36': 'load_remy_factor',
    'REMFF.35': 'load_remy_factor',
    'REMFF.34': 'load_remy_factor',
    'REMFF.33': 'load_remy_factor',
    'REMFF.32': 'load_remy_factor',
    'REMFF.31': 'load_remy_factor',
    'REMWB.12': 'load_remy_factor',
    'REMWB.11': 'load_remy_factor',
    'REMWB.10': 'load_remy_factor',
    'REMWB.09': 'load_remy_factor',
    'REMWB.08': 'load_remy_factor',
    'REMWB.07': 'load_remy_factor',
    'REMWB.06': 'load_remy_factor',
    'REMWB.05': 'load_remy_factor',
    'REMWB.04': 'load_remy_factor',
    'REMWB.03': 'load_remy_factor',
    'REMWB.02': 'load_remy_factor',
    'REMWB.01': 'load_remy_factor',
    'REMTK.30': 'load_remy_factor',
    'REMTK.29': 'load_remy_factor',
    'REMTK.28': 'load_remy_factor',
    'REMTK.27': 'load_remy_factor',
    'REMTK.26': 'load_remy_factor',
    'REMTK.25': 'load_remy_factor',
    'REMTK.24': 'load_remy_factor',
    'REMTK.23': 'load_remy_factor',
    'REMTK.22': 'load_remy_factor',
    'REMTK.21': 'load_remy_factor',
    'REMTK.20': 'load_remy_factor',
    'REMTK.19': 'load_remy_factor',
    'REMTK.18': 'load_remy_factor',
    'REMTK.17': 'load_remy_factor',
    'REMTK.16': 'load_remy_factor',
    'REMTK.15': 'load_remy_factor',
    'REMTK.14': 'load_remy_factor',
    'REMTK.13': 'load_remy_factor',
    'REMTK.12': 'load_remy_factor',
    'REMTK.11': 'load_remy_factor',
    'REMTK.10': 'load_remy_factor',
    'REMTK.09': 'load_remy_factor',
    'REMTK.08': 'load_remy_factor',
    'REMTK.07': 'load_remy_factor',
    'REMTK.06': 'load_remy_factor',
    'REMTK.05': 'load_remy_factor',
    'REMTK.04': 'load_remy_factor',
    'REMTK.03': 'load_remy_factor',
    'REMTK.02': 'load_remy_factor',
    'REMTK.01': 'load_remy_factor',
    'REMFF.30': 'load_remy_factor',
    'REMFF.29': 'load_remy_factor',
    'REMFF.28': 'load_remy_factor',
    'REMFF.27': 'load_remy_factor',
    'REMFF.26': 'load_remy_factor',
    'REMFF.25': 'load_remy_factor',
    'REMFF.24': 'load_remy_factor',
    'REMFF.23': 'load_remy_factor',
    'REMFF.22': 'load_remy_factor',
    'REMFF.21': 'load_remy_factor',
    'REMFF.20': 'load_remy_factor',
    'REMFF.19': 'load_remy_factor',
    'REMFF.18': 'load_remy_factor',
    'REMFF.17': 'load_remy_factor',
    'REMFF.16': 'load_remy_factor',
    'REMFF.15': 'load_remy_factor',
    'REMFF.14': 'load_remy_factor',
    'REMFF.13': 'load_remy_factor',
    'REMFF.12': 'load_remy_factor',
    'REMFF.11': 'load_remy_factor',
    'REMFF.10': 'load_remy_factor',
    'REMFF.09': 'load_remy_factor',
    'REMFF.08': 'load_remy_factor',
    'REMFF.07': 'load_remy_factor',
    'REMFF.06': 'load_remy_factor',
    'REMFF.05': 'load_remy_factor',
    'REMFF.04': 'load_remy_factor',
    'REMFF.03': 'load_remy_factor',
    'REMFF.02': 'load_remy_factor',
    'REMFF.01': 'load_remy_factor'
})
my_factor_dict.update(my_factor_dict_2)

if __name__ == '__main__':
    t1 = time.time()
    sum_pos_df = config_test()
    t2 = time.time()
    print(round(t2 - t1, 4))
