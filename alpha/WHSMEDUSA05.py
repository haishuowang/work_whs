import numpy as np
import pandas as pd
import os
import sys
from itertools import product, permutations, combinations
from datetime import datetime
import time
import matplotlib.pyplot as plt
import sys
sys.path.append("/mnt/mfs/LIB_ROOT")
import open_lib.shared_paths.path as pt
from open_lib.shared_tools import send_email


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
                 hold_time, lag, return_file, if_hedge, if_only_long):
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

        self.sector_df = self.load_sector_data()
        # print('Loaded sector DataFrame!')
        self.xnms = self.sector_df.columns
        self.xinx = self.sector_df.index

        index_df_1 = self.load_index_data('000300').fillna(0)
        index_df_2 = self.load_index_data('000905').fillna(0)
        hedge_df = 0.5 * index_df_1 + 0.5 * index_df_2

        return_choose = bt.AZ_Load_csv(os.path.join(root_path, 'EM_Funda/DERIVED_14/aadj_r.csv'))
        return_choose = return_choose.reindex(index=self.xinx, columns=self.xnms)
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
    def load_index_data(self, index_name):
        data = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Tab09/INDEX_TD_DAILYSYS/CHG.csv'))
        target_df = data[index_name].reindex(index=self.xinx)
        return target_df * 0.01

    def deal_mix_factor(self, mix_factor):
        if self.if_only_long:
            mix_factor = mix_factor[mix_factor > 0]
        # 下单日期pos
        order_df = mix_factor.replace(np.nan, 0)
        # 排除入场场涨跌停的影响
        order_df = order_df.div(order_df.abs().sum(axis=1).replace(0, np.nan), axis=0)
        order_df = order_df * self.sector_df * self.limit_buy_sell_df_c * self.suspendday_df_c
        order_df = order_df.astype(float)
        order_df[order_df > 0.1] = 0.1
        order_df[order_df < -0.1] = -0.1
        daily_pos = pos_daily_fun(order_df, n=self.hold_time)
        # 排除出场涨跌停的影响
        daily_pos = daily_pos * self.limit_buy_sell_df_c * self.suspendday_df_c
        daily_pos.fillna(method='ffill', inplace=True)
        return daily_pos


class FactorTestCRT(FactorTest):
    def __init__(self, *args):
        super(FactorTestCRT, self).__init__(*args)

    def load_change_factor(self, file_name):
        load_path = self.root_path + '/EM_Funda/daily/'
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
        load_path = self.root_path + '/EM_Funda/daily/'
        tmp_df = bt.AZ_Load_csv(os.path.join(load_path, file_name + '.csv')) \
            .reindex(index=self.xinx, columns=self.xnms)
        # target_df = bt.AZ_Row_zscore(tmp_df)
        target_df = self.row_extre(tmp_df, self.sector_df, 0.2)
        return target_df

    def load_tech_factor(self, file_name):
        load_path = os.path.join('/media/hdd1/DAT_PreCalc/PreCalc_whs/' + self.sector_name)
        # load_path = os.path.join('/mnt/mfs/dat_whs/data/new_factor_data/' + self.sector_name)
        target_df = pd.read_pickle(os.path.join(load_path, file_name + '.pkl')) \
            .reindex(index=self.xinx, columns=self.xnms)
        return target_df

    def single_test(self, fun_name, name1, name2, name3):
        fun_set = [add_fun, sub_fun, mul_fun]
        fun_mix_2_set = create_fun_set_2_(fun_set)
        fun = fun_mix_2_set[fun_name]
        change_factor = self.load_tech_factor(name1)
        ratio_factor = self.load_ratio_factor(name2)
        tech_factor = self.load_tech_factor(name3)
        mix_factor = fun(change_factor, ratio_factor, tech_factor)
        return mix_factor


def config_test():
    config_set = pd.read_pickle(f'/media/hdd1/DAT_PreCalc/PreCalc_whs/config_file/CRTMEDUSA05.pkl')
    config_data = config_set['factor_info']
    sector_name = config_set['sector_name']
    alpha_name = 'WHSMEDUSA05'
    cut_date = None
    begin_date = pd.to_datetime('20140601')
    end_date = datetime.now()
    # begin_date = datetime.now() - timedelta(300)

    sum_factor_df = pd.DataFrame()

    root_path = '/media/hdd1/DAT_EQT'
    # root_path = '/mnt/mfs/DAT_EQT'

    if_save = False
    if_new_program = True

    hold_time = 20
    lag = 2
    return_file = ''

    if_hedge = True
    if_only_long = False
    time_para_dict = dict()

    main = FactorTestCRT(root_path, if_save, if_new_program, begin_date, cut_date, end_date, time_para_dict,
                         sector_name, hold_time, lag, return_file, if_hedge, if_only_long)

    # print(len(config_data.index))
    for i in config_data.index:
        fun_name, name1, name2, name3, buy_sell = config_data.loc[i]
        mix_factor = main.single_test(fun_name, name1, name2, name3)

        if buy_sell > 0:
            sum_factor_df = sum_factor_df.add(mix_factor, fill_value=0)
        else:
            sum_factor_df = sum_factor_df.add(-mix_factor, fill_value=0)

    sum_pos_df_new = main.deal_mix_factor(sum_factor_df)
    sum_pos_df_new['IC01'] = -sum_pos_df_new.sum(axis=1)

    # pnl_df = (sum_pos_df_new.shift(2) * main.return_choose).sum(axis=1)
    # plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df), alpha_name)
    sum_pos_df_new.round(10).fillna(0).to_csv(f'/mnt/mfs/AAPOS/{alpha_name}.pos', sep='|', index_label='Date')
    return sum_pos_df_new


if __name__ == '__main__':
    t1 = time.time()
    sum_pos_df = config_test()
    t2 = time.time()
    print(round(t2 - t1, 4))
