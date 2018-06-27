import pandas as pd
import numpy as np
import os
from itertools import product, permutations, combinations
from multiprocessing import Pool, Manager, Lock
from collections import OrderedDict
import time
from datetime import datetime
import open_lib.shared_tools.back_test as bt
import random

# product 笛卡尔积　　（有放回抽样排列）
# permutations 排列　　（不放回抽样排列）
# combinations 组合,没有重复　　（不放回抽样组合）
# combinations_with_replacement 组合,有重复　　（有放回抽样组合）

root_path = '/mnt/mfs/dat_whs'


# 读取 sector(行业 最大市值等)
def load_stock_universe(begin_date):
    # end_date = pd.to_datetime('20160401')
    market_top_n = pd.read_pickle('/mnt/mfs/DAT_EQT/STK_Groups1/market_top_1000.pkl')
    # market_top_n = market_top_n[(market_top_n.index >= begin_date) & (market_top_n.index < end_date)]
    market_top_n = market_top_n[market_top_n.index >= begin_date]
    return market_top_n


# 读取 因涨跌停以及停牌等 不能变动仓位的日期信息
def load_locked_date(begin_date):
    # end_date = pd.to_datetime('20160401')
    suspendday_df = pd.read_pickle('/mnt/mfs/DAT_EQT/EM_Tab14/adj_data/TRAD_TD_SUSPENDDAY/SUSPENDREASON_adj.pkl')
    limit_updn_df = pd.read_pickle('/mnt/mfs/dat_whs/data/locked_date/limit_updn_table.pkl')
    locked_df = limit_updn_df * suspendday_df
    locked_df.dropna(how='all', axis=0, inplace=True)
    # locked_df = locked_df[(locked_df.index >= begin_date) & (locked_df.index < end_date)]
    locked_df = locked_df[locked_df.index >= begin_date]
    return locked_df


# 读取return
def load_pct(begin_date, stock_universe):
    # load_path = r'/media/hdd0/whs/data/adj_data/fnd_pct/pct_f5d.pkl'
    # end_date = pd.to_datetime('20160401')
    load_path_5 = os.path.join(root_path, 'data/adj_data/fnd_pct/pct_f5d.pkl')
    load_path_1 = os.path.join(root_path, 'data/adj_data/fnd_pct/pct_f1d.pkl')
    target_df_5 = pd.read_pickle(load_path_5)
    # target_df_1 = pd.read_pickle(load_path_1)
    # target_df_5 = target_df_5[(target_df.index >= begin_date) & (target_df.index < end_date)]
    target_df_5 = target_df_5[target_df_5.index >= begin_date]
    target_df_5 = target_df_5 * stock_universe
    target_df_5.dropna(how='all', axis=0, inplace=True)

    # target_df_1 = target_df_1[target_df_1.index >= begin_date]
    # target_df_1 = target_df_1 * stock_universe
    # target_df_1.dropna(how='all', axis=0, inplace=True)
    return target_df_5


# 读取部分factor
def load_part_factor(begin_date, stock_universe, locked_df, file_list):
    factor_set = OrderedDict()
    # end_date = pd.to_datetime('20160401')
    for file_name in file_list:
        load_path = os.path.join(root_path, 'data/adj_data/index_universe_f')
        target_df = pd.read_pickle(os.path.join(load_path, file_name + '.pkl'))
        # target_df = target_df[(target_df.index >= begin_date) & (target_df.index < end_date)]
        target_df = target_df[target_df.index >= begin_date]

        # sector筛选
        target_df = target_df * stock_universe
        target_df.dropna(how='all', axis=0, inplace=True)

        # 排除涨跌停和停牌对策略的影响
        target_df = (target_df.fillna(0) * locked_df)
        target_df.dropna(how='all', axis=0, inplace=True)
        # target_df.dropna(how='all', axis=1, inplace=True)
        target_df.fillna(method='ffill', inplace=True)

        factor_set[file_name] = target_df
    return factor_set


# 构建每天的position
def position_daily_fun(df, n=5):
    return df.rolling(window=n, min_periods=1).sum()/n


def mul_fun(a, b):
    return a.mul(b, fill_value=0)


def sub_fun(a, b):
    return a.sub(b, fill_value=0)


def add_fun(a, b):
    return a.add(b, fill_value=0)


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


def create_log_save_path(target_path):
    root_path = os.path.split(target_path)[0]
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    if not os.path.exists(target_path):
        os.mknod(target_path)


def out_sample_perf(pnl_df_out, way=1, cut_point_list=None):
    # 根据sharpe大小,统计样本外的表现
    if cut_point_list is None:
        cut_point_list = [0.30]

    if way == 1:
        rolling_sharpe, cut_sharpe = \
            bt.AZ_Rolling_sharpe(pnl_df_out, roll_year=0.5, year_len=250, cut_point_list=cut_point_list, output=True)
    else:
        rolling_sharpe, cut_sharpe = \
            bt.AZ_Rolling_sharpe(-pnl_df_out, roll_year=0.5, year_len=250, cut_point_list=cut_point_list, output=True)

    sharpe_quantile = cut_sharpe.values[0]
    out_condition = sharpe_quantile > 0.8
    return out_condition, sharpe_quantile * way


def filter_ic(cut_date, signal, pct_n, lag=1, cost=0):
    # signal向下移动一天,避免未来函数
    signal = signal.shift(lag)
    # 将所有的0替换为nan,使得计算ic时更加合理
    signal = signal.replace(0, np.nan)

    corr_df = signal.corrwith(pct_n, axis=1)
    pnl_df = (signal * pct_n).sum(axis=1)

    corr_df_in = corr_df[corr_df.index < cut_date]
    pnl_df_out = pnl_df[pnl_df.index >= cut_date]

    ic_rolling_5_y = bt.AZ_Rolling_mean(corr_df_in, 5 * 240)
    ic_rolling_5_y_mean = ic_rolling_5_y.iloc[-1]
    ic_rolling_half_y_list = bt.AZ_Rolling_mean(corr_df_in, int(0.5 * 240)).dropna().quantile([0.1, 0.9]).values

    ic_in_condition_1 = ic_rolling_5_y_mean > 0.04 and ic_rolling_half_y_list[0] > 0.01
    ic_in_condition_2 = ic_rolling_5_y_mean < -0.04 and ic_rolling_half_y_list[1] < -0.01
    ic_in_condition = ic_in_condition_1 | ic_in_condition_2
    if ic_rolling_5_y_mean > 0:
        way = 1
        ic_rolling_half_y_quantile = ic_rolling_half_y_list[0]
    else:
        way = -1
        ic_rolling_half_y_quantile = ic_rolling_half_y_list[1]

    out_condition, sharpe_quantile = out_sample_perf(pnl_df_out, way=way)
    return ic_in_condition, out_condition, ic_rolling_5_y_mean, ic_rolling_half_y_quantile, sharpe_quantile


def filter_ic_sharpe(cut_date, signal, pct_n, lag=1):
    signal = signal.shift(lag)
    signal = signal.replace(0, np.nan)

    corr_df = signal.corrwith(pct_n, axis=1)
    pnl_df = (signal * pct_n).sum(axis=1)

    corr_df_in = corr_df[corr_df.index < cut_date]
    pnl_df_in = pnl_df[pnl_df.index < cut_date]

    pnl_df_out = pnl_df[pnl_df.index >= cut_date]

    leve_ratio = bt.AZ_Leverage_ratio(pnl_df_in)
    if leve_ratio < 0:
        leve_ratio = 100

    ic_rolling_5_y = bt.AZ_Rolling_mean(corr_df_in, 5 * 240)
    ic_rolling_5_y_mean = ic_rolling_5_y.iloc[-1]
    ic_rolling_half_y_list = bt.AZ_Rolling_mean(corr_df_in, int(0.5 * 240)).dropna().quantile([0.1, 0.9]).values
    sharpe_in = bt.AZ_Rolling_sharpe(pnl_df_in, roll_year=1, year_len=250, min_periods=1,
                                     cut_point_list=[0.3, 0.7], output=False)

    ic_in_condition_1 = ic_rolling_5_y_mean > 0.04 and sharpe_in.values[0] > 1.5
    ic_in_condition_2 = ic_rolling_5_y_mean < -0.04 and sharpe_in.values[1] < -1.5
    ic_in_condition = ic_in_condition_1 | ic_in_condition_2
    if ic_rolling_5_y_mean > 0:
        way = 1
        ic_rolling_half_y_quantile = ic_rolling_half_y_list[0]
    else:
        way = -1
        ic_rolling_half_y_quantile = ic_rolling_half_y_list[1]

    out_condition, sharpe_quantile = out_sample_perf(pnl_df_out, way=way)
    return ic_in_condition, out_condition, ic_rolling_5_y_mean, ic_rolling_half_y_quantile, leve_ratio, sharpe_quantile


def filter_ic_leve(cut_date, signal, pct_n, lag=1):
    signal = signal.shift(lag)
    signal = signal.replace(0, np.nan)

    corr_df = signal.corrwith(pct_n, axis=1)
    pnl_df = (signal * pct_n).sum(axis=1)

    corr_df_in = corr_df[corr_df.index < cut_date]
    pnl_df_in = pnl_df[pnl_df.index < cut_date]

    pnl_df_out = pnl_df[pnl_df.index >= cut_date]

    leve_ratio = bt.AZ_Leverage_ratio(pnl_df_in)
    if leve_ratio < 0:
        leve_ratio = 100

    ic_rolling_5_y = bt.AZ_Rolling_mean(corr_df_in, 5 * 240)
    ic_rolling_5_y_mean = ic_rolling_5_y.iloc[-1]

    ic_in_condition_1 = ic_rolling_5_y_mean > 0.04 and leve_ratio >= 2
    ic_in_condition_2 = ic_rolling_5_y_mean < -0.04 and leve_ratio >= 2
    ic_in_condition = ic_in_condition_1 | ic_in_condition_2
    if ic_rolling_5_y_mean > 0:
        way = 1
    else:
        way = -1

    out_condition, sharpe_quantile = out_sample_perf(pnl_df_out, way=way)
    return ic_in_condition, out_condition, ic_rolling_5_y_mean, leve_ratio, sharpe_quantile


def create_all_para():
    load_path = os.path.join(root_path, 'data/adj_data/index_universe')
    file_list = sorted(os.listdir(load_path))
    file_name_list = [x[:-4] for x in file_list]
    return combinations(file_name_list, 3)


def part_test_index_3(key, name_1, name_2, name_3, begin_date, cut_date, stock_universe, locked_df, return_choose_5,
                      log_save_file, result_save_file, if_save=True):
    lock = Lock()
    start_time = time.time()
    load_time_1 = time.time()
    # load因子,同时根据stock_universe筛选数据.
    factor_set = load_part_factor(begin_date, stock_universe, locked_df, [name_1, name_2, name_3])
    load_time_2 = time.time()
    # 加载数据时间
    load_delta = round(load_time_2 - load_time_1, 4)

    fun_set = [sub_fun, add_fun, mul_fun]
    fun_mix_2_set = create_fun_set_2(fun_set)
    #################
    # 更换filter函数 #
    #################
    filter_fun = filter_ic
    filter_name = filter_fun.__name__
    for fun in fun_mix_2_set:
        mix_factor = fun(factor_set[name_1], factor_set[name_2], factor_set[name_3])

        # 返回样本内筛选结果
        in_condition, *filter_result = filter_fun(cut_date, mix_factor, return_choose_5, lag=1)
        # result 存储
        if in_condition:
            if if_save:
                with lock:
                    f = open(result_save_file, 'a')
                    write_list = [key, fun.__name__, name_1, name_2, name_3, filter_name, in_condition] + filter_result
                    f.write('|'.join([str(x) for x in write_list]) + '\n')
                    f.close()
            print([in_condition] + filter_result)
    end_time = time.time()

    # 参数存储
    if if_save:
        with lock:
            f = open(log_save_file, 'a')
            write_list = [key, name_1, name_2, name_3, filter_name, round(end_time - start_time, 4), load_delta]
            f.write('|'.join([str(x) for x in write_list]) + '\n')
            f.close()
    print('{}, {}, {}, {}, cost {} seconds, load_cost {} seconds'
          .format(key, name_1, name_2, name_3, round(end_time - start_time, 4), load_delta))


def test_index_3(stock_universe, locked_df, para_ready_df, begin_date, cut_date, log_save_file, result_save_file,
                 if_save):
    a_time = time.time()
    return_choose_5 = load_pct(begin_date, stock_universe)
    pool = Pool(20)
    # for key in sorted(random.sample(list(para_ready_df.index), 8000)):
    for key in list(para_ready_df.index)[:100]:
        name_1, name_2, name_3 = para_ready_df.loc[key]
        args_list = key, name_1, name_2, name_3, begin_date, cut_date, stock_universe, locked_df, \
                    return_choose_5, log_save_file, result_save_file, if_save

        # part_test_index_3_smart(*args_list)
        pool.apply_async(part_test_index_3, args=args_list)

    pool.close()
    pool.join()

    b_time = time.time()
    print('Success!Processing end, Cost {} seconds'.format(round(b_time - a_time, 2)))


def save_load_control(if_save=True, if_new_program=True):
    # 参数存储与加载的路径控制
    if if_new_program:
        now_time = datetime.now().strftime('%Y%m%d_%H%M')
        log_save_file = os.path.join(root_path, 'result/log/{}.txt'.format(now_time))
        result_save_file = os.path.join(root_path, 'result/result/{}.txt'.format(now_time))
        para_save_file = os.path.join(root_path, 'result/para/{}.txt'.format(now_time))
        para_ready_df = pd.DataFrame(list(create_all_para()))
        if if_save:
            create_log_save_path(log_save_file)
            create_log_save_path(result_save_file)
            create_log_save_path(para_save_file)
            para_ready_df.to_pickle(para_save_file)

    else:
        old_time = '20180613_1940'
        log_save_file = os.path.join(root_path, 'result/log/{}.txt'.format(old_time))
        result_save_file = os.path.join(root_path, 'result/result/{}.txt'.format(old_time))
        para_save_file = os.path.join(root_path, 'result/para/{}.txt'.format(old_time))

        para_tested_df = pd.read_table(log_save_file, sep='|', header=None, index_col=0)
        para_all_df = pd.read_pickle(para_save_file)
        para_ready_df = para_all_df.loc[sorted(list(set(para_all_df.index) - set(para_tested_df.index)))]
    return para_ready_df, log_save_file, result_save_file


if __name__ == '__main__':
    begin_date = pd.to_datetime('20100101')
    cut_date = pd.to_datetime('20160401')
    end_date = pd.to_datetime('20180401')

    if_save = True
    if_new_program = True

    para_ready_df, log_save_file, result_save_file = save_load_control(if_save, if_new_program)
    # sector
    stock_universe = load_stock_universe(begin_date)
    # suspend or limit up_dn
    locked_df = load_locked_date(begin_date)

    test_index_3(stock_universe, locked_df, para_ready_df, begin_date, cut_date, log_save_file, result_save_file,
                 if_save)


