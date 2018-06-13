import pandas as pd
import numpy as np
import os
import gc
from itertools import product, permutations, combinations
from multiprocessing import Pool, Manager, Lock
from collections import OrderedDict
import time
from datetime import datetime
import sys
import math
import feather

# product 笛卡尔积　　（有放回抽样排列）
# permutations 排列　　（不放回抽样排列）
# combinations 组合,没有重复　　（不放回抽样组合）
# combinations_with_replacement 组合,有重复　　（有放回抽样组合）

# 王海硕 = True


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


def create_log_save_path(log_path):
    if not os.path.exists(log_path):
        os.mknod(log_path)


def load_pct(begin_date, end_date):
    load_path = r'/media/hdd0/whs/data/adj_data/fnd_pct/pct_f5d.pkl'
    target_df = pd.read_pickle(load_path)
    target_df = target_df[(target_df.index <= end_date) & (target_df.index >= begin_date)]
    return target_df


def load_part_factor(begin_date, end_date, file_list):
    factor_set = OrderedDict()
    for file_name in file_list:
        load_path = '/media/hdd0/whs/data/adj_data/index_universe_f'
        target_df = pd.read_pickle(os.path.join(load_path, file_name + '.pkl'))
        factor_set[file_name] = target_df[(target_df.index <= end_date) & (target_df.index >= begin_date)]
    return factor_set


def filter_IR(signal, pct_n):
    """

    :param signal:
    :param pct_n:
    :param lag:
    :return:
    """
    signal = signal.replace(0, np.nan)
    corr_df = signal.corrwith(pct_n, axis=1).dropna()

    ic_mean = round(corr_df.mean(), 4)
    ic_std = round(corr_df.std(), 4)
    ir = ic_mean / ic_std
    condition = (ic_mean > 0.05) | (ic_mean < -0.05)
    return ic_mean, round(ir, 4)


def AZ_Leverage_ratio(pnl_df):
    """
    返回250天的return/(负的 一个月的return)
    :param pnl_df:
    :return:
    """
    pnl_df = pd.Series(pnl_df)
    pnl_df_20 = pnl_df - pnl_df.shift(20)
    pnl_df_250 = pnl_df - pnl_df.shift(250)
    if pnl_df_20.min().values != 0:
        return round(pnl_df_250.mean().values / (-pnl_df_20.min().values), 4)
    else:
        return 0


def rolling_sharpe_fun(pnl_df, roll_year=1, year_len=250, cut_point_list=None, output=False):
    """
    rolling sharpe
    :param pnl_df:
    :param roll_year:
    :param year_len:
    :param cut_point_list:
    :param output:
    :return:
    """
    if cut_point_list is None:
        cut_point_list = [0.05, 0.33, 0.5, 0.66, 0.95]
    rolling_sharpe = pnl_df.rolling(int(roll_year * year_len)) \
        .apply(lambda x: np.sqrt(year_len) * x.mean() / x.std())
    cut_sharpe = rolling_sharpe.quantile(cut_point_list).round(4).values
    if output:
        return rolling_sharpe, cut_sharpe
    else:
        return cut_sharpe


def index_filter_fun(mix_factor, return_choose, lag=1, pass_ratio=None):
    if pass_ratio is None:
        pass_ratio = [0.03, 30]
    mix_factor = mix_factor.shift(lag)
    ic_mean, ir = filter_IR(mix_factor, return_choose)
    condition = (ic_mean > 0.05) | (ic_mean < -0.05)
    return condition, ic_mean, ir


def create_all_para():
    load_path = r'/media/hdd0/whs/data/adj_data/index_universe'
    file_list = sorted(os.listdir(load_path))
    file_name_list = [x[:-4] for x in file_list]
    return combinations(file_name_list, 3)


def part_test_index_3_smart(name_1, name_2, name_3, return_choose, log_save_file, result_save_file, if_save=True):
    lock = Lock()
    start_time = time.time()
    load_time_1 = time.time()
    factor_set = load_part_factor(begin_date, end_date, [name_1, name_2, name_3])
    load_time_2 = time.time()
    load_delta = round(load_time_2 - load_time_1, 4)
    fun_set = [sub_fun, add_fun]
    fun_mix_2_set = create_fun_set_2(fun_set)
    for fun in fun_mix_2_set:
        mix_factor = fun(factor_set[name_1], factor_set[name_2], factor_set[name_3])
        condition, ic_mean, ir = index_filter_fun(mix_factor, return_choose, pass_ratio=None)
        if condition:
            if if_save:
                with lock:
                    f = open(result_save_file, 'a')
                    write_list = [fun.__name__, name_1, name_2, name_3, ic_mean, ir]

                    f.write('|'.join([str(x) for x in write_list]) + '\n')
                    f.close()
            print('{}, {}, {}, {}, {}, {}'
                  .format(fun.__name__, name_1, name_2, name_3, ic_mean, ir))

    end_time = time.time()
    if if_save:
        with lock:
            f = open(log_save_file, 'a')
            write_list = [name_1, name_2, name_3, round(end_time - start_time, 4), load_delta]
            f.write('|'.join([str(x) for x in write_list]) + '\n')
            f.close()
    print('{}, {}, {}, run 1 cost {} seconds, load_cost {} seconds'
          .format(name_1, name_2, name_3, round(end_time - start_time, 4), str(load_delta)))


def test_index_3_smart(begin_date, end_date, log_save_file, result_save_file, if_save):
    a_time = time.time()
    return_choose = load_pct(begin_date, end_date)
    pool = Pool(25)
    for name_1, name_2, name_3 in sorted(list(create_all_para())):
        # part_test_index_3_smart(name_1, name_2, name_3, return_choose, log_save_file, result_save_file)
        pool.apply_async(part_test_index_3_smart, args=(name_1, name_2, name_3, return_choose,
                                                        log_save_file, result_save_file, if_save))
    pool.close()
    pool.join()

    b_time = time.time()
    print('Success!Processing end, Cost {} seconds'.format(round(b_time - a_time, 2)))


if __name__ == '__main__':
    begin_date = pd.to_datetime('20080401')
    end_date = pd.to_datetime('20160401')
    now_time = datetime.now().strftime('%Y%m%d_%H%M')
    log_save_file = '/media/hdd0/whs/result/log/{}.txt'.format(now_time)
    result_save_file = '/media/hdd0/whs/result/result/{}.txt'.format(now_time)

    if_save = False
    if_new_program = True

    para_df = pd.DataFrame(list(create_all_para()))
    if if_save:
        if if_new_program:
            create_log_save_path(log_save_file)
            create_log_save_path(result_save_file)
        else:
            old_time = '20180613_0937'
            log_save_file = '/media/hdd0/whs/result/log/{}.txt'.format(old_time)
            result_save_file = '/media/hdd0/whs/result/result/{}.txt'.format(old_time)
            para_save_file = '/media/hdd0/whs/result/para/{}.txt'.format(now_time)

    if if_new_program:

    test_index_3_smart(begin_date, end_date, log_save_file, result_save_file, if_save)

