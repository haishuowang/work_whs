import pandas as pd
import numpy as np
import os
import gc
from itertools import product, permutations, combinations
from multiprocessing import Pool, Manager, Lock
from collections import OrderedDict
import time
import sys
import math


# product 笛卡尔积　　（有放回抽样排列）
# permutations 排列　　（不放回抽样排列）
# combinations 组合,没有重复　　（不放回抽样组合）
# combinations_with_replacement 组合,有重复　　（有放回抽样组合）


def mul_fun(a, b):
    return a * b


def dec_fun(a, b):
    return a - b


def add_fun(a, b):
    return a + b


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


def split_file_fun(limit=1000, file_size=40):
    load_path = r'/media/hdd0/whs/data/adj_data/index_universe'
    file_list = sorted(os.listdir(load_path))
    sub_step = int(limit / (3 * file_size))
    sub_n = int(math.ceil(len(file_list) / sub_step))
    return [file_list[x * sub_step:(x + 1) * sub_step] for x in range(sub_n)]


def load_split_index(begin_date, end_date, key_set, split_file_set):
    load_path = r'/media/hdd0/whs/data/adj_data/index_universe'
    index_set = OrderedDict()
    for key in key_set:
        sub_index_set = OrderedDict()
        for file_name in split_file_set[key]:
            target_df = pd.read_pickle(os.path.join(load_path, file_name))
            sub_index_set[file_name[:-4]] = target_df[(target_df.index <= end_date) & (target_df.index >= begin_date)]
            del target_df
        index_set[key] = sub_index_set
        del sub_index_set
    gc.collect()
    return index_set


def load_pct(begin_date, end_date):
    load_path = r'/media/hdd0/whs/data/adj_data/fnd_pct'
    file_list = sorted(os.listdir(load_path))
    return_set = OrderedDict()
    for i in range(len(file_list)):
        file_name = file_list[i]
        target_df = pd.read_pickle(os.path.join(load_path, file_name))
        target_df = target_df[(target_df.index <= end_date) & (target_df.index >= begin_date)]
        return_set[file_name[:-4]] = target_df
        del target_df
    return return_set


def filter_IR(signal, pct_n, lag=1):
    """

    :param signal:
    :param pct_n:
    :param lag:
    :return:
    """
    signal = signal.shift(lag)
    signal = signal.replace(0, np.nan)
    corr_df = signal.corrwith(pct_n, axis=1).dropna()

    IC_mean = round(corr_df.mean(), 4)
    IC_std = round(corr_df.std(), 4)
    IR = IC_mean / IC_std
    return IC_mean, IR


def AZ_Leverage_ratio(pnl_df):
    """
    返回250天的return/(负的 一个月的return)
    :param pnl_df:
    :return:
    """
    pnl_df = pd.Series(pnl_df)
    pnl_df_20 = pnl_df - pnl_df.shift(20)
    pnl_df_250 = pnl_df - pnl_df.shift(250)
    return pnl_df_250.mean() / (-pnl_df_20.min())


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
    cut_sharpe = rolling_sharpe.quantile(cut_point_list).values
    if output:
        return rolling_sharpe, cut_sharpe
    else:
        return cut_sharpe


def index_filter_fun(fun, index_choose_1, index_choose_2, index_choose_3, f_return_choose, pass_ratio=None):
    if pass_ratio is None:
        pass_ratio = [0.03, 30]

    mix_index_3 = fun(index_choose_1, index_choose_2, index_choose_3)

    ic_mean, ir = filter_IR(mix_index_3, f_return_choose, lag=1)
    pnl_df = (mix_index_3 * f_return_choose).sum(axis=1)
    cut_sharpe = rolling_sharpe_fun(pnl_df, roll_year=3, year_len=250)
    Leve_ratio = AZ_Leverage_ratio(pnl_df)
    condition = (ic_mean > 0.04) | (ic_mean < -0.04)
    return condition, ic_mean, ir, cut_sharpe, Leve_ratio


def part_test_index_3_smart(i, begin_date, end_date, key_set, split_file_set, return_set, save_file):
    lock = Lock()
    index_set = load_split_index(begin_date, end_date, key_set, split_file_set)
    fun_set = [mul_fun, dec_fun, add_fun]
    fun_mix_2_set = create_fun_set_2(fun_set)
    fun = fun_mix_2_set[i]
    if len(index_set) == 1:
        index_set_1 = index_set[list(index_set.keys())[0]]
        if len(index_set_1) >= 3:
            if 'mul' in fun.__name__ and fun.__name__[:3] != fun.__name__[4:7]:
                index_comb = permutations(list(index_set_1.keys()), 3)
            else:
                index_comb = combinations(list(index_set_1.keys()), 3)
            index_return_comb = product(list(index_comb), list(return_set.keys()))
            for (index_name_1, index_name_2, index_name_3), f_return_name in index_return_comb:
                index_choose_1 = index_set_1[index_name_1]
                index_choose_2 = index_set_1[index_name_2]
                index_choose_3 = index_set_1[index_name_3]
                f_return_choose = return_set[f_return_name]
                condition, ic_mean, ir, cut_sharpe, leve_ratio = \
                    index_filter_fun(fun, index_choose_1, index_choose_2, index_choose_3, f_return_choose)
                if condition:
                    with lock:
                        f = open(save_file, 'a')
                        f.write(','.join([fun.__name__, index_name_1, index_name_2, index_name_3, f_return_name,
                                          str(ic_mean), str(round(cut_sharpe[0], 4)),
                                          str(round(leve_ratio, 4))]) + '\n')
                        f.close()
                    print(fun.__name__, index_name_1, index_name_2, index_name_3, f_return_name, ic_mean,
                          round(cut_sharpe[0], 4), round(leve_ratio, 4))

                del index_choose_1, index_choose_2, index_choose_3, f_return_choose
                gc.collect()
        else:
            pass
    elif len(index_set) == 2:
        index_set_1 = index_set[list(index_set.keys())[0]]
        index_set_2 = index_set[list(index_set.keys())[1]]
        if 'mul' in fun.__name__ and fun.__name__[:3] != fun.__name__[4:7]:
            if len(index_set_1) >= 2:
                index_comb_2 = product(list(combinations(index_set_1.keys(), 2)),
                                       list(permutations(index_set_2.keys(), 1)))
                index_return_comb_2 = product(list(index_comb_2), list(return_set.keys()))

            if len(index_set_2) >= 2:
                index_comb_1 = product(list(combinations(index_set_1.keys(), 1)),
                                       list(permutations(index_set_2.keys(), 2)))
                index_return_comb_1 = product(list(index_comb_1), list(return_set.keys()))

        else:
            if len(index_set_1) >= 2:
                index_comb_2 = product(list(combinations(index_set_1.keys(), 2)),
                                       list(combinations(index_set_2.keys(), 1)))
                index_return_comb_2 = product(list(index_comb_2), list(return_set.keys()))

            if len(index_set_2) >= 2:
                index_comb_1 = product(list(combinations(index_set_1.keys(), 1)),
                                       list(combinations(index_set_2.keys(), 2)))
                index_return_comb_1 = product(list(index_comb_1), list(return_set.keys()))
        try:
            for ((index_name_1,), (index_name_2, index_name_3,)), f_return_name in index_return_comb_1:
                index_choose_1 = index_set_1[index_name_1]
                index_choose_2 = index_set_2[index_name_2]
                index_choose_3 = index_set_2[index_name_3]
                f_return_choose = return_set[f_return_name]
                condition, ic_mean, ir, cut_sharpe, leve_ratio = \
                    index_filter_fun(fun, index_choose_1, index_choose_2, index_choose_3, f_return_choose)
                if condition:
                    with lock:
                        f = open(save_file, 'a')
                        f.write(','.join([fun.__name__, index_name_1, index_name_2, index_name_3, f_return_name,
                                          str(ic_mean), str(round(cut_sharpe[0], 4)), str(round(leve_ratio, 4))]) + '\n')
                        f.close()
                    print(fun.__name__, index_name_1, index_name_2, index_name_3, f_return_name, ic_mean,
                          round(cut_sharpe[0], 4), round(leve_ratio, 4))
                del index_choose_1, index_choose_2, index_choose_3, f_return_choose
                gc.collect()
        except:
            pass

        try:
            for ((index_name_1, index_name_2,), (index_name_3,)), f_return_name in index_return_comb_2:
                index_choose_1 = index_set_1[index_name_1]
                index_choose_2 = index_set_1[index_name_2]
                index_choose_3 = index_set_2[index_name_3]
                f_return_choose = return_set[f_return_name]
                condition, ic_mean, ir, cut_sharpe, leve_ratio = \
                    index_filter_fun(fun, index_choose_1, index_choose_2, index_choose_3, f_return_choose)
                if condition:
                    with lock:
                        f = open(save_file, 'a')
                        f.write(','.join([fun.__name__, index_name_1, index_name_2, index_name_3, f_return_name,
                                          str(ic_mean), str(round(cut_sharpe[0], 4)), str(round(leve_ratio, 4))]) + '\n')
                        f.close()
                    print(fun.__name__, index_name_1, index_name_2, index_name_3, f_return_name, ic_mean,
                          round(cut_sharpe[0], 4), round(leve_ratio, 4))
                del index_choose_1, index_choose_2, index_choose_3, f_return_choose
                gc.collect()
        except:
            pass

    elif len(index_set) == 3:
        index_set_1 = index_set[list(index_set.keys())[0]]
        index_set_2 = index_set[list(index_set.keys())[1]]
        index_set_3 = index_set[list(index_set.keys())[2]]
        index_comb = product(list(index_set_1.keys()), list(index_set_2.keys()), list(index_set_3.keys()))
        for sub_index_comb in index_comb:
            if 'mul' in fun.__name__ and fun.__name__[:3] != fun.__name__[4:7]:
                index_comb_1 = permutations(sub_index_comb, 3)
            else:
                index_comb_1 = combinations(sub_index_comb, 3)
            index_return_comb = product(list(index_comb_1), list(return_set.keys()))
            for (index_name_1, index_name_2, index_name_3), f_return_name in index_return_comb:
                index_choose_1 = index_set_1[index_name_1]
                index_choose_2 = index_set_2[index_name_2]
                index_choose_3 = index_set_3[index_name_3]
                f_return_choose = return_set[f_return_name]
                condition, ic_mean, ir, cut_sharpe, leve_ratio = \
                    index_filter_fun(fun, index_choose_1, index_choose_2, index_choose_3, f_return_choose)
                if condition:
                    with lock:
                        f = open(save_file, 'a')
                        f.write(','.join([fun.__name__, index_name_1, index_name_2, index_name_3, f_return_name,
                                          str(ic_mean), str(round(cut_sharpe[0], 4)),
                                          str(round(leve_ratio, 4))]) + '\n')
                        f.close()
                    print(fun.__name__, index_name_1, index_name_2, index_name_3, f_return_name, ic_mean,
                          round(cut_sharpe[0], 4), round(leve_ratio, 4))
                del index_choose_1, index_choose_2, index_choose_3, f_return_choose
                gc.collect()
    else:
        print('error')


def test_index_3_smart(begin_date, end_date):
    split_file_list = split_file_fun(limit=1000, file_size=40)
    split_file_set = dict(zip(range(len(split_file_list)), split_file_list))
    return_set = load_pct(begin_date, end_date)

    key_comb = list(combinations(range(len(split_file_list)), 1)) + \
               list(combinations(range(len(split_file_list)), 2)) + \
               list(combinations(range(len(split_file_list)), 3))
    save_file = '/home/whs/Work/result/factor_search/tmp.txt'
    if not os.path.exists(save_file):
        os.mknod(save_file)
    pool = Pool(6)
    for key_set in key_comb:
        fun_set = [mul_fun, dec_fun, add_fun]
        fun_mix_2_set = create_fun_set_2(fun_set)
        for i in range(len(fun_mix_2_set)):
            # part_test_index_3_smart(lock, i, begin_date, end_date, key_set, split_file_set, return_set, save_file)
            pool.apply_async(part_test_index_3_smart,
                             args=(i, begin_date, end_date, key_set, split_file_set, return_set, save_file))
    pool.close()
    pool.join()


if __name__ == '__main__':
    begin_date = pd.to_datetime('20080401')
    end_date = pd.to_datetime('20160401')
    test_index_3_smart(begin_date, end_date)
