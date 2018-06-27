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


def load_stock_universe():
    market_top_n = pd.read_pickle('/mnt/mfs/DAT_EQT/STK_Groups1/market_top_1000.pkl')
    return market_top_n


def load_pct(begin_date, stock_universe):
    # load_path = r'/media/hdd0/whs/data/adj_data/fnd_pct/pct_f5d.pkl'
    load_path = os.path.join(root_path, 'data/adj_data/fnd_pct/pct_f5d.pkl')
    target_df = pd.read_pickle(load_path)
    target_df = target_df[target_df.index >= begin_date]
    target_df = target_df * stock_universe
    target_df.dropna(how='all', axis=0, inplace=True)
    target_df.dropna(how='all', axis=1, inplace=True)
    return target_df


def load_part_factor(begin_date, stock_universe, file_list):
    factor_set = OrderedDict()
    for file_name in file_list:
        # load_path = '/media/hdd0/whs/data/adj_data/index_universe_f'
        load_path = os.path.join(root_path, 'data/adj_data/index_universe_f')
        target_df = pd.read_pickle(os.path.join(load_path, file_name + '.pkl'))
        target_df = target_df[target_df.index >= begin_date]
        target_df = target_df * stock_universe
        target_df.dropna(how='all', axis=0, inplace=True)
        target_df.dropna(how='all', axis=1, inplace=True)
        factor_set[file_name] = target_df
    return factor_set


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
    return ic_in_condition, out_condition, ic_rolling_5_y_mean, sharpe_quantile, leve_ratio


def create_all_para():
    load_path = os.path.join(root_path, 'data/adj_data/index_universe')
    file_list = sorted(os.listdir(load_path))
    file_name_list = [x[:-4] for x in file_list]
    return combinations(file_name_list, 3)


def part_test_index_3_smart(key, name_1, name_2, name_3, begin_date, cut_date, stock_universe, return_choose,
                            log_save_file, result_save_file, if_save=True):
    lock = Lock()
    start_time = time.time()
    load_time_1 = time.time()

    factor_set = load_part_factor(begin_date, stock_universe, [name_1, name_2, name_3])
    load_time_2 = time.time()
    load_delta = round(load_time_2 - load_time_1, 4)
    fun_set = [sub_fun, add_fun]
    fun_mix_2_set = create_fun_set_2(fun_set)
    filer_name = filter_ic_leve.__name__
    for fun in fun_mix_2_set:
        mix_factor = fun(factor_set[name_1], factor_set[name_2], factor_set[name_3])
        in_condition, *filter_result = filter_ic_leve(cut_date, mix_factor, return_choose, lag=1)
        if in_condition:
            if if_save:
                with lock:
                    f = open(result_save_file, 'a')
                    write_list = [key, fun.__name__, name_1, name_2, name_3, filer_name, in_condition] + filter_result
                    f.write('|'.join([str(x) for x in write_list]) + '\n')
                    f.close()
            print([in_condition] + filter_result)
    end_time = time.time()
    if if_save:
        with lock:
            f = open(log_save_file, 'a')
            write_list = [key, name_1, name_2, name_3, filer_name, round(end_time - start_time, 4), load_delta]
            f.write('|'.join([str(x) for x in write_list]) + '\n')
            f.close()
    print('{}, {}, {}, {}, cost {} seconds, load_cost {} seconds'
          .format(key, name_1, name_2, name_3, round(end_time - start_time, 4), load_delta))


def test_index_3_smart(stock_universe, para_ready_df, begin_date, cut_date, log_save_file, result_save_file, if_save):
    a_time = time.time()
    return_choose = load_pct(begin_date, stock_universe)
    pool = Pool(20)
    for key in sorted(random.sample(list(para_ready_df.index), 4000)):
        name_1, name_2, name_3 = para_ready_df.loc[key]

        args_list = key, name_1, name_2, name_3, begin_date, cut_date, stock_universe, \
                    return_choose, log_save_file, result_save_file, if_save

        # part_test_index_3_smart(*args_list)
        pool.apply_async(part_test_index_3_smart, args=args_list)

    pool.close()
    pool.join()

    b_time = time.time()
    print('Success!Processing end, Cost {} seconds'.format(round(b_time - a_time, 2)))


def save_load_control(if_save=True, if_new_program=True):
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
    stock_universe = load_stock_universe()
    test_index_3_smart(stock_universe, para_ready_df, begin_date, cut_date, log_save_file, result_save_file, if_save)
