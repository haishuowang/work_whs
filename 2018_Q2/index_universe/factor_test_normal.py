import pandas as pd
import numpy as np
import os
from itertools import product, permutations, combinations
from multiprocessing import Pool, Lock, cpu_count
import time
from datetime import datetime
import open_lib.shared_tools.back_test as bt
import random
# 读取数据的函数 以及
from index_universe.script_load_data import load_index_data, load_sector_data, load_locked_data, load_pct, \
    load_part_factor, create_log_save_path

from index_universe.script_filter_fun import pos_daily_fun, out_sample_perf, \
    filter_ic, filter_ic_sharpe, filter_ic_leve, filter_pot_sharpe
# product 笛卡尔积　　（有放回抽样排列）
# permutations 排列　　（不放回抽样排列）
# combinations 组合,没有重复　　（不放回抽样组合）
# combinations_with_replacement 组合,有重复　　（有放回抽样组合）

root_path = '/mnt/mfs/dat_whs'


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


def create_all_para():
    load_path = os.path.join(root_path, 'data/adj_data/index_universe')
    file_list = sorted(os.listdir(load_path))
    file_name_list = [x[:-4] for x in file_list]
    return combinations(file_name_list, 3)


def part_test_index_3(key, name_1, name_2, name_3, sector_df, locked_df, return_choose, index_df,
                      begin_date, cut_date, end_date, log_save_file, result_save_file, if_save):
    lock = Lock()
    start_time = time.time()
    load_time_1 = time.time()
    # load因子,同时根据stock_universe筛选数据.
    factor_set = load_part_factor(begin_date,  end_date, sector_df, locked_df, [name_1, name_2, name_3])
    load_time_2 = time.time()
    # 加载数据时间
    load_delta = round(load_time_2 - load_time_1, 4)
    # 生成混合函数集
    fun_set = [sub_fun, add_fun, mul_fun]
    fun_mix_2_set = create_fun_set_2(fun_set)
    #################
    # 更换filter函数 #
    #################
    filter_fun = filter_pot_sharpe
    filter_name = filter_fun.__name__
    for fun in fun_mix_2_set:
        mix_factor = fun(factor_set[name_1], factor_set[name_2], factor_set[name_3])
        # 返回样本内筛选结果
        in_condition, *filter_result = filter_fun(cut_date, mix_factor, return_choose, index_df, lag=1, hedge_ratio=1)
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


def test_index_3(sector_df, locked_df, return_choose, index_df, para_ready_df, begin_date, cut_date, end_date,
                 log_save_file, result_save_file, if_save):

    a_time = time.time()

    pool = Pool(12)
    # for key in sorted(random.sample(list(para_ready_df.index), 8000)):
    for key in list(para_ready_df.index)[:8000]:
        name_1, name_2, name_3 = para_ready_df.loc[key]

        args_list = (key, name_1, name_2, name_3, sector_df, locked_df, return_choose, index_df,
                     begin_date, cut_date, end_date, log_save_file, result_save_file, if_save)
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
    sector_df = load_sector_data(begin_date, end_date)
    sector_set = sector_df.columns
    # suspend or limit up_dn
    locked_df = load_locked_data(begin_date, end_date, sector_set)
    # return
    return_choose = load_pct(begin_date, end_date, sector_set)
    # index data
    index_df = load_index_data(begin_date, end_date)
    test_index_3(sector_df, locked_df, return_choose, index_df, para_ready_df, begin_date, cut_date, end_date,
                 log_save_file, result_save_file, if_save)


