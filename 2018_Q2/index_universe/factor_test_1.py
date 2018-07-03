import pandas as pd
import numpy as np
import os
from itertools import product, permutations, combinations
from multiprocessing import Pool, Lock
import time
from datetime import datetime
import open_lib.shared_tools.back_test as bt
import random
# 读取数据的函数 以及
from index_universe.script_load_data import load_sector_df, load_locked_date, load_pct, \
    load_part_factor, create_log_save_path

from index_universe.script_filter_fun import pos_daily_fun, out_sample_perf, \
    filter_ic, filter_ic_sharpe, filter_ic_leve, filter_pot_sharpe
# product 笛卡尔积　　（有放回抽样排列）
# permutations 排列　　（不放回抽样排列）
# combinations 组合,没有重复　　（不放回抽样组合）
# combinations_with_replacement 组合,有重复　　（有放回抽样组合）


def test_index_1(begin_date, cut_date, file_name, stock_universe, locked_df, return_choose, result_save_file):
    lock = Lock()
    file_name = file_name[:-4]
    factor = load_part_factor(begin_date, stock_universe, locked_df, [file_name])[file_name]
    filter_result = filter_pot_sharpe(cut_date, factor, return_choose, lag=1)
    write_list = [file_name] + list(filter_result)
    with lock:
        f = open(result_save_file, 'a')
        f.write('|'.join([str(x) for x in write_list]) + '\n')
        f.close()
    print(write_list)
    return write_list


if __name__ == '__main__':
    root_path = '/mnt/mfs/dat_whs'
    factor_path = os.path.join(root_path, 'data/adj_data/index_universe_f')
    single_result_save_path = os.path.join(root_path, '/mnt/mfs/dat_whs/result/single_factor')

    begin_date = pd.to_datetime('20100101')
    cut_date = pd.to_datetime('20160401')
    # sector
    stock_universe = load_sector_df(begin_date)
    # suspend or limit up_dn
    locked_df = load_locked_date(begin_date)
    # return
    return_choose = load_pct(begin_date, stock_universe)
    file_list = [x for x in os.listdir(factor_path) if 'rr' in x or 'rc' in x]

    now_time = datetime.now().strftime('%Y%m%d_%H%M')
    result_save_file = os.path.join(single_result_save_path, now_time + 'txt')

    # pool = Pool(20)
    for file_name in sorted(file_list):
        test_index_1(begin_date, cut_date, file_name, stock_universe, locked_df, return_choose, result_save_file)
    # pool.close()
    # pool.join()
