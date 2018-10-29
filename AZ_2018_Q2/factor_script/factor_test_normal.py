import pandas as pd
import numpy as np
import os
from itertools import product, permutations, combinations
from multiprocessing import Pool, Lock, cpu_count
import time
import sys
sys.path.append('/mnt/mfs/work_whs')
sys.path.append('/mnt/mfs/work_whs/AZ_2018_Q2')
from datetime import datetime
import loc_lib.shared_tools.back_test as bt
import random
# 读取数据的函数 以及
from factor_script.script_load_data import load_index_data, load_sector_data, load_locked_data, load_pct, \
    load_part_factor, create_log_save_path, deal_mix_factor, deal_mix_factor_both, load_locked_data_both

from factor_script.script_filter_fun import pos_daily_fun, out_sample_perf, filter_all

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


def create_all_para(use_factor_set_path, new_factor_list, add_factor_list, choos_num=3):
    file_name_list = list(pd.read_pickle(use_factor_set_path)) + add_factor_list
    # file_name_list = ['R_BusinessCycle_First_row_extre_0.3',
    #                   'R_CurrentAssets_TotAssets_First_row_extre_0.3',
    #                   'R_CurrentAssetsTurnover_First_row_extre_0.3',
    #                   'R_DebtAssets_First_row_extre_0.3',
    #                   'R_DebtEqt_First_row_extre_0.3',
    #                   'R_OPCF_IntDebt_QTTM_row_extre_0.3',
    #                   'R_IntDebt_Mcap_First_row_extre_0.3',
    #                   'R_OPEX_sales_TTM_Y3YGR_row_extre_0.3',
    #                   'R_EBITDA_sales_TTM_First_row_extre_0.3',
    #                   'R_EBITDA_sales_TTM_QTTM_row_extre_0.3',
    #                   'R_ROA_TTM_Y3YGR_row_extre_0.3',
    #                   'R_OPCF_TotDebt_QYOY_row_extre_0.3',
    #                   'R_OPCF_TotDebt_First_row_extre_0.3',
    #                   'R_OPCF_NetInc_s_First_row_extre_0.3',
    #                   'R_OPCF_sales_First_row_extre_0.3',
    #                   'R_EBITDA_QTTM_and_R_SUMASSET_First_0.3',
    #                   'R_EBIT2_Y3YGR_and_MCAP_0.3',
    #                   'R_EBITDA_QYOY_and_MCAP_0.3',
    #                   'R_IntDebt_Y3YGR_and_R_SUMASSET_First_0.3',
    #                   'CCI_p120d_limit_12',
    #                   'MACD_20_100',
    #                   'log_price_0.2',
    #                   'bias_turn_p20d',
    #                   'vol_p20d',
    #                   'vol_p20d',
    #                   'evol_p20d',
    #                   'TVOL_p60d_col_extre_0.2',
    #                   'TVOL_p20d_col_extre_0.2']

    if len(new_factor_list) == 0:
        print('{} factor num:{}'.format(sector_name, len(file_name_list)))
        return combinations(sorted(file_name_list), choos_num)
    else:
        target_list = []
        old_factor_list = sorted(set(file_name_list) - set(new_factor_list))
        for factor_name in new_factor_list:
            for value in combinations(old_factor_list, 2):
                target_list += [[factor_name] + list(value)]
        return target_list


def part_test_index_3(sector_name, key, name_1, name_2, name_3, sector_df, suspendday_df, limit_buy_sell_df,
                      return_choose, index_df, cut_date, log_save_file, result_save_file, if_save, if_hedge, hold_time,
                      if_only_long, xnms, xinx, total_para_num):
    lock = Lock()
    lag = 2
    start_time = time.time()
    load_time_1 = time.time()
    # load因子,同时根据stock_universe筛选数据.
    factor_set = load_part_factor(sector_name, xnms, xinx, [name_1, name_2, name_3])
    load_time_2 = time.time()
    # 加载花费数据时间
    load_delta = round(load_time_2 - load_time_1, 2)
    # 生成混合函数集
    fun_set = [sub_fun, add_fun, mul_fun]
    fun_mix_2_set = create_fun_set_2(fun_set)
    #################
    # 更换filter函数 #
    #################
    filter_fun = filter_all
    filter_name = filter_fun.__name__
    for fun in fun_mix_2_set:
        mix_factor = fun(factor_set[name_1], factor_set[name_2], factor_set[name_3])
        if len(mix_factor.abs().sum(axis=1).replace(0, np.nan).dropna()) / len(mix_factor) < 0.1:
            # print('{}%, {}, {}, {}, {}, ERROR pos not enough, {}'
            #       .format(round(key / total_para_num, 4) * 100, key, name_1, name_2, name_3,
            #               mix_factor.sum(axis=1).mean()))
            continue
        daily_pos = deal_mix_factor(mix_factor, sector_df, suspendday_df, limit_buy_sell_df, hold_time, lag,
                                    if_only_long)
        # 返回样本内筛选结果
        in_condition, *filter_result = filter_fun(cut_date, daily_pos, return_choose, index_df, if_hedge, hedge_ratio=1,
                                                  if_return_pnl=False, if_only_long=if_only_long)
        # result 存储
        if in_condition:
            if if_save:
                with lock:
                    f = open(result_save_file, 'a')
                    write_list = [key, fun.__name__, name_1, name_2, name_3, filter_name, sector_name, in_condition] \
                                 + filter_result
                    f.write('|'.join([str(x) for x in write_list]) + '\n')
                    f.close()
            print([in_condition] + filter_result)
    end_time = time.time()
    # 参数存储
    if if_save:
        with lock:
            f = open(log_save_file, 'a')
            write_list = [key, name_1, name_2, name_3, filter_name, sector_name, round(end_time - start_time, 4),
                          load_delta]
            f.write('|'.join([str(x) for x in write_list]) + '\n')
            f.close()
    print('{}%, {}, {}, {}, {}, cost {} seconds, load_cost {} seconds'
          .format(round(key / total_para_num * 100, 4) , key, name_1, name_2, name_3,
                  round(end_time - start_time, 2), load_delta))


def test_index_3(sector_name, sector_df, suspendday_df, limit_buy_sell_df, return_choose, index_df,
                 para_ready_df, cut_date, log_save_file, result_save_file, if_save, if_hedge, hold_time, if_only_long,
                 xnms, xinx, total_para_num):
    a_time = time.time()
    pool = Pool(20)
    for key in list(para_ready_df.index):
        name_1, name_2, name_3 = para_ready_df.loc[key]

        args_list = (sector_name, key, name_1, name_2, name_3, sector_df, suspendday_df, limit_buy_sell_df,
                     return_choose, index_df, cut_date, log_save_file, result_save_file, if_save, if_hedge, hold_time,
                     if_only_long, xnms, xinx, total_para_num)
        # part_test_index_3(*args_list)
        pool.apply_async(part_test_index_3, args=args_list)
    pool.close()
    pool.join()

    b_time = time.time()
    print('Success!Processing end, Cost {} seconds'.format(round(b_time - a_time, 2)))


def save_load_control(use_factor_set_path, sector_name, new_factor_list, add_factor_list,
                      if_save=True, if_new_program=True, if_hedge=True, hold_time=5,
                      return_file='pct_file', if_only_long=False):
    # 参数存储与加载的路径控制
    result_save_path = '/mnt/mfs/dat_whs/result'
    if if_new_program:
        now_time = datetime.now().strftime('%Y%m%d_%H%M')
        if if_only_long:
            if len(new_factor_list) != 0:
                file_name = '{}_{}_{}_hold_{}_{}_long_new.txt' \
                    .format(sector_name, if_hedge, now_time, hold_time, return_file)
            else:
                file_name = '{}_{}_{}_hold_{}_{}_long.txt' \
                    .format(sector_name, if_hedge, now_time, hold_time, return_file)
        else:
            if len(new_factor_list) != 0:
                file_name = '{}_{}_{}_hold_{}_{}_new.txt' \
                    .format(sector_name, if_hedge, now_time, hold_time, return_file)
            else:
                file_name = '{}_{}_{}_hold_{}_{}.txt' \
                    .format(sector_name, if_hedge, now_time, hold_time, return_file)

        log_save_file = os.path.join(result_save_path, 'log', file_name)
        result_save_file = os.path.join(result_save_path, 'result', file_name)
        para_save_file = os.path.join(result_save_path, 'para', file_name)

        para_ready_df = pd.DataFrame(list(create_all_para(use_factor_set_path, new_factor_list, add_factor_list)))
        if if_save:
            create_log_save_path(log_save_file)
            create_log_save_path(result_save_file)
            create_log_save_path(para_save_file)
            para_ready_df.to_pickle(para_save_file)

    else:
        file_name = 'market_top_2000_True_20180823_0910_hold_20_aadj_r.txt'

        log_save_file = os.path.join(result_save_path, 'log', file_name)
        result_save_file = os.path.join(result_save_path, 'result', file_name)
        para_save_file = os.path.join(result_save_path, 'para', file_name)

        para_tested_df = pd.read_table(log_save_file, sep='|', header=None, index_col=0)
        para_all_df = pd.read_pickle(para_save_file)
        para_ready_df = para_all_df.loc[sorted(list(set(para_all_df.index) - set(para_tested_df.index)))]
    print(file_name)
    print(f'para_num:{len(para_ready_df)}')
    return para_ready_df, log_save_file, result_save_file


def main_fun(sector_name, index_name, hold_time, return_file, new_factor_list, add_factor_list,
             if_hedge=False, if_only_long=False):
    begin_date = pd.to_datetime('20100101')
    cut_date = pd.to_datetime('20160401')
    end_date = pd.to_datetime('20180401')

    if_save = True
    if_new_program = False
    use_factor_set_path = '/mnt/mfs/dat_whs/data/use_factor_set/market_top_500_201808171920.pkl'
    return_file = 'close'

    para_ready_df, log_save_file, result_save_file = \
        save_load_control(use_factor_set_path, sector_name, new_factor_list, add_factor_list,
                          if_save, if_new_program, if_hedge, hold_time, return_file, if_only_long)
    total_para_num = len(para_ready_df)
    # sector
    sector_df = load_sector_data(begin_date, end_date, sector_name)

    xnms = sector_df.columns
    xinx = sector_df.index

    # suspend or limit up_dn
    # suspendday_df, limit_buy_df, limit_sell_df = load_locked_data(xnms, xinx)
    suspendday_df, limit_buy_sell_df = load_locked_data_both(xnms, xinx)
    # return
    return_choose = pd.read_table('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv', sep='|', index_col=0)\
        .astype(float)
    return_choose.index = pd.to_datetime(return_choose.index)
    return_choose = return_choose.reindex(columns=xnms, index=xinx, fill_value=0)

    # index data
    index_df = load_index_data(xinx, index_name)

    # index_df = pd.Series(index_df)
    test_index_3(sector_name, sector_df, suspendday_df, limit_buy_sell_df, return_choose, index_df,
                 para_ready_df, cut_date, log_save_file, result_save_file, if_save, if_hedge, hold_time, if_only_long,
                 xnms, xinx, total_para_num)


if __name__ == '__main__':
    # sector_name = 'market_top_100'
    # index_name = '000016'

    sector_name = 'market_top_2000'
    index_name = '000300'
    return_file = 'pct_f1d'
    # new_factor_list = ['pub_info_all']
    new_factor_list = []
    add_factor_list = []
    for hold_time in [10]:
        return_file = 'pct_f1d'
        # main_fun(sector_name, index_name, hold_time, return_file, new_factor_list, add_factor_list, if_hedge=True,
        #          if_only_long=False)
        main_fun(sector_name, index_name, hold_time, return_file, new_factor_list, add_factor_list, if_hedge=True,
                 if_only_long=True)
