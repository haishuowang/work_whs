import pandas as pd
import numpy as np
import os
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from itertools import product, permutations, combinations
import random
import sys
sys.path.append('/mnt/mfs/open_lib/shared_tools')
import send_email


def mul_fun(a, b):
    return a.mul(b, fill_value=0)


def sub_fun(a, b):
    return a.sub(b, fill_value=0)


def add_fun(a, b):
    return a.add(b, fill_value=0)


def create_fun_set_2(fun_set):
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


def AZ_Rolling_sharpe(pnl_df, roll_year=1, year_len=250, cut_point_list=None, output=False):
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
    rolling_sharpe = pnl_df.rolling(int(roll_year * year_len))\
        .apply(lambda x: np.sqrt(year_len) * x.mean() / x.std())
    cut_sharpe = rolling_sharpe.quantile(cut_point_list)
    if output:
        return cut_sharpe
    else:
        return rolling_sharpe.dropna(), cut_sharpe


def para_result(i, fun_name, factor_load_path, return_load_path, name_1, name_2, name_3, return_name, *result):
    figure_save_path = '/media/hdd0/whs/tmp_figure'
    fun_set = [mul_fun, sub_fun, add_fun]
    mix_fun_set = create_fun_set_2(fun_set)
    fun = mix_fun_set[fun_name]
    choose_1 = pd.read_pickle(os.path.join(factor_load_path, name_1 + '.pkl'))
    choose_2 = pd.read_pickle(os.path.join(factor_load_path, name_2 + '.pkl'))
    choose_3 = pd.read_pickle(os.path.join(factor_load_path, name_3 + '.pkl'))
    return_data = pd.read_pickle(os.path.join(return_load_path, return_name + '.pkl'))
    mix_factor = fun(choose_1, choose_2, choose_3).shift(1)
    trade_time_df = mix_factor.diff().abs().sum(axis=1)
    pnl_df = (return_data * mix_factor).sum(axis=1)
    AZ_Rolling_sharpe(pnl_df, roll_year=1, year_len=250)
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('{} factor figure'.format(i), fontsize=40)
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    ax1.plot(pnl_df.index, pnl_df.cumsum(),
             label='fun_name={},\nname1={},\nname2={},\nname3={}\nresult={}'
             .format(fun_name, name_1, name_2, name_3, *result))
    ax1.grid(1)
    ax1.legend()

    rolling_sharpe, cut_sharpe = AZ_Rolling_sharpe(pnl_df, roll_year=3, year_len=250)
    ax2.hist(rolling_sharpe.values, bins=200)
    ax2.grid(axis='y')
    # plt.show()

    plt.savefig(os.path.join(figure_save_path, '|'.join([fun_name, name_1, name_2, name_3]))+'.png')

    text = '|'.join([fun_name, name_1, name_2, name_3])
    to = ['whs@malpha.info']
    subject = '|'.join([fun_name, name_1, name_2, name_3])
    filepath = [os.path.join(figure_save_path, '|'.join([fun_name, name_1, name_2, name_3]))+'.png']
    send_email.send_email(text, to, filepath, subject)


def result_get_random(result_load_path, n_random=20):
    result_data = pd.read_table(result_load_path, sep='|', header=None)
    return_name = 'pct_f5d'
    for i in random.sample(list(result_data.index), n_random):
        fun_name, name_1, name_2, name_3, *result = result_data.loc[i]
        print(fun_name, name_1, name_2, name_3, *result)
        para_result(i, fun_name, factor_load_path, return_load_path, name_1, name_2, name_3, return_name, *result)


def result_compare_extreme(result_load_path):
    result_data = pd.read_table(result_load_path, sep='|', header=None)
    return_name = 'pct_f5d'
    # result_data[result_data[4] < 0].sort_values(4)
    # for
    # result_data[result_data[4] > 0].sort_values(4)


if __name__ == '__main__':
    factor_load_path = '/media/hdd0/whs/data/adj_data/index_universe_f'
    return_load_path = '/media/hdd0/whs/data/adj_data/fnd_pct'
    result_load_path = '/media/hdd0/whs/result/result/20180614_1847.txt'
    # result_get_random(result_load_path, n_random=20)
