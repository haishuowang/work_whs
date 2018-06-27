import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import product, permutations, combinations
import random
import sys
from open_lib.shared_tools import send_email
import open_lib.shared_tools.back_test as bt


def mul_fun(a, b):
    return a.mul(b, fill_value=0)


def sub_fun(a, b):
    return a.sub(b, fill_value=0)


def add_fun(a, b):
    return a.add(b, fill_value=0)


# 构建每天的position
def position_daily_fun(df, n=5):
    return df.rolling(window=n, min_periods=1).sum()


def load_stock_universe(begin_date=pd.to_datetime('20100101')):
    market_top_n = pd.read_pickle('/mnt/mfs/DAT_EQT/STK_Groups1/market_top_1000.pkl')
    market_top_n = market_top_n[market_top_n.index >= begin_date]
    return market_top_n


def load_locked_date(begin_date=pd.to_datetime('20100101')):
    suspendday_df = pd.read_pickle('/mnt/mfs/DAT_EQT/EM_Tab14/adj_data/TRAD_TD_SUSPENDDAY/SUSPENDREASON_adj.pkl')
    limit_updn_df = pd.read_pickle('/mnt/mfs/dat_whs/data/locked_date/limit_updn_table.pkl')
    locked_df = limit_updn_df * suspendday_df
    locked_df.dropna(how='all', axis=0, inplace=True)
    locked_df = locked_df[locked_df.index >= begin_date]
    return locked_df


def load_factor(file_name, stock_universe):
    load_path = os.path.join(root_path, 'data/adj_data/index_universe_f')
    target_df = pd.read_pickle(os.path.join(load_path, file_name + '.pkl'))
    target_df = target_df * stock_universe
    target_df.dropna(how='all', axis=0, inplace=True)
    target_df = target_df[target_df.index >= begin_date]
    return target_df


def load_pct(stock_universe):
    load_path = os.path.join(root_path, 'data/adj_data/fnd_pct/pct_f5d.pkl')
    target_df = pd.read_pickle(load_path)
    target_df = target_df * stock_universe
    target_df.dropna(how='all', axis=0, inplace=True)
    target_df = target_df[target_df.index >= begin_date]
    return target_df


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


def plot_send_result(pnl_df, sharpe_ratio):
    figure_save_path = os.path.join(root_path, 'tmp_figure')
    plt.plot(pnl_df.index, pnl_df.cumsum(), label='sharpe_ratio='.format(sharpe_ratio))
    plt.savefig(os.path.join(figure_save_path, 'multy_mix_factor.png'))
    text = 'multy_mix_factor'
    subject = 'multy_mix_factor'
    to = ['whs@malpha.info']
    filepath = [os.path.join(figure_save_path, 'multy_mix_factor.png')]
    send_email.send_email(text, to, filepath, subject)


def para_result(stock_universe, locked_df, i, fun_name, factor_load_path, return_load_path,
                name_1, name_2, name_3, return_name, *result):
    cost_1 = 0.001
    cost_2 = 0.002
    figure_save_path = os.path.join(root_path, 'tmp_figure')
    fun_set = [mul_fun, sub_fun, add_fun]
    mix_fun_set = create_fun_set_2(fun_set)
    fun = mix_fun_set[fun_name]
    choose_1 = load_factor(name_1, stock_universe)
    choose_2 = load_factor(name_2, stock_universe)
    choose_3 = load_factor(name_3, stock_universe)

    return_data = load_pct(stock_universe)
    mix_factor = fun(choose_1, choose_2, choose_3).shift(1)
    mix_factor = mix_factor.fillna(0) * locked_df
    mix_factor.dropna(how='all', axis=0, inplace=True)
    mix_factor.fillna(method='ffill', inplace=True)

    pnl_df = (return_data * mix_factor).sum(axis=1)
    AZ_Rolling_sharpe(pnl_df, roll_year=1, year_len=250)
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('{} factor figure'.format(i), fontsize=40)
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    asset = pnl_df.cumsum()
    if asset.values[-1] < 0:
        asset = -asset
    mix_factor_day = position_daily_fun(mix_factor, n=5)
    cost_matrix_1 = (mix_factor_day.diff().abs() * cost_1).sum(axis=1).cumsum()
    cost_matrix_2 = (mix_factor_day.diff().abs() * cost_2).sum(axis=1).cumsum()

    ax1.plot(pnl_df.index, asset,
             label='fun_name={},\nname1={},\nname2={},\nname3={}\nfilter_fun={}'
             .format(fun_name, name_1, name_2, name_3, result[0]))
    ax1.plot(pnl_df.index, asset - cost_matrix_1)
    ax1.plot(pnl_df.index, asset - cost_matrix_2)

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
    stock_universe = load_stock_universe()
    locked_df = load_locked_date()
    result_data = pd.read_table(result_load_path, sep='|', header=None)
    return_name = 'pct_f5d'
    for i in random.sample(list(result_data.index), n_random):
        key, fun_name, name_1, name_2, name_3, *result = result_data.loc[i]
        print(key, fun_name, name_1, name_2, name_3, *result)
        para_result(stock_universe, locked_df, i, fun_name, factor_load_path, return_load_path, name_1, name_2, name_3, return_name, *result)


def result_analyse(file_list):
    for file_name in file_list:
        result_load_path = os.path.join(root_path, 'result/result/{}'.format(file_name))
        result_data = pd.read_table(result_load_path, sep='|', header=None)
        filter_fun = result_data.iloc[0, 5]
        survived = sum(result_data[7])
        choosed = result_data.shape[0]
        survived_ratio = survived / choosed
        print('{}, survived_num = {}, choosed_num = {}, survived_ratio = {}'
              .format(filter_fun, survived, choosed, round(survived_ratio, 4)))


def result_factor_sum(file_name='20180626_1148.txt', n_random=20, cut_date=pd.to_datetime('20160401')):
    stock_universe = load_stock_universe()
    locked_df = load_locked_date()
    return_data = load_pct(stock_universe)
    result_load_path = os.path.join(root_path, 'result/result/{}'.format(file_name))
    result_data = pd.read_table(result_load_path, sep='|', header=None)
    multy_mix_factor = pd.DataFrame()
    for i in random.sample(list(result_data.index), n_random):
        key, fun_name, name_1, name_2, name_3, *result, sharpe_quantile, leve_ratio = result_data.loc[i]
        fun_set = [mul_fun, sub_fun, add_fun]
        mix_fun_set = create_fun_set_2(fun_set)
        fun = mix_fun_set[fun_name]
        choose_1 = load_factor(name_1, stock_universe)
        choose_2 = load_factor(name_2, stock_universe)
        choose_3 = load_factor(name_3, stock_universe)

        mix_factor = fun(choose_1, choose_2, choose_3).shift(1)
        mix_factor = mix_factor.fillna(0) * locked_df
        mix_factor.dropna(how='all', axis=0, inplace=True)
        mix_factor.fillna(method='ffill', inplace=True)
        if sharpe_quantile > 0:
            multy_mix_factor = multy_mix_factor.add(mix_factor, fill_value=0)
        else:
            multy_mix_factor = multy_mix_factor.add(-mix_factor, fill_value=0)

    pnl_df = (return_data * multy_mix_factor).sum(axis=1)
    pnl_df_out = pnl_df[pnl_df.index >= cut_date]

    sharpe_ratio = bt.AZ_Sharpe_y(pnl_df_out)
    plot_send_result(pnl_df_out, sharpe_ratio)
    print(sharpe_ratio)


root_path = '/mnt/mfs/dat_whs'

if __name__ == '__main__':
    begin_date = pd.to_datetime('20100101')
    cut_date = pd.to_datetime('20160401')

    factor_load_path = os.path.join(root_path, 'data/adj_data/index_universe_f')
    return_load_path = os.path.join(root_path, 'data/adj_data/fnd_pct')
    result_load_path = os.path.join(root_path, 'result/result/20180626_1224.txt')

    result_get_random(result_load_path, n_random=5)

    # result_factor_sum()

    # file_list = ['20180626_1047.txt', '20180626_1106.txt', '20180626_1148.txt', '20180626_1224.txt',
    #              '20180626_1555.txt']
    # result_analyse(file_list)
