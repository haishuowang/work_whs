import pandas as pd
import numpy as np
import time
import os
from multiprocessing import Pool
from sklearn.cluster import KMeans
from collections import OrderedDict
import matplotlib.pyplot as plt
from itertools import product, permutations, combinations
import random
from open_lib.shared_tools import send_email
import loc_lib.shared_tools.back_test as bt

from factor_script.script_load_data import load_sector_data, load_locked_data, load_pct, \
    load_part_factor, create_log_save_path, load_index_data, deal_mix_factor

from factor_script.script_filter_fun import pos_daily_fun, out_sample_perf, filter_all
from factor_script.main_file import FactorTest


class AutoCreateAlpha:
    def __init__(self, result_file_name, n):
        self.n = n
        self.result_file_name = result_file_name

        data = pd.read_csv('/mnt/mfs/dat_whs/result/result/{}.txt'.format(self.result_file_name), sep='|', header=None)

        data.columns = ['time_para', 'key', 'fun_name', 'name1', 'name2', 'name3', 'filter_fun_name', 'sector_name',
                        'con_in', 'con_out_1', 'con_out_2', 'con_out_3', 'con_out_4',
                        'ic', 'sp_u', 'sp_m', 'sp_d', 'pot_in', 'fit_ratio', 'leve_ratio',
                        # 'sp_in',
                        'sp_out_1', 'sp_out_2', 'sp_out_3', 'sp_out_4']
        data['buy_sell'] = (data['sp_m'] > 0).astype(int).replace(0, -1)
        filter_cond = data[['name1', 'name2', 'name3']] \
            .apply(lambda x: not ('R_COMPANYCODE_First_row_extre_0.3' in set(x)), axis=1)
        data = data[filter_cond]
        data['pnl_file_name'] = data[['key', 'fun_name']].apply(lambda x: '|'.join(x.astype(str)), axis=1)
        self.data = data

    def get_sum_pnl(self, file_name_list):
        sum_pnl_df = pd.DataFrame()
        pnl_save_path = '/mnt/mfs/dat_whs/data/mix_factor_pnl/{}'.format(self.result_file_name)
        # file_name_list = os.listdir(pnl_save_path)
        for file_name in file_name_list:
            pnl_df = pd.DataFrame(pd.read_pickle(os.path.join(pnl_save_path, file_name + '.csv')),
                                  columns=[file_name])
            sum_pnl_df = pd.concat([sum_pnl_df, pnl_df], axis=1)
        target_df = (sum_pnl_df > 0).astype(int)
        kmeans = KMeans(n_clusters=self.n).fit(target_df.T)

        kmeans_result = kmeans.labels_
        columns_list = target_df.columns
        # return pd.DataFrame([kmeans_result, sum_pnl_df.std()], index=columns_list)
        return pd.DataFrame(kmeans_result, index=columns_list)

    def get_config(self, time_para, use_factor_ratio=0.25, save_config=None):
        print('______________________________________')
        print(time_para)
        data_n = self.data[self.data['time_para'] == time_para]
        a_n = data_n[(data_n['ic'].abs() > 0.01) & (data_n['pot_in'].abs() > 40)]
        survive_ratio = a_n['con_out_1'].sum() / len(a_n)
        file_name_list = a_n['pnl_file_name'].values
        group_df = self.get_sum_pnl(file_name_list)
        a_n['group_key'] = group_df.loc[file_name_list].values
        print(f'survive_ratio: {survive_ratio}, total num: {len(a_n)}')
        target_df = pd.DataFrame()
        for i in range(self.n):
            part_a_n = a_n[a_n['group_key'] == i].sort_values(by='pot_in')
            part_num = int(len(part_a_n) * use_factor_ratio)

            part_target_df = part_a_n[['fun_name', 'name1', 'name2', 'name3', 'buy_sell']].iloc[:part_num]
            print(part_num)
            target_df = target_df.append(part_target_df)
        print(len(target_df))
        if save_config is not None:
            config_info = dict()
            config_info['factor_info'] = target_df
            pd.to_pickle(config_info, '/mnt/mfs/alpha_whs/{}.pkl'.format(save_config))
        return target_df


def mul_fun(a, b):
    return a.mul(b, fill_value=0)


def sub_fun(a, b):
    return a.sub(b, fill_value=0)


def add_fun(a, b):
    return a.add(b, fill_value=0)


# 构建每天的position
def position_daily_fun(df, n=5):
    return df.rolling(window=n, min_periods=1).sum()


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


def plot_send_result(pnl_df, sharpe_ratio, subject):
    figure_save_path = os.path.join('/mnt/mfs/dat_whs', 'tmp_figure')
    plt.figure(figsize=[16, 8])
    plt.plot(pnl_df.index, pnl_df.cumsum(), label='sharpe_ratio='.format(sharpe_ratio))
    plt.savefig(os.path.join(figure_save_path, '{}.png'.format(subject)))
    text = ''
    to = ['whs@yingpei.com']
    filepath = [os.path.join(figure_save_path, '{}.png'.format(subject))]
    send_email.send_email(text, to, filepath, subject)


def part_pnl_create_and_save(cls, file_name, key, fun_name, name1, name2, name3, tmp_cut):
    fun_set = [mul_fun, sub_fun, add_fun]
    mix_fun_set = create_fun_set_2(fun_set)
    fun = mix_fun_set[fun_name]
    factor_set = load_part_factor(cls.sector_name, cls.xnms, cls.xinx, [name1, name2, name3])
    choose_1 = factor_set[name1]
    choose_2 = factor_set[name2]
    choose_3 = factor_set[name3]
    pnl_root_path = '/mnt/mfs/dat_whs/data/mix_factor_pnl/{}'.format(file_name)
    bt.AZ_Path_create(pnl_root_path)
    mix_factor = fun(choose_1, choose_2, choose_3)
    daily_pos = cls.deal_mix_factor(mix_factor)
    in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
    fit_ratio, leve_ratio, sharpe_q_out, pnl_df = filter_all(tmp_cut, daily_pos,
                                                             cls.return_choose,
                                                             cls.index_df,
                                                             if_hedge=cls.if_hedge,
                                                             hedge_ratio=1,
                                                             if_return_pnl=True,
                                                             if_only_long=cls.if_only_long)
    pnl_root_path = '/mnt/mfs/dat_whs/data/mix_factor_pnl/{}'.format(file_name)
    pnl_save_path = os.path.join(pnl_root_path, f'{key}|{fun_name}.csv')
    if not os.path.exists(pnl_save_path):
        pnl_df.to_csv(pnl_save_path, sep='|')
        print(f'{key}|{fun_name} is create!')
    else:
        print(f'{key}|{fun_name} is exists!')


def pnl_create_and_save(main_class, result_file_name, an, cut_date):

    pool = Pool(20)
    for i in an.index:
        key, fun_name, name1, name2, name3 = an.loc[i]
        args_list = (main_class, result_file_name, key, fun_name, name1, name2, name3, cut_date,)
        part_pnl_create_and_save(*args_list)
        pool.apply_async(part_pnl_create_and_save, args=args_list)
    pool.close()
    pool.join()


def survive_test(data, pot_in_num, leve_ratio_num, ic_num, sp_in):
    data_1 = data[data['time_para'] == 'time_para_1']
    data_2 = data[data['time_para'] == 'time_para_2']
    data_3 = data[data['time_para'] == 'time_para_3']

    a_1 = data_1[(data_1['ic'].abs() > ic_num) &
                 (data_1['pot_in'].abs() > pot_in_num) &
                 (data_1['leve_ratio'].abs() > leve_ratio_num) &
                 (data_1['sp_in'].abs() > sp_in)]

    a_2 = data_2[(data_2['ic'].abs() > ic_num) &
                 (data_2['pot_in'].abs() > pot_in_num) &
                 (data_2['leve_ratio'].abs() > leve_ratio_num) &
                 (data_2['sp_in'].abs() > sp_in)]

    a_3 = data_3[(data_3['ic'].abs() > ic_num) &
                 (data_3['pot_in'].abs() > pot_in_num) &
                 (data_3['leve_ratio'].abs() > leve_ratio_num) &
                 (data_3['sp_in'].abs() > sp_in)]

    print('_______________________________________')
    print(a_1['con_out_1'].sum() / len(a_1), len(a_1))
    print(a_2['con_out_1'].sum() / len(a_2), len(a_2))
    print(a_3['con_out_1'].sum() / len(a_3), len(a_3))

    print('_______________________________________')
    print(a_1['con_out_2'].sum() / len(a_1), len(a_1))
    print(a_2['con_out_2'].sum() / len(a_2), len(a_2))
    print(a_3['con_out_2'].sum() / len(a_3), len(a_3))

    print('_______________________________________')
    print(a_1['con_out_3'].sum() / len(a_1), len(a_1))
    print(a_2['con_out_3'].sum() / len(a_2), len(a_2))
    print(a_3['con_out_3'].sum() / len(a_3), len(a_3))

    print('_______________________________________')
    print(a_1['con_out_4'].sum() / len(a_1), len(a_1))
    print(a_2['con_out_4'].sum() / len(a_2), len(a_2))
    print(a_3['con_out_4'].sum() / len(a_3), len(a_3))


def get_factor_data(factor_name, sector_name):
    data = pd.read_pickle(f'/mnt/mfs/dat_whs/data/new_factor_data/{sector_name}/{factor_name}.pkl')
    if len(data.abs().sum(axis=1).replace(0, np.nan).dropna()) == 0:
        print(factor_name, sector_name)
    return data


if __name__ == '__main__':
    result_file_name = 'market_top_2000_moment_bot_1000_True_20180903_1840_hold_20__alpha2'
    data = pd.read_csv('/mnt/mfs/dat_whs/result/result/{}.txt'.format(result_file_name), sep='|', header=None)

    data.columns = ['time_para', 'key', 'fun_name', 'name1', 'name2', 'name3', 'filter_fun_name', 'sector_name',
                    'con_in', 'con_out_1', 'con_out_2', 'con_out_3', 'con_out_4', 'ic', 'sp_u', 'sp_m', 'sp_d',
                    'pot_in', 'fit_ratio', 'leve_ratio', 'sp_in', 'sp_out_1', 'sp_out_2', 'sp_out_3', 'sp_out_4']
    filter_cond = data[['name1', 'name2', 'name3']] \
        .apply(lambda x: not ('R_COMPANYCODE_First_row_extre_0.3' in set(x)), axis=1)
    data = data[filter_cond]
    ic_num = 0.005
    pot_in_num = 40
    leve_ratio_num = 1
    sp_in = 2
    survive_test(data, pot_in_num, leve_ratio_num, ic_num, sp_in)

    root_path = '/mnt/mfs/DAT_EQT'
    if_save = True
    if_new_program = True

    begin_date = pd.to_datetime('20100101')
    cut_date = pd.to_datetime('20160401')
    end_date = pd.to_datetime('20180901')

    sector_name = 'market_top_2000'
    index_name = '000905'
    return_file = 'pct_p1d'
    hold_time = 20
    lag = 2
    return_file = ''

    # if_hedge = True
    # if_only_long = False
    # time_para_dict = OrderedDict()
    #
    # time_para_dict['time_para_1'] = [pd.to_datetime('20110101'), pd.to_datetime('20150101'),
    #                                  pd.to_datetime('20150401'), pd.to_datetime('20150701'),
    #                                  pd.to_datetime('20151001'), pd.to_datetime('20160101')]
    # time_para_dict['time_para_2'] = [pd.to_datetime('20120101'), pd.to_datetime('20160101'),
    #                                  pd.to_datetime('20160401'), pd.to_datetime('20160701'),
    #                                  pd.to_datetime('20161001'), pd.to_datetime('20170101')]
    # time_para_dict['time_para_3'] = [pd.to_datetime('20130601'), pd.to_datetime('20170601'),
    #                                  pd.to_datetime('20170901'), pd.to_datetime('20171201'),
    #                                  pd.to_datetime('20180301'), pd.to_datetime('20180601')]
    # main_class = FactorTest(root_path, if_save, if_new_program, begin_date, cut_date, end_date, time_para_dict,
    #                         sector_name, index_name, hold_time, lag, return_file, if_hedge, if_only_long)
    #
    # an = data[(data['ic'].abs() > ic_num) &
    #           (data['pot_in'].abs() > pot_in_num) &
    #           (data['leve_ratio'].abs() > leve_ratio_num)]
    # bn = data[['key', 'fun_name', 'name1', 'name2', 'name3']].drop_duplicates()
    # a = time.time()
    # pnl_create_and_save(main_class, result_file_name, bn, cut_date)
    # b = time.time()
    # print(b-a)

    # n = 5
    # auto_create_alpha = AutoCreateAlpha(result_file_name, n)
    # para_list = ['time_para_1', 'time_para_2', 'time_para_3']
    # para_dict = {'time_para_1': '018JUN',
    #              'time_para_2': '018JUL',
    #              'time_para_3': '018AUG'}
    # for time_para in para_list:
    #     use_factor_ratio = 0.20
    #     save_config = para_dict[time_para]
    #     target_df = auto_create_alpha.get_config(time_para, use_factor_ratio, save_config)
