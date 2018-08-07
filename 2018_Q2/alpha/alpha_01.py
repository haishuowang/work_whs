import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from open_lib.shared_tools import send_email
from itertools import product, permutations, combinations
import os
from factor_script.script_load_data import load_sector_data, load_locked_data, load_pct, \
    load_part_factor, create_log_save_path, load_index_data, deal_mix_factor, deal_mix_factor_c

from factor_script.script_filter_fun import pos_daily_fun, out_sample_perf, \
    filter_ic, filter_ic_sharpe, filter_ic_leve, filter_pot_sharpe, filter_all
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


def create_pnl_file(all_use_factor, file_name, hold_time):
    pnl_root_path = '/mnt/mfs/dat_whs/tmp_pnl_file/{}'.format(file_name)
    bt.AZ_Path_create(pnl_root_path)
    use_factor_pot = all_use_factor[all_use_factor['pot_in'].abs() > 40]
    use_factor_pot.sort_values(by='sp_u', inplace=True)
    for i in use_factor_pot.index[:3]:
        key, fun_name, name1, name2, name3, filter_fun_name, sector_name, *result = all_use_factor.iloc[i]
        print('**************************************')
        print('now {}\'s is running, key={}'.format(i, key))
        save_file_name = '|'.join([str(key), fun_name, name1, name2, name3, filter_fun_name, sector_name]) + '.csv'
        fun_set = [mul_fun, sub_fun, add_fun]
        mix_fun_set = create_fun_set_2(fun_set)
        fun = mix_fun_set[fun_name]

        factor_set = load_part_factor(sector_name, xnms, xinx, [name1, name2, name3])
        choose_1 = factor_set[name1]
        choose_2 = factor_set[name2]
        choose_3 = factor_set[name3]
        mix_factor = fun(choose_1, choose_2, choose_3)
        pos_df_daily = deal_mix_factor_c(mix_factor, sector_df, locked_df, hold_time, 2, if_only_long)
        # pos_df_daily_c = deal_mix_factor_c(mix_factor, sector_df, locked_df, hold_time, 2, if_only_long)

        in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
        fit_ratio, leve_ratio, sharpe_q_out, pnl_df = filter_all(cut_date, pos_df_daily, return_data, index_df,
                                                                 if_hedge=True, hedge_ratio=1, if_return_pnl=True,
                                                                 if_only_long=if_only_long)
        # print(in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in,
        #       sharpe_q_out)

        # in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
        # fit_ratio, leve_ratio, sharpe_q_out, pnl_df = filter_all(cut_date, pos_df_daily_c, return_data, index_df,
        #                                                          if_hedge=True, hedge_ratio=1, if_return_pnl=True,
        #                                                          if_only_long=if_only_long)
        print(in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in,
              sharpe_q_out)
        print(result)
        # pnl_df.to_csv(os.path.join(pnl_root_path, save_file_name))


def plot_send_result(pnl_df, sharpe_ratio, subject):
    plt.figure(figsize=[16, 8])
    plt.plot(pd.to_datetime(pnl_df.index), pnl_df.cumsum(), label='sharpe_ratio='.format(sharpe_ratio))
    plt.savefig(os.path.join('/mnt/mfs/dat_whs/tmp_figure', 'multy_mix_factor.png'))
    text = ''
    to = ['whs@yingpei.com']
    filepath = [os.path.join('/mnt/mfs/dat_whs/tmp_figure', 'multy_mix_factor.png')]
    send_email.send_email(text, to, filepath, subject)


def pnl_sum(file_name):
    pnl_root_path = '/mnt/mfs/dat_whs/tmp_pnl_file/{}'.format(file_name)
    file_list = os.listdir(pnl_root_path)
    all_pnl_df = pd.DataFrame()
    for pnl_file in file_list:
        pnl_df = pd.read_csv(os.path.join(pnl_root_path, pnl_file), index_col=0, header=None)
        # pnl_df.columns = [pnl_file[:-4]]
        all_pnl_df = all_pnl_df.add(pnl_df, axis=1, fill_value=0)
    sharpe_ratio = bt.AZ_Sharpe_y(all_pnl_df)
    plot_send_result(all_pnl_df, sharpe_ratio, 'top_100_sharpe_pnl')


def pos_sum(all_use_factor, hold_time):
    all_use_factor.sort_values(by='sp_u', inplace=True)
    sum_pos_df = pd.DataFrame()
    for i in all_use_factor.index[:100]:
        key, fun_name, name1, name2, name3, filter_fun_name, sector_name, con_in, con_out, ic, sp_u, sp_m, sp_d, \
        pot_in, fit_ratio, leve_ratio, sp_out = all_use_factor.iloc[i]
        print('**************************************')
        print('now {}\'s is running, key={}'.format(i, key))
        fun_set = [mul_fun, sub_fun, add_fun]
        mix_fun_set = create_fun_set_2(fun_set)
        fun = mix_fun_set[fun_name]

        factor_set = load_part_factor(sector_name, xnms, xinx, [name1, name2, name3])
        choose_1 = factor_set[name1]
        choose_2 = factor_set[name2]
        choose_3 = factor_set[name3]
        mix_factor = fun(choose_1, choose_2, choose_3)
        pos_df_daily = deal_mix_factor_c(mix_factor, sector_df, locked_df, hold_time, 2, if_only_long)
        if sp_u>0:
            sum_pos_df = sum_pos_df.add(pos_df_daily, fill_value=0)
        else:
            sum_pos_df = sum_pos_df.add(-pos_df_daily, fill_value=0)

    in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
    fit_ratio, leve_ratio, sharpe_q_out, pnl_df = filter_all(cut_date, sum_pos_df, return_data, index_df,
                                                             if_hedge=True, hedge_ratio=1, if_return_pnl=True,
                                                             if_only_long=if_only_long)
    print(in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in,
          fit_ratio, leve_ratio, sharpe_q_out)
    return pnl_df, sum_pos_df


if __name__ == '__main__':
    root_path = '/media/hdd1/dat_whs'
    index_name = '000300'
    file_name = 'market_top_500_True_20180806_1134_hold_5_aadj_r'
    log_result = os.path.join(root_path, 'result/result/{}.txt'.format(file_name))
    sector_name = 'market_top_500'
    hold_time = 5
    lag = 2
    if_only_long = False
    begin_date = pd.to_datetime('20100101')
    cut_date = pd.to_datetime('20160401')
    end_date = pd.to_datetime('20180401')

    sector_df = load_sector_data(begin_date, end_date, sector_name)

    xnms = sector_df.columns
    xinx = sector_df.index

    # suspend or limit up_dn
    locked_df = load_locked_data(xnms, xinx)

    return_choose = pd.read_table('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv', sep='|', index_col=0).astype(float)
    return_choose.index = pd.to_datetime(return_choose.index)
    return_data = return_choose.reindex(columns=xnms, index=xinx, fill_value=0)

    # index data
    index_df = load_index_data(xinx, index_name)

    xnms, xinx = sector_df.columns, sector_df.index

    all_use_factor = pd.read_table(log_result, sep='|', header=None)
    all_use_factor.columns = ['key', 'fun_name', 'name1', 'name2', 'name3', 'filter_fun_name', 'sector_name',
                              'con_in', 'con_out', 'ic', 'sp_u', 'sp_m', 'sp_d', 'pot_in', 'fit_ratio', 'leve_ratio',
                              'sp_out']
    # 生成pnl文件
    create_pnl_file(all_use_factor, file_name, hold_time)
    # # 将生成的pnl文件组合
    # pnl_sum(file_name)
    # pnl_df, sum_pos_df = pos_sum(all_use_factor, hold_time)

