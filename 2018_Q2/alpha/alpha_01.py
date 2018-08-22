import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from open_lib.shared_tools import send_email
from itertools import product, permutations, combinations
import os
from factor_script.script_load_data import load_sector_data, load_locked_data, load_locked_data_both, load_pct, \
    load_part_factor, create_log_save_path, load_index_data, deal_mix_factor

from factor_script.script_filter_fun import pos_daily_fun, out_sample_perf, filter_all
import loc_lib.shared_tools.back_test as bt


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


def create_pnl_file(all_use_factor, file_name, hold_time):
    pnl_root_path = os.path.join(root_path, 'tmp_pnl_file/{}'.format(file_name))
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
        daily_pos = deal_mix_factor(mix_factor, sector_df, suspendday_df, limit_buy_sell_df, hold_time, lag,
                                    if_only_long)

        in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
        fit_ratio, leve_ratio, sharpe_q_out, pnl_df = filter_all(cut_date, daily_pos, return_choose, index_df,
                                                                 if_hedge=True, hedge_ratio=1, if_return_pnl=True,
                                                                 if_only_long=if_only_long)
        print(in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in,
              sharpe_q_out)
        print(result)
        # pnl_df.to_csv(os.path.join(pnl_root_path, save_file_name))


def plot_send_result(pnl_df, sharpe_ratio, subject):
    plt.figure(figsize=[16, 8])
    plt.plot(pd.to_datetime(pnl_df.index), pnl_df.cumsum(), label='sharpe_ratio={}'.format(sharpe_ratio))
    plt.legend()
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
    all_use_factor['sort_line'] = all_use_factor['sp_u'].abs()
    all_use_factor.sort_values(by='sort_line', inplace=True, ascending=False)
    all_use_factor.drop(columns='sort_line', inplace=True)
    filter_cond = all_use_factor.apply(lambda x: not ('volume_count_down_p60d' in set(x)), axis=1)
    all_use_factor = all_use_factor[filter_cond]

    a = all_use_factor[all_use_factor.pot_in.abs() > 20]
    a = a.iloc[:10]
    b = a.copy()
    b['buy_sell'] = (a['sp_m'] > 0).astype(int).replace(0, -1)
    print(b['con_out'].sum() / len(a), len(a))
    factor_info = b[['fun_name', 'name1', 'name2', 'name3', 'buy_sell']].replace(0, -1)
    config = dict()
    config['factor_info'] = factor_info
    pd.to_pickle(config, '/mnt/mfs/alpha_whs/config01.pkl')

    sum_factor_df = pd.DataFrame(columns=xnms, index=xinx)
    for i in a.index:
        key, fun_name, name1, name2, name3, filter_fun_name, sector_name, con_in, con_out, ic, sp_u, sp_m, sp_d, \
        pot_in, fit_ratio, leve_ratio, sp_out = a.loc[i]

        print('***************************************************')
        print('now {}\'s is running, key={}, {}, {}, {}, {}'.format(i, key, fun_name, name1, name2, name3))
        fun_set = [mul_fun, sub_fun, add_fun]
        mix_fun_set = create_fun_set_2(fun_set)
        fun = mix_fun_set[fun_name]

        factor_set = load_part_factor(sector_name, xnms, xinx, [name1, name2, name3])
        choose_1 = factor_set[name1]
        choose_2 = factor_set[name2]
        choose_3 = factor_set[name3]
        mix_factor = fun(choose_1, choose_2, choose_3)
        if len(mix_factor.abs().sum(axis=1).replace(0, np.nan).dropna()) / len(mix_factor) > 0.1:
            pos_daily = deal_mix_factor(mix_factor, sector_df, suspendday_df, limit_buy_sell_df, hold_time,
                                        lag, if_only_long)
            in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
            fit_ratio, leve_ratio, sharpe_q_out, pnl_df = filter_all(cut_date, pos_daily, return_choose, index_df,
                                                                     if_hedge=True, hedge_ratio=1, if_return_pnl=True,
                                                                     if_only_long=if_only_long)
            plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df), '{}, key={}'.format(i, key))
            print(con_in, con_out, ic, sp_u, sp_m, sp_d, pot_in, fit_ratio, leve_ratio, sp_out)
            print(in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in,
                  fit_ratio, leve_ratio, sharpe_q_out)
            if sp_m > 0:
                sum_factor_df = sum_factor_df.add(mix_factor, fill_value=0)
            else:
                sum_factor_df = sum_factor_df.add(-mix_factor, fill_value=0)
        else:
            print('pos not enough!')
    sum_pos_df = deal_mix_factor(sum_factor_df, sector_df, suspendday_df, limit_buy_sell_df, hold_time, lag,
                                 if_only_long).round(14)
    in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
    fit_ratio, leve_ratio, sharpe_q_out, pnl_df = filter_all(cut_date, sum_pos_df, return_choose, index_df,
                                                             if_hedge=True, hedge_ratio=1, if_return_pnl=True,
                                                             if_only_long=if_only_long)
    print(in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in,
          fit_ratio, leve_ratio, sharpe_q_out)
    return pnl_df, sum_pos_df


def factor_test(all_use_factor, xnms, xinx):
    print(all_use_factor['con_out'].sum() / len(all_use_factor), len(all_use_factor))
    a = all_use_factor[all_use_factor.pot_in.abs() > 20]
    print(a['con_out'].sum() / len(a), len(a))
    b = a[(a.ic.abs() > 0.04) & (a.ic.abs() < 1)]
    print(b['con_out'].sum() / len(b), len(b))
    c = b[b.fit_ratio.abs() > 1.5]
    print(c['con_out'].sum() / len(c), len(c))

    error_line = all_use_factor[all_use_factor.pot_in.abs() > 400]
    name_1, name_2, name_3 = error_line.loc[2585][['name1', 'name2', 'name3']]
    factor_set = load_part_factor(sector_name, xnms, xinx, [name_1, name_2, name_3])
    mix_factor = factor_set[name_1].mul(factor_set[name_2].mul(factor_set[name_3], fill_value=0), fill_value=0)
    mix_factor.sum(axis=1).replace(0, np.nan).dropna()


def create_config_info(all_use_factor):
    all_use_factor.sort_values(by='sp_u', inplace=True)

    a = all_use_factor[all_use_factor.pot_in.abs() > 20]
    print(a['con_out'].sum() / len(a))
    b = a[a.ic.abs() > 0.04]
    print(b['con_out'].sum() / len(b))

    config = {}
    b['buy_sell'] = (b['sp_m'] > 0).astype(int)
    factor_info = b[['fun_name', 'name1', 'name2', 'name3', 'buy_sell']]
    config['factor_info'] = factor_info
    pd.to_pickle(config, '/mnt/mfs/config01.pkl')


if __name__ == '__main__':
    root_path = '/mnt/mfs/dat_whs'
    index_name = '000300'
    file_name = 'market_top_500_True_20180816_0200_hold_10_aadj_r.txt'
    log_result = os.path.join(root_path, 'result/result/{}'.format(file_name))
    sector_name = 'market_top_500'
    hold_time = 10
    lag = 2
    if_only_long = False
    begin_date = pd.to_datetime('20100101')
    cut_date = pd.to_datetime('20160401')
    end_date = pd.to_datetime('20180801')

    sector_df = load_sector_data(begin_date, end_date, sector_name)

    xnms = sector_df.columns
    xinx = sector_df.index

    # suspend or limit up_dn
    suspendday_df, limit_buy_sell_df = load_locked_data_both(xnms, xinx)

    # return
    return_choose = pd.read_table('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv', sep='|', index_col=0).astype(float)
    return_choose.index = pd.to_datetime(return_choose.index)
    return_choose = return_choose.reindex(columns=xnms, index=xinx, fill_value=0)

    # index data
    index_df = load_index_data(xinx, index_name)

    all_use_factor = pd.read_table(log_result, sep='|', header=None)
    all_use_factor.columns = ['key', 'fun_name', 'name1', 'name2', 'name3', 'filter_fun_name', 'sector_name',
                              'con_in', 'con_out', 'ic', 'sp_u', 'sp_m', 'sp_d', 'pot_in', 'fit_ratio', 'leve_ratio',
                              'sp_out']

    # 生成pnl文件
    # create_pnl_file(all_use_factor, file_name, hold_time)
    # 将生成的pnl文件组合
    pnl_df, sum_pos_df = pos_sum(all_use_factor, hold_time)
    plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df), 'sum_pos_plot')

