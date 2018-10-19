import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import product, permutations, combinations
from sklearn.cluster import KMeans
import random
import sys

sys.path.append('/mnt/mfs/work_whs/2018_Q2')
sys.path.append('/mnt/mfs/work_whs')
from loc_lib.shared_tools import send_email
import loc_lib.shared_tools.back_test as bt

from factor_script.script_load_data import load_sector_data, load_locked_data, load_pct, \
    load_part_factor, create_log_save_path, load_index_data, deal_mix_factor

from factor_script import main_file_update as mf


def AZ_Leverage_ratio(asset_df):
    """
    返回250天的return/(负的 一个月的return)
    :param asset_df:
    :return:
    """
    asset_20 = asset_df - asset_df.shift(20)
    asset_250 = asset_df - asset_df.shift(250)
    if asset_250.mean() > 0:
        return round(asset_250.mean() / (-asset_20.min()), 2)
    else:
        return round(asset_250.mean() / (-asset_20.max()), 2)


def AZ_Pot(pos_df_daily, last_asset):
    trade_times = pos_df_daily.diff().abs().sum().sum()
    if trade_times == 0:
        return 0
    else:
        pot = last_asset / trade_times * 10000
        return round(pot, 2)


def out_sample_perf_c(pnl_df_out, way=1):
    if way == 1:
        sharpe_out = bt.AZ_Sharpe_y(pnl_df_out)
    else:
        sharpe_out = bt.AZ_Sharpe_y(-pnl_df_out)
    out_condition = sharpe_out > 0.8
    return out_condition, round(sharpe_out * way, 2)


def filter_all(cut_date, pos_df_daily, pct_n, if_return_pnl=False, if_only_long=False):
    pnl_df = (pos_df_daily * pct_n).sum(axis=1)
    pnl_df = pnl_df.replace(np.nan, 0)
    # pnl_df = pd.Series(pnl_df)
    # 样本内表现
    return_in = pct_n[pct_n.index < cut_date]

    pnl_df_in = pnl_df[pnl_df.index < cut_date]
    asset_df_in = pnl_df_in.cumsum()
    last_asset_in = asset_df_in.iloc[-1]
    pos_df_daily_in = pos_df_daily[pos_df_daily.index < cut_date]
    pot_in = AZ_Pot(pos_df_daily_in, last_asset_in)

    leve_ratio = AZ_Leverage_ratio(asset_df_in)
    if leve_ratio < 0:
        leve_ratio = 100
    sharpe_q_in_df = bt.AZ_Rolling_sharpe(pnl_df_in, roll_year=1, year_len=250, min_periods=1,
                                          cut_point_list=[0.3, 0.5, 0.7], output=False)
    sp_in = bt.AZ_Sharpe_y(pnl_df_in)
    fit_ratio = bt.AZ_fit_ratio(pos_df_daily_in, return_in)
    ic = round(bt.AZ_Normal_IC(pos_df_daily_in, pct_n, min_valids=None, lag=0).mean(), 6)
    sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d = sharpe_q_in_df.values
    in_condition_u = sharpe_q_in_df_u > 0.9 and leve_ratio > 1
    in_condition_d = sharpe_q_in_df_d < -0.9 and leve_ratio > 1
    # 分双边和只做多
    if if_only_long:
        in_condition = in_condition_u
    else:
        in_condition = in_condition_u | in_condition_d

    if sharpe_q_in_df_m > 0:
        way = 1
    else:
        way = -1

    # 样本外表现
    pnl_df_out = pnl_df[pnl_df.index >= cut_date]
    out_condition, sharpe_q_out = out_sample_perf_c(pnl_df_out, way=way)
    if if_return_pnl:
        return in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
               fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df
    else:
        return in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
               fit_ratio, leve_ratio, sp_in, sharpe_q_out


def mul_fun(a, b):
    a_l = a.where(a > 0, 0)
    a_s = a.where(a < 0, 0)

    b_l = b.where(b > 0, 0)
    b_s = b.where(b < 0, 0)

    pos_l = a_l.mul(b_l)
    pos_s = a_s.mul(b_s)

    pos = pos_l.sub(pos_s)
    return pos


def sub_fun(a, b):
    return a.sub(b, fill_value=0)


def add_fun(a, b):
    return a.add(b, fill_value=0)


def create_fun_set_2_crt():
    fun_2 = mul_fun
    mix_fun_set = dict()
    for fun_1 in [add_fun, sub_fun, mul_fun]:
        exe_str_1 = """def {0}_{1}_fun(a, b, c):
                mix_1 = {0}_fun(a, b)
                mix_2 = {1}_fun(mix_1, c)
                return mix_2
            """.format(fun_1.__name__.split('_')[0], fun_2.__name__.split('_')[0])
        exec(compile(exe_str_1, '', 'exec'))
        exec('mix_fun_set[\'{0}_{1}_fun\'] = {0}_{1}_fun'
             .format(fun_1.__name__.split('_')[0], fun_2.__name__.split('_')[0]))
    return mix_fun_set


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
    plt.plot(pnl_df.index, pnl_df.cumsum(), label='sharpe_ratio={}'.format(sharpe_ratio))
    plt.legend()
    plt.savefig(os.path.join(figure_save_path, '{}.png'.format(subject)))
    text = ''
    to = ['whs@yingpei.com']
    filepath = [os.path.join(figure_save_path, '{}.png'.format(subject))]
    send_email.send_email(text, to, filepath, subject)


def config_create(result_file_name, config_name, data, time_para, pot_in_num, leve_ratio_num, sp_in, ic_num, fit_ratio,
                  n,
                  use_factor_ratio=0.25):
    time_para_dict = dict()
    time_para_dict['time_para_4'] = [pd.to_datetime('20140601'), pd.to_datetime('20180601'),
                                     pd.to_datetime('20180901')]

    time_para_dict['time_para_5'] = [pd.to_datetime('20140701'), pd.to_datetime('20180701'),
                                     pd.to_datetime('20180901')]

    time_para_dict['time_para_6'] = [pd.to_datetime('20140801'), pd.to_datetime('20180801'),
                                     pd.to_datetime('20180901')]

    data_n = data[data['time_para'] == time_para]
    a_n = data_n[(data_n['ic'].abs() > ic_num) &
                 (data_n['pot_in'].abs() > pot_in_num) &
                 (data_n['leve_ratio'].abs() > leve_ratio_num) &
                 (data_n['sp_in'].abs() > sp_in) &
                 (data_n['fit_ratio'].abs() > fit_ratio)]
    a_n['pnl_file_name'] = a_n[['time_para', 'key', 'fun_name']].apply(lambda x: '|'.join(x.astype(str)), axis=1)
    print(a_n['con_out_2'].sum() / len(a_n), len(a_n))
    a_n['buy_sell'] = (a_n['sp_m'] > 0).astype(int).replace(0, -1)

    pnl_save_path = '/mnt/mfs/dat_whs/data/mix_factor_pnl/' + result_file_name
    sum_pnl_df = pd.DataFrame()
    for i in a_n.index:
        pnl_file_name = a_n['pnl_file_name'].loc[i]
        print('***************************************************')
        print('now {}\'s is running, key={}'.format(i, pnl_file_name))
        pnl_df = pd.read_pickle(os.path.join(pnl_save_path, '{}.csv'.format(pnl_file_name)))
        pnl_df.name = pnl_file_name
        sum_pnl_df = pd.concat([sum_pnl_df, pnl_df], axis=1)

    target_df = (sum_pnl_df > 0).astype(int)
    kmeans = KMeans(n_clusters=n).fit(target_df.T)

    kmeans_result = kmeans.labels_
    columns_list = target_df.columns
    group_df = pd.DataFrame(kmeans_result, index=columns_list)
    file_name_list = a_n['pnl_file_name'].values
    a_n['group_key'] = group_df.loc[file_name_list].values
    target_df = pd.DataFrame()

    for i in range(n):
        part_a_n = a_n[a_n['group_key'] == i].sort_values(by='pot_in')
        part_num = int(len(part_a_n) * use_factor_ratio)

        part_target_df = part_a_n[['fun_name', 'name1', 'name2', 'name3', 'buy_sell']].iloc[:part_num]
        print(part_num)
        target_df = target_df.append(part_target_df)
    print(len(target_df))

    config_info = dict()
    config_info['factor_info'] = target_df
    pd.to_pickle(config_info, '/mnt/mfs/alpha_whs/{}.pkl'.format(config_name))


def pos_sum_c(data, time_para, result_file_name, pot_in_num, leve_ratio_num, sp_in, ic_num, fit_ratio):
    time_para_dict = dict()

    time_para_dict['time_para_1'] = [pd.to_datetime('20110101'), pd.to_datetime('20150101'),
                                     pd.to_datetime('20150701')]

    time_para_dict['time_para_2'] = [pd.to_datetime('20120101'), pd.to_datetime('20160101'),
                                     pd.to_datetime('20160701')]

    time_para_dict['time_para_3'] = [pd.to_datetime('20130601'), pd.to_datetime('20170601'),
                                     pd.to_datetime('20171201')]

    time_para_dict['time_para_4'] = [pd.to_datetime('20140601'), pd.to_datetime('20180601'),
                                     pd.to_datetime('20180901')]

    time_para_dict['time_para_5'] = [pd.to_datetime('20140701'), pd.to_datetime('20180701'),
                                     pd.to_datetime('20180901')]

    time_para_dict['time_para_6'] = [pd.to_datetime('20140801'), pd.to_datetime('20180801'),
                                     pd.to_datetime('20180901')]

    data_n = data[data['time_para'] == time_para]
    begin_date, cut_date, end_date = time_para_dict[time_para]
    a_n = data_n[(data_n['ic'].abs() > ic_num) &
                 (data_n['pot_in'].abs() > pot_in_num) &
                 (data_n['leve_ratio'].abs() > leve_ratio_num) &
                 (data_n['sp_in'].abs() > sp_in) &
                 (data_n['fit_ratio'].abs() > fit_ratio)]
    print(a_n['con_out_2'].sum() / len(a_n), len(a_n))

    sum_factor_df = pd.DataFrame()
    pnl_save_path = '/mnt/mfs/dat_whs/data/mix_factor_pnl/' + result_file_name
    bt.AZ_Path_create(pnl_save_path)

    root_path = '/mnt/mfs/DAT_EQT'
    if_save = False
    if_new_program = True

    sector_name = 'market_top_2000'
    index_name = '000905'
    hold_time = 20
    lag = 2
    return_file = ''

    if_hedge = True
    if_only_long = False
    time_para_dict = dict()

    main = mf.FactorTestCRT(root_path, if_save, if_new_program, begin_date, cut_date, end_date, time_para_dict,
                            sector_name, hold_time, lag, return_file, if_hedge, if_only_long)

    for i in a_n.index:
        x, key, fun_name, name1, name2, name3, filter_fun_name, sector_name, \
        con_in, con_out_1, con_out_2, con_out_3, con_out_4, ic, \
        sp_u, sp_m, sp_d, pot_in, fit_ratio, leve_ratio, \
        sp_in, sp_out_1, sp_out_2, sp_out_3, sp_out_4 = a_n.loc[i]
        print('***************************************************')
        print('now {}\'s is running, key={}, {}, {}, {}, {}'.format(i, key, fun_name, name1, name2, name3))

        mix_factor, con_in_c, con_out_c, ic_c, sp_u_c, sp_m_c, sp_d_c, pot_in_c, fit_ratio_c, leve_ratio_c, sp_in_c, \
        sp_out_c, pnl_df_c = main.single_test(fun_name, name1, name2, name3)
        # plot_send_result(pnl_df_c, bt.AZ_Sharpe_y(pnl_df_c), '{}, key={}'.format(i, key))

        print(con_in, con_out_1, ic, sp_u, sp_m, sp_d, pot_in, fit_ratio, leve_ratio, sp_out_1)
        print(con_in_c, con_out_c, ic_c, sp_u_c, sp_m_c, sp_d_c, pot_in_c, fit_ratio_c, leve_ratio_c, sp_out_c)

        if sp_m > 0:
            sum_factor_df = sum_factor_df.add(mix_factor, fill_value=0)
            if not os.path.exists(os.path.join(pnl_save_path, '{}|{}|{}.csv'.format(x, key, fun_name))):
                pnl_df_c.to_pickle(os.path.join(pnl_save_path, '{}|{}|{}.csv'.format(x, key, fun_name)))
        else:
            sum_factor_df = sum_factor_df.add(-mix_factor, fill_value=0)
            if not os.path.exists(os.path.join(pnl_save_path, '{}|{}|{}.csv'.format(x, key, fun_name))):
                (-pnl_df_c).to_pickle(os.path.join(pnl_save_path, '{}|{}|{}.csv'.format(x, key, fun_name)))
        pass

    sum_pos_df = main.deal_mix_factor(sum_factor_df).shift(2)
    in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
    fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df = filter_all(cut_date, sum_pos_df, main.return_choose,
                                                                    if_return_pnl=True, if_only_long=False)
    print(in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in,
          fit_ratio, leve_ratio, sp_in, sharpe_q_out)
    plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df), 'mix_factor')
    return sum_pos_df, pnl_df


def survive_rario(data, pot_in_num, leve_ratio_num, sp_in, ic_num, fit_ratio):
    data_1 = data[data['time_para'] == 'time_para_1']
    data_2 = data[data['time_para'] == 'time_para_2']
    data_3 = data[data['time_para'] == 'time_para_3']
    data_4 = data[data['time_para'] == 'time_para_4']
    data_5 = data[data['time_para'] == 'time_para_5']
    data_6 = data[data['time_para'] == 'time_para_6']

    a_1 = data_1[(data_1['ic'].abs() > ic_num) &
                 (data_1['pot_in'].abs() > pot_in_num) &
                 (data_1['leve_ratio'].abs() > leve_ratio_num) &
                 (data_1['sp_in'].abs() > sp_in) &
                 (data_1['fit_ratio'].abs() > fit_ratio)]
    a_2 = data_2[(data_2['ic'].abs() > ic_num) &
                 (data_2['pot_in'].abs() > pot_in_num) &
                 (data_2['leve_ratio'].abs() > leve_ratio_num) &
                 (data_2['sp_in'].abs() > sp_in) &
                 (data_2['fit_ratio'].abs() > fit_ratio)]
    a_3 = data_3[(data_3['ic'].abs() > ic_num) &
                 (data_3['pot_in'].abs() > pot_in_num) &
                 (data_3['leve_ratio'].abs() > leve_ratio_num) &
                 (data_3['sp_in'].abs() > sp_in) &
                 (data_3['fit_ratio'].abs() > fit_ratio)]
    a_4 = data_4[(data_4['ic'].abs() > ic_num) &
                 (data_4['pot_in'].abs() > pot_in_num) &
                 (data_4['leve_ratio'].abs() > leve_ratio_num) &
                 (data_4['sp_in'].abs() > sp_in) &
                 (data_4['fit_ratio'].abs() > fit_ratio)]
    a_5 = data_5[(data_5['ic'].abs() > ic_num) &
                 (data_5['pot_in'].abs() > pot_in_num) &
                 (data_5['leve_ratio'].abs() > leve_ratio_num) &
                 (data_5['sp_in'].abs() > sp_in) &
                 (data_5['fit_ratio'].abs() > fit_ratio)]
    a_6 = data_6[(data_6['ic'].abs() > ic_num) &
                 (data_6['pot_in'].abs() > pot_in_num) &
                 (data_6['leve_ratio'].abs() > leve_ratio_num) &
                 (data_6['sp_in'].abs() > sp_in) &
                 (data_6['fit_ratio'].abs() > fit_ratio)]
    print('_______________________________________')
    print(a_1['con_out_1'].sum() / len(a_1), len(a_1))
    print(a_2['con_out_1'].sum() / len(a_2), len(a_2))
    print(a_3['con_out_1'].sum() / len(a_3), len(a_3))
    print(a_4['con_out_1'].sum() / len(a_4), len(a_4))
    print(a_5['con_out_1'].sum() / len(a_5), len(a_5))
    print(a_6['con_out_1'].sum() / len(a_6), len(a_6))

    print('_______________________________________')
    print(a_1['con_out_2'].sum() / len(a_1), len(a_1))
    print(a_2['con_out_2'].sum() / len(a_2), len(a_2))
    print(a_3['con_out_2'].sum() / len(a_3), len(a_3))
    print(a_4['con_out_2'].sum() / len(a_4), len(a_4))
    print(a_5['con_out_2'].sum() / len(a_5), len(a_5))
    print(a_6['con_out_2'].sum() / len(a_6), len(a_6))

    print('_______________________________________')
    print(a_1['con_out_3'].sum() / len(a_1), len(a_1))
    print(a_2['con_out_3'].sum() / len(a_2), len(a_2))
    print(a_3['con_out_3'].sum() / len(a_3), len(a_3))
    print(a_4['con_out_3'].sum() / len(a_4), len(a_4))
    print(a_5['con_out_3'].sum() / len(a_5), len(a_5))
    print(a_6['con_out_3'].sum() / len(a_6), len(a_6))

    print('_______________________________________')
    print(a_1['con_out_4'].sum() / len(a_1), len(a_1))
    print(a_2['con_out_4'].sum() / len(a_2), len(a_2))
    print(a_3['con_out_4'].sum() / len(a_3), len(a_3))
    print(a_4['con_out_4'].sum() / len(a_4), len(a_4))
    print(a_5['con_out_4'].sum() / len(a_5), len(a_5))
    print(a_6['con_out_4'].sum() / len(a_6), len(a_6))


def config_test(config_name, time_para):
    config_set = pd.read_pickle(f'/mnt/mfs/alpha_whs/{config_name}.pkl')
    config_data = config_set['factor_info']

    time_para_dict = dict()

    time_para_dict['time_para_1'] = [pd.to_datetime('20110101'), pd.to_datetime('20150101'),
                                     pd.to_datetime('20150701')]

    time_para_dict['time_para_2'] = [pd.to_datetime('20120101'), pd.to_datetime('20160101'),
                                     pd.to_datetime('20160701')]

    time_para_dict['time_para_3'] = [pd.to_datetime('20130601'), pd.to_datetime('20170601'),
                                     pd.to_datetime('20171201')]

    time_para_dict['time_para_4'] = [pd.to_datetime('20140601'), pd.to_datetime('20180601'),
                                     pd.to_datetime('20180901')]

    time_para_dict['time_para_5'] = [pd.to_datetime('20140701'), pd.to_datetime('20180701'),
                                     pd.to_datetime('20180901')]

    time_para_dict['time_para_6'] = [pd.to_datetime('20140801'), pd.to_datetime('20180801'),
                                     pd.to_datetime('20180901')]

    begin_date, cut_date, end_date = time_para_dict[time_para]

    sum_factor_df = pd.DataFrame()
    pnl_save_path = '/mnt/mfs/dat_whs/data/mix_factor_pnl/' + result_file_name
    bt.AZ_Path_create(pnl_save_path)

    root_path = '/mnt/mfs/DAT_EQT'
    if_save = False
    if_new_program = True

    sector_name = 'market_top_2000'
    index_name = '000905'
    hold_time = 20
    lag = 2
    return_file = ''

    if_hedge = True
    if_only_long = False
    time_para_dict = dict()

    main = mf.FactorTestCRT(root_path, if_save, if_new_program, begin_date, cut_date, end_date, time_para_dict,
                            sector_name, hold_time, lag, return_file, if_hedge, if_only_long)

    for i in config_data.index:
        fun_name, name1, name2, name3, buy_sell = config_data.loc[i]
        print('***************************************************')
        print('now {}\'s is running, key={}, {}, {}, {}'.format(i, fun_name, name1, name2, name3))

        mix_factor, con_in_c, con_out_c, ic_c, sp_u_c, sp_m_c, sp_d_c, pot_in_c, fit_ratio_c, leve_ratio_c, sp_in_c, \
        sp_out_c, pnl_df_c = main.single_test(fun_name, name1, name2, name3)
        plot_send_result(pnl_df_c, bt.AZ_Sharpe_y(pnl_df_c), '{}, {}, {}, {}, {}'
                         .format(fun_name, name1, name2, name3, buy_sell))
        print(con_in_c, con_out_c, ic_c, sp_u_c, sp_m_c, sp_d_c, pot_in_c, fit_ratio_c, leve_ratio_c, sp_out_c)

        if buy_sell > 0:
            sum_factor_df = sum_factor_df.add(mix_factor, fill_value=0)

        else:
            sum_factor_df = sum_factor_df.add(-mix_factor, fill_value=0)

    sum_pos_df = main.deal_mix_factor(sum_factor_df).shift(2)
    in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
    fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df = filter_all(cut_date, sum_pos_df, main.return_choose,
                                                                    if_return_pnl=True, if_only_long=False)
    print(in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in,
          fit_ratio, leve_ratio, sp_in, sharpe_q_out)
    plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df), 'mix_factor')
    return sum_pos_df, pnl_df


if __name__ == '__main__':
    # WHS018JUN JUL AUG三个alpha的生成数据
    # result_file_name = 'market_top_2000_True_20180903_1018_hold_20_aadj_r'
    # alpha2
    # result_file_name = 'market_top_2000_True_20180904_0856_hold_20__'
    # result_file_name = 'market_top_2000_True_20180906_1024_hold_20__'
    result_file_name = 'market_top_2000_True_20180909_1346_hold_20__'
    # result_file_name = 'market_top_2000_industry_10_15_True_20180910_1355_hold_20__'
    # result_file_name = 'market_top_2000_True_20180827_1555_hold_20_aadj_r'
    # result_file_name = 'market_top_2000_True_20180830_1133_hold_20_aadj_r'
    # result_file_name = 'market_top_2000_True_20180827_1555_hold_20_aadj_r'

    data = pd.read_csv('/mnt/mfs/dat_whs/result/result/{}.txt'.format(result_file_name),
                       sep='|', header=None, error_bad_lines=False)

    data.columns = ['time_para', 'key', 'fun_name', 'name1', 'name2', 'name3', 'filter_fun_name', 'sector_name',
                    'con_in', 'con_out_1', 'con_out_2', 'con_out_3', 'con_out_4', 'ic', 'sp_u', 'sp_m', 'sp_d',
                    'pot_in', 'fit_ratio', 'leve_ratio', 'sp_in', 'sp_out_1', 'sp_out_2', 'sp_out_3', 'sp_out_4']

    # data['name1'] = data['name1'] + '_and_QTTM_0.3'
    # data['name2'] = data['name2'] + '_row_extre_0.2'

    filter_cond = data[['name1', 'name2', 'name3']] \
        .apply(lambda x: not (('R_COMPANYCODE_First_row_extre_0.3' in set(x)) or
                              ('return_p20d_0.2' in set(x)) or
                              ('price_p120d_hl' in set(x)) or
                              ('return_p60d_0.2' in set(x)) or
                              ('wgt_return_p120d_0.2' in set(x)) or
                              ('wgt_return_p20d_0.2' in set(x)) or
                              ('log_price_0.2' in set(x)) or
                              ('TVOL_row_extre_0.2' in set(x))), axis=1)

    data = data[filter_cond]
    pot_in_num = 30
    leve_ratio_num = 1
    sp_in = 1
    ic_num = 0.002
    fit_ratio = 1

    # pot_in_num = 50
    # leve_ratio_num = 2
    # sp_in = 1
    # ic_num = 0.002
    # fit_ratio = 2

    # #############################################################################
    # 结果分析
    survive_rario(data, pot_in_num, leve_ratio_num, sp_in, ic_num, fit_ratio)

    #############################################################################
    # 回测函数
    time_para_list = ['time_para_4', 'time_para_5', 'time_para_6']
    for time_para in time_para_list[:1]:
        print(time_para)
        sum_pos_df, pnl_df = pos_sum_c(data, time_para, result_file_name,
                                       pot_in_num, leve_ratio_num, sp_in, ic_num, fit_ratio)

    #############################################################################
    # 生成config文件
    time_para_list = ['time_para_4', 'time_para_5', 'time_para_6']
    # CRT alpha1
    # config_dict = {'time_para_4': 'CRTJUN01', 'time_para_5': 'CRTJUL01', 'time_para_6': 'CRTAUG01'}
    # CRT alpha2
    config_dict = {'time_para_4': 'CRTJUN02', 'time_para_5': 'CRTJUL02', 'time_para_6': 'CRTAUG02'}
    for time_para in time_para_list[:1]:
        config_name = config_dict[time_para]
        config_create(result_file_name, config_name, data, time_para, pot_in_num, leve_ratio_num, sp_in, ic_num,
                      fit_ratio, 5, use_factor_ratio=0.10)

    #############################################################################
    # 测试config结果
    # sum_pos_df, pnl_df_CRTJUN02 = config_test('CRTJUN02', 'time_para_4')
    # sum_pos_df, pnl_df_CRTJUL01 = config_test('CRTJUL01', 'time_para_5')
