import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import product, permutations, combinations
from sklearn.cluster import KMeans
from multiprocessing import Pool
import random
import sys
from datetime import datetime
import time

# import matplotlib

# matplotlib.use('Qt5Agg')
# matplotlib.use('Agg')

sys.path.append('/mnt/mfs/work_whs/AZ_2018_Q2')
sys.path.append('/mnt/mfs')
from work_whs.loc_lib.shared_tools import send_email
import work_whs.loc_lib.shared_tools.back_test as bt
from collections import Counter


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
    if if_only_long:
        pnl_df = (pos_df_daily[pos_df_daily > 0] * pct_n).sum(axis=1)
        pnl_df = pnl_df.replace(np.nan, 0)
    else:
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
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(figure_save_path, '{}.png'.format(subject)))
    plt.close()
    text = ''
    to = ['whs@yingpei.com']
    filepath = [os.path.join(figure_save_path, '{}.png'.format(subject))]
    send_email.send_email(text, to, filepath, subject)


def config_create(main_model, sector_name, result_file_name, config_name, data, time_para, pot_in_num, leve_ratio_num,
                  sp_in, ic_num, fit_ratio, n, use_factor_num):
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
                 (data_n['fit_ratio'].abs() > fit_ratio) &
                 (data_n['sp_in'] * data_n['sp_out_4'] > 0)]

    a_n['pnl_file_name'] = a_n[['time_para', 'key', 'fun_name']].apply(lambda x: '|'.join(x.astype(str)), axis=1)
    print(a_n['con_out_2'].sum() / len(a_n), len(a_n))
    a_n['buy_sell'] = (a_n['sp_m'] > 0).astype(int).replace(0, -1)
    use_factor_ratio = use_factor_num / len(a_n.index)

    pnl_save_path = '/mnt/mfs/dat_whs/data/mix_factor_pnl/' + result_file_name
    sum_pnl_df = pd.DataFrame()
    for i in a_n.index:
        pnl_file_name = a_n['pnl_file_name'].loc[i]
        print('***************************************************')
        print('now {}\'s is running, key={}'.format(i, pnl_file_name))
        pnl_df = pd.read_pickle(os.path.join(pnl_save_path, '{}.csv'.format(pnl_file_name)))
        pnl_df.name = pnl_file_name
        sum_pnl_df = pd.concat([sum_pnl_df, pnl_df], axis=1)
    # _________________________________________________________________________________
    part_sum_pnl_df = sum_pnl_df.loc[:pd.to_datetime('20180601')]
    sharpe_df_after = part_sum_pnl_df.iloc[-100:].apply(bt.AZ_Sharpe_y)
    sharpe_df_after.name = 'sharpe_df_after'
    sharpe_df_before = part_sum_pnl_df.iloc[:-100].apply(bt.AZ_Sharpe_y)
    sharpe_df_before.name = 'sharpe_df_before'
    sharpe_df = part_sum_pnl_df.apply(bt.AZ_Sharpe_y)
    sharpe_df.name = 'sharpe_df'
    # info_df = pd.concat([sharpe_df_before, sharpe_df_after], axis=1)

    # _________________________________________________________________________________
    target_df = (sum_pnl_df > 0).astype(int)
    kmeans = KMeans(n_clusters=n).fit(target_df.T)

    kmeans_result = kmeans.labels_
    columns_list = target_df.columns
    group_df = pd.DataFrame(kmeans_result, index=columns_list)
    file_name_list = a_n['pnl_file_name'].values
    a_n['group_key'] = group_df.loc[file_name_list].values
    target_df = pd.DataFrame()

    for i in range(n):
        part_a_n = a_n[a_n['group_key'] == i].sort_values(by='sp_in')
        part_num = int(len(part_a_n) * use_factor_ratio)

        part_target_df = part_a_n[['fun_name', 'name1', 'name2', 'name3', 'buy_sell']].iloc[:part_num]
        print(part_num)
        target_df = target_df.append(part_target_df)

    print(len(target_df))
    print(Counter(target_df['name1'].values))
    print(Counter(target_df['name2'].values))
    print(Counter(target_df['name3'].values))

    config_info = dict()
    config_info['factor_info'] = target_df
    config_info['sector_name'] = sector_name
    config_info['result_file_name'] = result_file_name
    config_info['if_weight'] = main_model.if_weight
    config_info['ic_weight'] = main_model.ic_weight
    config_info['hold_time'] = main_model.hold_time
    config_info['if_hedge'] = main_model.if_hedge
    config_info['if_only_long'] = main_model.if_only_long
    pd.to_pickle(config_info, '/mnt/mfs/dat_whs/alpha_data/{}.pkl'.format(config_name))


def bkt_fun(main_model, pnl_save_path, a_n, i):
    x, key, fun_name, name1, name2, name3, filter_fun_name, sector_name, \
    con_in, con_out_1, con_out_2, con_out_3, con_out_4, ic, \
    sp_u, sp_m, sp_d, pot_in, fit_ratio, leve_ratio, \
    sp_in, sp_out_1, sp_out_2, sp_out_3, sp_out_4 = a_n.loc[i]

    mix_factor, con_in_c, con_out_c, ic_c, sp_u_c, sp_m_c, sp_d_c, pot_in_c, fit_ratio_c, leve_ratio_c, \
    sp_in_c, sp_out_c, pnl_df_c = main_model.single_test(fun_name, name1, name2, name3)
    # plot_send_result(pnl_df_c, bt.AZ_Sharpe_y(pnl_df_c), '{}, key={}'.format(i, key))

    print('***************************************************')
    print('now {}\'s is running, key={}, {}, {}, {}, {}'.format(i, key, fun_name, name1, name2, name3))
    print(con_in_c, con_out_c, ic_c, sp_u_c, sp_m_c, sp_d_c, pot_in_c, fit_ratio_c, leve_ratio_c, sp_out_c)
    print(con_in, con_out_1, ic, sp_u, sp_m, sp_d, pot_in, fit_ratio, leve_ratio, sp_out_1)

    if sp_m > 0:
        if not os.path.exists(os.path.join(pnl_save_path, '{}|{}|{}.csv'.format(x, key, fun_name))):
            pnl_df_c.to_pickle(os.path.join(pnl_save_path, '{}|{}|{}.csv'.format(x, key, fun_name)))

        else:
            pnl_df_c.to_pickle(os.path.join(pnl_save_path, '{}|{}|{}.csv'.format(x, key, fun_name)))
            print('file exist!')
        return mix_factor
    else:
        if not os.path.exists(os.path.join(pnl_save_path, '{}|{}|{}.csv'.format(x, key, fun_name))):
            (-pnl_df_c).to_pickle(os.path.join(pnl_save_path, '{}|{}|{}.csv'.format(x, key, fun_name)))
        else:
            (-pnl_df_c).to_pickle(os.path.join(pnl_save_path, '{}|{}|{}.csv'.format(x, key, fun_name)))
            print('file exist!')
        return -mix_factor


def pos_sum_c(main_model, data, time_para, result_file_name, pot_in_num, leve_ratio_num, sp_in, ic_num, fit_ratio):
    time_para_dict = dict()

    time_para_dict['time_para_1'] = [pd.to_datetime('20110101'), pd.to_datetime('20150101'),
                                     pd.to_datetime('20150701')]

    time_para_dict['time_para_2'] = [pd.to_datetime('20120101'), pd.to_datetime('20160101'),
                                     pd.to_datetime('20160701')]

    time_para_dict['time_para_3'] = [pd.to_datetime('20130601'), pd.to_datetime('20170601'),
                                     pd.to_datetime('20171201')]

    time_para_dict['time_para_4'] = [pd.to_datetime('20140601'), pd.to_datetime('20180601'),
                                     pd.to_datetime('20181001')]

    time_para_dict['time_para_5'] = [pd.to_datetime('20140701'), pd.to_datetime('20180701'),
                                     pd.to_datetime('20181001')]

    time_para_dict['time_para_6'] = [pd.to_datetime('20140801'), pd.to_datetime('20180801'),
                                     pd.to_datetime('20181001')]

    data_n = data[data['time_para'] == time_para]
    begin_date, cut_date, end_date = time_para_dict[time_para]
    a_n = data_n[(data_n['ic'].abs() > ic_num) &
                 (data_n['pot_in'].abs() > pot_in_num) &
                 (data_n['leve_ratio'].abs() > leve_ratio_num) &
                 (data_n['sp_in'].abs() > sp_in) &
                 (data_n['fit_ratio'].abs() > fit_ratio) &
                 (data_n['sp_in'] * data_n['sp_out_4'] > 0)]

    sum_factor_df = pd.DataFrame()
    pnl_save_path = '/mnt/mfs/dat_whs/data/mix_factor_pnl/' + result_file_name
    bt.AZ_Path_create(pnl_save_path)

    result_list = []
    pool = Pool(10)
    for i in a_n.index:
        # bkt_fun(main_model, pnl_save_path, a_n, i,)
        result_list.append(pool.apply_async(bkt_fun, args=(main_model, pnl_save_path, a_n, i,)))
    pool.close()
    pool.join()

    for res in result_list:
        sum_factor_df = sum_factor_df.add(res.get(), fill_value=0)

    sum_pos_df = main_model.deal_mix_factor(sum_factor_df).shift(2)
    in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
    fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df = filter_all(cut_date, sum_pos_df, main_model.return_choose,
                                                                    if_return_pnl=True, if_only_long=False)
    print(in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in,
          fit_ratio, leve_ratio, sp_in, sharpe_q_out)
    # plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df), 'mix_factor')
    return sum_pos_df, pnl_df


def survive_ratio(data, pot_in_num, leve_ratio_num, sp_in, ic_num, fit_ratio):
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

    return a_1, a_2, a_3, a_4, a_5, a_6


def survive_ratio_test(data, para_adj_set_list):
    for para_adj_set in para_adj_set_list:
        a_1, a_2, a_3, a_4, a_5, a_6 = survive_ratio(data, **para_adj_set)
        for con_out_name in ['con_out_4', 'con_out_3']:
            sr_1 = a_1[con_out_name].sum() / len(a_1)
            sr_2 = a_2[con_out_name].sum() / len(a_2)
            sr_3 = a_3[con_out_name].sum() / len(a_3)
            sr_4 = a_4[con_out_name].sum() / len(a_4)
            sr_5 = a_5[con_out_name].sum() / len(a_5)
            sr_6 = a_6[con_out_name].sum() / len(a_6)
            print(sr_1, sr_2, sr_3, sr_4, sr_5, sr_6)
            print(len(a_1), len(a_2), len(a_3), len(a_4), len(a_5), len(a_6))
            sr_list_in = np.array([sr_1, sr_2, sr_3])
            sr_list_out = np.array([sr_4, sr_5, sr_6])
            cond_1 = sum(sr_list_in > 0.5) >= 2  # and sum(sr_list_in > 0.2) == 3
            # cond_2 = sr_5 > 0.55

            cond_3_1 = sum(sr_list_out > 0.55) >= 1
            cond_3_2 = sum(sr_list_out > 0.3) >= 2
            cond_3_3 = sum(sr_list_out > 0.15) >= 3

            cond_3 = cond_3_1 and cond_3_2 and cond_3_3
            cond_4 = (len(a_4) > 20) and (len(a_5) > 20) and (len(a_6) > 20)
            print(cond_1, cond_3, cond_4)
            # if cond_1 and cond_3 and cond_4:
            if cond_3 and cond_4:
                return para_adj_set
    return None


def config_test(main_model, config_name, result_file_name, cut_date):
    config_set = pd.read_pickle(f'/mnt/mfs/dat_whs/alpha_data/{config_name}.pkl')
    config_data = config_set['factor_info']
    sum_factor_df = pd.DataFrame()
    for i in config_data.index:
        fun_name, name1, name2, name3, buy_sell = config_data.loc[i]
        print('***************************************************')
        print('now {}\'s is running, key={}, {}, {}, {}'.format(i, fun_name, name1, name2, name3))

        mix_factor, con_in_c, con_out_c, ic_c, sp_u_c, sp_m_c, sp_d_c, pot_in_c, fit_ratio_c, leve_ratio_c, sp_in_c, \
        sp_out_c, pnl_df_c = main_model.single_test(fun_name, name1, name2, name3)
        # plot_send_result(pnl_df_c, bt.AZ_Sharpe_y(pnl_df_c), '{}, {}, {}, {}, {}'
        #                  .format(fun_name, name1, name2, name3, buy_sell))
        # print(con_in_c, con_out_c, ic_c, sp_u_c, sp_m_c, sp_d_c, pot_in_c, fit_ratio_c, leve_ratio_c, sp_out_c)

        if buy_sell > 0:
            sum_factor_df = sum_factor_df.add(mix_factor, fill_value=0)
        else:
            sum_factor_df = sum_factor_df.add(-mix_factor, fill_value=0)

    sum_pos_df = main_model.deal_mix_factor(sum_factor_df).shift(2)
    in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
    fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df = filter_all(cut_date, sum_pos_df, main_model.return_choose,
                                                                    if_return_pnl=True, if_only_long=False)
    print(in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in,
          fit_ratio, leve_ratio, sp_in, sharpe_q_out)
    sp = bt.AZ_Sharpe_y(pnl_df)
    pnl_df.to_csv(f'/mnt/mfs/dat_whs/tmp_pnl_file/{result_file_name}.csv')
    send_list = [in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in,
                 fit_ratio, leve_ratio, sp_in, sharpe_q_out]
    send_email.send_email(','.join([str(x) for x in send_list]), ['whs@yingpei.com'], [], result_file_name)
    plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df), result_file_name)
    return sum_pos_df, pnl_df, sp


def load_index_data(index_name, xinx):
    data = bt.AZ_Load_csv(os.path.join('/mnt/mfs/DAT_EQT', 'EM_Tab09/INDEX_TD_DAILYSYS/CHG.csv'))
    target_df = data[index_name].reindex(index=xinx)
    return target_df * 0.01


def get_corr_matrix(cut_date=None):
    pos_file_list = [x for x in os.listdir('/mnt/mfs/AAPOS') if x.startswith('WHS')]
    return_df = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv').astype(float)

    index_df_1 = load_index_data('000300', return_df.index).fillna(0)
    index_df_2 = load_index_data('000905', return_df.index).fillna(0)

    sum_pnl_df = pd.DataFrame()
    for pos_file_name in pos_file_list:
        pos_df = bt.AZ_Load_csv('/mnt/mfs/AAPOS/{}'.format(pos_file_name))

        cond_1 = 'IF01' in pos_df.columns
        cond_2 = 'IC01' in pos_df.columns
        if cond_1 and cond_2:
            hedge_df = 0.5 * index_df_1 + 0.5 * index_df_2
            return_df_c = return_df.sub(hedge_df, axis=0)
        elif cond_1:
            hedge_df = index_df_1
            return_df_c = return_df.sub(hedge_df, axis=0)
        elif cond_2:
            hedge_df = index_df_2
            return_df_c = return_df.sub(hedge_df, axis=0)
        else:
            print('alpha hedge error')
            continue
        pnl_df = (pos_df.shift(2) * return_df_c).sum(axis=1)
        pnl_df.name = pos_file_name
        sum_pnl_df = pd.concat([sum_pnl_df, pnl_df], axis=1)
        # plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df), 'mix_factor')
    if cut_date is not None:
        sum_pnl_df = sum_pnl_df[sum_pnl_df.index > cut_date]
    return sum_pnl_df


def get_all_pnl_corr(pnl_df, col_name):
    all_pnl_df = pd.read_csv('/mnt/mfs/AATST/corr_tst_pnls', sep='|', index_col=0, parse_dates=True)
    all_pnl_df_c = pd.concat([all_pnl_df, pnl_df], axis=1)
    a = all_pnl_df_c.iloc[-600:].corr()[col_name]
    print(a[a > 0.6])
    return a


def find_sector_name(result_file_name):
    if 'True' in result_file_name:
        sector_name = result_file_name.split('True')[0][:-1]
        if_hedge = True
    elif 'False' in result_file_name:
        sector_name = result_file_name.split('False')[0][:-1]
        if_hedge = False
    else:
        print('sector_name ERROR!')
        sector_name = None
        if_hedge = None
    return sector_name, if_hedge


def find_target_file(begin_time, end_time, time_type, endswith=None):
    def time_judge(begin_time, end_time, target_time):
        # print(datetime.fromtimestamp(target_time))
        if end_time >= datetime.fromtimestamp(target_time) >= begin_time:
            return True
        else:
            return False

    result_root_path = '/mnt/mfs/dat_whs/result/result'
    raw_file_name_list = [x for x in os.listdir(result_root_path)
                          if os.path.getsize(os.path.join(result_root_path, x)) != 0]

    if endswith is not None:
        raw_file_name_list = [x for x in os.listdir(result_root_path)
                              if x[:-4].endswith(endswith)]

    if time_type == 'm':
        result_file_name_list = [x[:-4] for x in raw_file_name_list if
                                 time_judge(begin_time, end_time, os.path.getmtime(os.path.join(result_root_path, x)))]
        return sorted(result_file_name_list)
    elif time_type == 'c':
        result_file_name_list = [x[:-4] for x in raw_file_name_list if
                                 time_judge(begin_time, end_time, os.path.getctime(os.path.join(result_root_path, x)))]

        return sorted(result_file_name_list)

    elif time_type == 'a':
        result_file_name_list = [x[:-4] for x in raw_file_name_list if
                                 time_judge(begin_time, end_time, os.path.getatime(os.path.join(result_root_path, x)))]

        return sorted(result_file_name_list)

    else:
        return []


def main(result_file_name, time_para_dict):
    print('*******************************************************************************************************')
    root_path = '/mnt/mfs/DAT_EQT'
    # result_file_name = 'market_top_800plus_True_20181104_0237_hold_5__7'
    # root_path = '/media/hdd1/DAT_EQT'
    config_name = result_file_name
    if_save = False
    if_new_program = True

    hold_time = int(result_file_name.split('hold')[-1].split('_')[1])
    # 加载对应脚本

    if result_file_name.split('_')[-1] == 'long':
        script_num = result_file_name.split('_')[-2]
        if_only_long = True
    else:
        script_num = result_file_name.split('_')[-1]
        if_only_long = False
    # script_num = '15'
    print('script_num : ', script_num)
    print('hold_time : ', hold_time)
    loc = locals()
    try:
        exec(f'from work_whs.AZ_2018_Q2.factor_script.main_file import main_file_sector_{script_num} as mf')
    except Exception as error:
        print(error)
        return -1
    mf = loc['mf']
    time_para_dict = mf.time_para_dict
    # from work_whs.AZ_2018_Q2.factor_script.main_file import main_file_sector_6 as mf

    lag = 2
    return_file = ''

    sector_name, if_hedge = find_sector_name(result_file_name)
    if_hedge = True
    # sector_name = 'market_top_800plus_industry_10_15'
    print(result_file_name)
    print(sector_name)
    result_path = '/mnt/mfs/dat_whs/result/result/{}.txt'.format(result_file_name)
    # #############################################################################
    # 判断文件大小
    if os.path.getsize(result_path):
        try:
            data = pd.read_csv(result_path, sep='|', header=None, error_bad_lines=False)

            data.columns = ['time_para', 'key', 'fun_name', 'name1', 'name2', 'name3', 'filter_fun_name', 'sector_name',
                            'con_in', 'con_out_1', 'con_out_2', 'con_out_3', 'con_out_4', 'ic', 'sp_u', 'sp_m', 'sp_d',
                            'pot_in', 'fit_ratio', 'leve_ratio', 'sp_in', 'sp_out_1', 'sp_out_2', 'sp_out_3',
                            'sp_out_4']
        except Exception as error:
            print(error)
            return -1
    else:
        return -1

    filter_cond = data[['name1', 'name2', 'name3']] \
        .apply(lambda x: not (('R_COMPANYCODE_First_row_extre_0.3' in set(x)) or
                              ('return_p20d_0.2' in set(x)) or
                              ('price_p120d_hl' in set(x)) or
                              ('return_p60d_0.2' in set(x)) or
                              ('wgt_return_p120d_0.2' in set(x)) or
                              ('wgt_return_p20d_0.2' in set(x)) or
                              ('log_price_0.2' in set(x)) or
                              ('TVOL_row_extre_0.2' in set(x)) or
                              ('TVOL_row_extre_0.2' in set(x)) or
                              ('tab2_11_row_extre_0.3' in set(x)) or
                              ('tab1_8_row_extre_0.3' in set(x)) or
                              ('intra_dn_vol_col_score_row_extre_0.3' in set(x)) or
                              ('intra_dn_vol_row_extre_0.3' in set(x)) or
                              ('turn_p30d_0.24' in set(x)) or
                              ('evol_p30d' in set(x))
                              # ('CMO_40_0' in set(x))
                              # ('ATR_40_0.2' in set(x))
                              # ('ADX_200_40_20' in set(x))
                              # ('ATR_140_0.2' in set(x))
                              ), axis=1)
    data = data[filter_cond]

    para_adj_set_list = [{'pot_in_num': 50, 'leve_ratio_num': 2, 'sp_in': 1.5, 'ic_num': 0.0, 'fit_ratio': 2},
                         {'pot_in_num': 40, 'leve_ratio_num': 2, 'sp_in': 1.5, 'ic_num': 0.0, 'fit_ratio': 2},
                         {'pot_in_num': 50, 'leve_ratio_num': 2, 'sp_in': 1, 'ic_num': 0.0, 'fit_ratio': 1},
                         {'pot_in_num': 50, 'leve_ratio_num': 1, 'sp_in': 1, 'ic_num': 0.0, 'fit_ratio': 2},
                         {'pot_in_num': 50, 'leve_ratio_num': 1, 'sp_in': 1, 'ic_num': 0.0, 'fit_ratio': 1},
                         {'pot_in_num': 40, 'leve_ratio_num': 1, 'sp_in': 1, 'ic_num': 0.0, 'fit_ratio': 1}]

    # para_adj_set_list_9 = [{'pot_in_num': 30, 'leve_ratio_num': 2, 'sp_in': 1.5, 'ic_num': 0.0, 'fit_ratio': 2},
    #                        {'pot_in_num': 20, 'leve_ratio_num': 2, 'sp_in': 1.5, 'ic_num': 0.0, 'fit_ratio': 2},
    #                        {'pot_in_num': 30, 'leve_ratio_num': 2, 'sp_in': 1, 'ic_num': 0.0, 'fit_ratio': 1},
    #                        {'pot_in_num': 30, 'leve_ratio_num': 1, 'sp_in': 1, 'ic_num': 0.0, 'fit_ratio': 2},
    #                        {'pot_in_num': 30, 'leve_ratio_num': 1, 'sp_in': 1, 'ic_num': 0.0, 'fit_ratio': 1},
    #                        {'pot_in_num': 20, 'leve_ratio_num': 1, 'sp_in': 1, 'ic_num': 0.0, 'fit_ratio': 1}]

    time_para = 'time_para_5'
    print(time_para)

    # #############################################################################
    # 结果分析
    print('结果分析')
    survive_result = survive_ratio_test(data, para_adj_set_list)
    if survive_result is None:
        print(f'{result_file_name} not satisfaction!!!!!!!!')
        return 0
    else:
        pass
    print(hold_time)
    #############################################################################
    # 回测函数
    if sector_name.startswith('market_top_300plus'):
        if_weight = 1
        ic_weight = 0

    elif sector_name.startswith('market_top_300to800plus'):
        if_weight = 0
        ic_weight = 1

    else:
        if_weight = 0.5
        ic_weight = 0.5
    print('回测函数')
    begin_date, cut_date, end_date, end_date, end_date, end_date = time_para_dict[time_para]
    main_model = mf.FactorTestSector(root_path, if_save, if_new_program, begin_date, cut_date, end_date,
                                     time_para_dict, sector_name, hold_time, lag, return_file, if_hedge,
                                     if_only_long, if_weight, ic_weight)
    try:
        sum_pos_df, pnl_df = pos_sum_c(main_model, data, time_para, result_file_name, **survive_result)
    except Exception as error:
        print(error)
        return -1
    #############################################################################
    # 生成config文件

    config_create(main_model, sector_name, result_file_name, config_name, data, time_para, **survive_result,
                  n=5, use_factor_num=40)
    ###########################################################################
    # 测试config结果
    begin_date, cut_date, end_date, end_date, end_date, end_date = time_para_dict[time_para]
    sum_pos_df, pnl_df, sp = config_test(main_model, config_name, result_file_name, cut_date)
    if sp < 2:
        return 0
    pnl_df.name = result_file_name
    # #############################################################################
    # 计算相关性
    # pnl_df_CRTSECJUN
    #
    sum_pnl_df = get_corr_matrix(cut_date=None)
    sum_pnl_df_c = pd.concat([sum_pnl_df, pnl_df], axis=1)

    corr_self = sum_pnl_df_c.corr()[[result_file_name]]
    print(corr_self)
    print('______________________________________')
    print(corr_self[corr_self > 0.7].dropna(axis=0))
    if len(corr_self[corr_self > 0.7].dropna(axis=0)) >= 2:
        print('FAIL!')
        send_email.send_email('FAIL!\n' + pd.DataFrame(corr_self).to_html(),
                              ['whs@yingpei.com'],
                              [],
                              result_file_name)
    else:
        print('SUCCESS!')
        send_email.send_email('SUCCESS!\n' + pd.DataFrame(corr_self).to_html(),
                              ['whs@yingpei.com'],
                              [],
                              result_file_name)
    print('______________________________________')
    return 0


if __name__ == '__main__':
    time_para_dict = dict()
    time_para_dict['time_para_1'] = [pd.to_datetime('20100101'), pd.to_datetime('20150101'),
                                     pd.to_datetime('20151001')]

    time_para_dict['time_para_2'] = [pd.to_datetime('20110101'), pd.to_datetime('20160101'),
                                     pd.to_datetime('20161001')]

    time_para_dict['time_para_3'] = [pd.to_datetime('20120601'), pd.to_datetime('20170601'),
                                     pd.to_datetime('20180301')]

    time_para_dict['time_para_4'] = [pd.to_datetime('20130601'), pd.to_datetime('20180601'),
                                     pd.to_datetime('20181201')]

    time_para_dict['time_para_5'] = [pd.to_datetime('20130701'), pd.to_datetime('20180701'),
                                     pd.to_datetime('20181201')]

    time_para_dict['time_para_6'] = [pd.to_datetime('20130801'), pd.to_datetime('20180801'),
                                     pd.to_datetime('20181201')]

    # # begin_time = datetime(2018, 11, 1, 6, 11, 13)
    # # end_time = datetime(2018, 11, 6, 6, 30, 13)

    # begin_time = datetime(2018, 11, 8, 6, 11, 13)
    # end_time = datetime(2018, 11, 11, 6, 30, 13)

    # begin_time = datetime(2018, 10, 11, 6, 11, 13)
    # end_time = datetime(2018, 11, 13, 6, 30, 13)

    # begin_time = datetime(2018, 11, 19, 6, 11, 13)
    # end_time = datetime(2018, 11, 26, 6, 30, 13)

    begin_time = datetime(2018, 12, 26, 00, 00, 00)
    end_time = datetime(2019, 1, 1, 00, 00, 00)

    time_type = 'm'
    endswith = None
    end_pass_list = ['']

    result_file_name_list = find_target_file(begin_time, end_time, time_type, endswith)
    # result_file_name_list = ['market_top_300to800plus_True_20181228_1404_hold_5__16']
    print(result_file_name_list)
    for result_file_name in result_file_name_list:
        pass_result_list = ['market_top_300plus_True_20181204_0930_hold_5__11',
                            'market_top_300plus_True_20181205_0955_hold_5__7_long',
                            'market_top_300plus_True_20181206_2321_hold_20__7_long',
                            'market_top_300plus_True_20181206_2349_hold_20__11',

                            'market_top_300to800plus_True_20181203_1659_hold_20__13',
                            'market_top_300plus_industry_45_50_True_20181204_2039_hold_5__11',
                            'market_top_300plus_industry_20_25_30_35_True_20181207_0610_hold_20__11',
                            'market_top_300plus_industry_20_25_30_35_True_20181204_1639_hold_5__11',
                            'market_top_300plus_True_20181224_1822_hold_5__13',
                            'market_top_300plus_True_20181221_1454_hold_5__15_long',
                            'market_top_300to800plus_True_20181225_0352_hold_5__13',
                            'market_top_300to800plus_industry_10_15_True_20181202_0107_hold_20__7',
                            'market_top_300to800plus_industry_10_15_True_20181203_2013_hold_20__13',

                            'market_top_300plus_True_20181227_0336_hold_20__17',
                            ]
        if result_file_name in pass_result_list:
            pass
        else:
            fun_result = main(result_file_name, time_para_dict)

    # sum_pnl_df = get_corr_matrix()
