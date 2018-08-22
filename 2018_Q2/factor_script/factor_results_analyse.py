import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import product, permutations, combinations
import random
from open_lib.shared_tools import send_email
import open_lib.shared_tools.back_test as bt

from factor_script.script_load_data import load_sector_data, load_locked_data, load_pct, \
    load_part_factor, create_log_save_path, load_index_data, deal_mix_factor

from factor_script.script_filter_fun import pos_daily_fun, out_sample_perf, filter_all


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
    rolling_sharpe = pnl_df.rolling(int(roll_year * year_len)) \
        .apply(lambda x: np.sqrt(year_len) * x.mean() / x.std())
    cut_sharpe = rolling_sharpe.quantile(cut_point_list)
    if output:
        return cut_sharpe
    else:
        return rolling_sharpe.dropna(), cut_sharpe


def plot_send_result(pnl_df, sharpe_ratio, subject):
    figure_save_path = os.path.join(root_path, 'tmp_figure')
    plt.figure(figsize=[16, 8])
    plt.plot(pnl_df.index, pnl_df.cumsum(), label='sharpe_ratio='.format(sharpe_ratio))
    plt.savefig(os.path.join(figure_save_path, '{}.png'.format(subject)))
    text = ''
    to = ['whs@yingpei.com']
    filepath = [os.path.join(figure_save_path, 'multy_mix_factor.png')]
    send_email.send_email(text, to, filepath, subject)


def para_result(begin_date, end_date, sector_df, locked_df, index_df, hold_time, i,
                fun_name, name_1, name_2, name_3, sector_name, result, if_only_long=False):
    cost_1 = 0.001
    lag = 1
    xnms, xinx = sector_df.columns, sector_df.index
    r_con_in, r_con_out, r_ic, r_sp_u, r_sp_m, r_sp_d, r_pot_in, r_fit_ratio, r_leve_ratio, r_sharpe_out = result
    figure_save_path = '/mnt/mfs/dat_whs/result/tmp'
    # figure_save_path = os.path.join(root_path, 'tmp_figure')
    fun_set = [mul_fun, sub_fun, add_fun]
    mix_fun_set = create_fun_set_2(fun_set)
    fun = mix_fun_set[fun_name]

    factor_set = load_part_factor(sector_name, begin_date, end_date, xnms, xinx, [name_1, name_2, name_3])
    choose_1 = factor_set[name_1]
    choose_2 = factor_set[name_2]
    choose_3 = factor_set[name_3]

    return_data = load_pct(begin_date, end_date, xnms, xinx)
    mix_factor = fun(choose_1, choose_2, choose_3)
    pos_df_daily = deal_mix_factor(mix_factor, sector_df, locked_df, hold_time, lag, if_only_long)

    in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
    fit_ratio, leve_ratio, sharpe_q_out, pnl_df = filter_all(cut_date, pos_df_daily, return_data, index_df,
                                                             if_hedge=False, hedge_ratio=1, if_return_pnl=True,
                                                             if_only_long=False)

    hedge_df = 1 * index_df.mul(pos_df_daily.sum(axis=1), axis=0)
    pnl_df_h = -hedge_df.sub((pos_df_daily * return_data).sum(axis=1), axis=0)
    pnl_df_h = pnl_df_h[pnl_df_h.columns[0]]

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('{} factor figure'.format(i), fontsize=40)
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    asset = pnl_df.cumsum()
    asset_hedge = pnl_df_h.cumsum()

    pot = bt.AZ_Pot(pos_df_daily, asset.iloc[-1])
    sharpe = bt.AZ_Sharpe_y(pnl_df)

    pot_h = bt.AZ_Pot(pos_df_daily, asset_hedge.iloc[-1])
    sharpe_h = bt.AZ_Sharpe_y(pnl_df_h)

    if asset.values[-1] < 0:
        asset = -asset
        asset_hedge = -asset_hedge

    cost_matrix_1 = (pos_df_daily.diff().abs() * cost_1).sum(axis=1).cumsum()

    ax1.plot(pnl_df.index, asset, label='raw_line, pot={}, sharpe={}'.format(pot, sharpe))
    ax1.plot(pnl_df.index, asset - cost_matrix_1, label='cost_line, pot={}, sharpe={}'.format(pot, sharpe))
    ax1.plot(pnl_df.index, asset_hedge, label='helge_line, pot_h={}, sharpe_h={}'.format(pot_h, sharpe_h))

    ax1.set_title('{}, {}, {}, {}'
                  .format(fun_name, name_1, name_2, name_3))
    ax1.grid(1)
    ax1.legend()

    rolling_sharpe, cut_sharpe = AZ_Rolling_sharpe(pnl_df, roll_year=3, year_len=250)
    ax2.hist(rolling_sharpe.values, bins=200)
    ax2.set_title('{},{},{},{},{},{},{},{}'.format(round(r_ic, 4), r_sp_u, r_sp_m, r_sp_d, r_pot_in,
                                                   round(r_fit_ratio, 4), round(r_leve_ratio, 4), r_sharpe_out))
    ax2.grid(axis='y')

    result_filter = filter_pot_sharpe(cut_date, fun(choose_1, choose_2, choose_3), return_data, index_df, lag=1,
                                      hold_time=5, if_hedge=False, hedge_ratio=1)
    ax3.set_title(','.join([str(x) for x in result_filter]))
    ax4.set_title('pot_in={}, leve_ratio={}, sharpe_q_in_df_m={}, fit_ratio={}'
                  .format(pot_in, round(leve_ratio, 4), sharpe_q_in_df_m, round(fit_ratio, 4)))
    # plt.show()

    # ax3.plot(pos_df_daily.sum(axis=1), label=','.join([str(x) for x in result]))
    ax3.plot(bt.AZ_Rolling(pos_df_daily.sum(axis=1), 250).mean())
    ax3.legend()
    plt.savefig(os.path.join(figure_save_path, '|'.join([fun_name, name_1, name_2, name_3])) + '.png')

    # text = '|'.join([str(x) for x in [fun_name, name_1, name_2, name_3] + list(result)])
    text = '.'
    to = ['whs@yingpei.com']
    subject = '|'.join([fun_name, name_1, name_2, name_3])
    filepath = [os.path.join(figure_save_path, '|'.join([fun_name, name_1, name_2, name_3])) + '.png']
    send_email.send_email(text, to, filepath, subject)


def result_get_random(begin_date, end_date, sector_df, locked_df, index_df, file_name, n_random=20,
                      hold_time=20, if_only_long=False):
    result_load_path = os.path.join('/mnt/mfs/dat_whs/result/result', file_name)
    result_data = pd.read_table(result_load_path, sep='|', header=None)
    for i in random.sample(list(result_data.index), n_random):
        # for i in result_data.index:
        key, fun_name, name_1, name_2, name_3, filter_fun, sector_name, *result = result_data.loc[i]
        print(key, fun_name, name_1, name_2, name_3, *result)
        para_result(begin_date, end_date, sector_df, locked_df, index_df, hold_time, key,
                    fun_name, name_1, name_2, name_3, sector_name, result, if_only_long=if_only_long)


def analyse_result(file_list):
    for file_name in file_list:
        result_load_path = os.path.join(root_path, 'result/result/{}'.format(file_name))

        if os.path.getsize(result_load_path) != 0:
            result_data = pd.read_table(result_load_path, sep='|', header=None)
            result_data.columns = ['key', 'fun_name', 'name_1', 'name_2', 'name_3', 'filter_fun', 'sector_name',
                                   'cond_in', 'cond_out', 'ic', 'sp_u', 'sp_m', 'sp_d', 'pot', 'fit_ratio',
                                   'leve_ratio', 'sp_out']
            filter_fun = result_data.iloc[0, 5]
            survived = sum(result_data['cond_out'])
            pot_mean = np.nanmean(result_data['pot'].abs())
            pot_mean = round(pot_mean, 4)
            choosed = result_data.shape[0]
            survived_ratio = survived / choosed
        else:
            filter_fun = None
            survived = 0
            pot_mean = 0
            choosed = 0
            survived_ratio = 0
        print('{}, {}, survived_num = {}, choose_num = {}, survived_ratio = {}, pot_mean = {}'
              .format(file_name, filter_fun, survived, choosed, round(survived_ratio, 4), pot_mean))


def analyse_pot(result_load_path):
    result_data = pd.read_table(result_load_path, sep='|', header=None)
    pot_data = result_data[11].abs()
    figure_save_path = os.path.join(root_path, 'tmp_figure')
    plt.hist(pot_data.values, bins=200)
    file_name = os.path.split(result_load_path)[1]
    plt.savefig(os.path.join(figure_save_path, file_name.split('.')[1] + '.png'))

    text = file_name
    to = ['whs@malpha.info']
    subject = ''
    filepath = [os.path.join(figure_save_path, file_name.split('.')[1] + '.png')]
    send_email.send_email(text, to, filepath, subject)

    bt.AZ_Delete_file(figure_save_path)


def result_filter(all_use_factor):
    all_use_factor = pd.read_table(log_result, sep='|', header=None)
    all_use_factor.columns = ['key', 'fun_name', 'name1', 'name2', 'name3', 'filter_fun_name', 'sector_name',
                              'con_in', 'con_out', 'ic', 'sp_u', 'sp_m', 'sp_d', 'pot_in', 'fit_ratio', 'leve_ratio',
                              'sp_out']
    print(all_use_factor['con_out'].sum() / len(all_use_factor), all_use_factor.pot_in.mean())
    a = all_use_factor[all_use_factor.pot_in.abs() > 10]
    print(a['con_out'].sum() / len(a), len(a), a.pot_in.mean())
    b = a[a.ic.abs() > 0.04]
    print(b['con_out'].sum() / len(b), len(b), b.pot_in.mean())


if __name__ == '__main__':
    # # root_path = '/media/hdd1/dat_whs'
    # root_path = '/mnt/mfs/dat_whs'
    # index_name = '000300'
    # file_name = 'market_top_500_True_20180815_1023_hold_5_aadj_r.txt'
    # log_result = os.path.join(root_path, 'result/result/{}'.format(file_name))
    # sector_name = 'market_top_500'
    # hold_time = 5
    # lag = 2
    # if_only_long = False
    # begin_date = pd.to_datetime('20100101')
    # cut_date = pd.to_datetime('20160401')
    # end_date = pd.to_datetime('20180401')
    #
    # sector_df = load_sector_data(begin_date, end_date, sector_name)
    #
    # xnms = sector_df.columns
    # xinx = sector_df.index
    #
    # # suspend or limit up_dn
    # suspendday_df, limit_buy_df, limit_sell_df = load_locked_data(xnms, xinx)
    #
    # # return
    # return_choose = pd.read_table('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv', sep='|', index_col=0).astype(float)
    # return_choose.index = pd.to_datetime(return_choose.index)
    # return_choose = return_choose.reindex(columns=xnms, index=xinx, fill_value=0)
    #
    # # index data
    # index_df = load_index_data(xinx, index_name)

    # all_use_factor = pd.read_table(log_result, sep='|', header=None)
    # all_use_factor.columns = ['key', 'fun_name', 'name1', 'name2', 'name3', 'filter_fun_name', 'sector_name',
    #                           'con_in', 'con_out', 'ic', 'sp_u', 'sp_m', 'sp_d', 'pot_in', 'fit_ratio', 'leve_ratio',
    #                           'sp_out']
    # a = all_use_factor[all_use_factor.pot_in.abs() > 20]
    # print(a['con_out'].sum() / len(a))
    # b = a[a.ic.abs() > 0.01]
    # print(b['con_out'].sum() / len(b))

    root_path = '/mnt/mfs/dat_whs'
    begin_date = pd.to_datetime('20100101')
    cut_date = pd.to_datetime('20160401')
    end_date = pd.to_datetime('20180401')
    #
    # file_list = ['market_top_500_True_20180815_1510_hold_5_aadj_r.txt',
    #              'market_top_500_True_20180815_1823_hold_5_aadj_r_long.txt',
    #              'market_top_500_True_20180815_1919_hold_5_aadj_r.txt',
    #              'market_top_500_True_20180815_2233_hold_5_aadj_r_long.txt',
    #              'market_top_500_True_20180816_0200_hold_10_aadj_r.txt',
    #              'market_top_500_True_20180816_0512_hold_10_aadj_r_long.txt',
    #              'market_top_500_True_20180816_0834_hold_20_aadj_r.txt']
    # analyse_result(file_list)

    # fun_name, name_1, name_2, name_3 = 'sub_add_fun|RQMCL_p345d_continue_ud|RQMCL_p5d_roll_mean_row_ext_2|' \
    #                                    'intra_most_volume_vwap_compare'.split('|')
    # para_result(begin_date, end_date, sector_df, locked_df, index_df, 5, 0,
    #             fun_name, name_1, name_2, name_3, sector_name, '', if_only_long=False)

    # name_1 = 'RZCHE_p345d_continue_ud'
    # name_2 = 'rr5_ext_1'
    # name_3 = 'rr5_ext_2'
    # result = None
    # key = None
    # index_name = '000300'
    # sector_name = 'market_top_500'
    # sector_df = load_sector_data(begin_date, end_date, 'market_top_500')
    # locked_df = load_locked_data(begin_date, end_date, sector_df.columns)
    # index_df = load_index_data(begin_date, end_date, index_name)
    # result_get_random('000300', begin_date, end_date, result_load_path, n_random=20, hold_time=10)
    # hold_time = 5
    # i = 0
    # fun_name = 'add_mul_fun'
    # name_1 = 'rc5_ext_1'
    # name_2 = 'rr20_ext_2'
    # name_3 = 'vol_p60d'
    # result = ''
    # para_result(begin_date, end_date, sector_df, locked_df, index_df, hold_time, i,
    #             fun_name, name_1, name_2, name_3, sector_name, result, if_only_long=True)

    # analyse_result(['20180702_0915.txt'])
    # result_factor_sum()

    # file_list = ['market_top_500_False_20180706_1201.txt', 'market_top_500_True_20180706_1142.txt',
    #              'market_top_100_True_20180706_1515.txt', 'market_top_100_False_20180706_1514.txt']

    # file_list = ['market_top_100_True_20180713_1841.txt', 'market_top_500_True_20180713_1840.txt']
    # analyse_result(file_list)

    # sector_df = load_sector_data(begin_date, end_date)
    # locked_df = load_locked_data(begin_date, end_date, sector_df.columns)

# data = pd.read_csv('/mnt/mfs/dat_whs/result/result/market_top_2000_True_20180821_1605_hold_20_aadj_r.txt',
#                    sep='|', header=None)
# data = pd.read_csv('/mnt/mfs/dat_whs/result/result/market_top_2000_True_20180822_0859_hold_20_aadj_r.txt',
#                    sep='|', header=None)

data = pd.read_csv('/mnt/mfs/dat_whs/result/result/market_top_2000_True_20180822_0900_hold_20_aadj_r.txt',
                   sep='|', header=None)

data.columns = ['time_para', 'key', 'fun_name', 'name1', 'name2', 'name3', 'filter_fun_name', 'sector_name',
                'con_in', 'con_out', 'ic', 'sp_u', 'sp_m', 'sp_d', 'pot_in', 'fit_ratio', 'leve_ratio',
                'sp_out']
data_1 = data[data['time_para'] == 'time_para_1']
data_2 = data[data['time_para'] == 'time_para_2']
data_3 = data[data['time_para'] == 'time_para_3']

factor_list = sorted(list(set(data[['name1', 'name2', 'name3']].values.ravel())))
print(len(factor_list))
# POT
a_1 = data_1[data_1['pot_in'].abs() > 40]
print(a_1['con_out'].sum()/len(a_1), len(a_1))

a_2 = data_2[data_2['pot_in'].abs() > 40]
print(a_2['con_out'].sum()/len(a_2), len(a_2))

a_3 = data_3[data_3['pot_in'].abs() > 40]
print(a_3['con_out'].sum()/len(a_3), len(a_3))

# IC
a_1 = data_1[(data_1['ic'].abs() > 0.01) & (data_1['pot_in'].abs() > 40)]
print(a_1['con_out'].sum()/len(a_1), len(a_1))

a_2 = data_2[(data_2['ic'].abs() > 0.01) & (data_2['pot_in'].abs() > 40)]
print(a_2['con_out'].sum()/len(a_2), len(a_2))

a_3 = data_3[(data_3['ic'].abs() > 0.01) & (data_3['pot_in'].abs() > 40)]
print(a_3['con_out'].sum()/len(a_3), len(a_3))


# ['CCI_p120d_limit_12',
#  'MACD_20_100',
#  'R_COMPANYCODE_First_row_extre_0.3',
#  'R_EPS_s_YOY_First_row_extre_0.3',
#  'R_FinExp_sales_s_First_row_extre_0.3',
#  'R_GSCF_sales_s_First_row_extre_0.3',
#  'R_MgtExp_sales_s_First_row_extre_0.3',
#  'R_NetCashflowPS_s_First_row_extre_0.3',
#  'R_NetROA_s_First_row_extre_0.3',
#  'R_OperProfit_s_POP_First_row_extre_0.3',
#  'R_ParentProfit_s_POP_First_row_extre_0.3',
#  'R_Tax_TotProfit_s_First_row_extre_0.3',
#  'TVOL_p120d_col_extre_0.2',
#  'TVOL_p20d_col_extre_0.2',
#  'bias_turn_p120d',
#  'evol_p20d',
#  'turn_p120d_0.2',
#  'turn_p20d_0.2',
#  'vol_p20d',
#  'vol_p60d']
#
# ['CCI_p120d_limit_12',
#  'R_CFO_TotRev_s_First_row_extre_0.3',
#  'R_GSCF_sales_s_First_row_extre_0.3',
#  'R_NOTICEDATE_First_row_extre_0.3',
#  'R_NetAssets_s_POP_First_row_extre_0.3',
#  'R_NetCashflowPS_s_First_row_extre_0.3',
#  'R_OPCF_NetInc_s_First_row_extre_0.3',
#  'R_OPCF_sales_s_First_row_extre_0.3',
#  'R_Tax_TotProfit_s_First_row_extre_0.3',
#  'R_TotRev_s_POP_First_row_extre_0.3',
#  'R_TotRev_s_YOY_First_row_extre_0.3',
#  'TVOL_p120d_col_extre_0.2',
#  'TVOL_p20d_col_extre_0.2',
#  'bias_turn_p20d',
#  'log_price_0.2',
#  'price_p120d_hl',
#  'turn_p20d_0.2',
#  'vol_count_down_p60d',
#  'vol_p20d',
#  'vol_p60d']
