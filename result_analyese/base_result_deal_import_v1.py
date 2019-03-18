from work_whs.loc_lib.pre_load import *


def plot_send_result(pnl_df, sharpe_ratio, subject):
    figure_save_path = os.path.join('/mnt/mfs/dat_whs', 'tmp_figure')
    plt.figure(figsize=[16, 8])
    plt.plot(pnl_df.index, pnl_df.cumsum(), label='sharpe_ratio={}'.format(sharpe_ratio))
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(figure_save_path, '{}.png'.format(subject)))
    text = ''
    to = ['whs@yingpei.com']
    filepath = [os.path.join(figure_save_path, '{}.png'.format(subject))]
    send_email.send_email(text, to, filepath, subject)


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
            # cond_2 = (len(a_1) > 20) and (len(a_2) > 20) and (len(a_3) > 20)

            cond_3_1 = sum(sr_list_out > 0.55) >= 1
            cond_3_2 = sum(sr_list_out > 0.3) >= 2
            cond_3_3 = sum(sr_list_out > 0.1) >= 3

            cond_3 = cond_3_1 and cond_3_2 and cond_3_3
            cond_4 = (len(a_4) > 20) and (len(a_5) > 20) and (len(a_6) > 20)
            print(cond_1, cond_3, cond_4)
            if cond_1 and cond_3 and cond_4:
                return para_adj_set
    return None


def load_result_data(result_file_name):
    data = pd.read_csv('/mnt/mfs/dat_whs/result/result/{}.txt'.format(result_file_name),
                       sep='|', header=None, error_bad_lines=False)

    data.columns = ['time_para', 'key', 'fun_name', 'name1', 'name2', 'name3', 'filter_fun_name', 'sector_name',
                    'con_in', 'con_out_1', 'con_out_2', 'con_out_3', 'con_out_4', 'ic', 'sp_u', 'sp_m', 'sp_d',
                    'pot_in', 'fit_ratio', 'leve_ratio', 'sp_in', 'sp_out_1', 'sp_out_2', 'sp_out_3', 'sp_out_4']

    return data


def bkt_fun(pnl_save_path, a_n, i):
    x, key, fun_name, name1, name2, name3, filter_fun_name, sector_name, \
    con_in, con_out_1, con_out_2, con_out_3, con_out_4, ic, \
    sp_u, sp_m, sp_d, pot_in, fit_ratio, leve_ratio, \
    sp_in, sp_out_1, sp_out_2, sp_out_3, sp_out_4 = a_n.loc[i]

    mix_factor, con_in_c, con_out_c, ic_c, sp_u_c, sp_m_c, sp_d_c, pot_in_c, fit_ratio_c, leve_ratio_c, \
    sp_in_c, sp_out_c, pnl_df_c = main_model.single_test(fun_name, name1, name2, name3)
    plot_send_result(pnl_df_c, bt.AZ_Sharpe_y(pnl_df_c), '{}, key={}'.format(i, key))

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
                 (data_n['fit_ratio'].abs() > fit_ratio)]

    sum_factor_df = pd.DataFrame()
    pnl_save_path = '/mnt/mfs/dat_whs/data/mix_factor_pnl/' + result_file_name
    bt.AZ_Path_create(pnl_save_path)

    result_list = []
    pool = Pool(20)
    for i in a_n.index:
        # bkt_fun(pnl_save_path, a_n, i)
        result_list.append(pool.apply_async(bkt_fun, args=(pnl_save_path, a_n, i,)))
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
    plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df), 'mix_factor')
    return sum_pos_df, pnl_df


def resull_analyese(result_file_name, ):
    data = load_result_data(result_file_name)
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
                              ('turn_p30d_0.24' in set(x))
                              # ('RSI_140_30' in set(x)) or
                              # ('CMO_200_0' in set(x)) or
                              # ('CMO_40_0' in set(x))
                              # ('ATR_40_0.2' in set(x))
                              # ('ADX_200_40_20' in set(x))
                              # ('ATR_140_0.2' in set(x))
                              ), axis=1)
    data = data[filter_cond]
    # #############################################################################
    # 结果分析
    survive_result = survive_ratio_test(data, para_adj_set_list)
    if survive_result is None:
        print(f'{result_file_name} not satisfaction!!!!!!!!')
    else:
        pass
