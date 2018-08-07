import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from open_lib.shared_tools import send_email
import open_lib.shared_tools.back_test as bt
import random
import datetime as datetime
from factor_script.script_load_data import load_sector_data, load_locked_data, load_pct, \
    load_part_factor, create_log_save_path, load_index_data, deal_mix_factor


def pos_daily_fun(df, n=5):
    return df.rolling(window=n, min_periods=1).sum()


def AZ_Pot(pos_df, asset_last):
    """
    计算 pnl/turover*10000的值,衡量cost的影响
    :param pos_df: 仓位信息
    :param asset_last: 最后一天的收益
    :return:
    """
    trade_times = pos_df.diff().abs().sum().sum()
    if trade_times == 0:
        return 0
    else:
        pot = asset_last / trade_times * 10000
        return pot


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


def get_use_company(data2):
    # all_company_list = list(data2['COMPANYCODE'].values)
    # use_company_list = []
    # for company in sorted(set(data2['COMPANYCODE'].values)):
    #     print(company)
    #     if all_company_list.count(company) > 50:
    #         use_company_list += [company]
    # pd.to_pickle(use_company_list, '/mnt/mfs/dat_whs/data/adj_data/table14/use_company_list.pkl')
    use_company_list = pd.read_pickle('/mnt/mfs/dat_whs/data/adj_data/table14/use_company_list.pkl')
    return use_company_list


def split_fun(use_company_list):
    company_set = {}
    address_set = {}

    for x in use_company_list:

        if len(x.split('公司')) == 1:
            pass
        else:
            company, address = x.split('公司', 1)
            if company not in company_set.keys():
                company_set[company] = [x]
            else:
                company_set[company] += [x]
            if address[:2] not in address_set.keys():
                address_set[address[:2]] = [x]
            else:
                address_set[address[:2]] += [x]
    jgzy_list = [x for x in use_company_list if '机构' in x]
    jydy_list = [x for x in use_company_list if '交易单元' in x]
    hgt_list = [x for x in use_company_list if '股通专用' in x]
    return company_set, address_set, jgzy_list, jydy_list, hgt_list


def plot_save_fun(suject, tmp_df, return_df, locked_df, hold_time_list, index_df, now_time, if_hedge=False):
    fig_save_path = '/mnt/mfs/dat_whs/tmp_figure/'
    plt.figure(figsize=[16, 8])
    a = 0
    for hold_time in hold_time_list:
        # daily_pos = pos_daily_fun(tmp_df, hold_time)
        daily_pos = deal_mix_factor(tmp_df, None, locked_df, hold_time, 1, if_only_long=False)

        if if_hedge:
            hedge_df = index_df.mul(daily_pos.shift(1).sum(axis=1), axis=0)
            pnl_df = -hedge_df.sub((daily_pos.shift(1) * return_df).sum(axis=1), axis=0)['000300']
        else:
            pnl_df = (daily_pos.shift(1) * return_df).sum(axis=1)
        asset_df = pnl_df.cumsum()
        sp_y = bt.AZ_Sharpe_y(pnl_df)
        pot = AZ_Pot(daily_pos, asset_df.values[-1])
        if sp_y > 1:
            a = 1
            print(1)
        plt.plot(pnl_df.index, asset_df, label='hold_time={}, pot={}, sharpe={}'.format(hold_time, pot, sp_y))
    plt.legend()
    plt.title(suject + '_' + now_time)
    plt.savefig(fig_save_path + 'company.png')
    plt.close()
    if a == 1:
        text = ''
        to = ['whs@yingpei.com']
        filepath = [fig_save_path + 'company.png']
        subject = suject
        send_email.send_email(text, to, filepath, subject)


def main_fun(key, company_list, data1, data2, xnms, xinx, cond_d, cond_u, vol30, return_df, locked_df,
             hold_time_list, index_df):
    all_tmp_df = pd.DataFrame()
    for company in company_list:
        print(company)
        part_data2 = data2[data2['COMPANYNAME'] == company][['RID', 'TRSDIR']]

        part_data1 = data1.loc[part_data2['RID'].values][['SECURITYCODE', 'TRADEDATE']]
        part_data1['TRSDIR'] = part_data2['TRSDIR'].values

        tmp_df = part_data1.groupby(['TRADEDATE', 'SECURITYCODE'])['TRSDIR'].apply(lambda x: x.iloc[-1]).unstack()
        tmp_df = (tmp_df == '0').astype(int)
        delete_index = sorted(set(tmp_df.index) - set(xinx))
        tmp_df = tmp_df.reindex(columns=xnms, index=xinx, fill_value=0)
        if len(delete_index) != 0:
            delete_tmp_df = tmp_df.reindex(columns=xnms, index=delete_index, fill_value=0)
            delete_tmp_df.index = [(xinx[xinx < x])[-1] for x in delete_tmp_df.index]
            tmp_df = tmp_df.add(delete_tmp_df, fill_value=0)
            tmp_df[tmp_df > 1] = 1

        all_tmp_df = all_tmp_df.add(tmp_df, fill_value=0)

    now_time = datetime.datetime.now().strftime('%H%M%S')
    all_tmp_df.to_pickle('/mnt/mfs/dat_whs/data/new_factor_data/market_top_500/pub_info/pub_info_{}.pkl'.format(key))
    all_tmp_df_d = cond_d * all_tmp_df / vol30
    all_tmp_df_u = cond_u * all_tmp_df / vol30
    all_tmp_df_d = all_tmp_df_d.replace(np.nan, 0)
    all_tmp_df_u = all_tmp_df_u.replace(np.nan, 0)
    plot_save_fun(key + 'test_d', all_tmp_df_d, return_df, locked_df, hold_time_list, index_df, now_time)
    plot_save_fun(key + 'test_d_h', all_tmp_df_d, return_df, locked_df, hold_time_list, index_df, now_time,
                  if_hedge=True)
    plot_save_fun(key + 'test_u', all_tmp_df_u, return_df, locked_df, hold_time_list, index_df, now_time)
    plot_save_fun(key + 'test_u_h', all_tmp_df_u, return_df, locked_df, hold_time_list, index_df, now_time,
                  if_hedge=True)


if __name__ == '__main__':
    begin_date = pd.to_datetime('20100101')
    end_date = pd.to_datetime('20170101')
    data1 = pd.read_pickle(r'/mnt/mfs/DAT_EQT/EM_Tab14/raw_data/TIT_S_PUB_STOCK.pkl')
    data1['SECURITYCODE'] = bt.AZ_add_stock_suffix(data1['SECURITYCODE'])
    data1.index = data1['OLDID']

    data2 = pd.read_pickle(r'/mnt/mfs/DAT_EQT/EM_Tab14/raw_data/TIT_S_PUB_SALES.pkl')
    data2.sort_values(by='TRADEDATE', inplace=True)
    data2 = data2[(data2['TRADEDATE'] > begin_date) & (data2['TRADEDATE'] < end_date)]

    return_df = pd.read_pickle('/mnt/mfs/DAT_EQT/EM_Tab14/DERIVED/aadj_r.pkl')
    return_df = AZ_Cut_window(return_df, begin_date, end_date, column=None)

    stk_CR = return_df.replace(np.nan, 0).cumsum()
    ma5 = bt.AZ_Rolling_mean(stk_CR, 3)
    ma50 = bt.AZ_Rolling_mean(stk_CR, 12)

    vol30 = bt.AZ_Rolling(return_df, 30).std() * (250 ** 0.5)
    vol30[vol30 < 0.08] = 0.08

    cond_d = (ma5 < ma50).astype(int)
    cond_u = (ma5 > ma50).astype(int)

    hold_time_list = [1, 2, 5]
    xnms = return_df.columns
    xinx = return_df.index

    index_df = load_index_data(begin_date, end_date, xinx, '000300').shift(1)
    locked_df = load_locked_data(begin_date, end_date, xnms, xinx)

    use_company_list = get_use_company(data2)
    aa = random.sample(use_company_list, 200)
    # 数据分类
    company_set, address_set, jgzy_list, jydy_list, hgt_list = split_fun(use_company_list)
    for key in address_set.keys():
        print(key)
        aa = address_set[key]
        main_fun(key, aa, data1, data2, xnms, xinx, cond_d, cond_u, vol30, return_df, locked_df,
                 hold_time_list, index_df)

    for key in company_set.keys():
        print(key)
        aa = company_set[key]
        main_fun(key, aa, data1, data2, xnms, xinx, cond_d, cond_u, vol30, return_df, locked_df,
                 hold_time_list, index_df)
    print('机构专用')
    main_fun('机构专用', jgzy_list, data1, data2, xnms, xinx, cond_d, cond_u, vol30, return_df, locked_df,
             hold_time_list, index_df)
    print('交易单元')
    main_fun('交易单元', jydy_list, data1, data2, xnms, xinx, cond_d, cond_u, vol30, return_df, locked_df,
             hold_time_list, index_df)
    print('沪股通')
    main_fun('沪股通', hgt_list, data1, data2, xnms, xinx, cond_d, cond_u, vol30, return_df, locked_df,
             hold_time_list, index_df)
