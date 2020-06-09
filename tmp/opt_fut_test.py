from py_vollib.black.greeks.analytical import delta, gamma, rho, theta, vega
from py_vollib.black_scholes.implied_volatility import implied_volatility

import sys

sys.path.append('/mnt/mfs')
from work_dmgr_fut.loc_lib.pre_load import *

time_list = ['09:31', '09:36',
             '09:41', '09:46',
             '09:51', '09:56',
             '10:01', '10:06',
             '10:11', '10:16',
             '10:21', '10:26',
             '10:31', '10:36',
             '10:41', '10:46',
             '10:51', '10:56',
             '11:01', '11:06',
             '11:11', '11:16',
             '11:21', '11:26',
             '13:01', '13:06',
             '13:11', '13:16',
             '13:21', '13:26',
             '13:31', '13:36',
             '13:41', '13:46',
             '13:51', '13:56',
             '14:01', '14:06',
             '14:11', '14:16',
             '14:21', '14:26',
             '14:31', '14:36',
             '14:41', '14:46',
             '14:51', '14:56']


def id_to_price(opt_id, now_date, now_time):
    print(opt_id)
    a = pd.read_csv(f'/media/hdd1/DAT_OPT/Min/{opt_id}.csv', sep='|', index_col=0)
    a.index = pd.to_datetime(a.index)
    return a.loc[pd.to_datetime(f'{now_date} {now_time}')]['Close'] * 0.0001


daily_data = pd.read_csv('/mnt/mfs/DAT_OPT/option_close.csv', sep='|', index_col=0)
daily_data.index = pd.to_datetime(daily_data.index)


def judge_daily_data(id_list, now_date):
    return daily_data[id_list].loc[pd.to_datetime(now_date)].notna().values


map_df = pd.read_csv('/mnt/mfs/DAT_OPT/mapping_table.csv', sep='|')
map_df.index = pd.to_datetime(map_df['exercise_date'])
map_df['exe_price'] = map_df['option_code'].str.slice(-4).astype(int) * 0.001
map_df['type'] = map_df['option_code'].str.slice(6, 7)
map_df['d_type'] = map_df['option_code'].str.slice(11, 12)


def get_today_opt(now_date, spot_price):
    exe_date_array = np.array(sorted(set(map_df.index)))
    exe_date_list = exe_date_array[exe_date_array > pd.to_datetime(now_date)][:2]
    map_df_date = map_df.loc[exe_date_list]
    all_exe_price_list = np.array(sorted(set(map_df_date['exe_price'])))
    # target_exe_price_list = [all_exe_price_list[all_exe_price_list < spot_price][-1],
    #                          all_exe_price_list[all_exe_price_list > spot_price][0], ]
    target_exe_price_list = [all_exe_price_list[all_exe_price_list > spot_price][0]]
    tmp_fun = lambda x: True if x in target_exe_price_list else False
    map_df_date_price = map_df_date[[tmp_fun(x) for x in map_df_date['exe_price']]]
    return map_df_date_price[(map_df_date_price['type'] == 'C') & (map_df_date_price['d_type'] == 'M')]


def deal_fun(now_date='20191115', now_time='10:30'):
    print('_______________________________')
    print(now_date)
    eqt_min_path = '/mnt/mfs/DAT_EQT/intraday/eqt_1mbar'

    spot_id = 'SH510050'
    # now_time = '10:30'
    spot_data = pd.read_csv(f'{eqt_min_path}/{now_date[:4]}/{now_date[:6]}/{now_date}/Close.csv', index_col=0)
    if spot_id not in spot_data.columns:
        return None
    spot_price = spot_data[spot_id].round(6).loc[now_time]

    exe_date_array = np.array(sorted(set(map_df.index)))
    exe_date_list = exe_date_array[exe_date_array > pd.to_datetime(now_date)][:2]

    result_dict = pd.Series()
    result_dict['date'] = now_date
    result_dict['now_time'] = now_time
    result_dict['spot_px'] = spot_price
    for i, exe_date in enumerate(exe_date_list):
        result_dict[f'opt_mat_{i+1}'] = exe_date
        time_to_maturity = (exe_date - pd.to_datetime(now_date)).days / 365
        result_dict[f'opt_ttm_{i+1}'] = (exe_date - pd.to_datetime(now_date)).days
        map_df_date = map_df.loc[exe_date]
        all_exe_price_list = np.array(sorted(set(map_df_date['exe_price'])))
        target_exe_price_list = [all_exe_price_list[all_exe_price_list < spot_price][-1],
                                 all_exe_price_list[all_exe_price_list > spot_price][0], ]
        for ii, target_exe_price in enumerate(target_exe_price_list):
            # print(exe_date, target_exe_price)
            result_dict[f'opt_k_{i+1}_{ii+1}'] = target_exe_price
            map_df_date_price = map_df_date[map_df_date['exe_price'] == target_exe_price]
            # 剔除当天没有close的id
            map_df_date_price = map_df_date_price[judge_daily_data(map_df_date_price['option'].values, now_date)]
            if len(map_df_date_price) == 0:
                print('len map_df_date_price=0')
                result_dict[f'call_px_{i+1}_{ii+1}'] = np.nan
                result_dict[f'put_px_{i+1}_{ii+1}'] = np.nan

            elif len(map_df_date_price) == 1:
                print('len map_df_date_price=1')
                result_dict[f'call_px_{i+1}_{ii+1}'] = np.nan
                result_dict[f'put_px_{i+1}_{ii+1}'] = np.nan

            else:
                if len(map_df_date_price[map_df_date_price['type'] == 'C']['option']) > 1:
                    call_id = \
                        map_df_date_price[(map_df_date_price['type'] == 'C') & (map_df_date_price['d_type'] == 'M')][
                            'option'].iloc[0]
                else:
                    call_id = map_df_date_price[map_df_date_price['type'] == 'C']['option'].iloc[0]

                if len(map_df_date_price[map_df_date_price['type'] == 'P']['option']) > 1:
                    put_id = \
                        map_df_date_price[(map_df_date_price['type'] == 'P') & (map_df_date_price['d_type'] == 'M')][
                            'option'].iloc[0]
                else:
                    put_id = map_df_date_price[map_df_date_price['type'] == 'P']['option'].iloc[0]

                call_price = id_to_price(call_id, now_date, now_time)
                put_price = id_to_price(put_id, now_date, now_time)

                result_dict[f'call_px_{i+1}_{ii+1}'] = call_price
                result_dict[f'put_px_{i+1}_{ii+1}'] = put_price

            # rate = np.log(target_exe_price /
            #               (spot_price - (call_price - put_price))) / time_to_maturity
            # rate_list.append(rate)

            rate = np.log(result_dict[f'opt_k_{i+1}_{ii+1}'] /
                          (spot_price - (result_dict[f'call_px_{i+1}_{ii+1}'] - result_dict[f'put_px_{i+1}_{ii+1}']))) \
                   / time_to_maturity
            result_dict[f'rate_{i+1}_{ii+1}'] = round(rate, 6)

        rate_mean = (result_dict[f'rate_{i+1}_1'] + result_dict[f'rate_{i+1}_2']) / 2
        result_dict[f'rate_mean_{i+1}'] = rate_mean
    # return rate_mean_dict
    if result_dict[f'opt_ttm_1'] <= 5:
        result_dict['rate_weight'] = result_dict['rate_mean_2']

    elif 5 < result_dict[f'opt_ttm_1'] <= 29:
        a1 = (exe_date_list[0] - pd.to_datetime(now_date)).days
        a2 = (exe_date_list[1] - pd.to_datetime(now_date)).days
        x = (30 - a2) / (a1 - a2)
        # print(x)
        result_dict['rate_weight'] = x * result_dict[f'rate_mean_1'] + (1 - x) * result_dict[f'rate_mean_2']

    elif result_dict[f'opt_ttm_1'] > 30:
        result_dict['rate_weight'] = result_dict['rate_mean_1']

    else:
        result_dict['rate_weight'] = None
        # if 'rate_mean_1' and 'rate_mean_2':  # result_dict[f'r_mean_{i+1}']:
    #     a1 = (exe_date_list[0] - pd.to_datetime(now_date)).days
    #     a2 = (exe_date_list[1] - pd.to_datetime(now_date)).days
    #     x = (30 - a2) / (a1 - a2)
    #     # print(x)
    #     result_dict['rate_weight'] = x * result_dict[f'rate_mean_1'] + (1 - x) * result_dict[f'rate_mean_2']
    # elif 'rate_mean_1' in result_dict.keys():
    #     result_dict['rate_weight'] = result_dict['rate_mean_1']
    # elif 'rate_mean_2' in result_dict.keys():
    #     result_dict['rate_weight'] = result_dict['rate_mean_2']
    # else:
    #     result_dict['rate_weight'] = None
    return result_dict


def main_fun(now_date, now_time, percent):
    vol_sr = pd.Series()
    try:
        part_result_dict = deal_fun(now_date=now_date, now_time=now_time)
        F = part_result_dict['spot_px']
        vol_sr['S'] = F
        r = part_result_dict['rate_weight']
        vol_sr['r'] = r
        vol_sr['percent'] = percent
        tran_dict = {1: 'first', 2: 'second'}
        for i in range(1, 3):
            t = part_result_dict[f'opt_ttm_{i}'] / 365
            for ii in range(1, 3):
                K = part_result_dict[f'opt_k_{i}_{ii}']
                vol_sr[f'{tran_dict[i]}_K_{ii}'] = K
                if F >= K:
                    flag = 'call'
                    price = part_result_dict[f'{flag}_px_{i}_{ii}']
                    vol_sr[f'{tran_dict[i]}_{flag}_px_{ii}'] = price
                    sigma = implied_volatility(price, F, K, t, r, flag[0])
                    vol_sr[f'{tran_dict[i]}_{flag}_vol_{ii}'] = sigma
                    vol_sr[f'{tran_dict[i]}_vol_{ii}'] = vol_sr[f'{tran_dict[i]}_call_vol_{ii}']
                else:
                    flag = 'put'
                    price = part_result_dict[f'{flag}_px_{i}_{ii}']
                    vol_sr[f'{tran_dict[i]}_{flag}_px_{ii}'] = price
                    sigma = implied_volatility(price, F, K, t, r, flag[0])
                    vol_sr[f'{tran_dict[i]}_{flag}_vol_{ii}'] = sigma
                    vol_sr[f'{tran_dict[i]}_vol_{ii}'] = vol_sr[f'{tran_dict[i]}_put_vol_{ii}']

                # for flag in ['call', 'put']:
                #     price = part_result_dict[f'{flag}_px_{i}_{ii}']
                #     vol_sr[f'{tran_dict[i]}_{flag}_px_{ii}'] = price
                #     sigma = implied_volatility(price, F, K, t, r, flag[0])
                #     vol_sr[f'{tran_dict[i]}_{flag}_vol_{ii}'] = sigma
                #
                # vol_sr[f'{tran_dict[i]}_vol_{ii}'] = (vol_sr[f'{tran_dict[i]}_call_vol_{ii}'] +
                #                                       vol_sr[f'{tran_dict[i]}_put_vol_{ii}']) / 2
                # 计算希腊字母
                # for flag in ['call', 'put']:
                #     price = result_dict[now_date][f'{flag}_px_{i}_{ii}']
                #     vol_sr[f'{tran_dict[i]}_{flag}_px_{ii}'] = price
                #     sigma = implied_volatility(price, F, K, t, r, flag[0])
                #     vol_sr[f'{tran_dict[i]}_{flag}_vol_{ii}'] = sigma
                #     vol_sr[f'{tran_dict[i]}_{flag}_delta_{ii}'] = delta(flag[0], F, K, t, r, sigma)
                #     vol_sr[f'{tran_dict[i]}_{flag}_gamma_{ii}'] = gamma(flag[0], F, K, t, r, sigma)
                #     vol_sr[f'{tran_dict[i]}_{flag}_rho_{ii}'] = rho(flag[0], F, K, t, r, sigma)
                #     vol_sr[f'{tran_dict[i]}_{flag}_theta_{ii}'] = theta(flag[0], F, K, t, r, sigma)
                #     vol_sr[f'{tran_dict[i]}_{flag}_vega_{ii}'] = vega(flag[0], F, K, t, r, sigma)

            vol_sr[f'{tran_dict[i]}_K_weight'] = (F - vol_sr[f'{tran_dict[i]}_K_2']) / \
                                                 (vol_sr[f'{tran_dict[i]}_K_1'] - vol_sr[f'{tran_dict[i]}_K_2'])
            vol_sr[f'{tran_dict[i]}_vol'] = vol_sr[f'{tran_dict[i]}_K_weight'] * \
                                            vol_sr[f'{tran_dict[i]}_vol_1'] + \
                                            (1 - vol_sr[f'{tran_dict[i]}_K_weight']) * \
                                            vol_sr[f'{tran_dict[i]}_vol_2']

        vol_sr[f'opt_ttm_1'] = part_result_dict['opt_ttm_1']
        vol_sr[f'opt_ttm_2'] = part_result_dict['opt_ttm_2']
        if vol_sr[f'opt_ttm_1'] <= 5:
            vol_sr[f'time_weight'] = 0
            vol_sr[f'target_vol'] = vol_sr[f'second_vol']
        elif 5 < vol_sr[f'opt_ttm_1'] <= 29:
            x = (30 - part_result_dict['opt_ttm_2']) / \
                (part_result_dict['opt_ttm_1'] - part_result_dict['opt_ttm_2'])
            vol_sr[f'opt_ttm_1'] = part_result_dict['opt_ttm_1']
            vol_sr[f'opt_ttm_2'] = part_result_dict['opt_ttm_2']
            vol_sr[f'time_weight'] = x
            vol_sr[f'target_vol'] = x * vol_sr[f'first_vol'] + (1 - x) * vol_sr[f'second_vol']
        elif 29 < vol_sr[f'opt_ttm_1']:
            vol_sr[f'time_weight'] = 1
            vol_sr[f'target_vol'] = vol_sr[f'first_vol']
        else:
            vol_sr[f'time_weight'] = None
            vol_sr[f'target_vol'] = None

    except Exception as error:
        print(error)
    vol_sr.name = pd.to_datetime(f'{now_date} {now_time}')
    return vol_sr


eqt_min_path = '/mnt/mfs/DAT_EQT/intraday/eqt_1mbar'
spot_id = 'SH510050'


def min_price(part_date_list):
    min_list = []
    for now_date in part_date_list:
        spot_data = pd.read_csv(f'{eqt_min_path}/{now_date[:4]}/{now_date[:6]}/{now_date}/Low.csv', index_col=0)
        min_list.append(spot_data[spot_id].min())
    return min(min_list)


def simple_st(now_date, part_date_list):
    past_3_day_low = min_price(part_date_list)

    now_time = '10:30'
    deal_time_1 = '10:31'
    deal_time_2 = '15:00'

    spot_data = pd.read_csv(f'{eqt_min_path}/{now_date[:4]}/{now_date[:6]}/{now_date}/Close.csv', index_col=0)
    tmp_df = spot_data[spot_id].loc[:now_time]
    spot_price = spot_data[spot_id].loc[deal_time_1]
    # print(past_3_day_low, tmp_df)
    a = tmp_df[tmp_df < past_3_day_low]
    if len(a) > 0:
        opt_info_df = get_today_opt(now_date, spot_price)
        day_len = (opt_info_df.index - pd.to_datetime(now_date)).days
        if day_len[0]<5:
            opt_id = opt_info_df['option'].iloc[1]
        else:
            opt_id = opt_info_df['option'].iloc[0]
        buy_price = id_to_price(opt_id, now_date, deal_time_1)
        sell_price = id_to_price(opt_id, now_date, deal_time_2)
        today_pnl = sell_price - buy_price
        print(past_3_day_low, a, opt_info_df)
        return today_pnl
    else:
        return 0


if __name__ == '__main__':
    date_list = [x for x in sorted(os.listdir('/media/hdd1/DAT_OPT/Tick')) if x > '20190101']
    # date_list = ['20191115']
    # now_time_list = ['10:30', '15:00']
    now_time_list = time_list

    result_list = []
    for now_date in date_list:
        for now_time in now_time_list:
            result_list.append(deal_fun(now_date, now_time))
    result_df = pd.concat(result_list, axis=1, sort=False).T

    # date_list = ['20191115']
    # now_time_list = ['10:30', '15:00']
    # percent_list = [0.95, 1]
    # now_time = '10:30'
    # percent = 1
    # result_dict = dict()
    # vol_sr_list = []
    # for now_date in date_list:
    #     for now_time in now_time_list:
    #         for percent in percent_list:
    #             vol_sr = main_fun(now_date, now_time, percent)
    #             vol_sr_list.append(vol_sr)
    # vol_df = pd.concat(vol_sr_list, axis=1, sort=False).T

    # result_sr = pd.Series()
    # for i in range(3, len(date_list)):
    #     print('____________________________________')
    #
    #     now_date = date_list[i]
    #     print(now_date)
    #     part_date_list = date_list[i - 3:i]
    #
    #     try:
    #         result_sr[now_date] = simple_st(now_date, part_date_list)
    #     except Exception as error:
    #         print(error)
    #         result_sr[now_date] = 0

