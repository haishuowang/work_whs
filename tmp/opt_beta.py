from py_vollib.black.greeks.analytical import delta, gamma, rho, theta, vega
from py_vollib.black_scholes.implied_volatility import implied_volatility

import sys

sys.path.append('/mnt/mfs')
from work_dmgr_fut.loc_lib.pre_load import *


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


def deal_fun(now_date='20191115', now_time='10:30', percent=1):
    print('_______________________________')
    print(now_date)
    eqt_min_path = '/mnt/mfs/DAT_EQT/intraday/eqt_1mbar'

    spot_id = 'SH510050'
    # now_time = '10:30'
    spot_data = pd.read_csv(f'{eqt_min_path}/{now_date[:4]}/{now_date[:6]}/{now_date}/Close.csv', index_col=0)
    if spot_id not in spot_data.columns:
        return None
    spot_price = spot_data[spot_id].round(6).loc[now_time] * percent

    exe_date_array = np.array(sorted(set(map_df.index)))
    exe_date_list = exe_date_array[exe_date_array > pd.to_datetime(now_date)][:2]

    result_dict = dict()
    result_dict['date'] = now_date
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
    if 'rate_mean_1' and 'rate_mean_2':  # result_dict[f'r_mean_{i+1}']:
        a1 = (exe_date_list[0] - pd.to_datetime(now_date)).days
        a2 = (exe_date_list[1] - pd.to_datetime(now_date)).days
        x = (30 - a2) / (a1 - a2)
        # print(x)
        result_dict['rate_weight'] = x * result_dict[f'rate_mean_1'] + (1 - x) * result_dict[f'rate_mean_2']
    elif 'rate_mean_1' in result_dict.keys():
        result_dict['rate_weight'] = result_dict['rate_mean_1']
    elif 'rate_mean_2' in result_dict.keys():
        result_dict['rate_weight'] = result_dict['rate_mean_2']
    else:
        result_dict['rate_weight'] = None
    return result_dict


def main_fun(now_date, now_time, percent):
    vol_sr = pd.Series()
    try:
        vol_sr['percent'] = percent
        part_result_dict = deal_fun(now_date=now_date, now_time=now_time, percent=percent)
        F = part_result_dict['spot_px']
        vol_sr['S'] = F
        r = part_result_dict['rate_weight']
        vol_sr['r'] = r

        tran_dict = {1: 'first', 2: 'second'}
        for i in range(1, 3):
            t = part_result_dict['opt_ttm_1'] / 365
            for ii in range(1, 3):
                K = part_result_dict[f'opt_k_{i}_{ii}']
                vol_sr[f'{tran_dict[i]}_K_{ii}'] = K
                # 计算 vol
                # if F >= K:
                #     flag = 'call'
                #     price = part_result_dict[f'{flag}_px_{i}_{ii}']
                #     vol_sr[f'{tran_dict[i]}_{flag}_px_{ii}'] = price
                #     sigma = implied_volatility(price, F, K, t, r, flag[0])
                #     vol_sr[f'{tran_dict[i]}_{flag}_vol_{ii}'] = sigma
                #     vol_sr[f'{tran_dict[i]}_vol_{ii}'] = vol_sr[f'{tran_dict[i]}_call_vol_{ii}']
                # else:
                #     flag = 'put'
                #     price = part_result_dict[f'{flag}_px_{i}_{ii}']
                #     vol_sr[f'{tran_dict[i]}_{flag}_px_{ii}'] = price
                #     sigma = implied_volatility(price, F, K, t, r, flag[0])
                #     vol_sr[f'{tran_dict[i]}_{flag}_vol_{ii}'] = sigma
                #     vol_sr[f'{tran_dict[i]}_vol_{ii}'] = vol_sr[f'{tran_dict[i]}_put_vol_{ii}']

                # 计算希腊字母
                for flag in ['call', 'put']:
                    price = result_dict[now_date][f'{flag}_px_{i}_{ii}']
                    vol_sr[f'{tran_dict[i]}_{flag}_px_{ii}'] = price
                    sigma = implied_volatility(price, F, K, t, r, flag[0])
                    vol_sr[f'{tran_dict[i]}_{flag}_vol_{ii}'] = sigma
                    vol_sr[f'{tran_dict[i]}_{flag}_delta_{ii}'] = delta(flag[0], F, K, t, r, sigma)
                    vol_sr[f'{tran_dict[i]}_{flag}_gamma_{ii}'] = gamma(flag[0], F, K, t, r, sigma)
                    vol_sr[f'{tran_dict[i]}_{flag}_rho_{ii}'] = rho(flag[0], F, K, t, r, sigma)
                    vol_sr[f'{tran_dict[i]}_{flag}_theta_{ii}'] = theta(flag[0], F, K, t, r, sigma)
                    vol_sr[f'{tran_dict[i]}_{flag}_vega_{ii}'] = vega(flag[0], F, K, t, r, sigma)

            vol_sr[f'{tran_dict[i]}_K_weight'] = (F - vol_sr[f'{tran_dict[i]}_K_2']) / \
                                                 (vol_sr[f'{tran_dict[i]}_K_1'] - vol_sr[f'{tran_dict[i]}_K_2'])
            vol_sr[f'{tran_dict[i]}_vol'] = vol_sr[f'{tran_dict[i]}_K_weight'] * \
                                            vol_sr[f'{tran_dict[i]}_vol_1'] + \
                                            (1 - vol_sr[f'{tran_dict[i]}_K_weight']) * \
                                            vol_sr[f'{tran_dict[i]}_vol_2']
        x = (30 - part_result_dict['opt_ttm_2']) / \
            (part_result_dict['opt_ttm_1'] - part_result_dict['opt_ttm_2'])
        vol_sr[f'opt_ttm_1'] = part_result_dict['opt_ttm_1']
        vol_sr[f'opt_ttm_2'] = part_result_dict['opt_ttm_2']
        vol_sr[f'time_weight'] = x
        vol_sr[f'target_vol'] = x * vol_sr[f'first_vol'] + (1 - x) * vol_sr[f'second_vol']
    except Exception as error:
        print(error)
    vol_sr.name = pd.to_datetime(f'{now_date} {now_time}')
    return vol_sr


if __name__ == '__main__':
    date_list = [x for x in sorted(os.listdir('/media/hdd1/DAT_OPT/Tick')) if x > '20190101']
    # date_list = ['20191115']
    now_time_list = ['10:30', '15:00']
    percent_list = [0.95, 1]
    now_time = '10:30'
    percent = 1
    result_dict = dict()
    vol_sr_list = []
    for now_date in date_list:
        for now_time in now_time_list:
            for percent in percent_list:
                vol_sr = main_fun(now_date, now_time, percent)
                vol_sr_list.append(vol_sr)
    vol_df = pd.concat(vol_sr_list, axis=1, sort=False).T

# first:近月 second:远月
# first_K_1:近月 第一个执行价格 看涨期权 执行价格
# first_call_px_1:近月 第一个执行价格 看涨期权 价格
# first_call_vol_1:近月 第一个执行价格 看涨期权 vol
# first_K_weight:近月根据 执行价格 计算出的 weight
# first_vol_1:近月 第一个执行价格 期权 vol =(first_call_vol_1 + first_put_vol_1)/2
# first_vol:近月 期权 vol = first_K_weight * first_vol_1 + (1 - first_K_weight) * first_vol_2
# time_weight: 近月 远月 根据 时间 计算出的 weight
# target_vol: time_weight * first_vol + (1-time_weight) * second_vol
