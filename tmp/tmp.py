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


def deal_fun(now_date='20191115'):
    print('_______________________________')
    print(now_date)
    eqt_min_path = '/mnt/mfs/DAT_EQT/intraday/eqt_1mbar'

    spot_id = 'SH510050'
    now_time = '10:30'
    spot_data = pd.read_csv(f'{eqt_min_path}/{now_date[:4]}/{now_date[:6]}/{now_date}/Close.csv', index_col=0)
    if spot_id not in spot_data.columns:
        return None
    spot_price = spot_data[spot_id].round(6).loc[now_time]

    a = ['option_code', 'exercise_date', 'exe_price', 'type']

    exe_date_array = np.array(sorted(set(map_df.index)))
    exe_date_list = exe_date_array[exe_date_array > pd.to_datetime(now_date)][:2]
    rate_mean_dict = dict()
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
            print(exe_date, target_exe_price)
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
                    call_id = map_df_date_price[(map_df_date_price['type'] == 'C') & (map_df_date_price['d_type'] == 'M')][
                        'option'].iloc[0]
                else:
                    call_id = map_df_date_price[map_df_date_price['type'] == 'C']['option'].iloc[0]

                if len(map_df_date_price[map_df_date_price['type'] == 'P']['option']) > 1:
                    put_id = map_df_date_price[(map_df_date_price['type'] == 'P') & (map_df_date_price['d_type'] == 'M')][
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

        rate_mean = (result_dict[f'rate_{i+1}_1'] + result_dict[f'rate_{i+1}_2'])/2
        result_dict[f'rate_mean_{i+1}'] = rate_mean
    # return rate_mean_dict
    if 'rate_mean_1' and 'rate_mean_2':#result_dict[f'r_mean_{i+1}']:
        a1 = (exe_date_list[0] - pd.to_datetime(now_date)).days
        a2 = (exe_date_list[1] - pd.to_datetime(now_date)).days
        x = (30 - a2) / (a1-a2)
        print(x)
        result_dict['rate_weight'] = x * result_dict[f'rate_mean_1'] + (1-x) * result_dict[f'rate_mean_2']
    elif 'rate_mean_1' in result_dict.keys():
        result_dict['rate_weight'] = result_dict['rate_mean_1']
    elif 'rate_mean_2' in result_dict.keys():
        result_dict['rate_weight'] = result_dict['rate_mean_2']
    else:
        result_dict['rate_weight'] = None
    return result_dict


if __name__ == '__main__':
    date_list = [x for x in sorted(os.listdir('/media/hdd1/DAT_OPT/Tick')) if x > '20190101']
    result_dict = dict()
    for now_date in date_list:
        result_dict[now_date] = deal_fun(now_date)

    x = pd.DataFrame().from_dict(result_dict).T
    a = ['first_call_px_1', 'first_call_px_2', 'second_call_px_1',
         'second_call_px_2', 'date', 'first_opt_k_1', 'first_opt_k_2',
         'second_opt_k_1', 'second_opt_k_2', 'first_opt_mat', 'second_opt_mat',
         'first_opt_ttm', 'second_opt_ttm', 'first_put_px_1', 'first_put_px_2',
         'second_put_px_1', 'second_put_px_2', 'first_rate_1', 'first_rate_2',
         'second_rate_1', 'second_rate_2', 'first_rate_mean', 'second_rate_mean',
         'rate_weight', 'spot_px']
    x.columns = a
    b = ['date',
         'first_opt_mat', 'first_opt_ttm', 'first_opt_k_1', 'first_call_px_1', 'first_put_px_1', 'first_opt_k_2',
         'first_call_px_2', 'first_put_px_2',
         'second_opt_mat', 'second_opt_ttm', 'second_opt_k_1', 'second_call_px_1', 'second_put_px_1', 'second_opt_k_2',
         'second_call_px_2', 'second_put_px_2',
         'first_rate_1', 'first_rate_2', 'second_rate_1', 'second_rate_2', 'first_rate_mean', 'second_rate_mean',
         'rate_weight', 'spot_px']
    result_df = x[b].round(6)
