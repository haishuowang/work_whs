import sys

sys.path.append('/mnt/mfs')
from work_dmgr_fut.loc_lib.pre_load import *
from work_dmgr_fut.fut_script.FutDataLoad import FutData, ReshapeData
from work_dmgr_fut.loc_lib.pre_load.plt import savfig_send
from work_whs.loc_lib.pre_load.senior_tools import SignalAnalysis

fut_data = FutData()

fut_name_dict = {}
root_path = '/mnt/mfs/dat_whs/DAT_FUT/201906/sc'
root_path_2 = '/mnt/mfs/dat_whs/DAT_FUT/201906/ine'
all_con_file = os.listdir(root_path)
# fut_name_list = sorted(list(set([re.sub('\d', '', x.split('_')[0]) for x in all_con_id
#                             if len(re.sub('\d', '', x.split('_')[0])) < 3])))

# for con_file in all_con_file:
#     fut_name = re.sub('\d', '', con_file.split('_')[0])
#     con_id = con_file.split('_')[0]
#     if len(fut_name) < 3:
#         if fut_name not in fut_name_dict.keys():
#             fut_name_dict[fut_name] = {con_id: [con_file]}
#         else:
#             print(con_file)
#
#             if con_id not in fut_name_dict[fut_name].keys():
#                 fut_name_dict[fut_name].update({con_id: [con_file]})
#             else:
#                 fut_name_dict[fut_name][con_id].append(con_file)
#     else:
#         pass
#

#
# for fut_name in list(fut_name_dict.keys())[:1]:
#     fut_name = 'fu'
#     fut_name_2 = 'sc'
#     print(fut_data.act_info_df[fut_name.upper() + '01'])
#     active_df = fut_data.act_info_df[fut_name.upper() + '01']\
#         .truncate(datetime(2019, 6, 1), datetime(2019, 6, 30))
#     sub_dict = fut_name_dict[fut_name]
#     for target_date, target_id in active_df.dropna().items():
#         target_id_low = target_id.split('.')[0].lower()
#         # print(target_id)
#         # print(sub_dict)
#         # print(sub_dict[target_id_low])
#         daily_path = f"{root_path}/{target_id_low}_{target_date.strftime('%Y%m%d')}.csv"
#         data = pd.read_csv(daily_path, encoding='gbk')
#         data.index = pd.to_datetime(data['时间'])
#
#         target_id_low_2 = fut_data.act_info_df[f'{fut_name_2.upper()}01'].loc[target_date].split('.')[0].lower()
#         print(target_id_low, target_id_low_2)
#         daily_path_2 = f"{root_path_2}/{target_id_low_2}_{target_date.strftime('%Y%m%d')}.csv"
#         data_2 = pd.read_csv(daily_path_2, encoding='gbk')
#         data_2.index = pd.to_datetime(data_2['时间'])
#         data_2 = data_2[~data_2.index.duplicated('last')]
#
#         tick_data = data[['时间', '最新', '成交量', '持仓', '增仓', '买一价', '买一量', '卖一价', '卖一量']]
#         tick_data.index = pd.to_datetime(tick_data['时间'])
#         t10_min_vol = tick_data.truncate(before=pd.to_datetime('2019-06-27 20:59'),
#                                          after=pd.to_datetime('2019-06-27 21:10'))['成交量']
#         big_vol = t10_min_vol.quantile(0.7)
#         print(big_vol)
#         data['diff'] = data['最新'] - data['最新'].iloc[0]
#         data_2['diff'] = data_2['最新'] - data_2['最新'].iloc[0]
#
#         cut_num = 2000
#
#
#         # for i in range(int(len(data['最新']) / cut_num)):
#         #     try:
#         #         val_1 = data['最新'].iloc[i * cut_num:(i + 1) * cut_num]
#         #         print(val_1.index[0], val_1.index[-1])
#         #
#         #         val_2 = data_2['最新'].reindex(index=val_1.index)
#         #         fig = plt.figure(figsize=[16, 10])
#         #         ax1 = fig.add_subplot(111)
#         #         ax2 = ax1.twinx()
#         #         ax1.plot(val_1.values, c='blue')
#         #         ax2.plot(val_2.values, c='red')
#         #         ax1.grid()
#         #         savfig_send(f'{target_date} figure {i}')
#         #     except:
#         #         print(1)


def get_big_vol_and_rt(data, tmp_begin, tmp_end):
    t10_min_data = data.truncate(before=tmp_begin,
                                 after=tmp_end)
    big_vol = t10_min_data['成交量'].quantile(0.90)
    a = t10_min_data[t10_min_data['成交量'] > big_vol][['成交量', '成交额', 'way']]

    sum_turn = a['成交额'].sum()
    bs_turn = (a['成交额'] * a['way']).sum()
    bs_rt = bs_turn / sum_turn
    return bs_rt


def send_cdf_fun(window, bar_num):
    for fut_name in list(fut_name_dict.keys()):
        print(fut_name)
        # print(fut_data.act_info_df[fut_name.upper() + '01'])

        if fut_name.upper() + '01' not in fut_data.act_info_df.columns:
            continue

        active_df = fut_data.act_info_df[fut_name.upper() + '01'] \
            .truncate(datetime(2019, 6, 1), datetime(2019, 6, 30))
        sub_dict = fut_name_dict[fut_name]
        signal_list = []
        return_list = []
        for target_date, target_id in list(active_df.dropna().items()):
            con_intra_data = fut_data.load_intra_data(target_id, usecols_list=['Close'])
            con_time_range = con_intra_data.index
            # target_date = datetime(2019, 6, 27)
            target_id_low = target_id.split('.')[0].lower()
            # print(target_id)
            # print(sub_dict)
            # print(sub_dict[target_id_low])
            daily_path = f"{root_path}/{target_id_low}_{target_date.strftime('%Y%m%d')}.csv"
            data = pd.read_csv(daily_path, encoding='gbk')
            data['way'] = data['方向'].apply(lambda x: 1 if x == 'B' else (-1 if x == 'S' else 0))
            data.index = pd.to_datetime(data['时间'])

            today_begin = pd.to_datetime(f"{data.index[0].strftime('%Y%m%d')} 21:00")
            today_end = pd.to_datetime(f"{data.index[-1].strftime('%Y%m%d')} 16:00")
            today_min_range = con_time_range[(con_time_range > today_begin) & (con_time_range < today_end)]
            # print(target_id_low, target_date)

            # tick_data = data[['时间', '最新', '成交量', '持仓', '增仓', '买一价', '买一量', '卖一价', '卖一量', '方向']]
            # tick_data.index = pd.to_datetime(tick_data['时间'])

            # print(big_vol)

            # for i in range(int(len(today_min_range) / bar_num)):
            #     # print(today_min_range[bar_num * i:bar_num * (i + 1)])
            #     tmp_begin = today_min_range[bar_num * i] - timedelta(minutes=1)
            #     tmp_end = today_min_range[bar_num * (i + 1) - 1]
            #     bs_rt = get_big_vol_and_rt(data, tmp_begin, tmp_end)
            #     print(tmp_begin, tmp_end, bs_rt)

            # bs_rt_list_1 = [get_big_vol_and_rt(data, today_min_range[bar_num * i] - timedelta(minutes=1),
            #                                    today_min_range[bar_num * (i + 1) - 1])
            #                 for i in range(int(len(today_min_range) / bar_num))]

            bs_rt_list = [get_big_vol_and_rt(data, i_time + timedelta(minutes=-(window - bar_num) - 1),
                                             i_time + timedelta(minutes=bar_num - 1))
                          for i_time in today_min_range[::bar_num]]
            # bs_rt_list = np.array(bs_rt_list)
            trade_time_list = today_min_range[::bar_num]
            # return_df = con_intra_data['']
            price_diff = con_intra_data['Close'].shift(-bar_num).loc[trade_time_list] - \
                         con_intra_data['Close'].loc[trade_time_list]
            price_diff_list = list(price_diff.values)
            signal_list += bs_rt_list[:-1]
            return_list += price_diff_list[1:]

            SignalAnalysis.CDF(pd.Series(signal_list), pd.Series(return_list), hold_time=1,
                           title=f'{fut_name} {window} {bar_num}{target_date} CDF Figure', lag=1)


# bar_num = 5
# window = 30
# for bar_num, window in [(5, 5), (5, 10), (5, 20), (15, 20), (15, 30)][:1]:
#     send_cdf_fun(window, bar_num)
