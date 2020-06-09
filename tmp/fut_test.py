import sys

sys.path.append('/mnt/mfs')
from work_dmgr_fut.loc_lib.pre_load import *
from work_dmgr_fut.loc_lib.pre_load.plt import savfig_send
from work_dmgr_fut.fut_script.FutDataLoad import FutData, FutClass
from work_dmgr_fut.fut_script.signal_fut_fun import FutIndex, Signal, Position
from work_dmgr_fut.loc_lib.pre_load.senior_tools import SignalAnalysis

fut_data = FutData()

tick_path = '/media/hdd1/CTP_DATA_HUB/TICk_DATA'

date_list = sorted(os.listdir(tick_path))

date_now = date_list[-1]


def date_fun(date_now, fut_name):
    date_path = f'{tick_path}/{date_now}'

    con_list = [x for x in sorted(os.listdir(date_path)) if re.sub('\d+', '', x.split('_')[0]) == fut_name]

    con_data_list = []
    # plt.figure(figsize=[20, 10])
    for con_file in con_list:
        # if con_file.split('_')[0][-2:] in ['01', '05', '10']:
        # print(con_file)
        con_path = f'{date_path}/{con_file}'
        con_data = pd.read_csv(con_path, sep='|', header=None)

        col_list = ['Con_id',
                    'Date',
                    'Time',
                    'Now_px',
                    'Trade_num',
                    'OI',
                    'Turnover',
                    'bid_px',
                    'bid_num',
                    'ask_px',
                    'ask_num',
                    'Trade_date'
                    ]

        con_data.columns = col_list

        morn_begin = '09:00:00.0'
        morn_end = '11:30:00.0'
        noon_begin = '13:00:00.0'
        noon_end = '15:00:00.0'
        even_begin = '21:00:00.0'
        even_end = '23:00:00.0'
        con_data = con_data[((morn_end >= con_data['Time']) & (con_data['Time'] > morn_begin)) |
                            ((noon_end >= con_data['Time']) & (con_data['Time'] > noon_begin)) |
                            ((even_end >= con_data['Time']) & (con_data['Time'] > even_begin))
                            ]
        # try:
        con_data.index = pd.to_datetime(con_data['Date'].fillna(method='ffill').astype(int).astype(str)
                                        + ' ' + con_data['Time'])
        # except:
        #     print(1)
        price_sr = con_data['Now_px']
        price_sr.name = con_file.split('_')[0]
        price_sr = price_sr[~price_sr.index.duplicated('last')]
        # plt.plot(price_sr, label=price_sr.name)
        con_data_list.append(price_sr)
    # plt.legend()
    # savfig_send()

    con_num = fut_data.act_info_df[fut_name.upper() + '01'].loc[
        pd.to_datetime(date_now.replace('_', ''))].split('.')[0]
    con_num = re.sub('\D', '', con_num)
    print(con_num)
    # if con_num[-2:] == '01':
    #     next_num = con_num[:2] + '05'
    # elif con_num[-2:] == '05':
    #     next_num = con_num[:2] + '09'
    # elif con_num[-2:] == '09':
    #     next_num = str(int(con_num[:2]) + 1) + '01'
    # else:
    #     print('con error')
    #     return pd.DataFrame(), None, None, None

    if con_num[-2:] == '12':
        next_num = str(int(con_num[:2]) + 1) + '01'
    else:
        next_num = con_num[:2] + str(int(con_num[-2:]) + 1).zfill(2)

    price_df = pd.concat(con_data_list, axis=1)
    price_df = price_df.replace(0, np.nan).fillna(method='ffill')
    # diff_df = price_df.sub(price_df['rb2010'], axis=0)
    diff_df = price_df.sub(price_df[fut_name + con_num], axis=0)
    return diff_df, next_num, price_sr.iloc[-1], con_num


# diff_df_1 = date_fun(date_list[-1])
# diff_df_2 = date_fun(date_list[-2])
# diff_df_3 = date_fun(date_list[-3])
#
# for date_now in date_list[-10:]:
#     diff_df = date_fun(date_now)
#     diff_df['rb2101'].iloc[:2400].quantile([0.05, 0.20, 0.80, 0.95])
#
#     plt.figure(figsize=[40, 10])
#     plt.plot(diff_df['rb2101'].dropna().values)
#     plt.legend()
#     savfig_send(date_now)

# FutIndex.boll_fun(diff_df_1, 500)
#
# diff_df = diff_df_1
#
# plt.figure(figsize=[20, 10])
# plt.plot(diff_df)
# plt.legend()
# savfig_send()
#
# window = 60
# i = 3
# a = diff_df['rb2005'].iloc[window * i:window * (i + 1)]
# min_px = a.min()
# max_px = a.max()
# print(min_px, max_px)
#
# b = diff_df['rb2005'].iloc[window * (i + 1):window * (i + 2)]
# c = pd.concat([a, b], axis=0)
# d = diff_df['rb2005']
# d.name = 'px'
# e = pd.concat([d, FutIndex.boll_fun(diff_df['rb2005'], n=60)], axis=1)

# price_df = pd.concat(con_data_list, axis=1)
# price_df = price_df.fillna(method='ffill')
# diff_df = price_df.sub(price_df['rb2010'], axis=0)
# # diff_df = price_df.sub(price_df[price_df.columns[0]], axis=0)
#
# plt.figure(figsize=[20, 10])
# plt.plot(diff_df)
# plt.legend()
# savfig_send()

def pnl_fun(date_now, fut_name='rb', bkt_num=3600, buy_sell_list=[0.05, 0.90, 0.95, 0.10]):
    diff_df, next_num, close_px, con_num = date_fun(date_now, fut_name)
    # bkt_num, buy_sell_list = 3600, [0.05, 0.90, 0.95, 0.10]
    buy_open_px, buy_close_px, sell_open_px, sell_close_px = diff_df[f'{fut_name}{next_num}'].iloc[:bkt_num].quantile(
        buy_sell_list)
    pos = 0
    vol = diff_df[f'{fut_name}{next_num}'].iloc[:bkt_num].std()
    open_time = 0
    # for x in diff_df['rb2101'].iloc[2400:]:
    i = 0
    data_len = len(diff_df[f'{fut_name}{next_num}'].iloc[bkt_num:])
    pos_list = np.array([None] * data_len)
    while i < data_len:
        x = diff_df[f'{fut_name}{next_num}'].iloc[bkt_num:].iloc[i]
        if pos == 0:
            if x <= buy_open_px:
                pos = 1
                pos_list[i:i + 120] = 1
                i = i + 120
                open_time = open_time + 120
            elif x >= sell_open_px:
                pos = -1
                pos_list[i:i + 120] = -1
                i = i + 120
                open_time = open_time + 120
            else:
                pos = 0
                i += 1
                pos_list[i - 1] = pos

        elif pos == 1:
            if x >= buy_close_px:
                pos = 0
                open_time = 0
            elif x <= buy_open_px - 4:
                pos = 0
                open_time = 0
            elif open_time >= 1200:
                pos = 0
                open_time = 0
            else:
                pos = 1
                open_time += 1
            i += 1
            pos_list[i - 1] = pos
        elif pos == -1:
            if x <= sell_close_px:
                pos = 0
                open_time = 0
            elif x >= sell_open_px + 4:
                pos = 0
                open_time = 0
            elif open_time >= 1200:
                pos = 0
                open_time = 0
            else:
                pos = -1
                open_time += 1
            i += 1
            pos_list[i - 1] = pos
        else:
            pos = np.nan
            i += 1
            pos_list[i - 1] = pos
            print('error')

    pos_sr = pd.Series(pos_list, index=diff_df.index[bkt_num:])
    x = pd.concat([diff_df[fut_name + next_num].iloc[bkt_num:], pos_sr], axis=1)

    x[f'{fut_name}{next_num}_diff'] = x[fut_name + next_num].diff()
    x['pnl'] = x[0].shift(1) * x[f'{fut_name}{next_num}_diff']
    asset = x['pnl'].sum()
    turnover = x[0].diff().abs().sum() * close_px
    pot = round(asset / turnover * 10000, 6)
    # print(round(asset / turnover * 10000, 6), asset, turnover)
    return x['pnl'].sum(), turnover, pot, con_num, vol


# fut_name = 'zn'
if __name__ == '__main__':
    fut_name_list = ['al']
    # fut_name_list = ['ru', 'j', 'jm', 'jd']
    bkt_num_list = [1200, 2400, 3600]
    buy_sell_list_list = [[0.05, 0.90, 0.95, 0.10],
                          # [0.00, 0.90, 1, 0.10],
                          [0.01, 0.95, 0.99, 0.05],
                          ]
    tmp_list = []
    a_list = []
    for fut_name in fut_name_list:
        for bkt_num in bkt_num_list[-1:]:
            for buy_sell_list in buy_sell_list_list[:1]:
                result_list = []
                for date_now in date_list[30:-1]:
                    # date_now = '2019_09_27'
                    print('_________________')
                    print(date_now)
                    # diff_df = date_fun(date_now)
                    # diff_df['rb2101'].iloc[:2400].quantile([0.05, 0.10, 0.90, 0.95])
                    #
                    # plt.figure(figsize=[40, 10])
                    # plt.plot(diff_df['rb2101'].dropna().values)
                    # plt.legend()
                    # savfig_send(date_now)
                    # pnl, turnover, pot, con_num = pnl_fun(date_now, fut_name)
                    # pnl_fun(date_now, fut_name)

                    result_list.append(pnl_fun(date_now, fut_name, bkt_num, buy_sell_list))

                a = pd.DataFrame(result_list, columns=['pnl', 'turnover', 'pot', 'con_num', 'vol'], index=date_list[30:-1])
                a['pnl_real'] = a['pnl'] - a['turnover'] * 0.00005
                a_list.append(a)
                plt.figure(figsize=[20, 10])
                plt.plot(a['pnl_real'].cumsum())
                pot_all = a['pnl'].sum() / a['turnover'].sum() * 10000
                savfig_send(fut_name + str(round(pot_all, 6)))
                print(pot_all)
                tmp_list.append([bkt_num, buy_sell_list, a])
                SignalAnalysis.CDF(a['vol'], a['pot'], 1)
                SignalAnalysis.CDF(a['vol'], a['pnl_real'], 1)



