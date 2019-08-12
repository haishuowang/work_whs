import sys

sys.path.append('/mnf/mfs')
from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load import log
from work_whs.loc_lib.pre_load.plt import savfig_send
from work_whs.loc_lib.pre_load.senior_tools import SignalAnalysis
from work_whs.test_future.FutDataLoad import FutData, FutClass
from work_whs.test_future.signal_fut_fun import FutIndex, Signal, Position


def MA_LINE(Close, slowperiod, fastperiod):
    slow_line = bt.AZ_Rolling_mean(Close, slowperiod, min_periods=0)
    fast_line = bt.AZ_Rolling_mean(Close, fastperiod, min_periods=0)
    MA_diff = fast_line - slow_line
    target_df = MA_diff.copy()
    target_df[MA_diff > 0] = 1
    target_df[MA_diff < 0] = -1
    return target_df


def mul_fun(a, b):
    a_l = a.where(a > 0, 0)
    a_s = a.where(a < 0, 0)

    b_l = b.where(b > 0, 0)
    b_s = b.where(b < 0, 0)

    pos_l = a_l.mul(b_l)
    pos_s = a_s.mul(b_s)

    pos = pos_l.sub(pos_s)
    return pos


def get_active_df(data_name='Volume'):
    root_path = '/mnt/mfs/dat_whs/DAT_FUT'
    active_df = bt.AZ_Load_csv(f'{root_path}/Active_{data_name}')
    adj_factor_df = bt.AZ_Load_csv(f'{root_path}/adj_factor_{data_name}')
    aadj_factor_df = adj_factor_df.cumprod()
    return active_df, aadj_factor_df


# @log.try_catch
def part_test(con_id, begin_time, end_time,
              # p_window, p_limit,
              # v_window, v_limit,
              # voli_window, voli_limit,
              CCI_window, CCI_limit,
              hold_time
              ):
    try:
        print(con_id, begin_time, end_time)
        data_df = fut_data.load_intra_data(con_id, ['High', 'Low', 'Close', 'Volume', 'OpenInterest'])

        # begin_time = pd.to_datetime('20190101')
        # # end_time = begin_time + timedelta(1)
        # end_time = pd.to_datetime('20190401')

        # p_window = 1
        # p_limit = 1
        # v_window = 20
        # v_limit = 2

        part_data_df = data_df.truncate(before=begin_time, after=end_time)
        part_data_df['OpenInterest_core'] = FutIndex.boll_fun(data_df['OpenInterest'], 100)
        part_data_df['OpenInterest_signal'] = (part_data_df['OpenInterest_core'] > 0).astype(int)
        part_data_df['weekday'] = pd.to_datetime(part_data_df['Date']).dt.weekday
        part_data_df['weekday_pos'] = (
                (part_data_df['weekday'] != 2)
                & (part_data_df['weekday'] != 1)
            # & (part_data_df['weekday'] != 3)
            # (part_data_df['weekday'] != 4)
        ).astype(int)

        part_data_df['month'] = part_data_df['Date'].str.slice(5, 7)
        part_data_df['month_pos'] = (
                (part_data_df['month'] != '01')
                # & (part_data_df['month'] != '12')
        ).astype(int)

        # part_data_df['time_pos_cut'] = part_data_df.apply(lambda x: -1 if (((x['Time'] > '14:51')
        #                                                                     & (x['Time'] <= '15:00')
        #                                                                     # | (x['Time'] > '21:00')
        #                                                                     # & (x['Time'] <= '22:00')
        #                                                                     # | (x['Time'] > '21:00')
        #                                                                     # & (x['Time'] <= '22:00')
        #                                                                     )) else 1, axis=1)

        # v_window = 60

        # part_data_df['Volume_boll'] = FutIndex.boll_fun(part_data_df[['Volume']], 60)

        # part_data_df['Volume_zscore'] = bt.AZ_Col_zscore(part_data_df[['Volume']], v_window)
        # part_data_df['Volume_zscore_10'] = bt.AZ_Rolling(part_data_df[['Volume']], 10).mean()
        # part_data_df['Volume_zscore_20'] = bt.AZ_Rolling(part_data_df[['Volume']], 20).mean()
        # part_data_df['Volume_zscore_40'] = bt.AZ_Rolling(part_data_df[['Volume']], 40).mean()
        # part_data_df['Volume_zscore_100'] = bt.AZ_Rolling(part_data_df[['Volume']], 100).mean()

        # part_data_df['Volume_signal'] = (part_data_df['Volume_zscore'] < 2).astype(int).replace(0, -1)

        p_window = 3
        # part_data_df['past_min_pct_change'] = (part_data_df['Close'] - part_data_df['Close'].shift(p_window)) \
        #                                       / part_data_df['Close'].shift(p_window)
        part_data_df['past_min_pct_change'] = (part_data_df['Close'] - part_data_df['Close'].shift(p_window))
        part_data_df['past_min_pct_signal'] = part_data_df['past_min_pct_change'] \
            .apply(lambda x: 0 if abs(x) < 0.005 else (1 if x > 0.005 else -1))

        # part_data_df['past_high_low'] = bt.AZ_Rolling(part_data_df['Close'], 100) \
        #     .apply(lambda x: 2 * x[-1] / (max(x) + min(x)), raw=True)

        # part_data_df['past_high_low_signal'] = (((part_data_df['past_high_low'] < 1.01) &
        #                                         (part_data_df['past_high_low'] > 1.001)) | \
        #                                        ((part_data_df['past_high_low'] > 0.99) &
        #                                         (part_data_df['past_high_low'] < 0.999))).astype(int)

        # def con_fun(x):
        #     if ((x < 1.01) & (x > 1.0006)) | (x > 0.99) & (x < 0.9995):
        #         return 1
        #     else:
        #         return 0
        # part_data_df['past_high_low_signal'] = part_data_df['past_high_low'].apply(con_fun)

        # part_data_df['past_over_high'] = bt.AZ_Rolling(part_data_df['Close'], 100) \
        #     .apply(lambda x: x[-1] / max(x), raw=True)

        # part_data_df['past_over_low'] = bt.AZ_Rolling(part_data_df['Close'], 100) \
        #     .apply(lambda x: x[-1] / min(x), raw=True)

        part_data_df['CCI_score'] = FutIndex.CCI_fun(part_data_df['High'],
                                                     part_data_df['Low'],
                                                     part_data_df['Close'], CCI_window)

        part_data_df['CCI_signal'] = Signal.fun_1(part_data_df['CCI_score'], CCI_limit)

        part_data_df['CCI_pos'] = Position.fun_1(part_data_df['CCI_signal'])

        part_data_df['boll_score'] = FutIndex.boll_fun(part_data_df['Close'], CCI_window)
        part_data_df['boll_signal'] = Signal.fun_1(part_data_df['boll_score'], CCI_limit)
        part_data_df['boll_pos'] = Position.fun_1(part_data_df['boll_signal'])

        part_data_df['trend_signal'] = bt.AZ_Rolling(part_data_df['Close'], 500).std()
        part_data_df['trend_pos'] = (part_data_df['trend_signal'] < 20).astype(int)

        # part_data_df['position'] = part_data_df['boll_pos'] * part_data_df['weekday_pos'] * part_data_df['month_pos']
        part_data_df['signal'] = part_data_df['boll_signal'] * part_data_df['trend_pos'] \
                                 * part_data_df['OpenInterest_signal']

        # part_data_df['signal'] = part_data_df['CCI_signal'] * part_data_df['trend_pos'] \
        #                          * part_data_df['OpenInterest_signal']

        # part_data_df['position'] = part_data_df['boll_pos'] * part_data_df['trend_pos']
        part_data_df['position'] = bt.AZ_Rolling(part_data_df['signal'], hold_time).sum()

        part_data_df['position_sft'] = part_data_df['position'].shift(2) # * part_data_df['month_pos']
        part_data_df['price_return'] = part_data_df['Close'] - part_data_df['Close'].shift(1)

        part_data_df['price_return_sum'] = bt.AZ_Rolling(part_data_df['price_return'], hold_time).sum() \
            .shift(-hold_time + 1)
        # part_data_df['pnl'] = part_data_df['position_sft'] * part_data_df['price_return']

        part_data_df['pnl_test'] = part_data_df['signal'] * part_data_df['price_return_sum'].shift(-2)
        part_data_df['pnl'] = part_data_df['position_sft'] * part_data_df['price_return']

        part_data_df['asset'] = part_data_df['pnl'].cumsum()

        part_data_df['turnover'] = (part_data_df['position_sft'] - part_data_df['position_sft'].shift(1)) \
                                   * part_data_df['Close']

        # 绘图
        # plt.figure(figsize=[64, 64])
        # ax1 = plt.subplot(4, 1, 1)
        # ax2 = plt.subplot(4, 1, 2)
        # ax3 = plt.subplot(4, 1, 3)
        # ax1.plot(part_data_df['asset'].values)
        # # ax2.plot(part_data_df['Close'].values, '--', color='#75bbfd')
        # ax2.scatter(np.array(range(len(part_data_df.index)))[(part_data_df['position_sft'] > 0)],
        #             part_data_df['Close'][part_data_df['position_sft'] > 0], s=0.5, color='red')
        #
        # ax2.scatter(np.array(range(len(part_data_df.index)))[(part_data_df['position_sft'] == 0)],
        #             part_data_df['Close'][part_data_df['position_sft'] == 0], s=0.5, color='black')
        #
        # ax2.scatter(np.array(range(len(part_data_df.index)))[(part_data_df['position_sft'] < 0)],
        #             part_data_df['Close'][part_data_df['position_sft'] < 0], s=0.5, color='blue')
        # # ax2.scatter(np.array(range(len(part_data_df.index)))[(part_data_df['past_min_pct_signal'] > 0)],
        # #             part_data_df['Close'][part_data_df['past_min_pct_signal'] > 0], s=20, color='black')
        # # ax2.scatter(np.array(range(len(part_data_df.index)))[(part_data_df['past_min_pct_signal'] < 0)],
        # #             part_data_df['Close'][part_data_df['past_min_pct_signal'] < 0], s=20, color='y')
        #
        # ax3.bar(range(len(part_data_df.index)), part_data_df['Volume'].values)
        #
        # ax1.grid()
        # ax2.grid()
        # ax3.grid()
        # plt.title(con_id)
        # savfig_send(con_id)

        part_pnl_df = part_data_df.groupby('Date')['pnl'].sum()
        part_turnover = part_data_df.groupby('Date')['turnover'].apply(lambda x: sum(abs(x)))
        return part_pnl_df, part_turnover, part_data_df
    except Exception as error:
        print(error)
        return pd.Series(), pd.Series(), pd.Series()


def test_fun(fut_name, CCI_window, CCI_limit, hold_time):
    result_list = []
    pool = Pool(20)
    for con_id, part_info_df in active_df[[f'{fut_name}01']].groupby(f'{fut_name}01'):
        active_begin = fut_data.last_trade_date(part_info_df.index[0]) + timedelta(hours=17)
        active_end = part_info_df.index[-1] + timedelta(hours=17)
        args = [con_id, active_begin, active_end, CCI_window, CCI_limit, hold_time]
        # part_test(*args)
        # print(part_info_df.index[0], part_info_df.index[-1])
        result_list.append(pool.apply_async(part_test, args=args))
    pool.close()
    pool.join()

    pnl_df = pd.concat([res.get()[0] for res in result_list], axis=0)
    turnover_df = pd.concat([res.get()[1] for res in result_list], axis=0)
    data_df = pd.concat([res.get()[2] for res in result_list], axis=0)

    pot = pnl_df.sum() / turnover_df.sum() * 10000
    sp = bt.AZ_Sharpe_y(pnl_df)
    print(fut_name, sp, pot, CCI_window, CCI_limit)
    print(fut_name, pot, sp)
    for x, part_data_df in data_df.groupby(['weekday']):
        part_pnl = part_data_df['pnl'].fillna(0).values
        print(x, part_pnl.sum())
        # plt.plot(part_pnl.cumsum())
        # savfig_send(subject=f'{x}  {bt.AZ_Sharpe_y(part_pnl)}')

    for x, part_data_df in data_df.groupby(['month']):
        part_pnl = part_data_df['pnl'].fillna(0).values
        print(x, part_pnl.sum())
        # plt.plot(part_pnl.cumsum())
        # savfig_send(subject=f'{x}  {bt.AZ_Sharpe_y(part_pnl)}')

    # if abs(sp) > 1.5 and abs(pot) > 8:
    # for x, part_data_df in data_df.groupby(['weekday']):
    #     part_pnl = part_data_df['pnl'].fillna(0).values
    #     print(x, part_pnl.sum())
    #     plt.plot(part_pnl.cumsum())
    #     savfig_send(subject=f'{x}  {bt.AZ_Sharpe_y(part_pnl)}')
    #
    # for x, part_data_df in data_df.groupby(['month']):
    #     part_pnl = part_data_df['pnl'].fillna(0).values
    #     print(x, part_pnl.sum())
    #     plt.plot(part_pnl.cumsum())
    #     savfig_send(subject=f'{x}  {bt.AZ_Sharpe_y(part_pnl)}')
    plt.figure(figsize=[16, 8])
    pnl_df.index = pd.to_datetime(pnl_df.index)
    plt.plot(pnl_df.cumsum())
    plt.grid()
    savfig_send(f'{fut_name} sp:{sp} pot={pot} CCI_window:{CCI_window}, CCI_limit:{CCI_limit}')
    return data_df


def ana_fun(data_df, fut_name, col_name):
    raw_return_df = data_df['pnl']
    signal_df = data_df[col_name]
    SignalAnalysis.CDF(signal_df, raw_return_df, hold_time=1, title=f'{fut_name} {col_name} CDF Figure', lag=0)


@log.use_time
def main(fut_name_list, ban_name_list):
    # p_window = 5
    # p_limit = 0.001
    #
    # v_window = 120
    # v_limit = 0
    #
    # voli_window = 120
    # voli_limit = 0.000
    hold_time = 300

    CCI_w_list = [20, 60, 120, 180, 300, 400]
    CCI_l_list = [1, 1.5, 2, 2.5, 3]

    # CCI_w_list = []
    # CCI_l_list = [1, 1.5, 2, 2.5, 3]

    for fut_name in fut_name_list:
        data_df = test_fun(fut_name, 300, 1.5, hold_time)
        # part_data_df = data_df
        # col_name = 'Time'
        # raw_return_df = part_data_df['pnl']
        # signal_df = part_data_df[col_name].shift(2).replace(np.nan, '00:00')
        # SignalAnalysis.CDF_c(signal_df, raw_return_df, hold_time=1,
        #                      title=f'{fut_name} {col_name} CDF Figure', lag=0)
        # ana_fun(data_df, fut_name, 'Time')
        # ana_fun(data_df, fut_name, 'Volume_boll')

        # ana_fun(data_df, fut_name, 'past_min_pct_change')
        # ana_fun(data_df, fut_name, 'trend_signal')
        # ana_fun(data_df, fut_name, 'OpenInterest_core')

        # ana_fun(data_df, fut_name, 'past_over_low')
        # ana_fun(data_df, fut_name, 'past_over_high')
    return data_df
    # return data_df
    # for fut_name in fut_name_list:
    #     if fut_name in ban_name_list:
    #         continue
    #     else:
    #         for CCI_window, CCI_limit in product(CCI_w_list, CCI_l_list):
    #             print(fut_name)
    #
    #             data_df = test_fun(fut_name, CCI_window, CCI_limit, hold_time)
    #             # part_data_df = data_df
    #             # col_name = 'Time'
    #             # raw_return_df = part_data_df['pnl']
    #             # signal_df = part_data_df[col_name].shift(2).replace(np.nan, '00:00')
    #             # SignalAnalysis.CDF_c(signal_df, raw_return_df, hold_time=1,
    #             #                      title=f'{fut_name} {col_name} CDF Figure', lag=0)
    #             #
    #             # ana_fun(data_df, fut_name, 'OpenInterest_core')
    #             # ana_fun(data_df, fut_name, 'Volume_zscore')
    #             # ana_fun(data_df, fut_name, 'past_min_pct_change')
    #             # ana_fun(data_df, fut_name, 'trend_signal')
    #             pass


# @log.use_time
# def main_ana(fut_name_list, ban_name_list):
#     p_window = 1
#     # p_limit = 0.003
#     p_limit = 0
#     v_window = 20
#     v_limit = 0
#     ana_df_list = []
#     for fut_name in fut_name_list[:1]:
#         fut_name = 'IF'
#         pool = Pool(20)
#         result_list = []
#         if fut_name not in ban_name_list:
#             for con_id, part_info_df in fut_data.act_info_df[[f'{fut_name}01']].groupby(f'{fut_name}01'):
#                 args = [con_id, part_info_df.index[0], part_info_df.index[-1],
#                         p_window, p_limit, v_window, v_limit]
#                 # part_ana_df = part_test(*args)
#                 result_list.append(pool.apply_async(part_test, args=args))
#         pool.close()
#         pool.join()
#
#         ana_df = pd.concat([res.get() for res in result_list], axis=0)
#         ana_df_list.append(ana_df)
#
#     return ana_df_list


if __name__ == '__main__':
    root_path = '/mnt/mfs/DAT_FUT'
    fut_data = FutData(root_path)

    instrument_list = [
        'CU', 'ZN', 'AL', 'PB', 'AU', 'RB', 'RU', 'WR', 'FU', 'AG', 'BU', 'HC', 'NI', 'SN',
        'CF', 'SR', 'TA', 'WH', 'RI', 'JR', 'FG', 'OI', 'RM', 'RS', 'LR', 'SF', 'SM', 'MA',
        'ZC', 'CY', 'AP', 'A', 'B', 'C', 'J', 'L', 'M', 'P', 'V', 'Y', 'JD', 'JM', 'I',
        'FB', 'BB', 'PP', 'CS', 'SC', 'EG'
    ]
    error_list = [
        'JR', 'WR', 'FU', 'RI', 'LR', 'PB', 'CY', 'RS',
        'OI', 'ZC', 'SM', 'BB', 'FB', 'B',
    ]


    def get_active_df(data_name):
        root_path = '/mnt/mfs/dat_whs/DAT_FUT'
        active_df = bt.AZ_Load_csv(f'{root_path}/Active_{data_name}')
        adj_factor_df = bt.AZ_Load_csv(f'{root_path}/adj_factor_{data_name}')
        aadj_factor_df = adj_factor_df.cumprod()
        return active_df, aadj_factor_df


    active_df, aadj_factor_df = get_active_df('Volume')

    instrument_list = ['RB']
    data_df = main(instrument_list, error_list)

    # ana_fun(ana_df, 'RB')
    # for x, part_data_df in data_df.groupby(['weekday']):
    #     part_pnl = part_data_df['pnl'].fillna(0).values
    #     print(x, part_pnl.sum())
    #     plt.plot(part_pnl.cumsum())
    #     savfig_send(subject=f'{x}  {bt.AZ_Sharpe_y(part_pnl)}')
    #
    # for x, part_data_df in data_df.groupby(['month']):
    #     part_pnl = part_data_df['pnl'].fillna(0).values
    #     print(x, part_pnl.sum())
    #     plt.plot(part_pnl.cumsum())
    #     savfig_send(subject=f'{x}  {bt.AZ_Sharpe_y(part_pnl)}')
