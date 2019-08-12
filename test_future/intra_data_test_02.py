import sys

sys.path.append('/mnf/mfs')
from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load import log
from work_whs.loc_lib.pre_load.plt import savfig_send
from work_whs.loc_lib.pre_load.senior_tools import SignalAnalysis
from work_whs.test_future.FutDataLoad import FutData, FutClass
from work_whs.test_future.signal_fut_fun import FutIndex, Signal, Position
import talib as ta


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
              # hold_time
              ):
    print(begin_time, end_time)
    try:
        # print(con_id, begin_time, end_time)
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
        part_data_df['weekday'] = pd.to_datetime(part_data_df['Date']).dt.weekday
        part_data_df['weekday_pos'] = (
                (part_data_df['weekday'] != 2)
                # & (part_data_df['weekday'] != 3)
            # & (part_data_df['weekday'] != 3)
            # (part_data_df['weekday'] != 4)
        ).astype(int)
        part_data_df['month'] = part_data_df['Date'].str.slice(5, 7)
        part_data_df['month_pos'] = ((part_data_df['month'] != '10') & (part_data_df['month'] != '12')).astype(int)

        # part_data_df['time_pos_cut'] = part_data_df.apply(lambda x: -1 if (((x['Time'] > '14:51')
        #                                                                     & (x['Time'] <= '15:00')
        #                                                                     # | (x['Time'] > '21:00')
        #                                                                     # & (x['Time'] <= '22:00')
        #                                                                     # | (x['Time'] > '21:00')
        #                                                                     # & (x['Time'] <= '22:00')
        #                                                                     )) else 1, axis=1)

        part_data_df['Volume_boll'] = FutIndex.boll_fun(part_data_df[['Volume']], CCI_window)
        part_data_df['past_roll_Volume'] = bt.AZ_Rolling_mean(part_data_df[['Volume']], CCI_window)
        v_window = 60
        part_data_df['Volume_zscore'] = bt.AZ_Col_zscore(part_data_df[['Volume']], CCI_window)
        part_data_df['Volume_signal'] = part_data_df['Volume_zscore'][part_data_df['Volume_zscore'] < 1].astype(int)
        # part_data_df['Volume_signal'] = (part_data_df['Volume_zscore'] < 2).astype(int).replace(0, -1)

        p_window = 5
        # part_data_df['past_min_pct_change'] = (part_data_df['Close'] - part_data_df['Close'].shift(p_window)) \
        #                                       / part_data_df['Close'].shift(p_window)

        part_data_df['past_min_pct_change'] = (part_data_df['Close'] / part_data_df['Close'].shift(p_window) - 1)
        part_data_df['past_min_pct_signal'] = part_data_df['past_min_pct_change'] \
            .apply(lambda x: 0 if abs(x) < 0.004 else (1 if x > 0.004 else -1))

        part_data_df['CCI_score'] = FutIndex.CCI_fun(part_data_df['High'],
                                                     part_data_df['Low'],
                                                     part_data_df['Close'], CCI_window)

        part_data_df['CCI_signal'] = Signal.fun_1(part_data_df['CCI_score'], CCI_limit)

        part_data_df['CCI_pos'] = Position.fun_1(part_data_df['CCI_signal'])

        part_data_df['boll_score'], ma_n, md_n = FutIndex.boll_fun(part_data_df['Close'], CCI_window, return_line=True)
        part_data_df['boll_signal'] = Signal.fun_1(part_data_df['boll_score'], CCI_limit)

        part_data_df['boll_pos'] = Position.fun_1(part_data_df['boll_signal'])
        # part_data_df['boll_pos'] = Position.fun_3(part_data_df['boll_signal'], part_data_df['past_min_pct_signal'])

        # part_data_df['trend_signal'] = bt.AZ_Rolling(part_data_df['Close'], 500).std()
        # part_data_df['trend_pos'] = (part_data_df['trend_signal'] < 40).astype(int)

        part_data_df['trend_indicator'] = bt.AZ_Rolling(part_data_df['Close'], CCI_window). \
            apply(lambda x: (x[-1] / x[0] - 1) / (max(x) / min(x) - 1), raw=False)
        # part_data_df['trend_pos'] = (part_data_df['trend_signal'] < 40).astype(int)

        # part_data_df['position'] = part_data_df['boll_pos'] * part_data_df['weekday_pos'] * part_data_df['month_pos']

        part_data_df['position'] = part_data_df['boll_pos'] # * part_data_df['trend_pos']

        part_data_df['position_sft'] = part_data_df['position'].shift(2) # * part_data_df['weekday_pos']).fillna(0)
        part_data_df['price_return'] = part_data_df['Close'] - part_data_df['Close'].shift(1)

        # part_data_df['price_return_sum'] = bt.AZ_Rolling(part_data_df['price_return'], hold_time).sum() \
        #     .shift(-hold_time + 1)
        part_data_df['pnl'] = part_data_df['position_sft'] * part_data_df['price_return']
        part_data_df['asset'] = part_data_df['pnl'].cumsum()
        # part_data_df['pnl_test'] = part_data_df['signal'] * part_data_df['price_return_sum'].shift(-2)

        part_data_df['turnover'] = (part_data_df['position_sft'] - part_data_df['position_sft'].shift(1)) \
                                   * part_data_df['Close']

        # macd, macdsignal, macdhist = ta.MACD(part_data_df['Close'], 12, 26, 9)
        # part_data_df['macd'] = macd
        RSI = ta.RSI(part_data_df['Close'], CCI_window)
        RSI = RSI - 50
        RSI[RSI > 20] = 20
        RSI[RSI < -20] = -20
        part_data_df['RSI'] = RSI
        part_data_df['RSI_signal'] = Signal.fun_1(part_data_df['RSI'], 1)
        part_data_df['RSI_pos'] = Position.fun_1(part_data_df['RSI_signal'], limit=60)

        aroondown, aroonup = ta.AROON(part_data_df['High'], part_data_df['Low'], CCI_window)

        obv = ta.OBV(part_data_df['Close'], part_data_df['Volume'])
        part_data_df['obv'] = obv

        atr = ta.ATR(part_data_df['High'], part_data_df['Low'], part_data_df['Close'], CCI_window)
        part_data_df['atr'] = atr

        adx = ta.ADX(part_data_df['High'], part_data_df['Low'], part_data_df['Close'], CCI_window)
        part_data_df['adx'] = adx

        trix = ta.TRIX(part_data_df['Close'], CCI_window)
        part_data_df['trix'] = trix

        willr = ta.WILLR(part_data_df['High'], part_data_df['Low'], part_data_df['Close'], CCI_window)
        part_data_df['willr'] = willr

        # 剔除开盘收盘5min的signal
        # if '1910' in con_id:
        plt.figure(figsize=[64, 64])
        ax1 = plt.subplot(4, 1, 1)
        ax2 = plt.subplot(4, 1, 2)
        ax3 = plt.subplot(4, 1, 3)
        ax1.plot(part_data_df['asset'].values)
        # ax2.plot(part_data_df['Close'].values, '--', color='#75bbfd')
        ax2.plot(np.array(range(len(part_data_df.index))), part_data_df['Close'], color='b', linestyle='--')

        ax2.scatter(np.array(range(len(part_data_df.index)))[(part_data_df['position_sft'] > 0)],
                    part_data_df['Close'][part_data_df['position_sft'] > 0], s=0.5, color='red')

        ax2.scatter(np.array(range(len(part_data_df.index)))[(part_data_df['position_sft'] == 0)],
                    part_data_df['Close'][part_data_df['position_sft'] == 0], s=0.5, color='black')

        ax2.scatter(np.array(range(len(part_data_df.index)))[(part_data_df['position_sft'] < 0)],
                    part_data_df['Close'][part_data_df['position_sft'] < 0], s=0.5, color='blue')

        ax2.scatter(np.array(range(len(part_data_df.index)))[(part_data_df['past_min_pct_signal'] > 0)],
                    part_data_df['Close'][part_data_df['past_min_pct_signal'] > 0], s=20, color='y')
        ax2.scatter(np.array(range(len(part_data_df.index)))[(part_data_df['past_min_pct_signal'] < 0)],
                    part_data_df['Close'][part_data_df['past_min_pct_signal'] < 0], s=20, color='#f504c9')

        ax3.bar(range(len(part_data_df.index)), part_data_df['Volume'].values)

        ax1.grid()
        ax2.grid()
        ax3.grid()
        plt.title(con_id)
        savfig_send(con_id)
        #
        part_pnl_df = part_data_df.groupby('Date')['pnl'].sum()
        part_turnover = part_data_df.groupby('Date')['turnover'].apply(lambda x: sum(abs(x)))
        return part_pnl_df, part_turnover, part_data_df
    except Exception as error:
        print(error)
        return pd.Series(), pd.Series(), pd.Series()


def test_fun(fut_name, CCI_window, CCI_limit):
    result_list = []
    pool = Pool(20)
    for con_id, part_info_df in active_df[[f'{fut_name}01']].groupby(f'{fut_name}01'):
        args = [con_id, part_info_df.index[0] - timedelta(1), part_info_df.index[-1] + timedelta(1),
                CCI_window, CCI_limit]
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

    plt.figure(figsize=[16, 8])
    pnl_df.index = pd.to_datetime(pnl_df.index)
    plt.plot(pnl_df.cumsum())
    plt.grid()
    savfig_send(f'{fut_name} sp:{sp} pot={pot} CCI_window:{CCI_window}, CCI_limit:{CCI_limit}')
    return data_df


def ana_fun(data_df, fut_name, col_name):
    raw_return_df = data_df['price_return']
    signal_df = data_df[col_name]
    SignalAnalysis.CDF(signal_df, raw_return_df, hold_time=1, title=f'{fut_name} {col_name} CDF Figure', lag=2)


@log.use_time
def main(fut_name_list, ban_name_list):
    CCI_w_list = [20, 60, 120, 180, 300, 400]
    CCI_l_list = [1, 1.5, 2, 2.5, 3]
    CCI_limit = 1
    # CCI_w_list = []
    # CCI_l_list = [1, 1.5, 2, 2.5, 3]

    # for fut_name in fut_name_list:
    #     data_df = test_fun(fut_name, 300, 1.5)
    #     part_data_df = data_df
    #     col_name = 'Time'
    #     raw_return_df = part_data_df['pnl']
    #     signal_df = part_data_df[col_name].shift(2).replace(np.nan, '00:00')
    #     SignalAnalysis.CDF_c(signal_df, raw_return_df, hold_time=1,
    #                          title=f'{fut_name} {col_name} CDF Figure', lag=0)
    #     # ana_fun(data_df, fut_name, 'Time')

    for fut_name in fut_name_list:
        for CCI_window in CCI_w_list:
            print(fut_name)
            if fut_name in ban_name_list:
                continue
            data_df = test_fun(fut_name, CCI_window, CCI_limit)

            ana_fun(data_df, fut_name, 'boll_score')
            ana_fun(data_df, fut_name, 'CCI_score')

            ana_fun(data_df, fut_name, 'RSI')
            ana_fun(data_df, fut_name, 'obv')
            ana_fun(data_df, fut_name, 'atr')
            ana_fun(data_df, fut_name, 'adx')

            ana_fun(data_df, fut_name, 'trix')
            ana_fun(data_df, fut_name, 'willr')


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

def single_main():
    fut_name, CCI_window, CCI_limit = 'SC', 30, 1
    data_df = test_fun(fut_name, CCI_window, CCI_limit)

    ana_fun(data_df, fut_name, 'boll_score')
    ana_fun(data_df, fut_name, 'CCI_score')
    ana_fun(data_df, fut_name, 'OpenInterest_core')
    ana_fun(data_df, fut_name, 'Volume_zscore')
    ana_fun(data_df, fut_name, 'Volume_boll')
    ana_fun(data_df, fut_name, 'past_min_pct_change')
    ana_fun(data_df, fut_name, 'trend_indicator')
    return data_df


def get_active_df(data_name):
    root_path = '/mnt/mfs/dat_whs/DAT_FUT'
    active_df = bt.AZ_Load_csv(f'{root_path}/Active_{data_name}')
    adj_factor_df = bt.AZ_Load_csv(f'{root_path}/adj_factor_{data_name}')
    aadj_factor_df = adj_factor_df.cumprod()
    return active_df, aadj_factor_df


if __name__ == '__main__':
    root_path = '/mnt/mfs/DAT_FUT'
    fut_data = FutData(root_path)
    # fut_name_list = FutClass['黑色']
    fut_name_list = FutClass['化工']
    # fut_name_list = FutClass['有色']
    # fut_name_list = FutClass['农产品']
    # fut_name_list = FutClass['金融']
    #
    # fut_name_list = ['RB']

    instrument_list = [
        'CU', 'ZN', 'AL', 'PB', 'AU', 'RB', 'RU', 'WR', 'FU', 'AG', 'BU', 'HC', 'NI', 'SN',
        'CF', 'SR', 'TA', 'WH', 'RI', 'JR', 'FG', 'OI', 'RM', 'RS', 'LR', 'SF', 'SM', 'MA',
        'ZC', 'CY', 'AP', 'A', 'B', 'C', 'J', 'L', 'M', 'P', 'V', 'Y', 'JD', 'JM', 'I',
        'FB', 'BB', 'PP', 'CS', 'SC', 'EG'
    ]

    active_df, aadj_factor_df = get_active_df('Volume')

    cut_date = '20150101'
    active_df = active_df.truncate(before=pd.to_datetime(cut_date))
    print(active_df)
    # instrument_list = ['SC']
    instrument_list = ['RB', 'HC', 'JM', 'J', 'ZN', 'SC', 'BU', 'RU', 'MA', 'I']
    error_list = [
        'JR', 'WR', 'FU', 'RI', 'LR', 'PB', 'CY', 'RS',
        'OI', 'ZC', 'SM', 'BB', 'FB', 'B',
    ]
    # main(fut_name_list, ban_name_list)
    data_df = main(instrument_list, error_list)

    # data_df = single_main()
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
