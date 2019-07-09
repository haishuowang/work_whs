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


# @log.try_catch
def part_test(con_id, begin_time, end_time, CCI_window, CCI_limit):
    try:
        data_df = fut_data.load_intra_data(con_id, ['High', 'Low', 'Close', 'Volume'])

        part_data_df = data_df.truncate(before=begin_time, after=end_time)

        part_data_df['weekday'] = pd.to_datetime(part_data_df['Date']).dt.weekday
        part_data_df['weekday_pos'] = (
                (part_data_df['weekday'] != 2)
                & (part_data_df['weekday'] != 1)
            # & (part_data_df['weekday'] != 3)
            # (part_data_df['weekday'] != 4)
        ).astype(int).shift(-2)

        part_data_df['boll_score'] = FutIndex.boll_fun(part_data_df['Close'], CCI_window)
        part_data_df['boll_signal'] = Signal.fun_1(part_data_df['boll_score'], CCI_limit)
        part_data_df['boll_pos'] = Position.fun_1(part_data_df['boll_signal'])

        part_data_df['trend_signal'] = bt.AZ_Rolling(part_data_df['Close'], 500).std()
        part_data_df['trend_pos'] = (part_data_df['trend_signal'] < 20).astype(int)

        part_data_df['position'] = part_data_df['boll_pos'] * part_data_df['trend_pos'] * part_data_df['weekday_pos']
        part_data_df['position_exe'] = part_data_df['position'].shift(1).fillna(0)
        part_data_df['position_sft'] = part_data_df['position'].shift(2).fillna(0)
        part_data_df['price_return'] = part_data_df['Close'] / part_data_df['Close'].shift(1) - 1

        part_data_df['pnl'] = part_data_df['position_sft'] * part_data_df['price_return']
        part_data_df['asset'] = part_data_df['pnl'].cumsum()

        part_data_df['turnover'] = (part_data_df['position_sft'] - part_data_df['position_sft'].shift(1))
        part_data_df['con_id'] = con_id
        part_pnl_df = part_data_df.groupby('Date')['pnl'].sum()
        part_turnover = part_data_df.groupby('Date')['turnover'].apply(lambda x: sum(abs(x)))
        # 剔除开盘收盘5min的signal
        plt.figure(figsize=[64, 64])
        ax1 = plt.subplot(4, 1, 1)
        ax2 = plt.subplot(4, 1, 2)
        ax3 = plt.subplot(4, 1, 3)
        ax1.plot(part_data_df['asset'].values)
        # ax2.plot(part_data_df['Close'].values, '--', color='#75bbfd')
        ax2.scatter(np.array(range(len(part_data_df.index)))[(part_data_df['position_sft'] > 0)],
                    part_data_df['Close'][part_data_df['position_sft'] > 0], s=0.5, color='red')

        ax2.scatter(np.array(range(len(part_data_df.index)))[(part_data_df['position_sft'] == 0)],
                    part_data_df['Close'][part_data_df['position_sft'] == 0], s=0.5, color='black')

        ax2.scatter(np.array(range(len(part_data_df.index)))[(part_data_df['position_sft'] < 0)],
                    part_data_df['Close'][part_data_df['position_sft'] < 0], s=0.5, color='blue')
        # ax2.scatter(np.array(range(len(part_data_df.index)))[(part_data_df['past_min_pct_signal'] > 0)],
        #             part_data_df['Close'][part_data_df['past_min_pct_signal'] > 0], s=20, color='black')
        # ax2.scatter(np.array(range(len(part_data_df.index)))[(part_data_df['past_min_pct_signal'] < 0)],
        #             part_data_df['Close'][part_data_df['past_min_pct_signal'] < 0], s=20, color='y')

        ax3.bar(range(len(part_data_df.index)), part_data_df['Volume'].values)

        ax1.grid()
        ax2.grid()
        ax3.grid()
        plt.title(con_id)

        savfig_send(con_id)
        return part_pnl_df, part_turnover, part_data_df
    except Exception as error:
        print(error)
        return None


def test_fun(fut_name, CCI_window, CCI_limit):
    result_list = []
    pool = Pool(20)
    for con_id, part_info_df in fut_data.act_info_df[[f'{fut_name}01']].groupby(f'{fut_name}01'):
        args = [con_id, part_info_df.index[0] - timedelta(1), part_info_df.index[-1] + timedelta(1),
                CCI_window, CCI_limit]
        result_list.append(pool.apply_async(part_test, args=args))
    pool.close()
    pool.join()

    pnl_df = pd.concat([res.get()[0] for res in result_list], axis=0)
    turnover_df = pd.concat([res.get()[1] for res in result_list], axis=0)
    data_df = pd.concat([res.get()[2] for res in result_list], axis=0)

    pot = pnl_df.sum() / turnover_df.sum() * 10000
    sp = bt.AZ_Sharpe_y(pnl_df)
    print(fut_name, sp, pot, CCI_window, CCI_limit)

    plt.figure(figsize=[16, 10])
    pnl_df.index = pd.to_datetime(pnl_df.index)
    plt.plot(pnl_df.cumsum())
    plt.grid()
    savfig_send(f'{fut_name} sp:{sp} pot={pot} CCI_window:{CCI_window}, CCI_limit:{CCI_limit}')
    return data_df


# @log.use_time
# def update_fun(con_id, begin_time, end_time, boll_window, boll_limit):
#     data_df = fut_data.load_intra_data(con_id, ['High', 'Low', 'Close', 'Volume'])
#
#     part_data_df = data_df.truncate(before=begin_time, after=end_time)
#
#     part_data_df['weekday'] = pd.to_datetime(part_data_df['Date']).dt.weekday
#     part_data_df['weekday_pos'] = (
#             (part_data_df['weekday'] != 2)
#             & (part_data_df['weekday'] != 1)
#     ).astype(int).shift(-2)
#
#     part_data_df['boll_score'] = FutIndex.boll_fun(part_data_df['Close'], boll_window)
#     part_data_df['boll_signal'] = Signal.fun_1(part_data_df['boll_score'], boll_limit)
#     part_data_df['boll_pos'] = Position.fun_1(part_data_df['boll_signal'])
#
#     part_data_df['trend_signal'] = bt.AZ_Rolling(part_data_df['Close'], 500).std()
#     part_data_df['trend_pos'] = (part_data_df['trend_signal'] < 20).astype(int)
#
#     part_data_df['position'] = part_data_df['boll_pos'] * part_data_df['trend_pos'] * part_data_df['weekday_pos']
#     part_data_df['position_exe'] = part_data_df['position'].shift(1).fillna(0)
#     part_data_df['position_sft'] = part_data_df['position'].shift(2).fillna(0)
#     part_data_df['price_return'] = part_data_df['Close'] - part_data_df['Close'].shift(1)
#
#     part_data_df['pnl'] = part_data_df['position_sft'] * part_data_df['price_return']
#     part_data_df['asset'] = part_data_df['pnl'].cumsum()
#
#     part_data_df['turnover'] = (part_data_df['position_sft'] -
#                                 part_data_df['position_sft'].shift(1)) * part_data_df['Close']
#     part_data_df['con_id'] = con_id
#     part_pnl_df = part_data_df.groupby('Date')['pnl'].sum()
#     part_turnover = part_data_df.groupby('Date')['turnover'].apply(lambda x: sum(abs(x)))
#     return part_pnl_df, part_turnover, part_data_df


@log.use_time
def main():
    fut_name = 'J'
    data_df = test_fun(fut_name, 300, 1.5)
    exe_data_df = data_df.loc[data_df['position'].diff().replace(0, np.nan).dropna().index]
    pos_df = exe_data_df.groupby(by=['TradeDate', 'con_id'])['position'].last().unstack().fillna(0)
    all_con_list = list(pos_df.columns)
    # pos_df['e_time'] = pos_df.index.strftime('%Y%m%d %H:%M')
    # pos_df['w_time'] = datetime.now().strftime('%Y%m%d %H:%M')
    # pos_df = pos_df[['w_time', 'e_time'] + all_con_list]
    # pos_df.to_csv('/mnt/mfs/AAFUTPOS/BKT/WHSFUTTST01.pos', sep='|', index=None)
    print(pos_df)
    return pos_df, data_df


@log.use_time
def update_fun(con_id, begin_time, end_time, boll_window, boll_limit):
    part_pnl_df, part_turnover, part_data_df = part_test(con_id, begin_time, end_time, boll_window, boll_limit)
    exe_data_df = data_df.loc[data_df['position'].diff().replace(0, np.nan).dropna().index]
    pos_df = exe_data_df.groupby(by=['TradeDate', 'con_id'])['position_exe'].last().unstack().fillna(0)
    all_con_list = list(pos_df.columns)
    pos_df['e_time'] = pos_df.index
    pos_df['w_time'] = datetime.now()
    pos_df = pos_df[['w_time', 'e_time'] + all_con_list]


if __name__ == '__main__':
    root_path = '/mnt/mfs/DAT_FUT'
    fut_data = FutData(root_path)
    pos_df, data_df = main()
