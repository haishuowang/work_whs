import sys

sys.path.append('/mnf/mfs')
from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load import log
from work_whs.loc_lib.pre_load.plt import savfig_send
from work_whs.loc_lib.pre_load.senior_tools import SignalAnalysis
from work_whs.test_future.FutDataLoad import FutData, FutClass
from work_whs.test_future.signal_fut_fun import FutIndex, Signal, Position


def get_active_df(data_name):
    root_path = '/mnt/mfs/dat_whs/DAT_FUT'
    active_df = bt.AZ_Load_csv(f'{root_path}/Active_{data_name}')
    adj_factor_df = bt.AZ_Load_csv(f'{root_path}/adj_factor_{data_name}')
    aadj_factor_df = adj_factor_df.cumprod()
    return active_df, aadj_factor_df


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
instrument_list = sorted(list(set(instrument_list) - set(error_list)))

data_name = 'OpenInterest'
active_df, aadj_factor_df = get_active_df(data_name)

# FutIndex.CCI_fun(high, low, close, n)
# FutIndex.boll_fun(close, n)


def part_test_fun(instrument):
    adj_close_df = FutData().load_act_adj_fut_data(instrument, 'Close', active_df, aadj_factor_df)
    adj_high_df = FutData().load_act_adj_fut_data(instrument, 'High', active_df, aadj_factor_df)
    adj_low_df = FutData().load_act_adj_fut_data(instrument, 'Low', active_df, aadj_factor_df)

    adj_return_df = adj_close_df/adj_close_df.shift(1) - 1

    for n in window_list:
        raw_factor = FutIndex.boll_fun(adj_close_df, n)
        for limit in limit_list:
            signal = Signal.fun_1(raw_factor, limit)
            pos = Position.fun_1(signal)
            pnl_df = (pos.shift(2) * adj_return_df).fillna(0)
            sp = bt.AZ_Sharpe_y(pnl_df)
            pot = bt.AZ_Pot(pos, pnl_df.sum())
            print(instrument, sp, pot, n, limit)
            if abs(sp) > 1.4:
                print('!!!!!!!!!!!!!!!!')


window_list = [10, 20, 40, 100]
limit_list = [1, 1.5, 2]
for instrument in instrument_list:
    # part_test_fun(instrument)
    adj_close_df = FutData().load_act_adj_fut_data(instrument, 'Close', active_df, aadj_factor_df)
    adj_high_df = FutData().load_act_adj_fut_data(instrument, 'High', active_df, aadj_factor_df)
    adj_low_df = FutData().load_act_adj_fut_data(instrument, 'Low', active_df, aadj_factor_df)

    adj_return_df = adj_close_df / adj_close_df.shift(1) - 1
    month_signal = pd.DataFrame(adj_return_df.index.strftime('%m'), index=adj_return_df.index, columns=['month'])

    for n in window_list:
        raw_factor = FutIndex.boll_fun(adj_close_df, n)
        for limit in limit_list:
            signal = Signal.fun_1(raw_factor, limit)
            pos = Position.fun_1(signal)
            pnl_df = (pos.shift(2) * adj_return_df).fillna(0)

            sp = bt.AZ_Sharpe_y(pnl_df)
            pot = bt.AZ_Pot(pos, pnl_df.sum())
            print(instrument, sp, pot, n, limit)
            if abs(sp) > 1.4:
                print('!!!!!!!!!!!!!!!!')
            print(f'sum:', sum(pnl_df))
            if abs(sp) > 0.7:
                for x, y in month_signal.groupby(by=['month']):
                    print(x, round(sum(pnl_df.loc[y.index]), 5))
                plt.plot(pnl_df.cumsum())
                savfig_send('{}, {}, {}, {}, {}'.format(instrument, sp, pot, n, limit))
