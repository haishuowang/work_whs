import sys

sys.path.append('/mnf/mfs')
from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load import log
from work_whs.loc_lib.pre_load.plt import savfig_send
from work_whs.loc_lib.pre_load.senior_tools import SignalAnalysis
from work_whs.test_future.FutDataLoad import FutData, FutClass
from work_whs.test_future.signal_fut_fun import FutIndex, Signal, Position
import talib as ta


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
fut_data = FutData()

act_info_df = fut_data.act_info_df

# instrument_list = ['SC', 'C', 'J', 'JD']
# for fut_name in ['FG']:
for fut_name in instrument_list:
    c12 = act_info_df[[fut_name + '01', fut_name + '02']]

    OI_01 = fut_data.load_act_fut_data_r(fut_name, 'OpenInterest', act_info_df, active_num='01')
    OI_02 = fut_data.load_act_fut_data_r(fut_name, 'OpenInterest', act_info_df, active_num='02')
    OI_Totle = OI_02 + OI_01
    OI_Totle_MA = bt.AZ_Rolling_mean(OI_Totle, 20)
    OI_rt = OI_Totle / OI_Totle_MA
    cond1 = (OI_rt < OI_rt.shift(1)).astype(int)
    adj_r_01 = fut_data.load_act_fut_data_r(fut_name, 'adj_r', act_info_df, active_num='01').fillna(0)
    # Close_01 = fut_data.load_act_fut_data_r(fut_name, 'Close', act_info_df, active_num='01')

    # sig_rtn = adj_r_01  # - bt.AZ_Rolling_mean(adj_r_01, 5)
    sig_rtn = bt.AZ_Rolling_mean(adj_r_01, 5)
    # sig_rtn = Close_01/bt.AZ_Rolling_mean(Close_01, 5) - 1

    cond2 = sig_rtn.apply(lambda x: 0 if abs(x) == 0 else (1 if x > 0 else 0))
    signal = cond1 * cond2
    pnl_df = signal.shift(1) * adj_r_01
    asset_df = pnl_df.cumsum()
    SignalAnalysis.CDF(sig_rtn * cond1, adj_r_01, hold_time=1, title=f'{fut_name} CDF Figure', lag=1, zero_drop=True)

    plt.figure(figsize=[16, 10])
    plt.plot(asset_df.index, asset_df.values)
    plt.legend()
    plt.grid()
    savfig_send(fut_name)
    # Close = fut_data.load_fut_data(fut_name, 'Close')
    # OpenInterest = fut_data.load_fut_data(fut_name, 'OpenInterest')

# def part_test_fun(instrument):
#     adj_close_df = FutData().load_act_adj_fut_data(instrument, 'Close', active_df, aadj_factor_df)
#     adj_high_df = FutData().load_act_adj_fut_data(instrument, 'High', active_df, aadj_factor_df)
#     adj_low_df = FutData().load_act_adj_fut_data(instrument, 'Low', active_df, aadj_factor_df)
#
#     adj_return_df = adj_close_df/adj_close_df.shift(1) - 1
#
#     for n in window_list:
#         raw_factor = FutIndex.boll_fun(adj_close_df, n)
#         for limit in limit_list:
#             signal = Signal.fun_1(raw_factor, limit)
#             pos = Position.fun_1(signal)
#             pnl_df = (pos.shift(2) * adj_return_df).fillna(0)
#             sp = bt.AZ_Sharpe_y(pnl_df)
#             pot = bt.AZ_Pot(pos, pnl_df.sum())
#             print(instrument, sp, pot, n, limit)
#             if abs(sp) > 1.4:
#                 print('!!!!!!!!!!!!!!!!')
#
#
# window_list = [10, 20, 40, 100]
# limit_list = [1, 1.5, 2]
# for instrument in instrument_list:
#     # part_test_fun(instrument)
#     adj_close_df = FutData().load_act_adj_fut_data(instrument, 'Close', active_df, aadj_factor_df)
#     adj_high_df = FutData().load_act_adj_fut_data(instrument, 'High', active_df, aadj_factor_df)
#     adj_low_df = FutData().load_act_adj_fut_data(instrument, 'Low', active_df, aadj_factor_df)
#
#     adj_return_df = adj_close_df / adj_close_df.shift(1) - 1
#     month_signal = pd.DataFrame(adj_return_df.index.strftime('%m'), index=adj_return_df.index, columns=['month'])
#
#     for n in window_list:
#         raw_factor = FutIndex.boll_fun(adj_close_df, n)
#         for limit in limit_list:
#             signal = Signal.fun_1(raw_factor, limit)
#             pos = Position.fun_1(signal)
#             pnl_df = (pos.shift(2) * adj_return_df).fillna(0)
#
#             sp = bt.AZ_Sharpe_y(pnl_df)
#             pot = bt.AZ_Pot(pos, pnl_df.sum())
#             print(instrument, sp, pot, n, limit)
#             if abs(sp) > 1.4:
#                 print('!!!!!!!!!!!!!!!!')
#             print(f'sum:', sum(pnl_df))
#             if abs(sp) > 0.7:
#                 for x, y in month_signal.groupby(by=['month']):
#                     print(x, round(sum(pnl_df.loc[y.index]), 5))
#                 plt.plot(pnl_df.cumsum())
#                 savfig_send('{}, {}, {}, {}, {}'.format(instrument, sp, pot, n, limit))
