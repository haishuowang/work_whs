import sys

sys.path.append('/mnt/mfs')
from work_dmgr_fut.loc_lib.pre_load import *

# now_date = datetime.now().strftime('%Y%m%d')
# root_path = '/mnt/mfs/DAT_EQT'
# data_path = f'{root_path}/intraday/eqt_1mbar/{now_date[:4]}/{now_date[:6]}/{now_date}'
#
# stock_intra_df = pd.read_csv(f'{data_path}/Close.csv', index_col=0, parse_dates=True)
# stock_intra_df.columns = bt.AZ_clear_columns(stock_intra_df.columns)
#
# stock_daily_df = pd.read_csv(f'{root_path}/EM_Funda/TRAD_SK_DAILY_JC/NEW.csv', sep='|', index_col=0, parse_dates=True)
from work_dmgr_fut.fut_script.FutDataLoad import FutData, ReshapeData

fut_data = FutData()
a = fut_data.act_info_df
x = a['AG01']


# import sys
#
# sys.path.append('/mnt/mfs')
# from work_dmgr_fut.loc_lib.pre_load import *
# from work_dmgr_fut.loc_lib.pre_load.plt import savfig_send
#
# from work_dmgr_fut.fut_script.FutDataLoad import FutData, FutClass
# from work_dmgr_fut.fut_script.signal_fut_fun import FutIndex, Signal, Position
#
#
# def MA_LINE(Close, slowperiod, fastperiod):
#     slow_line = bt.AZ_Rolling_mean(Close, slowperiod, min_periods=0)
#     fast_line = bt.AZ_Rolling_mean(Close, fastperiod, min_periods=0)
#     MA_diff = fast_line - slow_line
#     target_df = MA_diff.copy()
#     target_df[MA_diff > 0] = 1
#     target_df[MA_diff < 0] = -1
#     return target_df
#
#
# def mul_fun(a, b):
#     a_l = a.where(a > 0, 0)
#     a_s = a.where(a < 0, 0)
#
#     b_l = b.where(b > 0, 0)
#     b_s = b.where(b < 0, 0)
#
#     pos_l = a_l.mul(b_l)
#     pos_s = a_s.mul(b_s)
#
#     pos = pos_l.sub(pos_s)
#     return pos
#
#
# # @log.try_catch
# def part_test(con_id, begin_time, end_time, CCI_window, CCI_limit):
#     try:
#         data_df = fut_data.load_intra_data(con_id, ['High', 'Low', 'Close', 'Volume'])
#
#         part_data_df = data_df.truncate(before=begin_time, after=end_time)
#
#         part_data_df['weekday'] = pd.to_datetime(part_data_df['Date']).dt.weekday
#         part_data_df['weekday_pos'] = (
#                 (part_data_df['weekday'] != 2)
#                 & (part_data_df['weekday'] != 1)
#             # & (part_data_df['weekday'] != 3)
#             # (part_data_df['weekday'] != 4)
#         ).astype(int).shift(-2)
#
#         part_data_df['boll_score'] = FutIndex.boll_fun(part_data_df['Close'], CCI_window)
#         part_data_df['boll_signal'] = Signal.fun_1(part_data_df['boll_score'], CCI_limit)
#         part_data_df['boll_pos'] = Position.fun_1(part_data_df['boll_signal'])
#
#         part_data_df['trend_signal'] = bt.AZ_Rolling(part_data_df['Close'], 500).std()
#         part_data_df['trend_pos'] = (part_data_df['trend_signal'] < 20).astype(int)
#
#         part_data_df['position'] = part_data_df['boll_pos'] * part_data_df['trend_pos'] * part_data_df['weekday_pos']
#         part_data_df['position_exe'] = part_data_df['position'].shift(1).fillna(0)
#         part_data_df['position_sft'] = part_data_df['position'].shift(2).fillna(0)
#         part_data_df['price_return'] = part_data_df['Close'] / part_data_df['Close'].shift(1) - 1
#
#         part_data_df['pnl'] = part_data_df['position_sft'] * part_data_df['price_return']
#         part_data_df['asset'] = part_data_df['pnl'].cumsum()
#
#         part_data_df['turnover'] = (part_data_df['position_sft'] - part_data_df['position_sft'].shift(1))
#         part_data_df['con_id'] = con_id
#         part_pnl_df = part_data_df.groupby('Date')['pnl'].sum()
#         part_turnover = part_data_df.groupby('Date')['turnover'].apply(lambda x: sum(abs(x)))
#         return part_pnl_df, part_turnover, part_data_df
#     except Exception as error:
#         print(error)
#         return None
#
#
# def test_fun(fut_name, CCI_window, CCI_limit):
#     result_list = []
#     pool = Pool(20)
#     for con_id, part_info_df in fut_data.act_info_df[[f'{fut_name}01']].groupby(f'{fut_name}01'):
#         args = [con_id, part_info_df.index[0] - timedelta(1), part_info_df.index[-1] + timedelta(1),
#                 CCI_window, CCI_limit]
#         result_list.append(pool.apply_async(part_test, args=args))
#     pool.close()
#     pool.join()
#
#     pnl_df = pd.concat([res.get()[0] for res in result_list], axis=0)
#     turnover_df = pd.concat([res.get()[1] for res in result_list], axis=0)
#     data_df = pd.concat([res.get()[2] for res in result_list], axis=0)
#
#     pot = pnl_df.sum() / turnover_df.sum() * 10000
#     sp = bt.AZ_Sharpe_y(pnl_df)
#     print(fut_name, sp, pot, CCI_window, CCI_limit)
#
#     plt.figure(figsize=[16, 10])
#     pnl_df.index = pd.to_datetime(pnl_df.index)
#     plt.plot(pnl_df.cumsum())
#     plt.grid()
#     savfig_send(f'{fut_name} sp:{sp} pot={pot} CCI_window:{CCI_window}, CCI_limit:{CCI_limit}')
#     return data_df
#
#
# def main():
#     fut_name = 'J'
#     data_df = test_fun(fut_name, 300, 1.5)
#     plt.plot(data_df['pnl'].cumsum())
#     savfig_send()
#     exe_data_df = data_df.loc[data_df['position'].diff().replace(0, np.nan).dropna().index]
#     pos_df = exe_data_df.groupby(by=['TradeDate', 'con_id'])['position'].last().unstack().fillna(0)
#     all_con_list = list(pos_df.columns)
#     pos_df['e_time'] = pos_df.index.strftime('%Y-%m-%d %H:%M')
#     pos_df['w_time'] = datetime.now().strftime('%Y-%m-%d %H:%M')
#     pos_df = pos_df[['w_time', 'e_time'] + all_con_list]
#     pos_df.to_csv('/mnt/mfs/AAFUTPOS/BKT/WHSFUTTST01.pos', sep='|', index=None)
#     print(pos_df.tail())
#     return pos_df, data_df


#
#
# def update_fun(con_id, begin_time, end_time, boll_window, boll_limit):
#     part_pnl_df, part_turnover, part_data_df = part_test(con_id, begin_time, end_time, boll_window, boll_limit)
#     exe_data_df = data_df.loc[data_df['position'].diff().replace(0, np.nan).dropna().index]
#     pos_df = exe_data_df.groupby(by=['TradeDate', 'con_id'])['position_exe'].last().unstack().fillna(0)
#     all_con_list = list(pos_df.columns)
#     pos_df['e_time'] = pos_df.index
#     pos_df['w_time'] = datetime.now()
#     pos_df = pos_df[['w_time', 'e_time'] + all_con_list]
#
#
# if __name__ == '__main__':
#     root_path = '/mnt/mfs/DAT_FUT'
#     fut_data = FutData(root_path)
#     pos_df, data_df = main()

import sys

sys.path.append('/mnt/mfs')
from work_dmgr_fut.loc_lib.pre_load import *
from work_dmgr_fut.fut_script.signal_fut_fun import FutIndex, Signal, Position
from work_dmgr_fut.fut_script.WHSCOMMON00 import FutMinPosBase


class FutMinPos(FutMinPosBase):
    def __init__(self,  *args, **kwargs):
        super(FutMinPos, self).__init__(*args, **kwargs)
        self.alpha_name = os.path.basename(__file__).split('.')[0]

    def part_generation(self, fut_name, con_id, window, limit, update_live_data=False):
        try:
            old_min_trade_time = self.load_intra_time(con_id)

            part_data_df, active_begin, active_end = self.get_old_new_data(con_id, update_live_data)
            # part_data_df_2, active_begin, active_end = self.get_old_new_data('AU' + con_id[2:], update_live_data)[self.usecols_list]

            # part_data_df = pd.concat([part_data_df_1, part_data_df_2], axis=1)
            all_min_trade_time = list(old_min_trade_time) + \
                                 list(self.live_time_dict[fut_name][self.live_time_dict[fut_name]
                                                                    > old_min_trade_time[-1]])
            all_min_trade_time_c = pd.Series(all_min_trade_time, index=all_min_trade_time).apply(lambda x: x if
            x.strftime('%H:%M') not in self.end_time_dict[fut_name] else None).fillna(method='bfill').shift(-self.lag)

            part_data_df['exe_time'] = all_min_trade_time_c.reindex(part_data_df.index)
            all_min_df = self.load_intra_data(con_id, ['Close'])
            if update_live_data:
                part_data_df['next_trade_close'] = 0
            else:
                next_trade_close = all_min_df['Close'].shift(-1).loc[part_data_df['exe_time'].values]\
                    .fillna(method='ffill')
                part_data_df['next_trade_close'] = next_trade_close.values

            part_pnl_df, part_turnover, part_data_df = self.run(part_data_df, con_id, window, limit, active_begin)
            return part_pnl_df, part_turnover, part_data_df
        except Exception as error:
            print(error)
            return pd.Series(), pd.Series(), pd.Series()

    @staticmethod
    def run(part_data_df, contract_id, window, limit, active_begin=None):
        # part_data_df['weekday'] = pd.to_datetime(part_data_df['Date']).dt.weekday
        # part_data_df['weekday_pos'] = (
        #         (part_data_df['weekday'] != 2)
        #         & (part_data_df['weekday'] != 1)
        # ).astype(int)
        #
        # part_data_df['boll_score'] = FutIndex.boll_fun(part_data_df['Close'], window)
        # part_data_df['boll_signal'] = Signal.fun_1(part_data_df['boll_score'], limit)
        # part_data_df['boll_pos'] = Position.fun_1(part_data_df['boll_signal'])
        #
        # part_data_df['trend_signal'] = bt.AZ_Rolling(part_data_df['Close'], 500).std()
        # part_data_df['trend_pos'] = (part_data_df['trend_signal'] < 20).astype(int)
        #
        # part_data_df['position'] = part_data_df['boll_pos'] * part_data_df['trend_pos'] * part_data_df['weekday_pos']
        # part_data_df['position_exe'] = part_data_df['position'].shift(1).fillna(0)
        # part_data_df['position_sft'] = part_data_df['position'].shift(2).fillna(0)
        # # part_data_df['price_return'] = part_data_df['Close'] / part_data_df['Close'].shift(1) - 1
        # part_data_df['price_return'] = part_data_df['next_trade_close'] / part_data_df['next_trade_close'].shift(1) - 1
        #
        # part_data_df['pnl'] = part_data_df['position_exe'] * part_data_df['price_return']
        #
        # part_data_df = part_data_df.truncate(before=active_begin)
        #
        # part_data_df['asset'] = part_data_df['pnl'].cumsum()
        #
        # part_data_df['turnover'] = (part_data_df['position_sft'] - part_data_df['position_sft'].shift(1))
        # part_data_df['con_id'] = contract_id
        # part_pnl_df = part_data_df.groupby('Date')['pnl'].sum()
        # part_turnover = part_data_df.groupby('Date')['turnover'].apply(lambda x: sum(abs(x)))
        # return part_pnl_df, part_turnover, part_data_df
        return part_data_df, pd.Series(), pd.Series()


if __name__ == '__main__':
    # usecols_list = ['High', 'Low', 'Close', 'Volume', 'OpenInterest']
    usecols_list = ['Close']
    fut_name, window, limit = 'AG', 300, 1.5
    fut_min_pos = FutMinPos([fut_name], usecols_list)

    # part_pnl_df, part_turnover, part_data_df = fut_min_pos.update('J', window, limit)
    data_df_2, sp, pot = fut_min_pos.generation(fut_name, window, limit, concat=False)
    # part_pnl_df, part_turnover, part_data_df = fut_min_pos.update(fut_name, window, limit, update_live_data=True)
    # fut_min_pos.loop(fut_name, window, limit)
