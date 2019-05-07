import sys

sys.path.append('/mnf/mfs')
from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load import log
from work_whs.loc_lib.pre_load.plt import savfig_send
from work_whs.loc_lib.pre_load.senior_tools import SignalAnalysis
from work_whs.test_future.FutDataLoad import FutData, FutClass


def MA_LINE(Close, slowperiod, fastperiod):
    slow_line = bt.AZ_Rolling_mean(Close, slowperiod, min_periods=0)
    fast_line = bt.AZ_Rolling_mean(Close, fastperiod, min_periods=0)
    MA_diff = fast_line - slow_line
    target_df = MA_diff.copy()
    target_df[MA_diff > 0] = 1
    target_df[MA_diff < 0] = -1
    return target_df


# @log.try_catch
def part_test(con_id, begin_time, end_time, p_window, p_limit, v_window, v_limit):
    try:

        print(con_id)
        data_df = fut_data.load_intra_data(con_id, ['Close', 'Volume'])
        # begin_time = pd.to_datetime('20190101')
        # # end_time = begin_time + timedelta(1)
        # end_time = pd.to_datetime('20190401')

        # p_window = 1
        # p_limit = 1
        # v_window = 20
        # v_limit = 2

        part_data_df = data_df.truncate(before=begin_time, after=end_time)
        part_data_df['Volume_zscore'] = bt.AZ_Col_zscore(part_data_df[['Volume']], v_window)
        part_data_df['open_signal'] = part_data_df.apply(lambda x: 1 if x['Volume_zscore'] > v_limit and
                                                                        (((x['Time'] > '09:05')
                                                                          & (x['Time'] <= '11:25'))
                                                                         | ((x['Time'] > '13:35')
                                                                            & (x['Time'] <= '14:55'))
                                                                         | ((x['Time'] > '21:05')
                                                                            & (x['Time'] <= '22:55'))) else 0, axis=1)

        part_data_df['past_min_pct_change'] = (part_data_df['Close'] - part_data_df['Close'].shift(p_window)) \
                                              / part_data_df['Close'].shift(p_window)
        # part_data_df['way'] = (part_data_df['past_min_price_change'] > 0).astype(int).replace(0, -1)
        part_data_df['way'] = part_data_df['past_min_pct_change'].apply(
            lambda x: 0 if abs(x) < p_limit else (1 if x >= p_limit else -1))

        part_data_df['signal'] = part_data_df['open_signal'] * part_data_df['way'] * -1
        # part_data_df['signal'] = part_data_df['MA_LINE']
        part_data_df['position'] = bt.AZ_Rolling(part_data_df['signal'], 10, min_periods=0).sum()
        part_data_df['position'] = part_data_df['position']
        part_data_df['position_sft'] = part_data_df['position'].shift(2)
        part_data_df['price_return'] = part_data_df['Close'] - part_data_df['Close'].shift(1)
        part_data_df['price_return_sum'] = bt.AZ_Rolling(part_data_df['price_return'], 10).sum().shift(-9)
        part_data_df['pnl'] = part_data_df['position_sft'] * part_data_df['price_return']
        part_data_df['pnl_test'] = part_data_df['signal'] * part_data_df['price_return_sum'].shift(-2)

        part_data_df['turnover'] = (part_data_df['position_sft'] - part_data_df['position_sft'].shift(1)) \
                                   * part_data_df['Close']

        # 剔除开盘收盘5min的signal
        # plt.figure(figsize=[16, 8])
        # ax1 = plt.subplot(3, 1, 1)
        # ax2 = plt.subplot(3, 1, 2)
        # ax3 = plt.subplot(3, 1, 3)
        # ax1.plot(part_data_df['Close'].values)
        # ax2.bar(range(len(part_data_df.index)), part_data_df['Volume'].values)
        # ax3.plot(part_data_df['pnl'].cumsum().values)
        # ax1.grid()
        # ax2.grid()
        # ax3.grid()
        # plt.title(con_id)
        # savfig_send(con_id)
        part_ana_df = part_data_df[part_data_df['signal'] != 0][['signal', 'past_min_pct_change', 'pnl_test',
                                                                 'Volume', 'Volume_zscore', ]]
        part_pnl_df = part_data_df.groupby('Date')['pnl_test'].sum()
        part_turnover = part_data_df['turnover'].abs().sum()

        # return part_pnl_df, part_turnover
        return part_ana_df
    except Exception as error:
        print(error)
        return None


@log.use_time
def main(fut_name_list, ban_name_list):
    p_window = 1
    # p_limit = 0.003
    p_limit = 0
    v_window = 20
    v_limit = 2

    for fut_name in fut_name_list[:1]:
        fut_name = 'IF'
        pool = Pool(20)
        result_list = []
        if fut_name not in ban_name_list:
            for con_id, part_info_df in fut_data.act_info_df[[f'{fut_name}01']].groupby(f'{fut_name}01'):
                args = [con_id, part_info_df.index[0], part_info_df.index[-1],
                        p_window, p_limit, v_window, v_limit]
                # part_pnl_df, part_turnover = part_test(*args)
                result_list.append(pool.apply_async(part_test, args=args))

            pnl_df = pd.concat([res.get()[0] for res in result_list], axis=0)
            turnover = sum([res.get()[1] for res in result_list])
            pot = pnl_df.sum() / turnover * 10000
            sp = bt.AZ_Sharpe_y(pnl_df)
            print(pot, sp)

            plt.figure(figsize=[16, 8])
            pnl_df.index = pd.to_datetime(pnl_df.index)
            plt.plot(pnl_df.cumsum())
            plt.grid()
            savfig_send(f'{fut_name} sp:{sp} pot={pot}', text=f'p_window:{p_window}, p_limit:{p_limit}, '
                                                              f'v_window:{v_window}, v_limit:{v_limit}')
        # pool.close()
        # pool.join()


@log.use_time
def main_ana(fut_name_list, ban_name_list):
    p_window = 1
    # p_limit = 0.003
    p_limit = 0
    v_window = 20
    v_limit = 2
    ana_df_list = []
    for fut_name in fut_name_list[:1]:
        fut_name = 'IF'
        pool = Pool(20)
        result_list = []
        if fut_name not in ban_name_list:
            for con_id, part_info_df in fut_data.act_info_df[[f'{fut_name}01']].groupby(f'{fut_name}01'):
                args = [con_id, part_info_df.index[0], part_info_df.index[-1],
                        p_window, p_limit, v_window, v_limit]
                # part_ana_df = part_test(*args)
                result_list.append(pool.apply_async(part_test, args=args))
        pool.close()
        pool.join()

        ana_df = pd.concat([res.get() for res in result_list], axis=0)
        ana_df_list.append(ana_df)

    return ana_df_list


if __name__ == '__main__':
    root_path = '/mnt/mfs/DAT_FUT'
    fut_data = FutData(root_path)
    # fut_name_list = FutClass['黑色']
    # fut_name_list = FutClass['化工']
    # fut_name_list = FutClass['有色']
    # fut_name_list = FutClass['农产品']
    fut_name_list = FutClass['金融']
    #
    ban_name_list = ['WR', 'BB', 'ZC', 'SF', 'SM', 'FU', 'TA', 'SC', 'MA', 'OI', 'RS', 'IC']
    # main(fut_name_list, ban_name_list)
    ana_df_list = main_ana(fut_name_list, ban_name_list)
    ana_df = ana_df_list[0]
