import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *


def row_extre(raw_df, percent):
    raw_df = raw_df
    target_df = raw_df.rank(axis=1, pct=True)
    target_df[target_df >= 1 - percent] = 1
    target_df[target_df <= percent] = -1
    target_df[(target_df > percent) & (target_df < 1 - percent)] = 0
    return target_df.loc[0]


# def part_get_etf_data(day_path, day):
#     print(day)
#     target_list = ['SH510050', 'SH510300', 'SH510500']
#     close = pd.read_csv(os.path.join(day_path, 'Close.csv'), index_col=0).astype(float)
#     close.index = day + ' ' + close.index
#     part_df = close[target_list]
#     return part_df


def signal_fun(file_name, limit, n):
    raw_df = bt.AZ_Load_csv(f'/mnt/mfs/dat_whs/{file_name}')
    tmp_df = bt.AZ_Col_zscore(raw_df, n, min_periods=0)
    tmp_df[(tmp_df < limit)] = 0
    tmp_df[tmp_df >= limit] = 1
    # tmp_df[tmp_df <= -limit] = -1
    return tmp_df


def signal_fun_dn(file_name, limit, n):
    raw_df = bt.AZ_Load_csv(f'/mnt/mfs/dat_whs/{file_name}')
    tmp_df = bt.AZ_Col_zscore(raw_df, n, min_periods=0)
    tmp_df[(tmp_df > limit)] = 0
    tmp_df[tmp_df <= limit] = 1
    # tmp_df[tmp_df <= -limit] = -1
    return tmp_df


def find_target_time(dn_pct, up_pct, return_intra):
    tmp_mask = ((return_intra > dn_pct) & (return_intra < up_pct)).astype(int).replace(0, np.nan)
    tmp_mask = tmp_mask.fillna(method='ffill').fillna(0)
    mask_diff = tmp_mask - tmp_mask.shift(1)
    return mask_diff[mask_diff > 0]


def part_test_fun(day_path, day, pre_close, adj_factor, sector, signal, bar_num):
    print(day)
    close = pd.read_csv(os.path.join(day_path, 'Close.csv'), index_col=0).astype(float)

    close.columns = bt.AZ_clear_columns(close.columns)

    return_index_min = close['000905.SH'].iloc[-5] / close['000905.SH'].iloc[32] - 1
    return_index_open = close['000905.SH'].iloc[30] / close['000905.SH'].iloc[0] - 1
    if return_index_open > 0.000:
        close = close.mul(sector)

        # adj_close = close / adj_factor
        # return_intra = (adj_close / pre_close - 1).round(6)
        # return_intra.dropna(how='all', axis='columns', inplace=True)

        return_open = pd.DataFrame(close.iloc[30] / close.iloc[0].replace(0, np.nan) - 1)
        return_min = close.iloc[-5] / close.iloc[32].replace(0, np.nan) - 1 - return_index_min
        # return_index_min

        # 开盘指定时间 数据
        filter_df_1 = row_extre(return_open.T, 0.1)
        filter_df_1 = filter_df_1[filter_df_1 > 0].abs()

        # 新闻数据
        filter_df_2 = (bar_num < 15).astype(int)

        # 波动数据
        # filter_df_3 =

        # pos_df
        pos_df = signal * filter_df_1 * filter_df_2
        pos_df.name = pd.to_datetime(day)

        pnl_df = return_min * pos_df
        pnl_df.name = pd.to_datetime(day)

        trade_time = (pos_df).abs().sum() * 2

    else:
        pnl_df = pd.Series([0] * len(sector), index=sector.index)
        pnl_df.name = pd.to_datetime(day)
        trade_time = 0
        pos_df = pd.Series([np.nan] * len(sector), index=sector.index)
        pos_df.name = pd.to_datetime(day)
    return pnl_df, trade_time, pos_df


# def load_whs_data():
#
#

if __name__ == '__main__':
    begin_str = '20160101'
    end_str = '20190322'
    root_path = '/mnt/mfs/DAT_EQT'
    index_name = '000905'
    limit, n = 2.4, 10

    begin_year, begin_month, begin_day = begin_str[:4], begin_str[:6], begin_str
    end_year, end_month, end_day = end_str[:4], end_str[:6], end_str
    result_list = []

    intraday_path = '/mnt/mfs/DAT_EQT/intraday/eqt_1mbar'
    # load_sector
    sector_df = bt.AZ_Load_csv(f'{root_path}/EM_Funda/IDEX_YS_WEIGHT_A/SECURITYNAME_{index_name}.csv')
    sector_df[sector_df == sector_df] = 1

    # load adj_factor
    adj_factor = bt.AZ_Load_csv(f'{root_path}/EM_Funda/TRAD_SK_FACTOR1/TAFACTOR.csv')

    # load daily close
    pre_close = bt.AZ_Load_csv(f'{root_path}/EM_Funda/DERIVED_14/aadj_p.csv').shift(1)
    # get signal
    signal_df = signal_fun('intra_open_10min_vol', limit, n)
    signal_df = signal_df.reindex(index=sector_df.index, columns=sector_df.columns)

    # get stock bar data
    bar_num_df = bt.AZ_Load_csv(f'/media/hdd1/DAT_EQT/EM_Funda/dat_whs/bar_num_df.csv')

    #
    # return_signal_df = signal_fun('intra_open_10min_return', 2, 15)

    pool = Pool(20)
    year_list = [x for x in os.listdir(intraday_path) if (x >= begin_year) & (x <= end_year)]
    for year in sorted(year_list):
        year_path = os.path.join(intraday_path, year)
        month_list = [x for x in os.listdir(year_path) if (x >= begin_month) & (x <= end_month)]
        for month in sorted(month_list):
            month_path = os.path.join(year_path, month)
            day_list = [x for x in os.listdir(month_path) if (x >= begin_day) & (x <= end_day)]
            for day in sorted(day_list):
                day_path = os.path.join(month_path, day)
                args = (day_path, day, pre_close.loc[pd.to_datetime(day)], adj_factor.loc[pd.to_datetime(day)],
                        sector_df.loc[pd.to_datetime(day)], signal_df.loc[pd.to_datetime(day)],
                        bar_num_df.loc[pd.to_datetime(day)])
                # part_test_fun(*args)
                result_list.append(pool.apply_async(part_test_fun, args=args))
    pool.close()
    pool.join()

    pnl_df = pd.concat([res.get()[0] for res in result_list], axis=1)
    trade_time = sum([res.get()[1] for res in result_list])
    pos_df = pd.concat([res.get()[2] for res in result_list], axis=1)
    pot = pnl_df.T.sum(1).cumsum().iloc[-1] / trade_time
    # print(pnl_df)

    plot_send_result(pnl_df.T.sum(1), bt.AZ_Sharpe_y(pnl_df.T.sum(1)), 'a', f'{index_name}|{limit}|{n}|{round(pot, 6)}')
    print(pot)
