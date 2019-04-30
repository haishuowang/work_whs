import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *


def row_extre(raw_df, percent):
    raw_df = raw_df
    target_df = raw_df.rank(axis=1, pct=True)
    target_df[target_df >= 1 - percent] = 1
    target_df[target_df <= percent] = -1
    target_df[(target_df > percent) & (target_df < 1 - percent)] = 0
    return target_df


def row_extre_rank(raw_df, rk_n=3, l_w=1, s_w=1):
    raw_df = raw_df
    target_df_rk1 = raw_df.rank(axis=1, ascending=True)
    target_df_rk2 = raw_df.rank(axis=1, ascending=False)
    target_df_up = (target_df_rk1 <= rk_n).astype(int)
    target_df_dn = (target_df_rk2 <= rk_n).astype(int)
    target_df = target_df_up * l_w - target_df_dn * s_w
    return target_df


def signal_fun(file_name, limit, n):
    raw_df = bt.AZ_Load_csv(f'/mnt/mfs/dat_whs/intra_data/{file_name}.csv')
    tmp_df = bt.AZ_Col_zscore(raw_df, n, min_periods=0)
    tmp_df[(tmp_df < limit) & (tmp_df > -limit)] = 0
    tmp_df[tmp_df >= limit] = 1
    tmp_df[tmp_df <= -limit] = -1
    return tmp_df


def find_target_time(dn_pct, up_pct, return_intra):
    tmp_mask = ((return_intra > dn_pct) & (return_intra < up_pct)).astype(int).replace(0, np.nan)
    tmp_mask = tmp_mask.fillna(method='ffill').fillna(0)
    mask_diff = tmp_mask - tmp_mask.shift(1)
    return mask_diff[mask_diff > 0]


def filter_2_fun(bar_num_df, target_num):
    bar_num_roll_df = bt.AZ_Rolling_mean(bar_num_df, 20)
    bar_num_roll_df[bar_num_roll_df < target_num] = -1
    bar_num_roll_df[bar_num_roll_df >= target_num] = 1
    return bar_num_roll_df


def part_get_all_ind_df(table_path, sector_df, all_file_list, xinx, i):
    a = int(i / 10)
    b = int(i - a * 10)
    i_str = str(a) + str(b)
    ind_df = pd.DataFrame()
    print(f'ZhongXing_Level2_CS{i_str}')
    for file_name in all_file_list:
        if f'ZhongXing_Level2_CS{i_str}' in file_name:
            tmp_df = bt.AZ_Load_csv(f'{table_path}/{file_name}.csv')
            xnms = sorted(list(set(sector_df.columns) & set(tmp_df.columns)))
            tmp_df = tmp_df.reindex(index=xinx, columns=xnms) * sector_df.reindex(columns=xnms)
            ind_df = ind_df.combine_first(tmp_df)
    return ind_df


pd.DataFrame().rank()


def get_all_ind_df(sector_df):
    xinx = sector_df.index
    table_path = f'{root_path}/EM_Funda/LICO_IM_INCHG'
    all_file_list = [x[:-4] for x in os.listdir(table_path) if 'ZhongXing_Level2' in x]
    all_ind_df_list = []
    for i in range(1, 30):
        ind_df = part_get_all_ind_df(table_path, sector_df, all_file_list, xinx, i)
        all_ind_df_list.append(ind_df)
    return all_ind_df_list


def filter_3_fun(target_close, pre_adj_close, pre_index_close, adj_factor, sector_df, percent):
    adj_target_close = target_close / adj_factor
    pre_close_pct_df = adj_target_close / pre_adj_close.replace(0, np.nan) - 1

    pre_idnex_close_pct_df = target_close['000905.SH'] / pre_index_close - 1

    all_ind_df_list = get_all_ind_df(sector_df)
    # 行业内 前percent 1 和 后percent -1
    all_ind_extre_df_list = [row_extre_rank(pre_close_pct_df * x, 3, 1, 1) for x in all_ind_df_list]

    all_ind_return_df = pd.concat([(ind_df * pre_close_pct_df).mean(1) for ind_df in all_ind_df_list], axis=1)
    all_ind_return_df.columns = range(1, 30)

    # all_ind_alpha_return_df = all_ind_return_df.sub(pre_idnex_close_pct_df, axis=0)
    all_ind_alpha_return_df = all_ind_return_df

    filter_table = (bt.AZ_Rolling_std(all_ind_alpha_return_df, 20) * (np.sqrt(250)) > 0.3).astype(int)

    all_ind_alpha_return_zscore_df = bt.AZ_Col_zscore(all_ind_alpha_return_df, 15)
    all_ind_alpha_return_zscore_df_up = (all_ind_alpha_return_zscore_df > 1.5).astype(int)
    all_ind_alpha_return_zscore_df_dn = (all_ind_alpha_return_zscore_df < -1.5).astype(int)

    all_ind_signal_df = (all_ind_alpha_return_zscore_df_up - all_ind_alpha_return_zscore_df_dn).replace(0, np.nan)
    all_ind_signal_df = all_ind_signal_df * filter_table
    # filter_3_df = pd.DataFrame()
    return all_ind_signal_df, all_ind_extre_df_list


def part_test_fun(day_path, day, sector, signal, bar_num, ind_signal, ind_extre):
    print(day, ind_signal)
    close = pd.read_csv(os.path.join(day_path, 'Close.csv'), index_col=0).astype(float)

    close.columns = bt.AZ_clear_columns(close.columns)

    return_index_min = close['000905.SH'].iloc[-5] / close['000905.SH'].iloc[32] - 1
    return_index_open = close['000905.SH'].iloc[30] / close['000905.SH'].iloc[0] - 1
    if return_index_open > 0.000 and ind_signal > 0:
        close = close.mul(sector)

        return_open = pd.DataFrame(close.iloc[30] / close.iloc[0].replace(0, np.nan) - 1)
        return_min = close.iloc[-5] / close.iloc[32].replace(0, np.nan) - 1 - return_index_min

        # 开盘指定时间 数据
        filter_1 = row_extre(return_open.T, 0.1)
        filter_1 = filter_1[filter_1 > 0].abs()

        # 新闻数据
        filter_2 = (bar_num < 15).astype(int)

        # 行业内
        filter_3 = (ind_extre > 0).astype(int)
        # 大盘涨幅过大

        # pos_df
        pos_df = filter_3
        pos_df.name = pd.to_datetime(day)

        pnl_df = return_min * pos_df
        pnl_df.name = pd.to_datetime(day)

        trade_time = pos_df.abs().sum() * 2

    else:
        pnl_df = pd.Series([0] * len(sector), index=sector.index)
        pnl_df.name = pd.to_datetime(day)
        trade_time = 0
        pos_df = pd.Series([np.nan] * len(sector), index=sector.index)
        pos_df.name = pd.to_datetime(day)

    return pnl_df, trade_time, pos_df


if __name__ == '__main__':
    begin_str = '20150101'
    end_str = '20190406'
    root_path = '/mnt/mfs/DAT_EQT'
    index_name = '000905'
    limit, n = 2.4, 10
    percent = 0.15
    begin_year, begin_month, begin_day = begin_str[:4], begin_str[:6], begin_str
    end_year, end_month, end_day = end_str[:4], end_str[:6], end_str
    result_list = []

    intraday_path = '/mnt/mfs/DAT_EQT/intraday/eqt_1mbar'
    # load_sector
    sector_df = bt.AZ_Load_csv(f'{root_path}/EM_Funda/IDEX_YS_WEIGHT_A/SECURITYNAME_{index_name}.csv')
    sector_df[sector_df == sector_df] = 1
    sector_df = sector_df[(sector_df.index > pd.to_datetime(begin_str))
                          & (sector_df.index < pd.to_datetime(end_str))]
    # load adj_factor
    adj_factor = bt.AZ_Load_csv(f'{root_path}/EM_Funda/TRAD_SK_FACTOR1/TAFACTOR.csv') \
        .reindex(index=sector_df.index)

    # load daily close
    pre_adj_close = bt.AZ_Load_csv(f'{root_path}/EM_Funda/DERIVED_14/aadj_p.csv').shift(1) \
        .reindex(index=sector_df.index)

    pre_index_close = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/NEW.csv')[index_name].shift(1) \
        .reindex(index=sector_df.index)
    # get signal
    signal_df = signal_fun('intra_open_min_vol|30', limit, n)
    signal_df = signal_df.reindex(index=sector_df.index, columns=sector_df.columns)

    # get stock bar data
    bar_num_df = bt.AZ_Load_csv(f'/mnt/mfs/DAT_EQT/EM_Funda/dat_whs/bar_num_df.csv')

    filter_2_df = filter_2_fun(bar_num_df, 20)

    # pre_close 到target_time的涨幅
    target_close = bt.AZ_Load_csv('/mnt/mfs/dat_whs/intra_data/intra_target_close|30.csv') \
        .reindex(index=sector_df.index)
    all_ind_signal_df, all_ind_extre_df_list = \
        filter_3_fun(target_close, pre_adj_close, pre_index_close, adj_factor, sector_df, percent)

    # all_ind_signal_df
    # all_ind_extre_df_list
    return_df = bt.AZ_Load_csv('/mnt/mfs/dat_whs/intra_data/intra_stock_return|32_240.csv')
    all_pnl_df = pd.DataFrame()
    for i in all_ind_signal_df.columns:
        ind_signal_df = all_ind_signal_df[i]
        ind_extre_df = all_ind_extre_df_list[i - 1]

        pos_df = ind_extre_df.replace(0, np.nan).dropna(how='all', axis=1).mul((ind_signal_df > 0).astype(int), axis=0)
        # pos_df = ind_extre_df.replace(0, np.nan).dropna(how='all', axis=1)\
        #     .mul((ind_signal_df < 0).astype(int), axis=0)
        pnl_df = (pos_df * return_df).sum(1)

        trade_time = pos_df.abs().sum().sum() * 2

        pot = pnl_df.cumsum().iloc[-1] / trade_time
        print(bt.AZ_Sharpe_y(pnl_df), pot)
        plot_send_result(pd.DataFrame(pnl_df), bt.AZ_Sharpe_y(pnl_df), f'manual_test_self_32_240_0.3{i}',
                         f'{index_name}|{limit}|{n}|{round(pot, 6)}')
        all_pnl_df = all_pnl_df.add(pd.DataFrame(pnl_df, columns=['all pnl']), fill_value=0)
    plot_send_result(all_pnl_df, bt.AZ_Sharpe_y(all_pnl_df), f'all_pnl', 'sadas')
    #     print(i)

    #
    #     pos_df = ind_extre_df > 0
    #     pnl_df = (ind_extre_df * return_df).sum(1)
    #     # pool = Pool(20)
    #     # year_list = [x for x in os.listdir(intraday_path) if (x >= begin_year) & (x <= end_year)]
    #     # for year in sorted(year_list):
    #     #     year_path = os.path.join(intraday_path, year)
    #     #     month_list = [x for x in os.listdir(year_path) if (x >= begin_month) & (x <= end_month)]
    #     #     for month in sorted(month_list):
    #     #         month_path = os.path.join(year_path, month)
    #     #         day_list = [x for x in os.listdir(month_path) if (x >= begin_day) & (x <= end_day)]
    #     #         for day in sorted(day_list):
    #     #             day_path = os.path.join(month_path, day)
    #     #             args = (day_path, day, sector_df.loc[pd.to_datetime(day)], signal_df.loc[pd.to_datetime(day)],
    #     #                     bar_num_df.loc[pd.to_datetime(day)],
    #     #                     ind_signal_df.loc[pd.to_datetime(day)], ind_extre_df.loc[pd.to_datetime(day)])
    #     #             # part_test_fun(*args)
    #     #             result_list.append(pool.apply_async(part_test_fun, args=args))
    #     # pool.close()
    #     # pool.join()
    #
    #     # pnl_df = pd.concat([res.get()[0] for res in result_list], axis=1)
    #     trade_time = sum(pos_df.abs().sum().sum()) * 2
    #     pos_df = pd.concat([res.get()[2] for res in result_list], axis=1)
    #     pot = pnl_df.T.sum(1).cumsum().iloc[-1] / trade_time
    #     print(bt.AZ_Sharpe_y(pnl_df.T.sum(1)), pot)
    #     plot_send_result(pnl_df.T.sum(1), bt.AZ_Sharpe_y(pnl_df.T.sum(1)), f'manual_test{i}',
    #                      f'{index_name}|{limit}|{n}|{round(pot, 6)}')

