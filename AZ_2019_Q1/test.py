import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *


# def part_get_etf_data(day_path, day):
#     print(day)
#     target_list = ['SH510050', 'SH510300', 'SH510500']
#     close = pd.read_csv(os.path.join(day_path, 'Close.csv'), index_col=0).astype(float)
#     close.index = day + ' ' + close.index
#     part_df = close[target_list]
#     return part_df

def find_target_time(dn_pct, up_pct, return_intra):
    tmp_mask = ((return_intra > dn_pct) & (return_intra < up_pct)).astype(int).replace(0, np.nan)
    tmp_mask = tmp_mask.fillna(method='ffill').fillna(0)
    mask_diff = tmp_mask - tmp_mask.shift(1)
    return mask_diff[mask_diff > 0]


def part_test_fun(day_path, day, pre_close, adj_factor, sector):
    print(day)
    # hold_at_least = 5
    close = pd.read_csv(os.path.join(day_path, 'Close.csv'), index_col=0).astype(float)
    close.columns = bt.AZ_clear_columns(close.columns)
    close = close.mul(sector)

    adj_close = close / adj_factor
    return_intra = (adj_close / pre_close - 1).round(6)
    return_intra.dropna(how='all', axis='columns', inplace=True)

    return_min = return_intra.diff()

    # buy_df_l_mask = find_target_time(0.08, 0.084, return_intra)
    #
    # buy_df_s_mask_1 = find_target_time(0.065, 0.069, return_intra)
    # buy_df_s_mask_2 = find_target_time(0.098, 0.12, return_intra)
    #
    # buy_deal_df = buy_df_l_mask.sub(buy_df_s_mask_1, fill_value=0) \
    #     .sub(buy_df_s_mask_2, fill_value=0).fillna(method='ffill')
    # buy_deal_df = buy_deal_df[buy_deal_df > 0]

    sell_df_l_mask = find_target_time(-0.084, -0.08, return_intra)

    sell_df_s_mask_1 = find_target_time(-0.069, -0.065, return_intra)
    sell_df_s_mask_2 = find_target_time(-0.12, -0.098, return_intra)

    sell_deal_df = sell_df_l_mask.sub(sell_df_s_mask_1, fill_value=0) \
        .sub(sell_df_s_mask_2, fill_value=0).fillna(method='ffill')
    sell_deal_df = sell_deal_df[sell_deal_df > 0]

    # pnl_min_df = buy_deal_df.shift(2) * return_min
    pnl_min_df = sell_deal_df.shift(2) * return_min
    pnl_df = pnl_min_df.sum()
    pnl_df.name = pd.to_datetime(day)
    # buy_deal_df = buy_deal_df

    # sell_df = (return_intra < -0.08)
    # sell_df = (return_intra < -0.08)

    a = return_intra[return_intra.abs() > 0.101].dropna(how='all', axis=1).dropna(how='all', axis=0)

    if len(a) != 0:
        print(day)
        # print(a)
    return pnl_df


if __name__ == '__main__':
    begin_str = '20100101'
    end_str = '20190322'
    root_path = '/mnt/mfs/DAT_EQT'
    index_name = '000905'

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
                        sector_df.loc[pd.to_datetime(day)])
                # part_test_fun(*args)
                result_list.append(pool.apply_async(part_test_fun, args=args))
    pool.close()
    pool.join()

    pnl_df = pd.concat([res.get() for res in result_list], axis=1)
    print(pnl_df)

    plot_send_result(pnl_df.T.sum(1), bt.AZ_Sharpe_y(pnl_df.T.sum(1)), 'a', 'a')
