import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *
from multiprocessing import Pool, Manager


def AZ_filter_stock(stock_list):  # 筛选相应股票池
    target_list = [x for x in stock_list if x[:2] == 'SH' and x[2] == '6' or
                   x[:2] == 'SZ' and x[2] in ['0', '3']]
    return target_list


def daily_deal_fun(day, day_path):
    try:
        print(day)
        volume = pd.read_csv(os.path.join(day_path, 'Volume.csv'), index_col=0).astype(float)
        volume = volume[AZ_filter_stock(volume.columns)]

        close = pd.read_csv(os.path.join(day_path, 'Close.csv'), index_col=0).astype(float)
        close = close[AZ_filter_stock(close.columns)]

        open_ = pd.read_csv(os.path.join(day_path, 'Open.csv'), index_col=0).astype(float)
        open_ = open_[AZ_filter_stock(open_.columns)]

        high = pd.read_csv(os.path.join(day_path, 'High.csv'), index_col=0).astype(float)
        high = open_[AZ_filter_stock(high.columns)]

        low = pd.read_csv(os.path.join(day_path, 'Low.csv'), index_col=0).astype(float)
        low = open_[AZ_filter_stock(low.columns)]

        pct_chg = close.pct_change()
        part_up_mask = pct_chg > 0
        part_dn_mask = pct_chg < 0

        part_up_chg_rank = pct_chg[part_up_mask].rank(ascending=False, na_option='bottom', axis=0)
        part_dn_chg_rank = pct_chg[part_dn_mask].rank(ascending=True, na_option='bottom', axis=0)

        part_up_chg_rank_15_mask = part_up_chg_rank <= 15
        part_dn_chg_rank_15_mask = part_dn_chg_rank <= 15

        part_up_vol_df = volume[part_up_mask].sum()
        part_up_vol_df.name = day
        part_dn_vol_df = volume[part_dn_mask].sum()
        part_dn_vol_df.name = day

        part_up_15_bar_vol_df = volume[part_up_chg_rank_15_mask].sum()
        part_up_15_bar_vol_df.name = day
        part_dn_15_bar_vol_df = volume[part_dn_chg_rank_15_mask].sum()
        part_dn_15_bar_vol_df.name = day

        part_daily_vwap = close.mul(volume).sum() / volume.sum()
        part_daily_vwap.name = day

        part_up_15_bar_vwap_df = volume[part_up_chg_rank_15_mask].mul(close[part_up_chg_rank_15_mask]).sum() \
                                 / volume[part_up_chg_rank_15_mask].sum()
        part_up_15_bar_vwap_df.name = day
        part_dn_15_bar_vwap_df = volume[part_dn_chg_rank_15_mask].mul(close[part_dn_chg_rank_15_mask]).sum() \
                                 / volume[part_dn_chg_rank_15_mask].sum()
        part_dn_15_bar_vwap_df.name = day

        part_up_vwap_df = volume[part_up_mask].mul(close[part_up_mask]).sum() \
                          / volume[part_up_mask].sum()
        part_up_vwap_df.name = day
        part_dn_vwap_df = volume[part_dn_mask].mul(close[part_dn_mask]).sum() \
                          / volume[part_dn_mask].sum()
        part_dn_vwap_df.name = day

        part_money_flow1_df = ((close > open_) * volume).sum() - ((close < open_) * volume).sum()
        part_money_flow1_df.name = day
        part_money_flow2_df = ((close > open_) * (close - open_) * volume).sum() + \
                              ((close < open_) * (close - open_) * volume).sum()
        part_money_flow2_df.name = day

        part_p_vol_df = close.std() / close.mean()
        part_p_vol_df[part_p_vol_df > 1] = np.nan
        part_p_vol_df.name = day

        part_p_vol_last_15min_df = close.iloc[-15:].std() / close.iloc[-15:].mean()
        part_p_vol_last_15min_df.name = day

        part_r_vol_df = pct_chg.std()
        part_r_vol_df.name = day

        part_r_vol_last_15min_df = pct_chg.iloc[-15:].std()
        part_r_vol_last_15min_df.name = day

        part_last_15_min_ud_df = pct_chg.iloc[-15:].sum()
        part_last_15_min_ud_df.name = day

        part_today_sharpe_df = pct_chg.apply(bt.AZ_Sharpe_y)
        part_today_sharpe_df.name = day

        part_hl_pct_df = ((high - low) / close).sum()
        part_hl_pct_df.name = day

        part_last_15min_volume_pct_df = volume.iloc[-15:].sum() / volume.sum()
        part_last_15min_volume_pct_df.name = day

        part_oc_15min_volume_div_df = volume.iloc[:15].sum()/volume.iloc[-15:].sum()
        part_oc_15min_volume_div_df.name = day

        return part_up_vol_df, part_dn_vol_df, part_up_15_bar_vol_df, part_dn_15_bar_vol_df, part_daily_vwap, \
               part_up_15_bar_vwap_df, part_dn_15_bar_vwap_df, part_up_vwap_df, part_dn_vwap_df, part_money_flow1_df, \
               part_money_flow2_df, part_p_vol_df, part_p_vol_last_15min_df, part_r_vol_df, part_r_vol_last_15min_df, \
               part_last_15_min_ud_df, part_today_sharpe_df, part_hl_pct_df, \
               part_last_15min_volume_pct_df, part_oc_15min_volume_div_df
    except Exception as error:
        send_email.send_email(str(error), ['whs@yingpei.com'], [], f'{day} intra data error')


def save_fun(data, target_path):
    data.columns = [x[2:] + '.' + x[:2] for x in data.columns]
    data = data.reindex(columns=sorted(data.columns))
    data.to_csv(target_path, sep='|')


def create_intra_data():
    begin_str = '20100101'
    end_str = '20190610'

    begin_year, begin_month, begin_day = begin_str[:4], begin_str[:6], begin_str
    end_year, end_month, end_day = end_str[:4], end_str[:6], end_str
    intraday_path = '/mnt/mfs/DAT_EQT/intraday/eqt_1mbar'
    index_save_path = '/mnt/mfs/DAT_EQT/EM_Funda/dat_whs'
    bt.AZ_Path_create(index_save_path)
    result_list = []
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
                # daily_deal_fun(day, day_path)
                result_list.append(pool.apply_async(daily_deal_fun, (day, day_path)))

    up_vol_df = pd.concat([x.get()[0] for x in result_list], axis=1, sort=True)
    dn_vol_df = pd.concat([x.get()[1] for x in result_list], axis=1, sort=True)

    up_15_bar_vol_df = pd.concat([x.get()[2] for x in result_list], axis=1, sort=True)
    dn_15_bar_vol_df = pd.concat([x.get()[3] for x in result_list], axis=1, sort=True)

    daily_vwap = pd.concat([x.get()[4] for x in result_list], axis=1, sort=True)

    up_15_bar_vwap_df = pd.concat([x.get()[5] for x in result_list], axis=1, sort=True)
    dn_15_bar_vwap_df = pd.concat([x.get()[6] for x in result_list], axis=1, sort=True)

    up_vwap_df = pd.concat([x.get()[7] for x in result_list], axis=1, sort=True)
    dn_vwap_df = pd.concat([x.get()[8] for x in result_list], axis=1, sort=True)

    money_flow1_df = pd.concat([x.get()[9] for x in result_list], axis=1, sort=True)
    money_flow2_df = pd.concat([x.get()[10] for x in result_list], axis=1, sort=True)

    p_vol_df = pd.concat([x.get()[11] for x in result_list], axis=1, sort=True)
    p_vol_last_15min_df = pd.concat([x.get()[12] for x in result_list], axis=1, sort=True)
    r_vol_df = pd.concat([x.get()[13] for x in result_list], axis=1, sort=True)
    r_vol_last_15min_df = pd.concat([x.get()[14] for x in result_list], axis=1, sort=True)
    last_15_min_ud_df = pd.concat([x.get()[15] for x in result_list], axis=1, sort=True)
    today_sharpe_df = pd.concat([x.get()[16] for x in result_list], axis=1, sort=True)
    hl_pct_df = pd.concat([x.get()[17] for x in result_list], axis=1, sort=True)
    last_15min_volume_pct_df = pd.concat([x.get()[18] for x in result_list], axis=1, sort=True)
    oc_15min_volume_div_df = pd.concat([x.get()[19] for x in result_list], axis=1, sort=True)

    up_div_dn = up_vwap_df / dn_vwap_df
    up_div_daily = up_vwap_df / daily_vwap
    dn_div_daily = dn_vwap_df / daily_vwap

    up_15_bar_div_dn_15_bar = up_15_bar_vwap_df / dn_15_bar_vwap_df
    up_15_bar_div_daily = up_15_bar_vwap_df / daily_vwap
    dn_15_bar_div_daily = dn_15_bar_vwap_df / daily_vwap

    save_fun(up_vol_df.T, f'{index_save_path}/intra_up_vol.csv')
    save_fun(dn_vol_df.T, f'{index_save_path}/intra_dn_vol.csv')

    save_fun(up_15_bar_vol_df.T, f'{index_save_path}/intra_up_15_bar_vol.csv')
    save_fun(dn_15_bar_vol_df.T, f'{index_save_path}/intra_dn_15_bar_vol.csv')

    save_fun(daily_vwap.T, f'{index_save_path}/intra_daily_vwap.csv')

    save_fun(up_15_bar_vwap_df.T, f'{index_save_path}/intra_up_15_bar_vwap.csv')
    save_fun(dn_15_bar_vwap_df.T, f'{index_save_path}/intra_dn_15_bar_vwap.csv')

    save_fun(up_vwap_df.T, f'{index_save_path}/intra_up_vwap.csv')
    save_fun(dn_vwap_df.T, f'{index_save_path}/intra_dn_vwap.csv')

    save_fun(up_div_dn.T, f'{index_save_path}/intra_up_div_dn.csv')
    save_fun(up_div_daily.T, f'{index_save_path}/intra_up_div_daily.csv')
    save_fun(dn_div_daily.T, f'{index_save_path}/intra_dn_div_daily.csv')

    save_fun(up_15_bar_div_dn_15_bar.T, f'{index_save_path}/intra_up_15_bar_div_dn_15_bar.csv')
    save_fun(up_15_bar_div_daily.T, f'{index_save_path}/intra_up_15_bar_div_daily.csv')
    save_fun(dn_15_bar_div_daily.T, f'{index_save_path}/intra_dn_15_bar_div_daily.csv')

    save_fun(money_flow1_df.T, f'{index_save_path}/intra_money_flow1.csv')
    save_fun(money_flow2_df.T, f'{index_save_path}/intra_money_flow2.csv')

    save_fun(p_vol_df.T, f'{index_save_path}/intra_p_vol.csv')
    save_fun(p_vol_last_15min_df.T, f'{index_save_path}/intra_p_vol_last_15min.csv')

    save_fun(r_vol_df.T, f'{index_save_path}/intra_r_vol.csv')
    save_fun(r_vol_last_15min_df.T, f'{index_save_path}/intra_r_vol_last_15min.csv')
    save_fun(last_15_min_ud_df.T, f'{index_save_path}/intra_last_15_min_ud.csv')

    save_fun(today_sharpe_df.T, f'{index_save_path}/intra_today_sharpe.csv')
    save_fun(hl_pct_df.T, f'{index_save_path}/intra_hl_pct.csv')

    save_fun(last_15min_volume_pct_df.T, f'{index_save_path}/intra_last_15min_volume_pct.csv')
    save_fun(oc_15min_volume_div_df.T, f'{index_save_path}/intra_oc_15min_volume_div.csv')


def intra_data_deal():
    load_path = '/mnt/mfs/DAT_EQT/EM_Funda/dat_whs'
    save_path = '/mnt/mfs/DAT_EQT/EM_Funda/dat_whs'
    return_df = bt.AZ_Load_csv(f'/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv')
    stock_list = return_df.columns
    intra_file_list = sorted([x for x in os.listdir(load_path) if x.startswith('intra')])
    for intra_file in intra_file_list:
        print(intra_file)
        data = bt.AZ_Load_csv(f'{load_path}/{intra_file}')
        data.columns = [x[2:] + '.' + x[:2] for x in data.columns]
        data = data.reindex(columns=stock_list)
        data.to_csv(f'{save_path}/{intra_file}', sep='|')


def esg_data_deal():
    path_1 = '/mnt/mfs/dat_whs/EM_Funda/my_data'
    path_2 = '/mnt/mfs/dat_whs/EM_Funda/my_data_test'
    esg_file_list = sorted(os.listdir(path_2))
    begin_date = pd.to_datetime('20120101')
    end_date = pd.to_datetime('20170101')
    for esg_file in esg_file_list:
        print(esg_file)
        data_1 = pd.read_csv(f'{path_1}/{esg_file}', sep='|', index_col=0, parse_dates=True)
        data_1 = data_1[(data_1.index > begin_date) & (data_1.index < end_date)]
        data_2 = pd.read_csv(f'{path_2}/{esg_file}', sep='|', index_col=0, parse_dates=True)
        data_2 = data_2[(data_2.index > begin_date) & (data_2.index < end_date)]
        xnms = sorted(list(set(data_1.columns) & set(data_2.columns)))
        data_1 = data_1[xnms]
        data_2 = data_2[xnms]
        mask = data_1 != data_2
        print(mask.sum(1))


if __name__ == '__main__':
    create_intra_data()
    # intra_data_deal()

