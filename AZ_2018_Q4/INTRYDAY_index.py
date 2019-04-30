import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *
from multiprocessing import Pool, Manager


def AZ_filter_stock(stock_list):  # 筛选相应股票池
    target_list = [x for x in stock_list if x[:2] == 'SH' and x[2] == '6' or
                   x[:2] == 'SZ' and x[2] in ['0', '3']]
    return target_list


def daily_deal_fun(day, day_path, up_vol_df, dn_vol_df, up_15_bar_vol_df, dn_15_bar_vol_df, daily_vwap,
                   up_15_bar_vwap_df, dn_15_bar_vwap_df, up_vwap_df, dn_vwap_df, money_flow1_df, money_flow2_df):
    volume = pd.read_csv(os.path.join(day_path, 'Volume.csv'), index_col=0).astype(float)
    volume = volume[AZ_filter_stock(volume.columns)]

    close = pd.read_csv(os.path.join(day_path, 'Close.csv'), index_col=0).astype(float)
    close = close[AZ_filter_stock(close.columns)]
    if os. path.exists(os.path.join(day_path, 'Open.csv')):
        open_ = pd.read_csv(os.path.join(day_path, 'Open.csv'), index_col=0).astype(float)
    else:
        open_ = pd.read_csv(os.path.join(day_path, 'Close.csv'), index_col=0).astype(float)
        send_email.send_email(day + ' date Dont have Open', ['whs@yingpei.com'], [], 'Intraday index')
    open_ = open_[AZ_filter_stock(open_.columns)]

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

    up_vol_df = pd.concat([up_vol_df, part_up_vol_df], axis=1)
    dn_vol_df = pd.concat([dn_vol_df, part_dn_vol_df], axis=1)

    up_15_bar_vol_df = pd.concat([up_15_bar_vol_df, part_up_15_bar_vol_df], axis=1)
    dn_15_bar_vol_df = pd.concat([dn_15_bar_vol_df, part_dn_15_bar_vol_df], axis=1)

    part_daily_vwap = close.mul(volume).sum() / volume.sum()
    part_daily_vwap.name = day

    daily_vwap = pd.concat([daily_vwap, part_daily_vwap], axis=1)

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

    up_15_bar_vwap_df = pd.concat([up_15_bar_vwap_df, part_up_15_bar_vwap_df], axis=1)
    dn_15_bar_vwap_df = pd.concat([dn_15_bar_vwap_df, part_dn_15_bar_vwap_df], axis=1)

    up_vwap_df = pd.concat([up_vwap_df, part_up_vwap_df], axis=1)
    dn_vwap_df = pd.concat([dn_vwap_df, part_dn_vwap_df], axis=1)

    part_money_flow1_df = ((close > open_) * volume).sum() - ((close < open_) * volume).sum()
    part_money_flow1_df.name = day
    part_money_flow2_df = ((close > open_) * (close - open_) * volume).sum() + \
                          ((close < open_) * (close - open_) * volume).sum()
    part_money_flow2_df.name = day
    money_flow1_df = pd.concat([money_flow1_df, part_money_flow1_df], axis=1)
    money_flow2_df = pd.concat([money_flow2_df, part_money_flow2_df], axis=1)

    return up_vol_df, dn_vol_df, up_15_bar_vol_df, dn_15_bar_vol_df, daily_vwap, up_15_bar_vwap_df, \
           dn_15_bar_vwap_df, up_vwap_df, dn_vwap_df, money_flow1_df, money_flow2_df


def create_intra_data():
    begin_str = '20050101'
    end_str = '20181001'

    begin_year, begin_month, begin_day = begin_str[:4], begin_str[:6], begin_str
    end_year, end_month, end_day = end_str[:4], end_str[:6], end_str
    intraday_path = '/mnt/mfs/DAT_EQT/intraday/eqt_1mbar'
    index_save_path = '/mnt/mfs/dat_whs/EM_Funda/my_data'

    up_vol_df = pd.DataFrame()
    dn_vol_df = pd.DataFrame()

    up_15_bar_vol_df = pd.DataFrame()
    dn_15_bar_vol_df = pd.DataFrame()

    daily_vwap = pd.DataFrame()

    up_15_bar_vwap_df = pd.DataFrame()
    dn_15_bar_vwap_df = pd.DataFrame()

    up_vwap_df = pd.DataFrame()
    dn_vwap_df = pd.DataFrame()

    money_flow1_df = pd.DataFrame()
    money_flow2_df = pd.DataFrame()

    year_list = [x for x in os.listdir(intraday_path) if (x >= begin_year) & (x <= end_year)]
    for year in sorted(year_list):
        year_path = os.path.join(intraday_path, year)
        month_list = [x for x in os.listdir(year_path) if (x >= begin_month) & (x <= end_month)]
        for month in sorted(month_list):
            month_path = os.path.join(year_path, month)
            day_list = [x for x in os.listdir(month_path) if (x >= begin_day) & (x <= end_day)]
            for day in sorted(day_list):
                print(day)
                day_path = os.path.join(month_path, day)
                up_vol_df, dn_vol_df, up_15_bar_vol_df, dn_15_bar_vol_df, daily_vwap, \
                up_15_bar_vwap_df, dn_15_bar_vwap_df, up_vwap_df, dn_vwap_df, money_flow1_df, money_flow2_df = \
                    daily_deal_fun(day, day_path, up_vol_df, dn_vol_df, up_15_bar_vol_df, dn_15_bar_vol_df, daily_vwap,
                                   up_15_bar_vwap_df, dn_15_bar_vwap_df, up_vwap_df, dn_vwap_df,
                                   money_flow1_df, money_flow2_df)


    up_div_dn = up_vwap_df / dn_vwap_df
    up_div_daily = up_vwap_df / daily_vwap
    dn_div_daily = dn_vwap_df / daily_vwap

    up_15_bar_div_dn_15_bar = up_15_bar_vwap_df / dn_15_bar_vwap_df
    up_15_bar_div_daily = up_15_bar_vwap_df / daily_vwap
    dn_15_bar_div_daily = dn_15_bar_vwap_df / daily_vwap

    up_vol_df.T.to_csv(os.path.join(index_save_path, 'intra_up_vol.csv'), sep='|')
    dn_vol_df.T.to_csv(os.path.join(index_save_path, 'intra_dn_vol.csv'), sep='|')

    up_15_bar_vol_df.T.to_csv(os.path.join(index_save_path, 'intra_up_15_bar_vol.csv'), sep='|')
    dn_15_bar_vol_df.T.to_csv(os.path.join(index_save_path, 'intra_dn_15_bar_vol.csv'), sep='|')

    daily_vwap.T.to_csv(os.path.join(index_save_path, 'intra_daily_vwap.csv'), sep='|')

    up_15_bar_vwap_df.T.to_csv(os.path.join(index_save_path, 'intra_up_15_bar_vwap.csv'), sep='|')
    dn_15_bar_vwap_df.T.to_csv(os.path.join(index_save_path, 'intra_dn_15_bar_vwap.csv'), sep='|')

    up_vwap_df.T.to_csv(os.path.join(index_save_path, 'intra_up_vwap.csv'), sep='|')
    dn_vwap_df.T.to_csv(os.path.join(index_save_path, 'intra_dn_vwap.csv'), sep='|')

    up_div_dn.T.to_csv(os.path.join(index_save_path, 'intra_up_div_dn.csv'), sep='|')
    up_div_daily.T.to_csv(os.path.join(index_save_path, 'intra_up_div_daily.csv'), sep='|')
    dn_div_daily.T.to_csv(os.path.join(index_save_path, 'intra_dn_div_daily.csv'), sep='|')

    up_15_bar_div_dn_15_bar.T.to_csv(os.path.join(index_save_path, 'intra_up_15_bar_div_dn_15_bar.csv'), sep='|')
    up_15_bar_div_daily.T.to_csv(os.path.join(index_save_path, 'intra_up_15_bar_div_daily.csv'), sep='|')
    dn_15_bar_div_daily.T.to_csv(os.path.join(index_save_path, 'intra_dn_15_bar_div_daily.csv'), sep='|')

    money_flow1_df.T.to_csv(os.path.join(index_save_path, 'money_flow1.csv'), sep='|')
    money_flow2_df.T.to_csv(os.path.join(index_save_path, 'money_flow2.csv'), sep='|')


def intra_data_deal():
    load_path = '/mnt/mfs/dat_whs/EM_Funda/my_data'
    save_path = '/mnt/mfs/dat_whs/EM_Funda/my_data_test'
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
        # esg_file = 'stock_code_df_tab5_14'
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
    intra_data_deal()

    # esg_data_deal()
