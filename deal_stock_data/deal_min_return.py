import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *

# root_path = '/media/hdd1/DAT_EQT'
root_path = '/mnt/mfs/DAT_EQT'

return_df = bt.AZ_Load_csv(f'{root_path}/EM_Funda/DERIVED_14/aadj_r.csv')
adj_factor = bt.AZ_Load_csv(f'{root_path}/EM_Funda/TRAD_SK_FACTOR1/TAFACTOR.csv')

vwap_data_raw = pd.read_pickle('/mnt/mfs/DAT_PUBLIC/dat_whs/intra_vwap_60_tab_1_2005_2018.pkl')

vwap_data = vwap_data_raw.reindex(columns=return_df.columns)
vwap_return = (vwap_data * adj_factor.reindex(vwap_data.index)).pct_change()
vwap_return.to_csv('/mnt/mfs/DAT_PUBLIC/dat_whs/vwap_return/open_1_hour.csv', sep='|')


def index_concat_fun(day_path, day, split_time, hs300_df, zz500_df):
    Close = pd.read_csv(os.path.join(day_path, 'Close.csv'), index_col=0).astype(float)
    if 'SH000001' in Close.columns:
        hs300_df.loc[pd.to_datetime(day), 'HS300'] = Close['SH000001'].iloc[:split_time].mean()
    if 'SZ399905' in Close.columns:
        zz500_df.loc[pd.to_datetime(day), 'ZZ500'] = Close['SZ399905'].iloc[:split_time].mean()
    return hs300_df, zz500_df


def twap_hs300_fun(split_time):
    begin_str = '20181001'
    end_str = '20181101'

    hs300_df = pd.DataFrame(columns=['HS300'])
    zz500_df = pd.DataFrame(columns=['ZZ500'])

    begin_year, begin_month, begin_day = begin_str[:4], begin_str[:6], begin_str
    end_year, end_month, end_day = end_str[:4], end_str[:6], end_str
    intraday_path = '/mnt/mfs/DAT_PUBLIC/intraday/eqt_1mbar'

    for i in range(int(240 / split_time)):
        exec('intra_vwap_tab_{}_df = pd.DataFrame()'.format(i + 1))
        exec('intra_close_tab_{}_df = pd.DataFrame()'.format(i + 1))

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
                hs300_df, zz500_df = index_concat_fun(day_path, day, split_time, hs300_df, zz500_df)
    return hs300_df, zz500_df


if __name__ == '__main__':
    hs300_df, zz500_df = twap_hs300_fun(60)
