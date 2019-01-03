import pandas as pd
from datetime import datetime, timedelta


def AZ_split_stock(stock_list):
    """
    在stock_list中寻找A股代码
    :param stock_list:
    :return:
    """
    eqa = [x for x in stock_list if (x.startswith('0') or x.startswith('30')) and x.endswith('SZ')
           or x.startswith('6') and x.endswith('SH')]
    return eqa


def concat_vwap():
    all_df = pd.DataFrame()
    for i in range(24):
        print(i)
        tmp_df = pd.read_pickle('/mnt/mfs/DAT_PUBLIC/dat_whs/intra_vwap_10_tab_{}_20181001.pkl'.format(i + 1))
        if i <= 11:
            tmp_df.index = tmp_df.index + timedelta(hours=9, minutes=30) + timedelta(minutes=(i + 1) * 10)
            all_df = all_df.append(tmp_df)
        else:
            tmp_df.index = tmp_df.index + timedelta(hours=13, minutes=00) + timedelta(minutes=(i + 1 - 12) * 10)
            all_df = all_df.append(tmp_df)
        all_df.sort_index(inplace=True)

    EQT_list = AZ_split_stock(all_df.columns)
    all_df = all_df[EQT_list]
    year_list = sorted(set([x.strftime('%Y') for x in all_df.index]))
    for year in year_list:
        index_list = [x for x in all_df.index if x.year == int(year)]
        part_all_df = all_df.loc[index_list]
        part_all_df.dropna(how='all', axis='columns', inplace=True)
        part_all_df.to_csv('/mnt/mfs/DAT_EQT/intraday/Vwap_10min_{}.csv'.format(year), sep='|')


def concat_close():
    all_df = pd.DataFrame()
    for i in range(24):
        print(i)
        tmp_df = pd.read_pickle('/mnt/mfs/DAT_PUBLIC/dat_whs/intra_close_10_tab_{}_20181001.pkl'.format(i + 1))
        if i <= 11:
            tmp_df.index = tmp_df.index + timedelta(hours=9, minutes=30) + timedelta(minutes=(i + 1) * 10)
            all_df = all_df.append(tmp_df)
        else:
            tmp_df.index = tmp_df.index + timedelta(hours=13, minutes=00) + timedelta(minutes=(i + 1 - 12) * 10)
            all_df = all_df.append(tmp_df)
        all_df.sort_index(inplace=True)

    EQT_list = AZ_split_stock(all_df.columns)
    all_df = all_df[EQT_list]
    year_list = sorted(set([x.strftime('%Y') for x in all_df.index]))
    for year in year_list:
        index_list = [x for x in all_df.index if x.year == int(year)]
        part_all_df = all_df.loc[index_list]
        part_all_df.dropna(how='all', axis='columns', inplace=True)
        part_all_df.to_csv('/mnt/mfs/DAT_EQT/intraday/Close_10min_{}.csv'.format(year), sep='|')


def concat_vwap_live_history():
    data_hist = pd.read_csv('/mnt/mfs/DAT_EQT/intraday/Vwap_10min_2018.csv', index_col=0, sep='|', parse_dates=True)
    data_live_1 = pd.read_csv('/media/hdd1/DAT_EQT_Live/Vwap_10min.csv', index_col=0, sep='|', parse_dates=True)
    data_live_2 = pd.read_csv('/media/hdd1/DAT_EQT_Live/Vwap_10min_2018.csv', index_col=0, sep='|', parse_dates=True)
    data_con_1 = data_hist.combine_first(data_live_1)
    data_con_2 = data_con_1.combine_first(data_live_2)
    data_con_2.to_csv('/mnt/mfs/DAT_EQT/intraday/Vwap_10min_2018.csv', sep='|')


def concat_close_live_history():
    data_hist = pd.read_csv('/mnt/mfs/DAT_EQT/intraday/Close_10min_2018.csv', index_col=0, sep='|', parse_dates=True)
    data_live_1 = pd.read_csv('/media/hdd1/DAT_EQT_Live/Close_10min.csv', index_col=0, sep='|', parse_dates=True)
    data_live_2 = pd.read_csv('/media/hdd1/DAT_EQT_Live/Close_10min_2018.csv', index_col=0, sep='|', parse_dates=True)
    data_con_1 = data_hist.combine_first(data_live_1)
    data_con_2 = data_con_1.combine_first(data_live_2)
    data_con_2.to_csv('/mnt/mfs/DAT_EQT/intraday/Close_10min_2018.csv', swep='|')


if __name__ == '__main__':
    concat_vwap()
    concat_close()
    # concat_vwap_live_history()
    # concat_close_live_history()

