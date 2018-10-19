import pandas as pd
import numpy as np
import os
import gc


def AZ_clear_columns(stock_list):
    return [x[2:] + '.' + x[:2]for x in stock_list]


def create_base_row_data():
    root_path = '/mnt/mfs/dat_whs'
    save_path = '/mnt/mfs/dat_whs/data/base_data'
    load_path = root_path + '/data/AllStock'
    all_open = pd.read_csv(os.path.join(load_path, 'all_open.csv'), index_col=0)
    all_high = pd.read_csv(os.path.join(load_path, 'all_high.csv'), index_col=0)
    all_low = pd.read_csv(os.path.join(load_path, 'all_low.csv'), index_col=0)
    all_close = pd.read_csv(os.path.join(load_path, 'all_close.csv'), index_col=0)
    all_amount = pd.read_csv(os.path.join(load_path, 'all_amount.csv'), index_col=0)

    all_volume = pd.read_csv(os.path.join(load_path, 'all_volume.csv'), index_col=0)
    all_adj_r = pd.read_csv(os.path.join(load_path, 'all_adj_r.csv'), index_col=0)
    root_save_path = '/mnt/mfs/dat_whs/data/factor_data'

    EQT_list = [x for x in all_open.columns if
                ((x[0] == '0' or x[0] == '3') and x[-2:] == 'SZ') or (x[0] == '6' and x[-2:] == 'SH')]

    date_index = pd.to_datetime(all_open.index.astype(str))
    all_open.index = date_index
    all_high.index = date_index
    all_low.index = date_index
    all_close.index = date_index
    all_amount.index = date_index
    all_volume.index = date_index
    all_adj_r.index = date_index

    begin_str = '20100101'
    end_str = '20180401'

    EQT_open = all_open[EQT_list][(date_index >= pd.to_datetime(begin_str)) & (date_index < pd.to_datetime(end_str))]
    EQT_high = all_high[EQT_list][(date_index >= pd.to_datetime(begin_str)) & (date_index < pd.to_datetime(end_str))]
    EQT_low = all_low[EQT_list][(date_index >= pd.to_datetime(begin_str)) & (date_index < pd.to_datetime(end_str))]
    EQT_close = all_close[EQT_list][(date_index >= pd.to_datetime(begin_str)) & (date_index < pd.to_datetime(end_str))]
    EQT_amount = all_amount[EQT_list][
        (date_index >= pd.to_datetime(begin_str)) & (date_index < pd.to_datetime(end_str))]
    EQT_volume = all_volume[EQT_list][
        (date_index >= pd.to_datetime(begin_str)) & (date_index < pd.to_datetime(end_str))]
    EQT_adj_r = all_adj_r[EQT_list][(date_index >= pd.to_datetime(begin_str)) & (date_index < pd.to_datetime(end_str))]

    EQT_open.to_pickle(os.path.join(save_path, 'EQT_open.pkl'))
    EQT_high.to_pickle(os.path.join(save_path, 'EQT_high.pkl'))
    EQT_low.to_pickle(os.path.join(save_path, 'EQT_low.pkl'))
    EQT_close.to_pickle(os.path.join(save_path, 'EQT_close.pkl'))
    EQT_amount.to_pickle(os.path.join(save_path, 'EQT_amount.pkl'))
    EQT_volume.to_pickle(os.path.join(save_path, 'EQT_volume.pkl'))
    EQT_adj_r.to_pickle(os.path.join(save_path, 'EQT_adj_r.pkl'))


def create_rzrq_row_data():
    root_path = '/mnt/mfs/DAT_EQT/EM_Tab14/adj_data/TRAD_MT_MARGIN'
    save_path = '/mnt/mfs/dat_whs/data/base_data'
    name_list = ['RZRQYE', 'RZMRE', 'RZYE', 'RQMCL', 'RQYE', 'RQYL', 'RQCHL', 'RZCHE']
    begin_str = '20100101'
    end_str = '20180401'
    for file_name in name_list:
        file_load_path = os.path.join(root_path, file_name + '.pkl')
        data = pd.read_pickle(file_load_path)
        data = data[(data.index >= pd.to_datetime(begin_str)) & (data.index < pd.to_datetime(end_str))]
        data.to_pickle(os.path.join(save_path, '{}.pkl').format(file_name))


def deal_intra_data():
    intra_raw_path = '/mnt/mfs/dat_whs/data/base_data'
    # volume_list = [x for x in os.listdir(intra_raw_path) if x.startswith('intra') and 'volume' in x]
    # vwap_list = [x for x in os.listdir(intra_raw_path) if x.startswith('intra') and 'vwap' in x]
    intra_list = [x for x in os.listdir(intra_raw_path) if x.startswith('intra')]

    for tab_name in intra_list:
        intra_data = pd.read_pickle(os.path.join(intra_raw_path, tab_name))
        a = [x for x in intra_data.columns if x.startswith('s')]
        print(tab_name, a)


def create_intra_data(split_time=20):
    begin_str = '20180901'
    # begin_str = '20180101'
    end_str = '20181001'

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
                volume = pd.read_csv(os.path.join(day_path, 'Volume.csv'), index_col=0).astype(float)
                close = pd.read_csv(os.path.join(day_path, 'Close.csv'), index_col=0).astype(float)
                for i in range(int(240 / split_time)):
                    tmp_volume = volume[i * split_time:(i + 1) * split_time]
                    tmp_volume_sum = tmp_volume.sum()
                    tmp_close = close[i * split_time:(i + 1) * split_time]
                    exec('intra_close_tab_{0}_df = intra_close_tab_{0}_df.append(pd.DataFrame([tmp_close.iloc[-1]], '
                         'index=[pd.to_datetime(day)]))'.format(i + 1))

                    tmp_vwap = (tmp_close * tmp_volume).sum() / tmp_volume_sum
                    exec('intra_vwap_tab_{0}_df = intra_vwap_tab_{0}_df.append(pd.DataFrame([tmp_vwap], '
                         'index=[pd.to_datetime(day)]))'.format(i + 1))
                    # print(tmp_close.iloc[-1].iloc[0], tmp_vwap.iloc[0])
                    del tmp_vwap, tmp_close, tmp_volume, tmp_volume_sum
                del close, volume
            gc.collect()
    for i in range(int(240 / split_time)):
        intra_save_path = '/mnt/mfs/DAT_PUBLIC/dat_whs'
        exec('intra_vwap_tab_{0}_df.columns = AZ_clear_columns(intra_vwap_tab_{0}_df.columns)'.format(i + 1))
        exec('intra_close_tab_{0}_df.columns = AZ_clear_columns(intra_close_tab_{0}_df.columns)'.format(i + 1))

        exec('intra_vwap_tab_{0}_df.to_pickle(os.path.join(intra_save_path, '
             '\'intra_vwap_{1}_tab_{0}_2018_2018.pkl\'))'
             .format(i + 1, split_time))
        exec('intra_close_tab_{0}_df.to_pickle(os.path.join(intra_save_path, '
             '\'intra_close_{1}_tab_{0}_2018_2018.pkl\'))'
             .format(i + 1, split_time))


def concat_data():
    for i in range(24):
        data_2005_2018 = pd.read_pickle('/mnt/mfs/DAT_PUBLIC/dat_whs/intra_vwap_10_tab_{}_2005_2018.pkl'.format(i+1))
        data_2018_2018 = pd.read_pickle('/mnt/mfs/DAT_PUBLIC/dat_whs/intra_vwap_10_tab_{}_2018_2018.pkl'.format(i+1))
        data_20181001 = data_2005_2018.combine_first(data_2018_2018)
        EQT_list = [x for x in data_20181001.columns if
                    ((x[0] == '0' or x[0] == '3') and x[-2:] == 'SZ') or (x[0] == '6' and x[-2:] == 'SH')]
        data_20181001_c = data_20181001[EQT_list]
        data_20181001_c.to_pickle('/mnt/mfs/DAT_PUBLIC/dat_whs/intra_vwap_10_tab_{}_20181001.pkl'.format(i+1))

    for i in range(24):
        data_2005_2018 = pd.read_pickle('/mnt/mfs/DAT_PUBLIC/dat_whs/c_vwap_10_tab_{}_2005_2018.pkl'.format(i+1))
        data_2018_2018 = pd.read_pickle('/mnt/mfs/DAT_PUBLIC/dat_whs/intra_vwap_10_tab_{}_2018_2018.pkl'.format(i+1))
        data_20181001 = data_2005_2018.combine_first(data_2018_2018)
        EQT_list = [x for x in data_20181001.columns if
                    ((x[0] == '0' or x[0] == '3') and x[-2:] == 'SZ') or (x[0] == '6' and x[-2:] == 'SH')]
        data_20181001_c = data_20181001[EQT_list]
        data_20181001_c.to_pickle('/mnt/mfs/DAT_PUBLIC/dat_whs/intra_vwap_10_tab_{}_20181001.pkl'.format(i+1))

def create_base_data():
    create_base_row_data()
    create_rzrq_row_data()


if __name__ == '__main__':
    # use_date = '20180329'
    # intra_data = pd.read_csv(f'/mnt/mfs/DAT_PUBLIC/intraday/eqt_1mbar/'
    #                          f'{use_date[:4]}/{use_date[:6]}/{use_date}/Close.csv')
    # create_intra_data(split_time=10)
    concat_data()
