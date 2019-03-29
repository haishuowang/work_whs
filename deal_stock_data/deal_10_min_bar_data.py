# import pandas as pd
# import numpy as np
# import os
import gc
import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *


def AZ_clear_columns(stock_list):
    return [x[2:] + '.' + x[:2] for x in stock_list]


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
    begin_str = '20180101'
    # begin_str = '20180101'
    end_str = '20180104'

    begin_year, begin_month, begin_day = begin_str[:4], begin_str[:6], begin_str
    end_year, end_month, end_day = end_str[:4], end_str[:6], end_str
    intraday_path = '/mnt/mfs/DAT_PUBLIC/intraday_test/eqt_1mbar'

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
                if len(volume.index) == 240:
                    for i in range(int(240 / split_time)):
                        tmp_volume = volume[i * split_time:(i + 1) * split_time]
                        tmp_volume_sum = tmp_volume.sum()
                        tmp_close = close[i * split_time:(i + 1) * split_time]
                        exec(
                            'intra_close_tab_{0}_df = intra_close_tab_{0}_df.append(pd.DataFrame([tmp_close.iloc[-1]], '
                            'index=[pd.to_datetime(day)]))'.format(i + 1))
                        # if sum(tmp_volume_sum == 0) != 0:
                        #     print(1)
                        tmp_vwap = (tmp_close * tmp_volume).sum() / tmp_volume_sum
                        tmp_vwap[tmp_volume_sum == 0] = tmp_close.iloc[0][tmp_volume_sum == 0]
                        exec('intra_vwap_tab_{0}_df = intra_vwap_tab_{0}_df.append(pd.DataFrame([tmp_vwap], '
                             'index=[pd.to_datetime(day)]))'.format(i + 1))
                        # print(tmp_close.iloc[-1].iloc[0], tmp_vwap.iloc[0])
                        del tmp_vwap, tmp_close, tmp_volume, tmp_volume_sum
                else:
                    for i in range(int(240 / split_time)):
                        tmp_value = [np.nan] * len(volume.iloc[-1])
                        exec('intra_close_tab_{0}_df = intra_close_tab_{0}_df.append(pd.DataFrame([tmp_value], '
                             'index=[pd.to_datetime(day)], columns=close.columns))'.format(i + 1))
                        exec('intra_vwap_tab_{0}_df = intra_vwap_tab_{0}_df.append(pd.DataFrame([tmp_value], '
                             'index=[pd.to_datetime(day)], columns=close.columns))'.format(i + 1))
                        del tmp_value
                del close, volume
            gc.collect()
    for i in range(int(240 / split_time)):
        intra_save_path = '/mnt/mfs/DAT_PUBLIC/dat_whs'
        exec('intra_vwap_tab_{0}_df.columns = AZ_clear_columns(intra_vwap_tab_{0}_df.columns)'.format(i + 1))
        exec('intra_close_tab_{0}_df.columns = AZ_clear_columns(intra_close_tab_{0}_df.columns)'.format(i + 1))

        exec('intra_vwap_tab_{0}_df.fillna(method=\'ffill\').to_pickle(os.path.join(intra_save_path, '
             '\'intra_vwap_{1}_tab_{0}_20180101_20190104.pkl\'))'
             .format(i + 1, split_time))
        exec('intra_close_tab_{0}_df.fillna(method=\'ffill\').to_pickle(os.path.join(intra_save_path, '
             '\'intra_close_{1}_tab_{0}_20180101_20190104.pkl\'))'
             .format(i + 1, split_time))


def save_fun(df, save_path, index_list):
    df.index = index_list
    df.fillna(method='ffill', inplace=True)
    df.to_csv(save_path, sep='|')


def part_fun_v2(month_path, day, save_path, split_time):
    index_list = ['09:40', '09:50', '10:00', '10:10', '10:20', '10:30', '10:40', '10:50',
                  '11:00', '11:10', '11:20', '11:30', '13:10', '13:20', '13:30', '13:40',
                  '13:50', '14:00', '14:10', '14:20', '14:30', '14:40', '14:50', '15:00']
    # index_list = ['10:30', '11:30', '14:00', '15:00']

    intra_vwap_df = pd.DataFrame()
    intra_close_df = pd.DataFrame()
    intra_high_df = pd.DataFrame()
    intra_low_df = pd.DataFrame()
    intra_open_df = pd.DataFrame()
    intra_volume_df = pd.DataFrame()

    day_path = os.path.join(month_path, day)
    volume = pd.read_csv(os.path.join(day_path, 'Volume.csv'), index_col=0).astype(float)
    close = pd.read_csv(os.path.join(day_path, 'Close.csv'), index_col=0).astype(float)
    high = pd.read_csv(os.path.join(day_path, 'High.csv'), index_col=0).astype(float)
    low = pd.read_csv(os.path.join(day_path, 'Low.csv'), index_col=0).astype(float)
    open_ = pd.read_csv(os.path.join(day_path, 'Open.csv'), index_col=0).astype(float)

    volume.columns = AZ_clear_columns(volume.columns)
    close.columns = AZ_clear_columns(close.columns)
    high.columns = AZ_clear_columns(high.columns)
    low.columns = AZ_clear_columns(low.columns)
    open_.columns = AZ_clear_columns(open_.columns)

    if len(close.index) == 240 or len(close.index) == 242:
        if len(close.index) == 242:
            volume = volume.astype(float).iloc[1:-1]
            close = close.astype(float).iloc[1:-1]
            high = high.astype(float).iloc[1:-1]
            low = low.iloc[1:-1]
            open_ = open_.iloc[1:-1]

        for i in range(int(240 / split_time)):
            tmp_volume = volume[i * split_time:(i + 1) * split_time]
            tmp_volume_sum = tmp_volume.sum()
            tmp_close = close[i * split_time:(i + 1) * split_time]
            tmp_high = high[i * split_time:(i + 1) * split_time]
            tmp_low = low[i * split_time:(i + 1) * split_time]
            tmp_open = open_[i * split_time:(i + 1) * split_time]

            intra_close_df = intra_close_df.append(pd.DataFrame([tmp_close.iloc[-1]], index=[i + 1]))
            intra_open_df = intra_open_df.append(pd.DataFrame([tmp_open.iloc[0]], index=[i + 1]))
            intra_high_df = intra_high_df.append(pd.DataFrame([tmp_high.max()], index=[i + 1]))
            intra_low_df = intra_low_df.append(pd.DataFrame([tmp_low.min()], index=[i + 1]))
            intra_volume_df = intra_volume_df.append(pd.DataFrame([tmp_volume.sum()], index=[i + 1]))

            tmp_vwap = (tmp_close * tmp_volume).sum() / tmp_volume_sum
            tmp_vwap[tmp_volume_sum == 0] = tmp_close.iloc[0][tmp_volume_sum == 0]
            intra_vwap_df = intra_vwap_df.append(pd.DataFrame([tmp_vwap], index=[i + 1]))

        daily_save_path = f'{save_path}/{day}'
        bt.AZ_Path_create(daily_save_path)
        save_fun(intra_high_df, f'{daily_save_path}/intra_high.csv', index_list)
        save_fun(intra_low_df, f'{daily_save_path}/intra_low.csv', index_list)
        save_fun(intra_close_df, f'{daily_save_path}/intra_close.csv', index_list)
        save_fun(intra_open_df, f'{daily_save_path}/intra_open.csv', index_list)
        save_fun(intra_vwap_df, f'{daily_save_path}/intra_vwap.csv', index_list)
        save_fun(intra_volume_df, f'{daily_save_path}/intra_volume.csv', index_list)

        print(day, 'deal')
    else:
        send_email.send_email(f'{day}', ['whs@yingpei.com'], [], '[intraday data error]')
        print(day, 'error')
        pass


def create_intra_data_v2(split_time=20):
    # begin_str = '20050101'
    begin_str = '20181216'
    end_str = '20190219'

    begin_year, begin_month, begin_day = begin_str[:4], begin_str[:6], begin_str
    end_year, end_month, end_day = end_str[:4], end_str[:6], end_str
    intraday_path = '/mnt/mfs/DAT_EQT/intraday/eqt_1mbar'
    save_path = f'/mnt/mfs/DAT_EQT/intraday/eqt_{split_time}mbar'
    bt.AZ_Path_create(save_path)
    year_list = [x for x in os.listdir(intraday_path) if (x >= begin_year) & (x <= end_year)]
    pool = Pool(25)
    for year in sorted(year_list):
        year_path = os.path.join(intraday_path, year)
        month_list = [x for x in os.listdir(year_path) if (x >= begin_month) & (x <= end_month)]
        for month in sorted(month_list):
            month_path = os.path.join(year_path, month)
            day_list = [x for x in os.listdir(month_path) if (x >= begin_day) & (x <= end_day)]
            for day in sorted(day_list):
                args = (month_path, day, save_path, split_time)
                # part_fun_v2(*args)
                pool.apply_async(part_fun_v2, args=args)
    pool.close()
    pool.join()


def concat_data():
    for i in range(24):
        data_2005_2018 = pd.read_pickle('/mnt/mfs/DAT_PUBLIC/dat_whs/intra_vwap_10_tab_{}_2005_2019.pkl'.format(i + 1))
        data_2005_2018 = data_2005_2018[data_2005_2018.index < pd.to_datetime('20180101')]
        data_2018_2018 = pd.read_pickle('/mnt/mfs/DAT_PUBLIC/dat_whs/intra_vwap_10_tab_{}_2018_2019.pkl'.format(i + 1))

        data_20181001 = data_2005_2018.combine_first(data_2018_2018)
        EQT_list = [x for x in data_20181001.columns if
                    ((x[0] == '0' or x[0] == '3') and x[-2:] == 'SZ') or (x[0] == '6' and x[-2:] == 'SH')]
        data_20181001_c = data_20181001[EQT_list]
        data_20181001_c.to_pickle('/mnt/mfs/DAT_PUBLIC/dat_whs/intra_vwap_10_tab_{}_20181101.pkl'.format(i + 1))


def create_base_data():
    create_base_row_data()
    create_rzrq_row_data()


def part_vwap_table(month_path, day, begin_time, end_time):
    try:
        day_path = os.path.join(month_path, day)
        volume = pd.read_csv(os.path.join(day_path, 'Volume.csv'), index_col=0).astype(float)
        volume = volume.loc[begin_time: end_time]

        close = pd.read_csv(os.path.join(day_path, 'Close.csv'), index_col=0).astype(float).fillna(method='ffill')
        close = close.loc[begin_time: end_time]

        volume.columns = AZ_clear_columns(volume.columns)
        close.columns = AZ_clear_columns(close.columns)
        vwap = ((volume * close).sum() / volume.sum()).round(4)
        vwap.name = day
        print(day, 'deal')
        return vwap
    except:
        print(day, 'error')
        send_email.send_email(day, ['whs@yingpei.com'], [], 'Wonderfully')
        return None


def get_vwap_table(begin_time, end_time):
    begin_str = '20100101'
    # begin_str = '20190218'
    end_str = '20190319'

    begin_year, begin_month, begin_day = begin_str[:4], begin_str[:6], begin_str
    end_year, end_month, end_day = end_str[:4], end_str[:6], end_str
    intraday_path = '/mnt/mfs/DAT_EQT/intraday/eqt_1mbar'
    save_path = f'/mnt/mfs/DAT_EQT/intraday/'
    bt.AZ_Path_create(save_path)
    year_list = [x for x in os.listdir(intraday_path) if (x >= begin_year) & (x <= end_year)]
    result_list = []
    pool = Pool(25)
    for year in sorted(year_list):
        year_path = os.path.join(intraday_path, year)
        month_list = [x for x in os.listdir(year_path) if (x >= begin_month) & (x <= end_month)]
        for month in sorted(month_list):
            month_path = os.path.join(year_path, month)
            day_list = [x for x in os.listdir(month_path) if (x >= begin_day) & (x <= end_day)]
            for day in sorted(day_list):
                args = (month_path, day, begin_time, end_time)
                # part_vwap_table(*args)
                result_list.append(pool.apply_async(part_vwap_table, args=args))
    pool.close()
    pool.join()

    vwap_df = pd.concat([res.get() for res in result_list], axis=1, sort=True).T
    vwap_df.to_csv(f'{save_path}/{begin_time}_{end_time}.csv', sep='|')
    return vwap_df


def part_info_table(month_path, day, begin_time, end_time):
    try:
        day_path = os.path.join(month_path, day)

        turnover = pd.read_csv(os.path.join(day_path, 'Turnover.csv'), index_col=0).astype(float)
        turnover_d = turnover.loc[begin_time: end_time].sum()
        turnover_d.name = day

        volume = pd.read_csv(os.path.join(day_path, 'Volume.csv'), index_col=0).astype(float)
        volume_d = volume.loc[begin_time: end_time].sum()
        volume_d.name = day

        close = pd.read_csv(os.path.join(day_path, 'Close.csv'), index_col=0).astype(float).fillna(method='ffill')
        close_d = close.loc[begin_time: end_time].iloc[-1]
        close_d.name = day

        high = pd.read_csv(os.path.join(day_path, 'High.csv'), index_col=0).astype(float)
        high_d = high.loc[begin_time: end_time].max()
        high_d.name = day

        low = pd.read_csv(os.path.join(day_path, 'Low.csv'), index_col=0).astype(float)
        low_d = low.loc[begin_time: end_time].min()
        low_d.name = day

        open = pd.read_csv(os.path.join(day_path, 'Open.csv'), index_col=0).astype(float)
        open_d = open.loc[begin_time: end_time].iloc[0]
        open_d.name = day
        print(day, 'deal')
        return turnover_d, volume_d, close_d, high_d, low_d, open_d
    except:
        print(day, 'error')
        send_email.send_email(day, ['whs@yingpei.com'], [], 'Wonderfully')
        return None, None, None


def get_info_table(begin_time, end_time):
    begin_str = '20100101'
    # begin_str = '20190218'
    end_str = '20190320'

    begin_year, begin_month, begin_day = begin_str[:4], begin_str[:6], begin_str
    end_year, end_month, end_day = end_str[:4], end_str[:6], end_str
    intraday_path = '/mnt/mfs/DAT_EQT/intraday/eqt_1mbar'
    save_path = f'/mnt/mfs/dat_whs/tmp'
    bt.AZ_Path_create(save_path)
    year_list = [x for x in os.listdir(intraday_path) if (x >= begin_year) & (x <= end_year)]
    result_list = []
    pool = Pool(25)
    for year in sorted(year_list):
        year_path = os.path.join(intraday_path, year)
        month_list = [x for x in os.listdir(year_path) if (x >= begin_month) & (x <= end_month)]
        for month in sorted(month_list):
            month_path = os.path.join(year_path, month)
            day_list = [x for x in os.listdir(month_path) if (x >= begin_day) & (x <= end_day)]
            for day in sorted(day_list):
                args = (month_path, day, begin_time, end_time)
                # part_info_table(*args)
                result_list.append(pool.apply_async(part_info_table, args=args))
    pool.close()
    pool.join()

    result_list = [res.get() for res in result_list]

    turnover_df = pd.concat([res[0] for res in result_list], axis=1, sort=True).T
    volume_df = pd.concat([res[1] for res in result_list], axis=1, sort=True).T
    close_df = pd.concat([res[2] for res in result_list], axis=1, sort=True).T
    high_df = pd.concat([res[3] for res in result_list], axis=1, sort=True).T
    low_df = pd.concat([res[4] for res in result_list], axis=1, sort=True).T
    open_df = pd.concat([res[5] for res in result_list], axis=1, sort=True).T

    turnover_df.columns = AZ_clear_columns(turnover_df.columns)
    volume_df.columns = AZ_clear_columns(volume_df.columns)
    close_df.columns = AZ_clear_columns(close_df.columns)
    high_df.columns = AZ_clear_columns(high_df.columns)
    low_df.columns = AZ_clear_columns(low_df.columns)
    open_df.columns = AZ_clear_columns(open_df.columns)

    turnover_df.to_csv(f'{save_path}/turnover_{begin_time}_{end_time}.csv', sep='|')
    volume_df.to_csv(f'{save_path}/volume_{begin_time}_{end_time}.csv', sep='|')
    close_df.to_csv(f'{save_path}/close_{begin_time}_{end_time}.csv', sep='|')
    high_df.to_csv(f'{save_path}/high_{begin_time}_{end_time}.csv', sep='|')
    low_df.to_csv(f'{save_path}/low_{begin_time}_{end_time}.csv', sep='|')
    open_df.to_csv(f'{save_path}/open_{begin_time}_{end_time}.csv', sep='|')


def adj_vwap(price_df):
    factor_1 = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_FACTOR1/TAFACTOR.csv')
    factor_1 = factor_1.reindex(index=price_df.index)
    price_df = price_df.reindex(columns=factor_1.columns)
    return (price_df / factor_1).pct_change()


def adj_price(price_df):
    factor_1 = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_FACTOR1/TAFACTOR.csv')
    factor_1 = factor_1.reindex(index=price_df.index)
    price_df = price_df.reindex(columns=factor_1.columns)
    return price_df / factor_1


def adj_columns(price_df):
    factor_1 = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_FACTOR1/TAFACTOR.csv')
    factor_1 = factor_1.reindex(index=price_df.index)
    price_df = price_df.reindex(columns=factor_1.columns)
    return price_df


if __name__ == '__main__':
    # use_date = '20180329'
    # intra_data = pd.read_csv(f'/mnt/mfs/DAT_PUBLIC/intraday/eqt_1mbar/'
    #                          f'{use_date[:4]}/{use_date[:6]}/{use_date}/Close.csv')
    # create_intra_data(split_time=60)
    # send_email.send_email('1 hour vwap', ['whs@yingpei.com'], [], '1 hour vwap created')
    # create_intra_data(split_time=10)

    # a = time.time()
    # begin_time, end_time = '09:40', '10:00'
    # vwap_df = get_vwap_table(begin_time, end_time)
    #
    # begin_time, end_time = '14:00', '14:15'
    # vwap_df = get_vwap_table(begin_time, end_time)
    #
    # begin_time, end_time = '14:30', '14:50'
    # vwap_df = get_vwap_table(begin_time, end_time)
    #
    # b = time.time()
    # print(b - a)

    # a = time.time()
    # create_intra_data_v2(split_time=10)
    # b = time.time()
    # print(b - a)

    for file_name in ['09:40_10:00', '14:00_14:15', '14:30_14:50']:
        data = bt.AZ_Load_csv(f'/mnt/mfs/DAT_EQT/intraday/{file_name}.csv')
        vwap_return = adj_vwap(data)
        file_name = file_name.replace(':', '')
        vwap_return.to_csv(f'/mnt/mfs/DAT_EQT/intraday/vwap_return/aadj_r{file_name}.csv', sep='|')

    # file_name_list = ['close_09:30_14:15',
    #                   'high_09:30_14:15',
    #                   'low_09:30_14:15',
    #                   'open_09:30_14:15']
    # for file_name in file_name_list:
    #     data = bt.AZ_Load_csv(f'/mnt/mfs/dat_whs/tmp/{file_name}.csv')
    #     adj_data = adj_price(data)
    #     f_name = file_name.split('_')[0]
    #     adj_data.to_csv(f'/mnt/mfs/DAT_EQT/intraday/daily_0930_1415/adj_{f_name}.csv', sep='|')

    # file_name_list = ['close', 'high', 'low', 'open', 'volume']
    # for file_name in file_name_list:
    #     data = bt.AZ_Load_csv(f'/mnt/mfs/DAT_EQT/intraday/daily_0930_1415/{file_name}.csv')
    #     adj_data = adj_columns(data)
    #     adj_data.to_csv(f'/mnt/mfs/DAT_EQT/intraday/daily_0930_1415/{file_name}.csv', sep='|')

    # begin_time, end_time = '09:30', '14:15'
    # get_info_table(begin_time, end_time)
