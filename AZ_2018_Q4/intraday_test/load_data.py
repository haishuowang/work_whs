import sys
import talib as ta

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *


# color_list = [
#     '#F0F8FF',
#     '#FAEBD7',
#     '#00FFFF',
#     '#7FFFD4',
#     '#F0FFFF',
#     '#F5F5DC',
#     '#FFE4C4',
#     '#000000',
#     '#FFEBCD',
#     '#0000FF',
#     '#8A2BE2',
#     '#A52A2A',
#     '#DEB887',
#     '#5F9EA0',
#     '#7FFF00',
#     '#D2691E',
#     '#FF7F50',
#     '#6495ED',
#     '#FFF8DC',
#     '#DC143C',
#     '#00FFFF',
#     '#00008B',
#     '#008B8B',
#     '#B8860B',
#     '#A9A9A9',
#     '#006400',
#     '#BDB76B',
#     '#8B008B',
#     '#556B2F',
#     '#FF8C00',
#     '#9932CC',
#     '#8B0000',
#     '#E9967A',
#     '#8FBC8F',
#     '#483D8B',
#     '#9ACD32']

def BBANDS(series, window=20):
    std_ser = series.rolling(window).std()
    ma_ser = series.rolling(window).mean()

    tmp_ser = (series - ma_ser).div(std_ser)
    # target_ser = tmp_ser.copy()
    # target_ser[tmp_ser >= 2] = 1
    # target_ser[tmp_ser <= -2] = -1
    # target_ser[(tmp_ser > -2) & (tmp_ser < 2)] = np.nan
    # target_ser.fillna(method='ffill', inplace=True)
    return tmp_ser


def plot_fun(df, color):
    for col in df.columns:
        plt.plot(df[col].values, c=color)


def get_group_index(group_df, stock_return_df):
    group_index_df = pd.DataFrame()
    # print(group_df)
    # figure_save_path_list = []
    for num_group, x in group_df.groupby(0):
        # print(num_group, len(x.index))
        if len(x.index) > 3:
            # group_df_index = pd.DataFrame(stock_return_df[x.index].mean())
            # group_df_index
            a = stock_return_df[x.index]
            group_index = pd.DataFrame(a.mean(1), columns=[num_group])
            group_index_df = pd.concat([group_index_df, group_index], axis=1)
    return group_index_df


def get_group_index_1(group_df, AR1):
    group_index_df = pd.DataFrame()
    # print(group_df)
    # figure_save_path_list = []
    for num_group, x in group_df.groupby(0):
        # print(num_group, len(x.index))
        if len(x.index) > 3:
            # group_df_index = pd.DataFrame(stock_return_df[x.index].mean())
            # group_df_index
            a = AR1[x.index]
            group_index = pd.DataFrame(a.mean(1), columns=[num_group])
            group_index_df = pd.concat([group_index_df, group_index], axis=1)
    return group_index_df


def get_group_signal(group_df, stock_return_df, today, long_short=0, percent=0.3):
    factor_df = pd.DataFrame()
    for num_group, x in group_df.groupby(0):
        if len(x.index) > 3:
            group_return = stock_return_df[x.index]
            window = int(9 * len(group_return) / 10)
            target_ser = group_return.apply(BBANDS, args=(window,))
            part_factor = target_ser.iloc[-1].replace(0, np.nan)
            factor_df = pd.concat([factor_df, part_factor], axis=0)
            # print(1)

    signal_df = factor_df[0].sort_values()
    signal_df[:int(len(signal_df) * percent)] = 1
    signal_df[-int(len(signal_df) * percent):] = -1
    signal_df[int(len(signal_df) * percent):-int(len(signal_df) * percent)] = 0
    signal_df.name = today
    if long_short == 0:
        return pd.DataFrame(signal_df).T
    elif long_short == 1:
        return pd.DataFrame(signal_df[signal_df > 0]).T
    elif long_short == -1:
        return pd.DataFrame(signal_df[signal_df < 0]).T

    else:
        return pd.DataFrame(columns=[today])


def get_train_data(date_file_list, back_look_num, i, intra_path, today):
    target_df = pd.DataFrame()
    for tmp_date in date_file_list[i - back_look_num: i]:
        tmp_path = f'{intra_path}/{tmp_date}'
        tmp_intra_vwap = pd.read_csv(f'{tmp_path}/intra_vwap.csv', index_col=0, sep='|')
        tmp_intra_vwap.index = pd.to_datetime(tmp_date + ' ' + tmp_intra_vwap.index)
        target_df = target_df.append(tmp_intra_vwap.pct_change().dropna(how='all', axis='index'), sort=True)

    tmp_intra_vwap = pd.read_csv(f'{intra_path}/{today}/intra_vwap.csv', index_col=0, sep='|')
    tmp_intra_vwap.index = pd.to_datetime(today + ' ' + tmp_intra_vwap.index)
    tmp_intra_vwap = tmp_intra_vwap.iloc[:12]

    target_df = target_df.append(tmp_intra_vwap.pct_change().dropna(how='all', axis='index'), sort=True)
    return target_df


def signal_fun(group_index_df):
    return group_index_df.cumsum().iloc[-1].sort_values()


def signal_fun_1(group_index_df):
    return group_index_df.apply(bt.AZ_Sharpe_y).sort_values()


def signal_fun_2(group_index_df):
    return group_index_df.apply(np.std).sort_values()


def get_part_signal(group_index_df, group_info_df, today, long_short=0):
    sorted_group_df = signal_fun_2(group_index_df)
    use_num = 2
    group_short = sorted_group_df.index[:use_num]
    group_long = sorted_group_df.index[-use_num:]
    if long_short == 0:
        stock_short = group_info_df.applymap(lambda x: -1 if x in group_short else 0)
        stock_short = stock_short / stock_short.abs().sum() / 2

        stock_long = group_info_df.applymap(lambda x: 1 if x in group_long else 0)
        stock_long = stock_long / stock_long.sum() / 2

        part_signal_df = stock_long + stock_short

    elif long_short == 1:

        stock_long = group_info_df.applymap(lambda x: 1 if x in group_long else 0)
        stock_long = stock_long / stock_long.sum()

        part_signal_df = stock_long

    elif long_short == -1:
        stock_short = group_info_df.applymap(lambda x: 1 if x in group_short else 0)
        stock_short = stock_short / stock_short.abs().sum()
        part_signal_df = stock_short

    else:
        part_signal_df = pd.DataFrame()
    part_signal_df.columns = [today]
    return part_signal_df


def daily_signal_fun(intra_path, i, date_file_list, sector_df, back_look_day, long_short):
    today = date_file_list[i]

    stock_list = sector_df.loc[pd.to_datetime(today)].dropna().index
    try:
        train_df = get_train_data(date_file_list, back_look_day, i, intra_path, today)
        train_df = train_df.reindex(columns=stock_list).dropna(how='any', axis='columns')
        train_df_cumsum = train_df.cumsum()
        train_df_cumsum.index = range(len(train_df_cumsum.index))
        kmeans = KMeans(n_clusters=30).fit(train_df.T)

        kmeans_result = kmeans.labels_
        columns_list = train_df.columns
        group_info_df = pd.DataFrame(kmeans_result, index=columns_list)
        # group_index_df = get_group_index(group_info_df, train_df)
        # part_signal_df = get_part_signal(group_index_df, group_info_df, today, long_short)
        group_info_df.to_csv(f'/mnt/mfs/dat_whs/tmp/kmean_intra_res/{today}.csv')
        return part_signal_df.T
    except Exception as error:
        print(error, today, i)
        return pd.DataFrame(columns=stock_list, index=[today])


def daily_signal_fun_1(intra_path, i, date_file_list, sector_df, back_look_day, long_short):
    today = date_file_list[i]

    stock_list = sector_df.loc[pd.to_datetime(today)].dropna().index
    try:
        train_df = get_train_data(date_file_list, back_look_day, i, intra_path, today)
        train_df = train_df.reindex(columns=stock_list).dropna(how='any', axis='columns')
        train_df_cumsum = train_df.cumsum()
        train_df_cumsum.index = range(len(train_df_cumsum.index))
        kmeans = KMeans(n_clusters=30).fit(train_df.T)

        kmeans_result = kmeans.labels_
        columns_list = train_df.columns
        group_info_df = pd.DataFrame(kmeans_result, index=columns_list)
        group_signal_df = get_group_signal(group_info_df, train_df, today, long_short)
        return group_signal_df
    except Exception as error:
        print(error, today, i)
        return pd.DataFrame(columns=stock_list, index=[today])


def daily_signal_fun_2(intra_path, i, date_file_list, sector_df, back_look_day, long_short, AR1):
    today = date_file_list[i]

    stock_list = sector_df.loc[pd.to_datetime(today)].dropna().index
    try:
        train_df = get_train_data(date_file_list, back_look_day, i, intra_path, today)
        train_df = train_df.reindex(columns=stock_list).dropna(how='any', axis='columns')
        train_df_cumsum = train_df.cumsum()
        train_df_cumsum.index = range(len(train_df_cumsum.index))
        kmeans = KMeans(n_clusters=30).fit(train_df.T)

        kmeans_result = kmeans.labels_
        columns_list = train_df.columns
        group_info_df = pd.DataFrame(kmeans_result, index=columns_list)
        group_index_df = get_group_index_1(group_info_df, AR1)

        part_signal_df = get_part_signal(group_index_df, group_info_df, today, long_short)
        return part_signal_df.T
    except Exception as error:
        print(error, today, i)
        return pd.DataFrame(columns=stock_list, index=[today])


def get_all_signal(root_path, sector_df, begin_date, end_date, long_short, back_look_day=5):
    intra_path = f'{root_path}/intraday/eqt_10mbar'

    date_file_list = sorted([x for x in os.listdir(intra_path) if
                             end_date.strftime('%Y%m%d') >= x >= begin_date.strftime('%Y%m%d')])

    result_list = []
    a = time.time()
    pool = Pool(25)
    for i in range(back_look_day, len(date_file_list)):
        args = (intra_path, i, date_file_list, sector_df, back_look_day, long_short)
        # daily_signal_fun(*args)
        # daily_signal_fun_1(*args)
        result_list.append(pool.apply_async(daily_signal_fun, args))
        # result_list.append(pool.apply_async(daily_signal_fun_1, args))
    pool.close()
    pool.join()

    signal_df = pd.concat([res.get() for res in result_list], sort=False).fillna(0)
    b = time.time()
    print(b - a)
    signal_df.index = pd.to_datetime(signal_df.index)
    return signal_df


def get_all_signal_1(root_path, sector_df, begin_date, end_date, long_short, back_look_day, AR1):
    intra_path = f'{root_path}/intraday/eqt_10mbar'

    date_file_list = sorted([x for x in os.listdir(intra_path) if
                             end_date.strftime('%Y%m%d') >= x >= begin_date.strftime('%Y%m%d')])

    result_list = []
    a = time.time()
    pool = Pool(25)
    for i in range(back_look_day, len(date_file_list)):
        args = (intra_path, i, date_file_list, sector_df, back_look_day, long_short, AR1)
        # daily_signal_fun(*args)
        daily_signal_fun_2(*args)
        # result_list.append(pool.apply_async(daily_signal_fun_2, args))
        # result_list.append(pool.apply_async(daily_signal_fun_1, args))
    # pool.close()
    # pool.join()

    signal_df = pd.concat([res.get() for res in result_list], sort=False).fillna(0)
    b = time.time()
    print(b - a)
    signal_df.index = pd.to_datetime(signal_df.index)
    return signal_df


if __name__ == '__main__':
    sector_name = 'market_top_300plus'
    begin_date = '20100101'
    end_date = '20190101'
    # get_all_signal(sector_name, begin_date, end_date)
