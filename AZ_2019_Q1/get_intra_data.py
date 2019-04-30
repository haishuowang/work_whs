import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *


def save_fun(data, target_path):
    data.columns = [x[2:] + '.' + x[:2] for x in data.columns]
    data = data.reindex(columns=sorted(data.columns))
    data.to_csv(target_path, sep='|')


def AZ_filter_stock(stock_list):  # 筛选相应股票池
    target_list = [x for x in stock_list if x[:2] == 'SH' and x[2] == '6' or
                   x[:2] == 'SZ' and x[2] in ['0', '3']]
    return target_list


def daily_deal_fun(day, day_path, cut_num):
    print(day)
    volume = pd.read_csv(os.path.join(day_path, 'Volume.csv'), index_col=0).astype(float)
    volume = volume[AZ_filter_stock(volume.columns)]
    part_open_min_vol = volume.iloc[:cut_num].sum()
    part_open_min_vol.name = day

    close = pd.read_csv(os.path.join(day_path, 'Close.csv'), index_col=0).astype(float)

    part_open_min_return = close.iloc[cut_num]/close.iloc[0].replace(0, np.nan) - 1
    part_open_min_return.name = day
    # part_open_min_return = part_open_10min_return
    return part_open_min_vol, part_open_min_return


def intra_data_get(begin_str, end_str):
    begin_year, begin_month, begin_day = begin_str[:4], begin_str[:6], begin_str
    end_year, end_month, end_day = end_str[:4], end_str[:6], end_str
    intraday_path = '/mnt/mfs/DAT_EQT/intraday/eqt_1mbar'
    save_path = '/mnt/mfs/dat_whs'
    cut_num = 15

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
                # daily_deal_fun(day, day_path, cut_num)
                result_list.append(pool.apply_async(daily_deal_fun, (day, day_path, cut_num)))

    part_open_min_vol = pd.concat([x.get()[0] for x in result_list], axis=1, sort=True)
    part_open_min_return = pd.concat([x.get()[1] for x in result_list], axis=1, sort=True)

    # target_df = pd.concat([x.get() for x in result_list], axis=1, sort=True)
    save_fun(part_open_min_vol.T, f'{save_path}/intra_open_{cut_num}min_vol')
    save_fun(part_open_min_return.T, f'{save_path}/intra_open_{cut_num}min_return')


if __name__ == '__main__':
    begin_str = '20150101'
    end_str = '20190322'
    intra_data_get(begin_str, end_str)
