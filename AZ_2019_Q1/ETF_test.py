import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *


def part_get_etf_data(day_path, day):

    target_list = ['SH510050', 'SH510300', 'SH510500']
    close = pd.read_csv(os.path.join(day_path, 'Close.csv'), index_col=0).astype(float)
    close.index = day + ' ' + close.index
    # part_df = close[target_list]
    if len(list(set(target_list) - set(close.columns)))!=0:
        print(day)
    # return part_df


if __name__ == '__main__':
    begin_str = '20100101'
    end_str = '20190322'

    begin_year, begin_month, begin_day = begin_str[:4], begin_str[:6], begin_str
    end_year, end_month, end_day = end_str[:4], end_str[:6], end_str
    intraday_path = '/mnt/mfs/DAT_EQT/intraday/eqt_1mbar'
    index_save_path = '/mnt/mfs/dat_whs/EM_Funda/dat_whs'
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
                result_list.append(pool.apply_async(part_get_etf_data, (day_path, day)))
    pool.close()
    pool.join()

    target_df = pd.concat([res.get() for res in result_list], axis=0)
    target_df.to_csv('/mnt/mfs/dat_whs/ETF_data.csv')
