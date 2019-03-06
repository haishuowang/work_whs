import sys
sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *

begin_str = '20100101'
# begin_str = '20180101'
end_str = '20190104'

begin_year, begin_month, begin_day = begin_str[:4], begin_str[:6], begin_str
end_year, end_month, end_day = end_str[:4], end_str[:6], end_str
intraday_path = '/mnt/mfs/DAT_PUBLIC/intraday/eqt_1mbar'
year_list = [x for x in os.listdir(intraday_path) if (x >= begin_year) & (x <= end_year)]
for year in sorted(year_list):
    year_path = os.path.join(intraday_path, year)
    month_list = [x for x in os.listdir(year_path) if (x >= begin_month) & (x <= end_month)]
    for month in sorted(month_list):
        month_path = os.path.join(year_path, month)
        day_list = [x for x in os.listdir(month_path) if (x >= begin_day) & (x <= end_day)]
        for day in sorted(day_list):
            print('............')
            day_path = os.path.join(month_path, day)
            volume = pd.read_csv(os.path.join(day_path, 'Volume.csv'), index_col=0).astype(float)
            data_len = len(volume.index)
            if data_len != 240:
                print(day)
                print(data_len)
