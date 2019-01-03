import sys
from sqlalchemy import create_engine

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *
from multiprocessing import Pool, Manager
import open_lib.shared_tools.back_test as bt

print(datetime.now().strftime('%Y%m%d'))
today_date = datetime.now().strftime('%Y%m%d')


# data = pd.read_csv('/media/hdd1/data_clear/20181221_TotalVolumeTrade_1min.txt', sep='|', header=None)

def index_deal(index):
    return [x[8:10] + ':' + x[10:12] for x in index.astype(str)]


def columns_deal(columns):
    return [x[-2:] + x[:6] for x in columns.astype(str)]


def mindata_to_file(today):
    file_dict = OrderedDict({'Close': 'Close',
                             'High': 'High',
                             'Low': 'Low',
                             'Open': 'Open',
                             'Turnover': 'TotalValueTrade',
                             'Volume': 'TotalVolumeTrade'})
    intra_save_path = f'/mnt/mfs/DAT_PUBLIC/intraday/eqt_1mbar/{today[:4]}/{today[:6]}/{today}'
    print(intra_save_path)
    if os.path.exists(f'/media/hdd1/data_clear/{today}_Close_1min.txt'):
        bt.AZ_Path_create(intra_save_path)
        for target_file in file_dict.keys():
            raw_file = pd.read_csv(f'/media/hdd1/data_clear/{today}_{file_dict[target_file]}_1min.txt',
                                   sep='|', header=None)
            tmp_df = raw_file.pivot(0, 1, 2)
            tmp_df.index = index_deal(tmp_df.index)
            tmp_df.columns = columns_deal(tmp_df.columns)
            tmp_df.to_csv(f'{intra_save_path}/{target_file}.csv')
    else:
        pass


if __name__ == '__main__':
    begin_date = pd.to_datetime('20181201')
    end_date = pd.to_datetime('20181225')
    # today_date = begin_date
    # while today_date <= end_date:
    #     print(today_date)
    #     today = today_date.strftime('%Y%m%d')
    #     mindata_to_file(today)
    #     today_date += timedelta(1)

    # today = datetime.now().strftime('%Y%m%d')
    # mindata_to_file(today)
