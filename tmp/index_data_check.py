import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *


def check_fun(today):

    target_path = f'/mnt/mfs/DAT_EQT/intraday/eqt_1mbar/{today[:4]}/{today[:6]}/{today}/Close.csv'
    if os.path.exists(target_path):
        target_data = pd.read_csv(target_path, index_col=0)
        index_set = {'SH000001', 'SH000300', 'SH000905', 'SH000906'}
        if index_set.issubset(set(target_data.columns)):
            print(today_date, 'success')
        else:
            print(today_date, 'error')
    else:
        pass


begin_date = pd.to_datetime('20150101')
end_date = pd.to_datetime('20190313')
today_date = begin_date
while today_date <= end_date:
    # print(today_date)
    today = today_date.strftime('%Y%m%d')
    check_fun(today)
    today_date += timedelta(1)