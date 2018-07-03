import pandas as pd
from EmQuantAPI import *
from datetime import timedelta, datetime
import time
import os

etf_fund_set = {'510050.SH': '华夏上证50ETF', '510330.SH': '华夏沪深300ETF', '512500.SH': '华夏中证500ETF'}

# loginResult = c.start("ForceLogin=1")
# # data = c.sector("507045", "2018-07-03")
# # data=c.sector("507046", "2018-07-03")
save_path = '/mnt/mfs/dat_whs/data/index_data'
# begin_date = "2008-01-01"
# end_date = "2018-07-01"
# for etf_code in etf_fund_set.keys():
#     data = c.csd(etf_code, "ADJUSTEDNAV,OPEN,CLOSE,HIGH,LOW,AMOUNT,VOLUME,TURN", begin_date, end_date,
#                  "period=1,adjustflag=1,curtype=1,pricetype=1,order=1,market=CNSESH")
#     target_df = pd.DataFrame(data.Data[etf_code], index=data.Indicators, columns=data.Dates).T
#     target_df.to_csv(os.path.join(save_path, etf_code + '.csv'))
#
# outResult = c.stop()


def etf_fnd_deal(etf_code_list, load_path, save_path, n=1):
    for etf_code in etf_code_list:
        load_path_file = os.path.join(load_path, etf_code + '.csv')
        etf_data = pd.read_csv(load_path_file, index_col=0)['ADJUSTEDNAV']
        etf_fnd = etf_data.shift(-n)/etf_data - 1
        etf_fnd.to_csv(os.path.join(save_path, etf_code + '_f{}d.csv'.format(n)))


if __name__ == '__main__':
    etf_fnd_deal(list(etf_fund_set.keys()), save_path, save_path, n=1)
