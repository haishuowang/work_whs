import sys

sys.path.append('/mnf/mfs')

from work_whs.loc_lib.pre_load import *
from EmQuantAPI import *

loginResult = c.start("ForceLogin=1")

factor_name_list = ['INFLOWRATE']
stock_code = c.sector("001004", '20190219').Codes
# stock_code_d = c.sector("001026", '20190219').Codes
stock_code_test = bt.AZ_get_stock_name()


def get_choice_data(stock_code):
    data = c.csd(stock_code, "INFLOWRATE",
                 "2010-01-01", "2019-02-20", "period=1,adjustflag=1,curtype=1,pricetype=1,order=1,market=CNSESH")
    if data.ErrorCode==0:
        target_df = pd.DataFrame(data.Data[stock_code], columns=data.Dates, index=[stock_code]).T
        return target_df
    else:
        print('error')
        return None


result_list = []

for stock_code in stock_code_test:
    print(stock_code)
    result_list.append(get_choice_data(stock_code))

target_df = pd.concat([x for x in result_list if x is not None], axis=1)
target_df.index = pd.to_datetime(target_df.index)

# print(data.Data)

# logoutResult = c.stop()
# data = c.csd("000001.SZ",
#              "INFLOWRATE",
#              "2019-02-19", "2019-02-20", "period=1,adjustflag=1,curtype=1,pricetype=1,order=1,market=CNSESH")
# data = c.csd("000001.SZ",
#              "INFLOWRATE",
#              "2010-01-19", "2019-02-20", "period=1,adjustflag=1,curtype=1,pricetype=1,order=1,market=CNSESH")
