import pandas as pd
import numpy as np


def AZ_split_stock(stock_list):
    """
    在stock_list中寻找A股代码
    :param stock_list:
    :return:
    """
    eqa = [x for x in stock_list if (x.startswith('0') or x.startswith('3')) and x.endswith('SZ')
           or x.startswith('6') and x.endswith('SH')]
    return eqa


def get_new_stock_info():
    begin_date = pd.to_datetime('20100101')
    end_date = pd.to_datetime('20180601')

    new_stock_data = pd.read_pickle('/mnt/mfs/DAT_EQT/EM_Tab01/CDSY_SECUCODE/LISTSTATE.pkl')
    new_stock_data.fillna(method='ffill', inplace=True)
    target_df = new_stock_data.shift(40)[(new_stock_data.index >= begin_date) &
                                         (new_stock_data.index < end_date)].notnull()
    eqa = AZ_split_stock(target_df.columns)
    target_df = target_df[eqa]
    return target_df


def get_st_stock_info():
    begin_date = pd.to_datetime('20100101')
    end_date = pd.to_datetime('20180601')
    data = pd.read_pickle('/mnt/mfs/DAT_EQT/EM_Tab01/CDSY_CHANGEINFO/CHANGEA.pkl')
    data.fillna(method='ffill', inplace=True)
    data = data[(data.index >= begin_date) & (data.index < end_date)]
    data = data.astype(str)
    target_df = data.applymap(lambda x: 0 if 'ST' in x or 'PT' in x else 1)
    # target_df.to_pickle('/mnt/mfs/dat_whs/data/error_stock_info/st_pt_stock.pkl')
    return target_df


if __name__ == '__main__':
    get_new_stock_info()
    get_st_stock_info()
