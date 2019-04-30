__author__ = 'jerry'

from EmQuantAPI import *

import pandas as pd
import datetime as datetime


######东方财富接口数据
def get_TradeDates(start_date, end_date):
    # names = {'SHF': 'CNFESF',
    #          'DCE': 'CNFEDC',
    #          'CZC': 'CNFEZC',
    #          'CFFE': 'CNFF00',
    #          'INE':  }
    c.start("ForceLogin=0")
    # EMI00069027 COMEX库存：银, EMI00069031 LME总库存：锌, EMI00069038 库存：阴极铜, EMI00069047 仓单数量：铅,
    # EMI00018808 焦炭库存：天津港, EMI00018809 焦炭库存：连云港, EMI00018810 焦炭库存：日照港, EMM00167660 仓单数量：天然橡胶

    QuantAPI_data = c.tradedates(f'{start_date}', f'{end_date}', period=1, order=1, market="CNFESF")

    c.stop()
    data = pd.DataFrame(1, index=QuantAPI_data.Data, columns=['Active'])
    data.rename_axis('Date', inplace=True, axis=0)
    return data


def main():
    save_path = '/mnt/mfs/DAT_FUT/DailyPX/TradeDates'
    start_date = str(datetime.datetime.now().year-8) + '-01-15'
    end_date = str(datetime.datetime.now().year + 1) + '-01-15'

    # access new data
    data_collected = get_TradeDates(start_date, end_date)
    data_collected.index = pd.to_datetime(data_collected.index)
    data_collected.to_csv(save_path, sep='|')
    # read old data
    # old_data = pd.read_csv(save_path, index_col=0, sep='|')
    # old_data.index = pd.to_datetime(old_data.index)
    # # update
    # new_data = data_collected.combine_first(old_data)
    # # save
    # new_data.to_csv(save_path, sep='|')


if __name__ == '__main__':
    main()
