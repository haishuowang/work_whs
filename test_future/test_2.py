import os
import pandas as pd

data = pd.read_csv('/mnt/mfs/DAT_EQT/EM_Funda/daily/R_WorkCapital_First.csv',
                   index_col=0, sep='|')

# (6 * 7)
root_path = '/mnt/mfs/DAT_EQT/intraday/eqt_1mbar'
for target_date in data.index[data.index > '2019-10-20']:
    target_date = target_date.replace('-', '')
    close_path = f'{root_path}/{target_date[:4]}/{target_date[:6]}/{target_date}/Close.csv'
    if os.path.exists(close_path):
        date_df = pd.read_csv(close_path, index_col=0)
        print(target_date, len(date_df.index))
    else:
        print(target_date, 'error')
