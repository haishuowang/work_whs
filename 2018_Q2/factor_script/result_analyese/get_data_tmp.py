import pandas as pd
import os

path1 = '/mnt/mfs/dat_whs/data/new_factor_data/market_top_1000_industry_10_15'
path2 = '/mnt/mfs/DAT_EQT/EM_Funda/daily'

# data1 = pd.read_pickle(os.path.join(path1, 'R_TotRev_TTM_QYOY.csv'))
data2 = pd.read_csv(os.path.join(path2, 'R_TotRev_TTM_QYOY.csv'), sep='|', index_col=0, parse_dates=True)
