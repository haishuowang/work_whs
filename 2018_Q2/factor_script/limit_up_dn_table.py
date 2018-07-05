import pandas as pd
import numpy as np

# data = pd.read_csv('/mnt/mfs/dat_whs/data/AllStock/all_close.csv', index_col=0)
# data.index = pd.to_datetime(data.index.astype(str))
# EQA = [x for x in data.columns if x.startswith('0') or x.startswith('3') or x.startswith('6')]
# data = data[EQA]
# data_pct = data/data.shift(1) -1
# data_copy = data_pct.copy()
# data_copy[(data_pct < -0.0975) | (data_pct > 0.0975)] = np.nan
# data_copy[(data_pct >= -0.0975) & (data_pct <= 0.0975)] = 1
# data_copy.to_pickle('/mnt/mfs/dat_whs/data/locked_date/limit_updn_table.pkl')
# date = pd.to_datetime('2008-01-22')

# 处理停牌数据
suspendday_df = pd.read_pickle('/mnt/mfs/DAT_EQT/EM_Tab14/adj_data/TRAD_TD_SUSPENDDAY/SUSPENDREASON.pkl')
limit_updn_df = pd.read_pickle('/mnt/mfs/dat_whs/data/locked_date/limit_updn_table.pkl')

begin_date = pd.to_datetime('20080101')
end_date = pd.to_datetime('20180401')

suspendday_df = suspendday_df.reindex(columns=limit_updn_df.columns, fill_value=None)
suspendday_df = suspendday_df[(suspendday_df.index >= begin_date) & (suspendday_df.index < end_date)]

mask = suspendday_df.notnull()
suspendday_df[mask] = np.nan
suspendday_df = suspendday_df.where(mask, other=1)
suspendday_df.to_pickle('/mnt/mfs/DAT_EQT/EM_Tab14/adj_data/TRAD_TD_SUSPENDDAY/SUSPENDREASON_adj.pkl')
