import pandas as pd
import numpy as np

root_path = '/media/hdd1/DAT_EQT'
root_path = '/mnt/mfs/DAT_EQT'

HS300 = pd.read_csv(root_path + '/EM_Funda/IDEX_YS_WEIGHT_A/SECURITYNAME_000300.csv',
                    index_col=0, parse_dates=True, sep='|', low_memory=False)

ZZ500 = pd.read_csv(root_path + '/EM_Funda/IDEX_YS_WEIGHT_A/SECURITYNAME_000905.csv',
                    index_col=0, parse_dates=True, sep='|', low_memory=False)

market_top_800 = pd.read_csv(root_path + '/EM_Funda/DERIVED_10/market_top_800.csv',
                             index_col=0, parse_dates=True, sep='|', low_memory=False)

xnms = sorted(set(market_top_800.columns) | set(ZZ500.columns) | set(HS300.columns))
xinx = sorted(set(market_top_800.index) | set(ZZ500.index) | set(HS300.index))

HS300 = HS300.reindex(columns=xnms, index=xinx)
ZZ500 = ZZ500.reindex(columns=xnms, index=xinx)
market_top_800 = market_top_800.reindex(columns=xnms, index=xinx)

ZZ500_mask = ZZ500.notna()
HS300_mask = HS300.notna()
market_top_800_mask = market_top_800.notna()

a = HS300_mask | ZZ500_mask | market_top_800_mask
b = a.astype(float).replace(0, np.nan)
# b.to_csv('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_10/market_top_888.csv', sep='|')
import matplotlib.pyplot as plt
