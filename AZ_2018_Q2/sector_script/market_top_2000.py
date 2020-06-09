import pandas as pd
import numpy as np
import time
import statsmodels.formula.api as sm
from sklearn import linear_model


def fun1(t):
    t = t.dropna()
    target_array = t/sum(t * t)
    return target_array


from scipy import stats
# import random
#
# # a = np.array([random.random() for i in range(1000)]).reshape(10, 100)
# # a = pd.DataFrame(a)
# # b = np.array([random.random() for i in range(1000)]).reshape(10, 100)
# # b = pd.DataFrame(b)
# # a = pd.DataFrame([[1, 2, 3, 4, 5]] * 10)
# # b = pd.DataFrame([[1, 2, 3, 4, 5]] * 10)
#
# # regr = linear_model.LinearRegression().fit([a[0].values], b[0].values)
#
# regr = linear_model.LinearRegression()
# market_df = pd.read_csv('/mnt/mfs/DAT_EQT/EM_Funda/LICO_YS_STOCKVALUE/AmarketCap.csv',
#                         sep='|', index_col=0, parse_dates=True)
# return_df = pd.read_csv('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
#                         sep='|', index_col=0, parse_dates=True)
# xnms = sorted(list(set(market_df.columns) | set(return_df.columns)))
# xinx = sorted(list(set(market_df.index) | set(return_df.index)))
#
# market_df = market_df.reindex(columns=xnms, index=xinx)
# return_df = return_df.reindex(columns=xnms, index=xinx)
#
# mask = market_df.notnull() | return_df.notnull()
# # mask.loc[pd.to_datetime('2018-09-28'), '601313.SH']
# market_df_c = market_df[mask]
# return_df_c = return_df[mask]
#
# a = market_df_c.iloc[-1]
# b = return_df_c.iloc[-1]
#
# a = a.dropna()
# b = b.dropna()
#
# market_df_c.apply(fun1, axis=1)
