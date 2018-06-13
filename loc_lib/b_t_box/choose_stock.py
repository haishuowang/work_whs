import pandas as pd
import numpy as np
import gc


def high_vol_stock(pre_price_data, pre_turnover_data,  p_num=200, t_num=300, v_num=500):
    df = pre_price_data.apply(lambda x: np.nanstd(x)/np.nanmean(x))
    universe_v = df.sort_values(ascending=False).index.values[:v_num]
    universe_v_t = pre_turnover_data[universe_v].sum().sort_values(ascending=False).index[:t_num]
    universe_v_t_p = pre_price_data[universe_v_t].mean().sort_values(ascending=False).index[:p_num]
    return universe_v_t_p


def pcf(df):
    pcf_df = (df - df.shift(1)) / df.shift(1)
    return


def HC300_universe():
    HC300_path = r'/media/hdd0/data/raw_data/equity/extraday/choice/index/000300.SH'
    HC300_data = pd.read_csv(HC300_path, index_col=0)
    return HC300_data

# begin_date = '20000101'
# end_date = '20000201'
#
# price, volume, vwap = gd.eqt_1mbar_data(begin_date, end_date)


# df.drop(columns='Time', inplace=True)
# high_vol_stock(data, num=300)
