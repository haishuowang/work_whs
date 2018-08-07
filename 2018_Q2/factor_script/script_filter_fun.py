import numpy as np
import loc_lib.shared_tools.back_test as bt
import pandas as pd


def AZ_Leverage_ratio(asset_df):
    """
    返回250天的return/(负的 一个月的return)
    :param asset_df:
    :return:
    """
    asset_20 = asset_df - asset_df.shift(20)
    asset_250 = asset_df - asset_df.shift(250)
    if asset_250.mean() > 0:
        return asset_250.mean() / (-asset_20.min())
    else:
        return asset_250.mean() / (-asset_20.max())


# 生成每天的position
def pos_daily_fun(df, n=5):
    return df.rolling(window=n, min_periods=1).sum()


def out_sample_perf(pnl_df_out, way=1, cut_point_list=None):
    # 根据sharpe大小,统计样本外的表现
    if cut_point_list is None:
        cut_point_list = [0.30]
    if way == 1:
        rolling_sharpe, cut_sharpe = \
            bt.AZ_Rolling_sharpe(pnl_df_out, roll_year=0.5, year_len=250, cut_point_list=cut_point_list, output=True)
    else:
        rolling_sharpe, cut_sharpe = \
            bt.AZ_Rolling_sharpe(-pnl_df_out, roll_year=0.5, year_len=250, cut_point_list=cut_point_list, output=True)

    sharpe_quantile = cut_sharpe.values[0]
    out_condition = sharpe_quantile > 0.8
    return out_condition, sharpe_quantile * way


def AZ_Pot(pos_df_daily, last_asset):
    trade_times = pos_df_daily.diff().abs().sum().sum()
    if trade_times == 0:
        return 0
    else:
        pot = last_asset / trade_times * 10000
        return round(pot, 2)


def filter_pot_sharpe(cut_date, signal, pct_n, index_df, lag=1, hold_time=5, if_hedge=True, hedge_ratio=1,
                      if_return_pnl=False):
    # signal向下移动一天,避免未来函数
    signal = signal.shift(lag)
    # # 将所有的0替换为nan,使得计算ic时更加合理, 计算pnl时没影响
    # signal = signal.replace(0, np.nan)
    pos_df_daily = pos_daily_fun(signal, n=hold_time)

    if if_hedge:
        hedge_df = hedge_ratio * index_df.mul(pos_df_daily.sum(axis=1), axis=0)
        # pnl_df = (signal * pct_n).sum(axis=1).sub(hedge_df, axis=0)
        pnl_df = -hedge_df.sub((pos_df_daily * pct_n).sum(axis=1), axis=0)
        pnl_df = pnl_df[pnl_df.columns[0]]
    else:
        pnl_df = (pos_df_daily * pct_n).sum(axis=1)
    # pnl_df = pd.Series(pnl_df)
    # 样本内表现
    pnl_df_in = pnl_df[pnl_df.index < cut_date]
    asset_df_in = pnl_df_in.cumsum()
    last_asset_in = asset_df_in.iloc[-1]
    pos_df_daily_in = pos_df_daily[pos_df_daily.index < cut_date]
    pot = AZ_Pot(pos_df_daily_in, last_asset_in)

    sharpe_q_in_df = bt.AZ_Rolling_sharpe(pnl_df_in, roll_year=1, year_len=250, min_periods=1,
                                          cut_point_list=[0.3, 0.5, 0.7], output=False)

    sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d = sharpe_q_in_df.values
    in_condition_u = sharpe_q_in_df_u > 0.9
    in_condition_d = sharpe_q_in_df_d < -0.9
    in_condition = in_condition_u | in_condition_d
    if sharpe_q_in_df_m > 0:
        way = 1
    else:
        way = -1

    # 样本外表现
    pnl_df_out = pnl_df[pnl_df.index >= cut_date]
    out_condition, sharpe_q_out = out_sample_perf(pnl_df_out, way=way)
    if if_return_pnl:
        return in_condition, out_condition, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot, sharpe_q_out, \
               pnl_df
    else:
        return in_condition, out_condition, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot, sharpe_q_out


def filter_all(cut_date, pos_df_daily, pct_n, index_df, if_hedge=True, hedge_ratio=1,
               if_return_pnl=False, if_only_long=False):

    if if_hedge:
        hedge_df = hedge_ratio * index_df.mul(pos_df_daily.sum(axis=1), axis=0)
        pnl_df = -hedge_df.sub((pos_df_daily * pct_n).sum(axis=1), axis=0)
        pnl_df = pnl_df[pnl_df.columns[0]]
    else:
        pnl_df = (pos_df_daily * pct_n).sum(axis=1)
    # pnl_df = pd.Series(pnl_df)
    # 样本内表现
    return_in = pct_n[pct_n.index < cut_date]

    pnl_df_in = pnl_df[pnl_df.index < cut_date]
    asset_df_in = pnl_df_in.cumsum()
    last_asset_in = asset_df_in.iloc[-1]
    pos_df_daily_in = pos_df_daily[pos_df_daily.index < cut_date]
    pot_in = AZ_Pot(pos_df_daily_in, last_asset_in)

    leve_ratio = AZ_Leverage_ratio(asset_df_in)
    if leve_ratio < 0:
        leve_ratio = 100
    sharpe_q_in_df = bt.AZ_Rolling_sharpe(pnl_df_in, roll_year=1, year_len=250, min_periods=1,
                                          cut_point_list=[0.3, 0.5, 0.7], output=False)
    fit_ratio = bt.AZ_fit_ratio(pos_df_daily_in, return_in)
    ic = bt.AZ_Normal_IC(pos_df_daily_in, pct_n, min_valids=None, lag=0).mean()
    sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d = sharpe_q_in_df.values
    in_condition_u = sharpe_q_in_df_u > 0.9 and leve_ratio > 1
    in_condition_d = sharpe_q_in_df_d < -0.9 and leve_ratio > 1
    # 分双边和只做多
    if if_only_long:
        in_condition = in_condition_u
    else:
        in_condition = in_condition_u | in_condition_d

    if sharpe_q_in_df_m > 0:
        way = 1
    else:
        way = -1

    # 样本外表现
    pnl_df_out = pnl_df[pnl_df.index >= cut_date]
    out_condition, sharpe_q_out = out_sample_perf(pnl_df_out, way=way)
    if if_return_pnl:
        return in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
               fit_ratio, leve_ratio, sharpe_q_out, pnl_df
    else:
        return in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
               fit_ratio, leve_ratio, sharpe_q_out


