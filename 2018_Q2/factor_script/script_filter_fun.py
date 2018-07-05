import numpy as np
import open_lib.shared_tools.back_test as bt


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


def filter_ic(cut_date, signal, pct_n, index_df, lag=1, hedge_ratio=1, if_hedge=True):
    # signal向下移动一天,避免未来函数
    signal = signal.shift(lag)
    # 将所有的0替换为nan,使得计算ic时更加合理
    signal = signal.replace(0, np.nan)

    corr_df = signal.corrwith(pct_n, axis=1)
    if if_hedge:
        pos_daily_df = pos_daily_fun(signal, n=5)
        hedge_df = hedge_ratio * pos_daily_df.sum(axis=1) * index_df
        pnl_df = (signal * pct_n).sum(axis=1) - hedge_df
    else:
        pnl_df = (signal * pct_n).sum(axis=1)
    # 样本内表现
    corr_df_in = corr_df[corr_df.index < cut_date]

    ic_rolling_5_y = bt.AZ_Rolling_mean(corr_df_in, 5 * 240)
    ic_rolling_5_y_mean = ic_rolling_5_y.iloc[-1]
    ic_rolling_half_y_list = bt.AZ_Rolling_mean(corr_df_in, int(0.5 * 240)).dropna().quantile([0.1, 0.9]).values

    in_condition_u = ic_rolling_5_y_mean > 0.04 and ic_rolling_half_y_list[0] > 0.01
    in_condition_d = ic_rolling_5_y_mean < -0.04 and ic_rolling_half_y_list[1] < -0.01
    in_condition = in_condition_u | in_condition_d
    if ic_rolling_5_y_mean > 0:
        way = 1
        ic_rolling_half_y_quantile = ic_rolling_half_y_list[0]
    else:
        way = -1
        ic_rolling_half_y_quantile = ic_rolling_half_y_list[1]

    # 样本外表现
    pnl_df_out = pnl_df[pnl_df.index >= cut_date]
    out_condition, sharpe_q_out = out_sample_perf(pnl_df_out, way=way)
    return in_condition, out_condition, ic_rolling_5_y_mean, ic_rolling_half_y_quantile, sharpe_q_out


def filter_ic_sharpe(cut_date, signal, pct_n,  index_df, lag=1, hedge_ratio=1, if_hedge=True):
    signal = signal.shift(lag)
    signal = signal.replace(0, np.nan)

    corr_df = signal.corrwith(pct_n, axis=1)
    if if_hedge:
        pos_daily_df = pos_daily_fun(signal, n=5)
        hedge_df = hedge_ratio * pos_daily_df.sum(axis=1) * index_df
        pnl_df = (signal * pct_n).sum(axis=1) - hedge_df
    else:
        pnl_df = (signal * pct_n).sum(axis=1)
    # 样本内表现
    corr_df_in = corr_df[corr_df.index < cut_date]
    pnl_df_in = pnl_df[pnl_df.index < cut_date]

    leve_ratio = bt.AZ_Leverage_ratio(pnl_df_in)
    if leve_ratio < 0:
        leve_ratio = 100

    ic_rolling_5_y = bt.AZ_Rolling_mean(corr_df_in, 5 * 240)
    ic_rolling_5_y_mean = ic_rolling_5_y.iloc[-1]
    ic_rolling_half_y_list = bt.AZ_Rolling_mean(corr_df_in, int(0.5 * 240)).dropna().quantile([0.1, 0.9]).values
    sharpe_in = bt.AZ_Rolling_sharpe(pnl_df_in, roll_year=1, year_len=250, min_periods=1,
                                     cut_point_list=[0.3, 0.5, 0.7], output=False)

    in_condition_u = ic_rolling_5_y_mean > 0.04 and sharpe_in.values[0] > 1.5
    in_condition_d = ic_rolling_5_y_mean < -0.04 and sharpe_in.values[1] < -1.5
    in_condition = in_condition_u | in_condition_d
    if in_condition_u:
        way = 1
        ic_rolling_half_y_quantile = ic_rolling_half_y_list[0]
    else:
        way = -1
        ic_rolling_half_y_quantile = ic_rolling_half_y_list[1]

    # 样本外表现
    pnl_df_out = pnl_df[pnl_df.index >= cut_date]
    out_condition, sharpe_q_out = out_sample_perf(pnl_df_out, way=way)
    return in_condition, out_condition, ic_rolling_5_y_mean, ic_rolling_half_y_quantile, leve_ratio, sharpe_q_out


def filter_ic_leve(cut_date, signal, pct_n, index_df, lag=1, hedge_ratio=1, if_hedge=True):
    signal = signal.shift(lag)
    signal = signal.replace(0, np.nan)

    corr_df = signal.corrwith(pct_n, axis=1)
    if if_hedge:
        pos_daily_df = pos_daily_fun(signal, n=5)
        hedge_df = hedge_ratio * pos_daily_df.sum(axis=1) * index_df
        pnl_df = (signal * pct_n).sum(axis=1) - hedge_df
    else:
        pnl_df = (signal * pct_n).sum(axis=1)
    # 样本内表现
    corr_df_in = corr_df[corr_df.index < cut_date]
    pnl_df_in = pnl_df[pnl_df.index < cut_date]

    leve_ratio = bt.AZ_Leverage_ratio(pnl_df_in)
    if leve_ratio < 0:
        leve_ratio = 100

    ic_rolling_5_y = bt.AZ_Rolling_mean(corr_df_in, 5 * 240)
    ic_rolling_5_y_mean = ic_rolling_5_y.iloc[-1]

    in_condition_u = ic_rolling_5_y_mean > 0.04 and leve_ratio >= 2
    in_condition_d = ic_rolling_5_y_mean < -0.04 and leve_ratio >= 2
    in_condition = in_condition_u | in_condition_d
    if ic_rolling_5_y_mean > 0:
        way = 1
    else:
        way = -1

    # 样本外表现
    pnl_df_out = pnl_df[pnl_df.index >= cut_date]
    out_condition, sharpe_q_out = out_sample_perf(pnl_df_out, way=way)
    return in_condition, out_condition, ic_rolling_5_y_mean, leve_ratio, sharpe_q_out


def filter_pot_sharpe(cut_date, signal, pct_n, index_df, lag=1, hedge_ratio=1, if_hedge=True):
    # signal向下移动一天,避免未来函数
    signal = signal.shift(lag)
    # # 将所有的0替换为nan,使得计算ic时更加合理, 计算pnl时没影响
    # signal = signal.replace(0, np.nan)
    if if_hedge:
        pos_daily_df = pos_daily_fun(signal, n=5)
        hedge_df = hedge_ratio * index_df.mul(pos_daily_df.sum(axis=1), axis=0)
        pnl_df = (signal * pct_n).sum(axis=1) - hedge_df
    else:
        pnl_df = (signal * pct_n).sum(axis=1)
    # 样本内表现
    pnl_df_in = pnl_df[pnl_df.index < cut_date]
    signal_in = signal[signal.index < cut_date]
    asset_df_in = pnl_df_in.cumsum()
    last_asset_in = asset_df_in.iloc[-1]
    pos_df_daily = pos_daily_fun(signal_in, n=5)
    pot = AZ_Pot(pos_df_daily, last_asset_in)

    sharpe_q_in_df = bt.AZ_Rolling_sharpe(pnl_df_in, roll_year=1, year_len=250, min_periods=1,
                                          cut_point_list=[0.3, 0.5, 0.7], output=False)
    sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d = sharpe_q_in_df.values
    in_condition_u = sharpe_q_in_df_u > 1.5
    in_condition_d = sharpe_q_in_df_d < -1.5
    in_condition = in_condition_u | in_condition_d
    if sharpe_q_in_df_m > 0:
        way = 1
    else:
        way = -1

    # 样本外表现
    pnl_df_out = pnl_df[pnl_df.index >= cut_date]
    out_condition, sharpe_q_out = out_sample_perf(pnl_df_out, way=way)
    return in_condition, out_condition, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot, sharpe_q_out


# def filter_ic_pot_sharpe_leve(cut_date, signal, pct_n, index_df, lag=1, hedge_ratio=1):
#     # signal向下移动一天,避免未来函数
#     signal = signal.shift(lag)
#     # # 将所有的0替换为nan,使得计算ic时更加合理, 计算pnl时没影响
#     # signal = signal.replace(0, np.nan)
#     pnl_df = (signal * pct_n).sum(axis=1) - signal.sum(axis=1) * index_df * hedge_ratio
#
#     corr_df = signal.corrwith(pct_n, axis=1)
#     pnl_df = (signal * pct_n).sum(axis=1) - signal.sum(axis=1) * index_df * hedge_ratio

