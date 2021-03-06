import numpy as np
import pandas as pd
# import dask.dataframe as dd
import os
import sys
from sklearn.cluster import KMeans

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.shared_tools import send_email
import matplotlib.pyplot as plt
from datetime import datetime
import time


def AZ_get_stock_name():
    return_df = AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv')
    return list(return_df.columns)


def AZ_Rolling_mean_multi(data, window, func, ncore=4):
    result_t = dd.from_pandas(data, npartitions=ncore).map_partitions(
        lambda df: (df.T.rolling(window, min_periods=0).apply(func)).T).compute()
    result = result_t.T
    return result


def AZ_Load_csv(target_path, parse_dates=True, index_col=0, sep='|', **kwargs):
    target_df = pd.read_table(target_path, sep=sep, index_col=index_col, low_memory=False,
                              parse_dates=parse_dates, **kwargs)
    return target_df


def AZ_Save_csv(target_df, target_path, **kwargs):
    target_df.to_csv(target_path, sep='|', **kwargs)


def AZ_Catch_error(func):
    def _deco(*args, **kwargs):
        try:
            ret = func(*args, **kwargs)
        except:
            ret = sys.exc_info()
            print(ret[0], ":", ret[1])
        return ret

    return _deco


def AZ_Time_cost(func):
    t1 = time.time()

    def _deco(*args, **kwargs):
        ret = func(*args, **kwargs)
        return ret

    t2 = time.time()
    print(f'cost_time: {t2-t1}')
    return _deco


def AZ_Sharpe_y(pnl_df):
    return round((np.sqrt(250) * pnl_df.mean()) / pnl_df.std(), 4)


def AZ_MaxDrawdown(asset_df):
    return asset_df - np.maximum.accumulate(asset_df)


def AZ_Col_zscore(df, n, cap=None, min_periods=1):
    df_mean = AZ_Rolling_mean(df, n, min_periods=min_periods).round(4)
    df_std = df.rolling(window=n, min_periods=min_periods).std().round(4).replace(0, np.nan)
    target = (df - df_mean) / df_std
    if cap is not None:
        target[target > cap] = cap
        target[target < -cap] = -cap
    return target


def AZ_Row_zscore(df, cap=None):
    df_mean = df.mean(axis=1)
    df_std = df.std(axis=1).replace(0, np.nan)
    target = df.sub(df_mean, axis=0).div(df_std, axis=0)
    if cap is not None:
        target[target > cap] = cap
        target[target < -cap] = -cap
    return target.replace(np.nan, 0)


def AZ_Rolling(df, n, min_periods=0):
    return df.rolling(window=n, min_periods=min_periods)


def AZ_Rolling_mean(df, window, min_periods=0):
    target = df.rolling(window=window, min_periods=min_periods).mean()
    target.iloc[:window - 1] = np.nan
    return target


def AZ_Rolling_min(df, window, min_periods=0):
    return AZ_Rolling(df, window, min_periods).min()


def AZ_Rolling_max(df, window, min_periods=0):
    return AZ_Rolling(df, window, min_periods).max()


def AZ_Rolling_sum(df, window, min_periods=0):
    return AZ_Rolling(df, window, min_periods).sum()


def AZ_Rolling_std(df, window, min_periods=0):
    return AZ_Rolling(df, window, min_periods).std().replace(0, np.nan)


def AZ_Rolling_corr(df1, df2, window, min_periods=0):
    return AZ_Rolling(df1, window, min_periods).corr(df2)


def AZ_Rolling_cov(df1, df2, window, min_periods=0):
    return AZ_Rolling(df1, window, min_periods).cov(df2)


def AZ_Rolling_apply(df, window, fun, min_periods=0):
    return AZ_Rolling(df, window, min_periods).apply(fun)


def AZ_Rolling_sharpe(pnl_df, roll_year=1, year_len=250, min_periods=0, cut_point_list=None, output=False):
    if cut_point_list is None:
        cut_point_list = [0.05, 0.33, 0.5, 0.66, 0.95]

    pnl_df_mean = pnl_df.rolling(int(roll_year * year_len), min_periods=min_periods).mean()
    pnl_df_std = pnl_df.rolling(int(roll_year * year_len), min_periods=min_periods).std().repalce(0, np.nan)
    rolling_sharpe = np.sqrt(year_len) * pnl_df_mean / pnl_df_std

    rolling_sharpe.iloc[:int(roll_year * year_len) - 1] = np.nan
    cut_sharpe = rolling_sharpe.quantile(cut_point_list)
    if output:
        return rolling_sharpe, cut_sharpe.round(4)
    else:
        return cut_sharpe.round(4)


def AZ_Pot(pos_df, asset_last):
    """
    计算 pnl/turover*10000的值,衡量cost的影响
    :param pos_df: 仓位信息
    :param asset_last: 最后一天的收益
    :return:
    """
    pos_df = pos_df.fillna(0)
    trade_times = pos_df.diff().abs().sum().sum()
    if trade_times == 0:
        return 0
    else:
        pot = asset_last / trade_times * 10000
        return round(pot, 2)


def AZ_Normal_IC(signal, pct_n, min_valids=None, lag=0):
    signal = signal.shift(lag)
    signal = signal.replace(0, np.nan)
    corr_df = signal.corrwith(pct_n, axis=1).dropna()

    if min_valids is not None:
        signal_valid = signal.count(axis=1)
        signal_valid[signal_valid < min_valids] = np.nan
        signal_valid[signal_valid >= min_valids] = 1
        corr_signal = corr_df * signal_valid
    else:
        corr_signal = corr_df
    return round(corr_signal, 6)


def AZ_Normal_IR(signal, pct_n, min_valids=None, lag=0):
    corr_signal = AZ_Normal_IC(signal, pct_n, min_valids, lag)
    ic_mean = corr_signal.mean()
    ic_std = corr_signal.std()
    ir = ic_mean / ic_std
    return ir, corr_signal


def AZ_Leverage_ratio(asset_df):
    """
    返回250天的return/(负的 一个月的return)
    :param asset_df:
    :return:
    """
    asset_20 = asset_df - asset_df.shift(20)
    asset_250 = asset_df - asset_df.shift(250)
    if asset_250.mean() > 0:
        return round(asset_250.mean() / (-asset_20.min()), 2)
    else:
        return round(asset_250.mean() / (-asset_20.max()), 2)


def AZ_Locked_date_deal(position_df, locked_df):
    """
    处理回测中停牌,涨停等 仓位需要锁死的情况
    :param position_df:仓位信息
    :param locked_df:停牌 涨跌停等不能交易信息(能交易记为1, 不能记为nan)
    :return:
    """

    position_df_adj = (position_df * locked_df).dropna(how='all', axis=0) \
        .fillna(method='ffill')
    return position_df_adj


def AZ_Path_create(target_path):
    """
    添加新路径
    :param target_path:
    :return:
    """
    if not os.path.exists(target_path):
        os.makedirs(target_path)


def AZ_split_stock(stock_list):
    """
    在stock_list中寻找A股代码
    :param stock_list:
    :return:
    """
    eqa = [x for x in stock_list if (x.startswith('0') or x.startswith('3')) and x.endswith('SZ')
           or x.startswith('6') and x.endswith('SH') or x in ['510050.SH', '510300.SH', '510500.SH',
                                                              '000001.SH', '000016.SH', '000300.SH', '000905.SH',
                                                              '000906.SH',
                                                              ]]
    return eqa


def AZ_add_stock_suffix(stock_list):
    """
    whs
    给stock_list只有数字的 A股代码 添加后缀
    如 000001 运行后 000001.SZ
    :param stock_list:
    :return:　　
    """
    return list(map(lambda x: x + '.SH' if x.startswith('6') else x + '.SZ', stock_list))


def AZ_cut_stock_suffix(stock_list, num=3, way=-1):
    if way == 1:
        return [x[num:] for x in stock_list]
    elif way == -1:
        return [x[:-num] for x in stock_list]
    else:
        print('ERROR')


def AZ_clear_columns(stock_list):
    return [x[2:] + '.' + x[:2] for x in stock_list]


def AZ_Delete_file(target_path, target_list=None, except_list=None):

    if except_list is None:
        except_list = []

    file_list = os.listdir(target_path)
    if target_list is None:
        file_list = list(set(file_list) - set(except_list))
    else:
        file_list = list(set(file_list) & set(target_list))
    for file_name in sorted(file_list):
        os.remove(os.path.join(target_path, file_name))
        # print(f'delete {file_name}~~')


def AZ_turnover(pos_df):
    diff_sum = pos_df.diff().abs().sum().sum()
    pos_sum = pos_df.abs().sum().sum()
    if pos_sum == 0:
        return .0
    return diff_sum / float(pos_sum)


def AZ_annual_return(pos_df, return_df, window=600):
    temp_pnl = (pos_df * return_df).iloc[-window:].sum().sum()
    temp_pos = pos_df.iloc[-window:].abs().sum().sum()
    if temp_pos == 0:
        return .0
    else:
        return temp_pnl * 250.0 / temp_pos


def AZ_ls_margin(pos_df, return_df, window=600):
    pos_long = pos_df[pos_df > 0]
    pos_short = pos_df[pos_df < 0]
    margin_l = AZ_annual_return(pos_long, return_df, window)
    margin_s = AZ_annual_return(pos_short, return_df, window)
    return margin_l, margin_s


def AZ_fit_ratio(pos_df, return_df):
    """
    传入仓位 和 每日收益
    :param pos_df:
    :param return_df:
    :return: 时间截面上的夏普 * sqrt（abs（年化）/换手率）， 当换手率为0时，返回0
    """
    sharp_ratio = AZ_Sharpe_y((pos_df * return_df).sum(axis=1))
    ann_return = AZ_annual_return(pos_df, return_df)
    turnover = AZ_turnover(pos_df)
    if turnover == 0:
        return .0
    else:
        return round(sharp_ratio * np.sqrt(abs(ann_return) / turnover), 2)


def AZ_fit_ratio_rolling(pos_df, pnl_df, roll_year=1, year_len=250, min_periods=1, cut_point_list=None, output=False):
    if cut_point_list is None:
        cut_point_list = [0.05, 0.33, 0.5, 0.66, 0.95]
    rolling_sharpe, cut_sharpe = AZ_Rolling_sharpe(pnl_df, roll_year=roll_year, year_len=year_len,
                                                   min_periods=min_periods, cut_point_list=cut_point_list, output=True)
    rolling_return = pnl_df.rolling(int(roll_year * year_len), min_periods=min_periods).apply(
        lambda x: 250.0 * x.sum().sum(), )

    rolling_diff_pos = pos_df.diff().abs().sum(axis=1).rolling(int(roll_year * year_len),
                                                               min_periods=min_periods).apply(
        lambda x: x.sum().sum())
    rolling_return.iloc[:int(roll_year * year_len) - 1] = np.nan
    rolling_diff_pos.iloc[:int(roll_year * year_len) - 1] = np.nan
    rolling_fit_ratio = rolling_sharpe * np.sqrt(abs(rolling_return) / rolling_diff_pos)
    rolling_fit_ratio = rolling_fit_ratio.replace(np.inf, np.nan)
    rolling_fit_ratio = rolling_fit_ratio.replace(-np.inf, np.nan)
    cut_fit = rolling_fit_ratio.quantile(cut_point_list)
    return cut_fit.round(4)


def AZ_VAR(pos_df, return_df, confidence_level, backward_len=500, forwward_len=250):
    tradeDayList = pos_df.index[:-forwward_len]
    col01 = return_df.columns[0]
    varList = []
    cut_point_list = [0.05, 0.33, 0.5, 0.66, 0.95]
    if len(tradeDayList) == 0:
        print('数据量太少')
    else:
        for tradeDay in tradeDayList:
            tempPos = pos_df.loc[tradeDay, :]
            dayIndex = list(return_df.loc[:tradeDay, col01].index[-backward_len:]) + list(
                return_df.loc[tradeDay:, col01].index[:forwward_len])
            return_df_c = return_df[list(tempPos.index)]
            historyReturn = list(return_df_c.mul(tempPos, axis=1).loc[dayIndex[0]:dayIndex[-1], :].sum(axis=1))
            historyReturn.sort()
            varList.append(historyReturn[int(len(historyReturn) * confidence_level)])
    var = pd.DataFrame({'var': varList}, index=tradeDayList)
    var = var.dropna()
    var_fit = var.quantile(cut_point_list)
    return list(var_fit['var'])


def AZ_holding_period_decay(pos_df, return_df, holding_period=[1, 3, 5, 7, 9, 11, 15, 20, 25, 30]):
    sharpList = []
    annual_return_list = []
    for holding in holding_period:
        sma_pos_df = AZ_Rolling_mean(pos_df, holding, min_periods=1)
        sharpList.append(AZ_Sharpe_y((sma_pos_df * return_df).sum(axis=1)))
        annual_return_list.append(AZ_annual_return(sma_pos_df, return_df))
    return sharpList, annual_return_list


# def AZ_pnl_kmean(all_pnl_df, n, ratio):
#     target_df = (all_pnl_df > 0).astype(int)
#     kmeans = KMeans(n_clusters=n).fit(target_df.T)
#     kmeans_result = kmeans.labels_
#     columns_list = target_df.columns
#     group_df = pd.DataFrame(kmeans_result, index=columns_list)
#     target_df = pd.DataFrame()
#
#     for i in range(n):
#         part_a_n = a_n[a_n['group_key'] == i].sort_values(by='sp_in')
#         part_num = max(int(len(part_a_n) * ratio), 1)
#
#         part_target_df = part_a_n[['fun_name', 'name1', 'name2', 'name3', 'buy_sell']].iloc[:part_num]
#         print(part_num)
#         target_df = target_df.append(part_target_df)


def commit_check(pnl_df, mod='o'):
    """
    pnl_df
    :param pnl_df:要求DataFrame格式,其中index为时间格式,columns为pnl的名称
    :param mod: 'o':多空,'h':对冲
    :return:result_df包含corr,sp5,sp2,lv5,lv2,其中0表示不满足,1表示满足,
            info_df为具体数值
    """
    assert type(pnl_df) == pd.DataFrame
    all_pnl_df = pd.read_csv('/mnt/mfs/AATST/corr_tst_pnls', sep='|', index_col=0, parse_dates=True)
    all_pnl_df_c = pd.concat([all_pnl_df, pnl_df], axis=1)
    all_pnl_df_c_ma3 = AZ_Rolling(all_pnl_df_c, 3).mean().iloc[-1250:]
    matrix_corr_o = all_pnl_df_c_ma3.corr()[pnl_df.columns].drop(index=pnl_df.columns)

    matrix_sp5 = pnl_df.iloc[-1250:].apply(AZ_Sharpe_y)
    matrix_lv5 = pnl_df.iloc[-1250:].cumsum().apply(AZ_Leverage_ratio)

    matrix_sp2 = pnl_df.iloc[-500:].apply(AZ_Sharpe_y)
    matrix_lv2 = pnl_df.iloc[-500:].cumsum().apply(AZ_Leverage_ratio)

    info_df = pd.concat([matrix_corr_o.max(), matrix_sp5, matrix_sp2, matrix_lv5, matrix_lv2], axis=1)
    info_df.columns = ['corr', 'sp5', 'sp2', 'lv5', 'lv2']
    info_df = info_df.T

    if mod == 'h':
        cond_matrix = pd.DataFrame([[0.45, 1.90, 1.66, 1.70, 1.70],
                                    [0.56, 2.00, 1.75, 1.75, 1.75],
                                    [0.62, 2.10, 1.80, 1.80, 1.80]])
    else:
        cond_matrix = pd.DataFrame([[0.45, 2.00, 1.75, 2.00, 2.00],
                                    [0.56, 2.10, 1.85, 2.10, 2.10],
                                    [0.62, 2.25, 1.95, 2.20, 2.20]])

    def result_deal(x):
        for i in range(len(cond_matrix)):
            if x[0] <= cond_matrix.iloc[i, 0]:
                corr, sp_5, sp_2, lv_5, lv_2 = cond_matrix.iloc[i]
                res = x > [-1, sp_5, sp_2, lv_5, lv_2]
                return res.astype(int)
        return [0, 0, 0, 0, 0]

    result_df = info_df.apply(result_deal)
    print('*******info_df*******')
    print(info_df)

    print('*******result_df*******')
    print(result_df)

    return result_df, info_df
