import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load.bkt import get_main_model


def moment_filter_fun(main_model, root_path):
    stock_return = bt.AZ_Load_csv(f'{root_path}/EM_Funda/DERIVED_14/aadj_r.csv')
    moment_df = bt.AZ_Rolling_mean(stock_return, 20).reindex(index=main_model.xinx, columns=main_model.xnms)

    percent = 0.2
    moment_df = main_model.sector_df * moment_df
    target_df_rank = moment_df.rank(axis=1, pct=True)
    target_df = target_df_rank.copy()
    target_df[(target_df_rank < 1 - percent)] = 1
    target_df[target_df_rank >= 1 - percent] = np.nan
    return target_df


def row_extre(raw_df, sector_df, percent):
    raw_df = raw_df * sector_df
    target_df = raw_df.rank(axis=1, pct=True)
    target_df[target_df >= 1 - percent] = 1
    target_df[target_df <= percent] = -1
    target_df[(target_df > percent) & (target_df < 1 - percent)] = 0
    return target_df


def load_whs_factor(file_name, main_model, dividend_ratio_df):
    load_path = '/mnt/mfs/DAT_EQT/EM_Funda/dat_whs/'
    tmp_df = bt.AZ_Load_csv(os.path.join(load_path, file_name + '.csv')) \
        .reindex(index=main_model.xinx, columns=main_model.xnms)
    tmp_df = tmp_df.fillna(method='ffill') * dividend_ratio_df
    target_df = row_extre(tmp_df, main_model.sector_df, 0.3)
    if main_model.if_only_long:
        target_df = target_df[target_df > 0]
    return target_df


def netprofit_filter_fun():
    netprofit_df = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/daily/R_NETPROFIT_s_First.csv')


def AZ_Cut_window(df, begin_date, end_date=None, column=None):
    if column is None:
        if end_date is None:
            return df[df.index > begin_date]
        else:
            return df[(df.index > begin_date) & (df.index < end_date)]
    else:
        if end_date is None:
            return df[df[column] > begin_date]
        else:
            return df[(df[column] > begin_date) & (df[column] < end_date)]


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


def pos_daily_fun(df, n=5):
    return df.rolling(window=n, min_periods=1).sum()


def AZ_Pot(pos_df_daily, last_asset):
    trade_times = pos_df_daily.diff().abs().sum().sum()
    if trade_times == 0:
        return 0
    else:
        pot = last_asset / trade_times * 10000
        return round(pot, 2)


def out_sample_perf_c(pnl_df_out, way=1):
    if way == 1:
        sharpe_out = bt.AZ_Sharpe_y(pnl_df_out)
    else:
        sharpe_out = bt.AZ_Sharpe_y(-pnl_df_out)
    out_condition = sharpe_out > 0.8
    return out_condition, round(sharpe_out * way, 2)


def filter_all(cut_date, pos_df_daily, pct_n, if_return_pnl=False, if_only_long=False):
    pnl_df = (pos_df_daily * pct_n).sum(axis=1)
    pnl_df = pnl_df.replace(np.nan, 0)

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
    sp_in = bt.AZ_Sharpe_y(pnl_df_in)
    fit_ratio = bt.AZ_fit_ratio(pos_df_daily_in, return_in)
    ic = round(bt.AZ_Normal_IC(pos_df_daily_in, pct_n, min_valids=None, lag=0).mean(), 6)
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
    out_condition, sharpe_q_out = out_sample_perf_c(pnl_df_out, way=way)
    if if_return_pnl:
        return in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
               fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df
    else:
        return in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
               fit_ratio, leve_ratio, sp_in, sharpe_q_out

def main_fun(sector_name, hold_time, if_only_long):
    root_path = '/media/hdd1/DAT_EQT'
    root_path = '/mnt/mfs/DAT_EQT'
    cut_date = '20180601'
    begin_date = pd.to_datetime('20140601')
    end_date = datetime.now()
    # sector_name = 'market_top_300plus'
    # hold_time = 5
    if_hedge = True
    # if_only_long = False
    if_weight = 1
    ic_weight = 0
    main_model = get_main_model(root_path, begin_date, cut_date, end_date, sector_name, hold_time,
                                if_hedge, if_only_long, if_weight, ic_weight)

    moment_filter = moment_filter_fun(main_model, root_path)
    dividend_ratio_df = load_whs_factor('dividend_ratio', main_model, moment_filter)
    pos_df = main_model.deal_mix_factor(dividend_ratio_df)
    in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
    fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df = filter_all(cut_date, pos_df, main_model.return_choose,
                                                                    if_return_pnl=True, if_only_long=False)
    print(in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in,
          fit_ratio, leve_ratio, sp_in, sharpe_q_out)
    result_df, info_df = bt.commit_check(pd.DataFrame(pnl_df))

    plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df), f'[RESULT]{sector_name}|{hold_time}|{if_only_long}',
                     info_df.to_html() + f'{pot_in}|{fit_ratio}|{leve_ratio}')


if __name__ == '__main__':


    # sector_name_list = [
    #     'market_top_300plus',
    #     'market_top_300plus_industry_10_15',
    #     'market_top_300plus_industry_20_25_30_35',
    #     'market_top_300plus_industry_40',
    #     'market_top_300plus_industry_45_50',
    #     'market_top_300plus_industry_55',
    #
    #     'market_top_300to800plus',
    #     'market_top_300to800plus_industry_10_15',
    #     'market_top_300to800plus_industry_20_25_30_35',
    #     'market_top_300to800plus_industry_40',
    #     'market_top_300to800plus_industry_45_50',
    #     'market_top_300to800plus_industry_55',
    # ]

    sector_name_list = [
        'market_top_300plus',
        'market_top_300plus_ind1',
        'market_top_300plus_ind10',
        'market_top_300plus_ind2',
        'market_top_300plus_ind3',
        'market_top_300plus_ind4',
        'market_top_300plus_ind5',
        'market_top_300plus_ind6',
        'market_top_300plus_ind7',
        'market_top_300plus_ind8',
        'market_top_300plus_ind9',
        'market_top_300plus_industry_10_15',
        'market_top_300plus_industry_20_25_30_35',
        'market_top_300plus_industry_40',
        'market_top_300plus_industry_45_50',
        'market_top_300plus_industry_55',

        'market_top_300to800plus',
        'market_top_300to800plus_ind1',
        'market_top_300to800plus_ind10',
        'market_top_300to800plus_ind2',
        'market_top_300to800plus_ind3',
        'market_top_300to800plus_ind4',
        'market_top_300to800plus_ind5',
        'market_top_300to800plus_ind6',
        'market_top_300to800plus_ind7',
        'market_top_300to800plus_ind8',
        'market_top_300to800plus_ind9',
        'market_top_300to800plus_industry_10_15',
        'market_top_300to800plus_industry_20_25_30_35',
        'market_top_300to800plus_industry_40',
        'market_top_300to800plus_industry_45_50',
        'market_top_300to800plus_industry_55',
    ]

    hold_time_list = [5, 20]
    for if_only_long in [False, True]:
        for hold_time in hold_time_list:
            for sector_name in sector_name_list:
                main_fun(sector_name, hold_time, if_only_long)
