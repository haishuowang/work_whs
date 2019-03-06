import pandas as pd
import numpy as np


def AZ_Load_csv(target_path, index_time_type=True):
    target_df = pd.read_table(target_path, sep='|', index_col=0, low_memory=False, parse_dates=index_time_type)
    return target_df


def AZ_Pot(pos_df, asset_last):
    trade_times = pos_df.diff().abs().sum().sum()
    if trade_times == 0:
        return 0
    else:
        pot = asset_last / trade_times * 10000
        return round(pot, 2)


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


def AZ_Sharpe_y(pnl_df):
    return round((np.sqrt(250) * pnl_df.mean()) / pnl_df.std(), 4)


def commit_check(pnl_df):
    """
    pnl_df
    :param pnl_df:要求DataFrame格式,其中index为时间格式,columns为pnl的名称
    :return:result_df包含corr,sp5,sp2,lv5,lv2,其中0表示不满足,1表示满足,
            info_df为具体数值
    """
    assert type(pnl_df) == pd.DataFrame
    all_pnl_df = pd.read_csv('/mnt/mfs/AATST/corr_tst_pnls', sep='|', index_col=0, parse_dates=True)
    all_pnl_df_c = pd.concat([all_pnl_df, pnl_df], axis=1)
    matrix_corr_o = all_pnl_df_c.iloc[-1250:].corr()[pnl_df.columns].drop(index=pnl_df.columns)
    # matrix_corr_s = all_pnl_df_c.iloc[-1250:].corr().loc[pnl_df.columns, pnl_df.columns]
    matrix_sp5 = pnl_df.iloc[-1250:].apply(AZ_Sharpe_y)
    matrix_lv5 = pnl_df.iloc[-1250:].cumsum().apply(AZ_Leverage_ratio)

    matrix_sp2 = pnl_df.iloc[-500:].apply(AZ_Sharpe_y)
    matrix_lv2 = pnl_df.iloc[-500:].cumsum().apply(AZ_Leverage_ratio)

    info_df = pd.concat([matrix_corr_o.max(), matrix_sp5, matrix_sp2, matrix_lv5, matrix_lv2], axis=1)
    info_df.columns = ['corr', 'sp5', 'sp2', 'lv5', 'lv2']
    info_df = info_df.T
    cond_matrix = pd.DataFrame([[0.5, 2.0, 1.9, 2.0, 2.0],
                                [0.6, 2.1, 2.0, 2.0, 2.0],
                                [0.7, 2.2, 2.1, 2.0, 2.0]])

    def result_deal(x):
        for i in range(len(cond_matrix)):
            if x[0] <= cond_matrix.iloc[i, 0]:
                corr, sp_5, sp_2, lv_5, lv_2 = cond_matrix.loc[i]
                res = x > [corr, sp_5, sp_2, lv_5, lv_2]
                return res.astype(int)
        return [0, 0, 0, 0, 0]

    result_df = info_df.apply(result_deal)
    print('*******info_df*******')
    print(info_df)

    print('*******result_df*******')
    print(result_df)

    return result_df, info_df


# def commit_check_pos(pos_df):


if __name__ == '__main__':
    pnl_df_1 = pd.read_csv('/media/hdd2/dat_whs/data/single_factor_pnl/market_top_300plus/R_WorkCapital_QTTM_'
                           'R_WorkCapital_First_R_TotProfit_EBIT_First_1_0.3_fun_6.pkl|market_top_300plus|5|True',
                           index_col=0, header=None, parse_dates=True)
    pnl_df_2 = pd.read_csv('/media/hdd2/dat_whs/data/single_factor_pnl/market_top_300to800plus_industry_20_25_30_35/'
                           'R_OPCF_TTM_QTTM_'
                           'R_ASSETOTHER_First_'
                           'R_FairValChg_TotProfit_s_First_-1_0.3_fun_6.pkl'
                           '|market_top_300to800plus_industry_20_25_30_35|20|True',
                           index_col=0, header=None, parse_dates=True)
    pnl_df = pd.concat([pnl_df_1, pnl_df_2], axis=1)
    pnl_df.columns = ['test_pnl_1', 'test_pnl_2']
    result_df = commit_check(pnl_df)
