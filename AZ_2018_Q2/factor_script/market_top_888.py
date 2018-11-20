import pandas as pd
import numpy as np

root_path = '/media/hdd1/DAT_EQT'

root_path = '/mnt/mfs/DAT_EQT'


def market_top_800plus_fun():
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

    market_top_800plus_mask = HS300_mask | ZZ500_mask | market_top_800_mask

    market_top_800plus = market_top_800plus_mask.astype(float).replace(0, np.nan)
    return market_top_800plus


def market_top_300plus_fun():
    HS300 = pd.read_csv(root_path + '/EM_Funda/IDEX_YS_WEIGHT_A/SECURITYNAME_000300.csv',
                        index_col=0, parse_dates=True, sep='|', low_memory=False)

    market_top_300 = pd.read_csv(root_path + '/EM_Funda/DERIVED_10/market_top_300.csv',
                                 index_col=0, parse_dates=True, sep='|', low_memory=False)

    xnms = sorted(set(market_top_300.columns) | set(HS300.columns))
    xinx = sorted(set(market_top_300.index) | set(HS300.index))

    HS300 = HS300.reindex(columns=xnms, index=xinx)
    market_top_300 = market_top_300.reindex(columns=xnms, index=xinx)

    HS300_mask = HS300.notna()
    market_top_300_mask = market_top_300.notna()

    market_top_300plus_mask = HS300_mask | market_top_300_mask
    market_top_300plus = market_top_300plus_mask.astype(float).replace(0, np.nan)
    return market_top_300plus


def market_top_500plus_fun():
    market_top_800plus = market_top_800plus_fun()
    market_top_300plus = market_top_300plus_fun()
    market_top_500plus = market_top_800plus.sub(market_top_300plus, fill_value=0) \
        .dropna(how='all', axis='columns')
    return market_top_300plus, market_top_500plus, market_top_800plus


# root_path = '/media/hdd1/DAT_EQT'
# root_path = '/mnt/mfs/DAT_EQT'


def AZ_Row_zscore(df, cap=None):
    df_mean = df.mean(axis=1)
    df_std = df.std(axis=1)
    target = df.sub(df_mean, axis=0).div(df_std, axis=0)
    if cap is not None:
        target[target > cap] = cap
        target[target < -cap] = -cap
    return target


def get_moment(short, long, aadj_CR, market_top_800plus):
    moment_short = aadj_CR.rolling(window=short, min_periods=0).mean()
    moment_long = aadj_CR.rolling(window=long, min_periods=0).mean()
    moment_short_long = (moment_short - moment_long) * market_top_800plus
    return moment_short_long


def market_top_n_MMT_fun():
    aadj_r = pd.read_csv(root_path + '/EM_Funda/DERIVED_14/aadj_r.csv',
                         index_col=0, parse_dates=True, sep='|', low_memory=False)
    aadj_CR_all = aadj_r.cumsum()

    # file_name_list = ['market_top_300plus', 'market_top_300to800plus', 'market_top_800plus']
    file_name_list = ['market_top_300to800plus']
    for file_name in file_name_list:
        market_top_n = pd.read_csv(root_path + f'/EM_Funda/DERIVED_10/{file_name}.csv',
                                         index_col=0, parse_dates=True, sep='|', low_memory=False)
        xnms = market_top_n.columns
        xinx = market_top_n.index

        aadj_CR = aadj_CR_all.reindex(index=xinx, columns=xnms)

        moment_20_100 = get_moment(20, 100, aadj_CR, market_top_n)
        moment_30_150 = get_moment(30, 150, aadj_CR, market_top_n)
        moment_50_250 = get_moment(50, 250, aadj_CR, market_top_n)

        moment_20_100_zscore = AZ_Row_zscore(moment_20_100, cap=4)
        moment_30_150_zscore = AZ_Row_zscore(moment_30_150, cap=4)
        moment_50_250_zscore = AZ_Row_zscore(moment_50_250, cap=4)

        MT_zscore = moment_20_100_zscore + moment_30_150_zscore + moment_50_250_zscore
        MT_zscore_rank = MT_zscore.rank(axis=1, pct=True)

        sector_1_mask = ((MT_zscore_rank > 0) & (MT_zscore_rank <= 1 / 3))
        sector_2_mask = (MT_zscore_rank > 1 / 3) & (MT_zscore_rank <= 2 / 3)
        sector_3_mask = (MT_zscore_rank > 2 / 3) & (MT_zscore_rank <= 1)

        sector_1 = sector_1_mask.astype(int).replace(0, np.nan)
        sector_2 = sector_2_mask.astype(int).replace(0, np.nan)
        sector_3 = sector_3_mask.astype(int).replace(0, np.nan)

        sector_1.to_csv(root_path + f'/EM_Funda/DERIVED_10/{file_name}_MM1.csv', sep='|')
        sector_2.to_csv(root_path + f'/EM_Funda/DERIVED_10/{file_name}_MM2.csv', sep='|')
        sector_3.to_csv(root_path + f'/EM_Funda/DERIVED_10/{file_name}_MM3.csv', sep='|')
    return 0


if __name__ == '__main__':
    # market_top_300plus, market_top_500plus, market_top_800plus = market_top_500plus_fun()
    market_top_n_MMT_fun()
    pass
