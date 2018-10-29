import pandas as pd
import numpy as np
import os
from datetime import datetime
import loc_lib.shared_paths.path as pt
import loc_lib.shared_tools.back_test as bt
from factor_script.script_load_data import load_index_data, load_sector_data, load_locked_data, load_pct, \
    load_part_factor, create_log_save_path, deal_mix_factor


class SectorSplit:
    def __init__(self, sector_name):
        begin_date = pd.to_datetime('20050505')
        end_date = datetime.now()
        # sector_name = 'market_top_2000'
        self.sector_name = sector_name
        market_top_n = bt.AZ_Load_csv(os.path.join('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_10/' + sector_name + '.csv'))
        market_top_n = market_top_n[(market_top_n.index >= begin_date) & (market_top_n.index < end_date)]
        self.sector_df = market_top_n
        xinx = self.sector_df.index
        xnms = self.sector_df.columns

        aadj_r = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv')
        self.aadj_r = aadj_r.reindex(index=xinx, columns=xnms)

        aadj_p = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv')
        self.aadj_p = aadj_p.reindex(index=xinx, columns=xnms)

        self.save_path = '/mnt/mfs/dat_whs/data/sector_data'

    @staticmethod
    def sort_fun(x, n=1000, top=True):
        target = np.array([float('nan')] * len(x))
        if top:
            target[x <= n] = 1
            target[x > n] = np.nan
        else:
            target[x <= n] = np.nan
            target[x > n] = 1
        return target

    def moment(self, n=1000):
        ma30 = bt.AZ_Rolling_mean(self.aadj_p, 30)
        ma300 = bt.AZ_Rolling_mean(self.aadj_p, 300)
        moment_factor = (ma30 - ma300) * self.sector_df
        moment_factor_rank = moment_factor.rank(ascending=False, axis=1)

        moment_top = moment_factor_rank.apply(self.sort_fun, args=(n, True), axis=1)
        moment_bot = moment_factor_rank.apply(self.sort_fun, args=(n, False), axis=1)

        moment_top.dropna(how='all', axis='columns', inplace=True)
        moment_bot.dropna(how='all', axis='columns', inplace=True)
        print(moment_bot.iloc[-1].dropna())
        moment_top.to_csv(os.path.join(self.save_path, f'{self.sector_name}_moment_top_{n}.csv'), sep='|')
        moment_bot.to_csv(os.path.join(self.save_path, f'{self.sector_name}_moment_bot_{n}.csv'), sep='|')

    def vol(self, n=1000):
        vol_factor = bt.AZ_Rolling(self.aadj_r, 120).std() * self.sector_df
        vol_factor_rank = vol_factor.rank(ascending=False, axis=1)
        vol_top = vol_factor_rank.apply(self.sort_fun, args=(n, True), axis=1)
        vol_bot = vol_factor_rank.apply(self.sort_fun, args=(n, False), axis=1)

        vol_top.dropna(how='all', axis='columns', inplace=True)
        vol_bot.dropna(how='all', axis='columns', inplace=True)
        print(vol_bot.iloc[-1].dropna())
        vol_top.to_csv(os.path.join(self.save_path, f'{self.sector_name}_vol_top_{n}.csv'), sep='|')
        vol_bot.to_csv(os.path.join(self.save_path, f'{self.sector_name}_vol_bot_{n}.csv'), sep='|')

    def industry(self, file_list):
        # self.sector_df
        industry_df_sum = pd.DataFrame()
        for file_name in file_list:
            industry_df = bt.AZ_Load_csv(f'/mnt/mfs/DAT_EQT/EM_Funda/LICO_IM_INCHG/Global_Level1_{file_name}.csv')
            industry_df_sum = industry_df_sum.add(industry_df, fill_value=0)
        industry_df_sum = self.sector_df.mul(industry_df_sum, fill_value=0).replace(0, np.nan)\
            .dropna(how='all', axis='columns')
        industry_df_sum.to_csv('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_10/{}_industry_{}.csv'
                               .format(self.sector_name, '_'.join([str(x) for x in file_list])))
        return industry_df_sum


if __name__ == '__main__':
    sector_split = SectorSplit('market_top_2000')
    for file_list in [[10, 15], [20, 25, 30, 35], [40], [45, 50], [55]]:
        industry_df_sum = sector_split.industry(file_list)
        market_top_n = bt.AZ_Load_csv('/mnt/mfs/dat_whs/data/sector_data/market_top_2000_industry_{}.csv'
                                      .format('_'.join([str(x) for x in file_list])))
        a = industry_df_sum.loc[pd.to_datetime('20180829')].dropna()
        b = market_top_n.loc[pd.to_datetime('20180829')].dropna()
        # print((industry_df_sum.loc[pd.to_datetime('20100829'):pd.to_datetime('20180829')]
        #        != market_top_n.loc[pd.to_datetime('20100829'):pd.to_datetime('20180829')]).sum().sum())
        # print((industry_df_sum > 1).sum().sum())
        # print((industry_df_sum == 0).sum().sum())
        # print((market_top_n > 1).sum().sum())
        # print((market_top_n == 0).sum().sum())
