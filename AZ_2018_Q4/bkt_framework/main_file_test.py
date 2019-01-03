import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *
import work_whs.AZ_2018_Q4.bkt_framework.bkt_base_import_script as bkt_base


class KeyFun:
    @staticmethod
    def load_guba_data(name_guba, xinx, xnms, sector_df):
        guba_data = bt.AZ_Load_csv('/media/hdd1/DAT_EQT/EM_Funda/dat_whs/{name}.csv')
        guba_data.reindex(index=xinx, columns=xnms)

    @staticmethod
    def row_extre(raw_df, sector_df, percent):
        raw_df = raw_df * sector_df
        target_df = raw_df.rank(axis=1, pct=True)
        target_df[target_df >= 1 - percent] = 1
        target_df[target_df <= percent] = -1
        target_df[(target_df > percent) & (target_df < 1 - percent)] = 0
        return target_df

    def create_mix_factor(self, name_guba, name_sector, xinx, xnms, sector_df, if_only_long, percent):
        factor_1 = self.load_daily_data(name_guba, xinx, xnms, sector_df)
        factor_2 = self.load_daily_data(name_2, xinx, xnms, sector_df)
        score_df_1 = bt.AZ_Row_zscore(factor_1, cap=5)
        score_df_2 = bt.AZ_Row_zscore(factor_2, cap=5)
        mix_df = score_df_1 + score_df_2
        target_df = self.row_extre(mix_df, sector_df, percent)
        if if_only_long:
            target_df = target_df[target_df > 0]
        return target_df
