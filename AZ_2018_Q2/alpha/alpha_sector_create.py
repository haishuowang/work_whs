import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import time

sys.path.append("/mnt/mfs/LIB_ROOT")
import open_lib_c.shared_paths.path as pt
from open_lib_c.shared_tools import send_email


class bt:
    @staticmethod
    def AZ_Load_csv(target_path, index_time_type=True):
        target_df = pd.read_table(target_path, sep='|', index_col=0, low_memory=False).round(8)
        if index_time_type:
            target_df.index = pd.to_datetime(target_df.index)
        return target_df

    @staticmethod
    def AZ_Rolling(df, n, min_periods=1):
        return df.rolling(window=n, min_periods=min_periods)

    @staticmethod
    def AZ_Rolling_mean(df, n, min_periods=1):
        target = df.rolling(window=n, min_periods=min_periods).mean()
        target.iloc[:n - 1] = np.nan
        return target

    @staticmethod
    def AZ_Path_create(target_path):
        """
        添加新路径
        :param target_path:
        :return:
        """
        if not os.path.exists(target_path):
            os.makedirs(target_path)


class SectorSplit:
    def __init__(self, root_path, sector_name):
        begin_date = pd.to_datetime('20050505')
        end_date = datetime.now()
        # sector_name = 'market_top_2000'
        self.root_path = root_path
        self.sector_name = sector_name
        market_top_n = bt.AZ_Load_csv(os.path.join(f'{self.root_path}/EM_Funda/DERIVED_10/' + sector_name + '.csv'))
        market_top_n = market_top_n[(market_top_n.index >= begin_date) & (market_top_n.index < end_date)]
        self.sector_df = market_top_n
        self.xinx = self.sector_df.index
        self.xnms = self.sector_df.columns

    def industry(self, file_list):
        industry_df_sum = pd.DataFrame()
        for file_name in file_list:
            industry_df = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/LICO_IM_INCHG/Global_Level1_{file_name}.csv') \
                .reindex(index=self.xinx, columns=self.xnms)
            industry_df_sum = industry_df_sum.add(industry_df, fill_value=0)
        industry_df_sum = industry_df_sum.reindex(index=self.xinx, columns=self.xnms)

        industry_df_sum = self.sector_df.mul(industry_df_sum, fill_value=0).replace(0, np.nan) \
            .dropna(how='all', axis='columns')
        industry_df_sum.to_csv('{}/EM_Funda/DERIVED_10/{}_industry_{}.csv'
                               .format(self.root_path, self.sector_name, '_'.join([str(x) for x in file_list])),
                               sep='|')

        return industry_df_sum


def create_sector():
    root_path = '/mnt/mfs/DAT_EQT'
    sector_split = SectorSplit(root_path, 'market_top_2000')
    for file_list in [[10, 15], [20, 25, 30, 35], [40], [45, 50], [55]]:
        sector_split.industry(file_list)

    sector_split = SectorSplit(root_path, 'market_top_1000')
    for file_list in [[10, 15], [20, 25, 30, 35], [40], [45, 50], [55]]:
        sector_split.industry(file_list)

    sector_split = SectorSplit(root_path, 'market_top_800plus')
    for file_list in [[10, 15], [20, 25, 30, 35], [40], [45, 50], [55]]:
        sector_split.industry(file_list)

if __name__ == '__main__':
    create_sector()