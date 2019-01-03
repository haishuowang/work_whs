import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import time

sys.path.append("/mnt/mfs/LIB_ROOT")


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
        self.sector_name = sector_name
        self.root_path = root_path
        market_top_n = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/DERIVED_10/{sector_name}.csv')
        market_top_n = market_top_n[(market_top_n.index >= begin_date) & (market_top_n.index < end_date)]
        self.sector_df = market_top_n
        self.xinx = self.sector_df.index
        self.xnms = self.sector_df.columns

    def industry01(self, file_list):
        industry_df_sum = pd.DataFrame()
        for file_name in file_list:
            industry_df = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/LICO_IM_INCHG/Global_Level1_{file_name}.csv') \
                .reindex(index=self.xinx, columns=self.xnms)
            industry_df_sum = industry_df_sum.add(industry_df, fill_value=0)
        industry_df_sum = industry_df_sum.reindex(index=self.xinx, columns=self.xnms)

        industry_df_sum = self.sector_df.mul(industry_df_sum, fill_value=0).replace(0, np.nan) \
            .dropna(how='all', axis='columns')
        industry_df_sum.to_csv(f"{self.root_path}/EM_Funda/DERIVED_10/{self.sector_name}_industry_"
                               f"{'_'.join([str(x) for x in file_list])}.csv", sep='|')
        return industry_df_sum

    def industry02(self):
        mapping = {
            'CS01': 6, 'CS02': 6, 'CS03': 10, 'CS04': 7, 'CS05': 10, 'CS06': 10, 'CS07': 10, 'CS08': 10,
            'CS09': 3, 'CS10': 3, 'CS11': 6, 'CS12': 3, 'CS13': 3, 'CS14': 1, 'CS15': 1, 'CS16': 1,
            'CS17': 1, 'CS18': 9, 'CS19': 9, 'CS20': 9, 'CS21': 5, 'CS22': 4, 'CS23': 2, 'CS24': 7,
            'CS25': 8, 'CS26': 8, 'CS27': 8, 'CS28': 8, 'CS29': 8
        }

        all_ind_df_raw = bt.AZ_Load_csv(f"{self.root_path}/EM_Funda/DERIVED_12/ZhongXing_Level1.csv")
        all_ind_df = all_ind_df_raw.applymap(lambda x: mapping[x] if isinstance(x, str) else x)
        for x in sorted(list(set(mapping.values()))):
            ind_df = all_ind_df == x
            ind_df = (self.sector_df * ind_df).replace(0, np.nan).dropna(how='all', axis='columns')
            ind_df.to_csv(f'{self.root_path}/EM_Funda/DERIVED_10/{self.sector_name}_ind{str(x)}.csv', sep='|')


def create_sector(root_path):
    # sector_split = SectorSplit(root_path, 'market_top_2000')
    # for file_list in [[10, 15], [20, 25, 30, 35], [40], [45, 50], [55]]:
    #     sector_split.industry01(file_list)
    #
    # sector_split = SectorSplit(root_path, 'market_top_1000')
    # for file_list in [[10, 15], [20, 25, 30, 35], [40], [45, 50], [55]]:
    #     sector_split.industry01(file_list)
    #
    # sector_split = SectorSplit(root_path, 'market_top_800plus')
    # sector_split.industry02()
    # for file_list in [[10, 15], [20, 25, 30, 35], [40], [45, 50], [55]]:
    #     sector_split.industry01(file_list)
    #
    # sector_split = SectorSplit(root_path, 'market_top_300')
    # for file_list in [[10, 15], [20, 25, 30, 35], [40], [45, 50], [55]]:
    #     sector_split.industry01(file_list)

    sector_split = SectorSplit(root_path, 'market_top_300plus')
    sector_split.industry02()
    # for file_list in [[10, 15], [20, 25, 30, 35], [40], [45, 50], [55]]:
    #     sector_split.industry01(file_list)

    sector_split = SectorSplit(root_path, 'market_top_300to800plus')
    sector_split.industry02()
    # for file_list in [[10, 15], [20, 25, 30, 35], [40], [45, 50], [55]]:
    #     sector_split.industry01(file_list)


def main(mod):
    if mod == 'bkt':
        root_path = '/mnt/mfs/DAT_EQT'
    elif mod == 'pro':
        root_path = '/media/hdd1/DAT_EQT'
    else:
        return -1
    a = time.time()
    create_sector(root_path)
    b = time.time()
    print('time cost:{} s'.format(b - a))


if __name__ == '__main__':
    # mod = 'pro'
    mod = 'bkt'
    main(mod)
