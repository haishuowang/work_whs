import pandas as pd
import numpy as np
import os

index_root_path = '/mnt/mfs/dat_whs/index_weight'
return_choose = pd.read_csv('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv', index_col=0, sep='|', parse_dates=True)
date_list = return_choose.index


def choice_index_load(index_code):
    data = pd.read_csv(f'/mnt/mfs/DAT_EQT/EM_Funda/IDEX_YS_WEIGHT_A/SECINDEXR_{index_code}.csv', index_col=0, sep='|')
    return data


def index_deal(index_code):
    index_file_list = sorted([x for x in os.listdir(index_root_path) if x.startswith(index_code)])
    target_df = pd.DataFrame()
    choice_data = choice_index_load(index_code)
    choice_data.dropna(how='all', axis='index', inplace=True)
    cut_date = choice_data.index[0]
    use_date_list = date_list[date_list < cut_date]
    for index_file in index_file_list:
        raw_df = pd.read_csv(f'/mnt/mfs/dat_whs/index_weight/{index_file}', index_col=0, encoding='gbk')
        part_target_df = raw_df.pivot('Date', 'Code', 'Weight')
        target_df = target_df.append(part_target_df)

    target_df.replace(np.nan, 0, inplace=True)
    target_df.index = pd.to_datetime(target_df.index)

    target_df = target_df[target_df.index < cut_date] * 0.01
    target_df = target_df.reindex(index=use_date_list).fillna(method='ffill')
    index_weight_df = target_df.append(choice_data).replace(0, np.nan).dropna(how='all', axis='index')
    index_weight_df.to_csv(f'/mnt/mfs/DAT_EQT/EM_Funda/IDEX_YS_WEIGHT_A/SECINDEXR_{index_code}plus.csv', sep='|')
    return index_weight_df


if __name__ == '__main__':
    target_df_300 = index_deal('000300')
    target_df_500 = index_deal('000905')

    class A:
        def __init__(self, a=1):
            self.x1 = a
            self._x2 = {}

        @property
        def x2(self):
            if self.x1 not in self._x2:
                print(f"not using cache, calculating: {self.x1}")
                self._x2[self.x1] = self.x1 ** 2
            return self._x2[self.x1]

    a = A(1)
    a.x2  # x1 = 1;

    a.x1 = 2  # reset x1 to another value
    a.x2  # not using cache at first time
    a.x2  # do it again, using cache