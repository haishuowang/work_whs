import sys

sys.path.append('/mnt/mfs')
import work_whs.loc_lib.shared_tools.back_test as bt
from sklearn import cluster, datasets, mixture


class Signal:
    @staticmethod
    def signal_fun(zscore_df, limit):
        zscore_df[(zscore_df < limit) & (zscore_df > -limit)] = 0
        zscore_df[zscore_df >= limit] = 1
        zscore_df[zscore_df <= -limit] = -1
        return zscore_df

    @staticmethod
    def row_extre(raw_df, percent):
        raw_df = raw_df
        target_df = raw_df.rank(axis=1, pct=True)
        target_df[target_df >= 1 - percent] = 1
        target_df[target_df <= percent] = -1
        target_df[(target_df > percent) & (target_df < 1 - percent)] = 0
        return target_df
