import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *


class DiscreteClass:
    """
    生成离散数据的公用函数
    """

    @staticmethod
    def pnd_con_ud(raw_df, sector_df, n_list):
        def fun(df, n):
            df_pct = df.diff()
            up_df = (df_pct > 0)
            dn_df = (df_pct < 0)
            target_up_df = up_df.copy()
            target_dn_df = dn_df.copy()

            for i in range(n - 1):
                target_up_df = target_up_df * up_df.shift(i + 1)
                target_dn_df = target_dn_df * dn_df.shift(i + 1)
            target_df = target_up_df.fillna(0).astype(int) - target_dn_df.fillna(0).astype(int)
            return target_df

        all_target_df = pd.DataFrame()
        for n in n_list:
            target_df = fun(raw_df, n)
            target_df = target_df * sector_df
            all_target_df = all_target_df.add(target_df, fill_value=0)
        return all_target_df

    @staticmethod
    def pnd_con_ud_pct(raw_df, sector_df, n_list):
        all_target_df = pd.DataFrame()
        for n in n_list:
            target_df = raw_df.rolling(window=n).apply(lambda x: 1 if (x >= 0).all() and sum(x) > 0
            else (-1 if (x <= 0).all() and sum(x) < 0 else 0))
            target_df = target_df * sector_df
            all_target_df = all_target_df.add(target_df, fill_value=0)
        return all_target_df

    @staticmethod
    def row_extre(raw_df, sector_df, percent):
        raw_df = raw_df * sector_df
        target_df = raw_df.rank(axis=1, pct=True)
        target_df[target_df >= 1 - percent] = 1
        target_df[target_df <= percent] = -1
        target_df[(target_df > percent) & (target_df < 1 - percent)] = 0
        return target_df

    @staticmethod
    def col_extre(raw_df, sector_df, window, percent, min_periods=1):
        dn_df = raw_df.rolling(window=window, min_periods=min_periods).quantile(percent)
        up_df = raw_df.rolling(window=window, min_periods=min_periods).quantile(1 - percent)
        dn_target = -(raw_df < dn_df).astype(int)
        up_target = (raw_df > up_df).astype(int)
        target_df = dn_target + up_target
        return target_df * sector_df

    @staticmethod
    def signal_fun(zscore_df, sector_df, limit):
        zscore_df[(zscore_df < limit) & (zscore_df > -limit)] = 0
        zscore_df[zscore_df >= limit] = 1
        zscore_df[zscore_df <= -limit] = -1
        return zscore_df * sector_df


class ContinueClass:
    """
    生成连续数据的公用函数
    """

    @staticmethod
    def roll_fun_20(raw_df, sector_df):
        return bt.AZ_Rolling_mean(raw_df, 20)

    @staticmethod
    def roll_fun_40(raw_df, sector_df):
        return bt.AZ_Rolling_mean(raw_df, 40)

    @staticmethod
    def roll_fun_100(raw_df, sector_df):
        return bt.AZ_Rolling_mean(raw_df, 100)

    @staticmethod
    def col_zscore(raw_df, sector_df, n, cap=5, min_periods=0):
        return bt.AZ_Col_zscore(raw_df, n, cap, min_periods)

    @staticmethod
    def row_zscore(raw_df, sector_df, cap=5):
        return bt.AZ_Row_zscore(raw_df * sector_df, cap)

    @staticmethod
    def pnd_vol(raw_df, sector_df, n):
        vol_df = bt.AZ_Rolling(raw_df, n).std().round(4) * (250 ** 0.5)
        return vol_df * sector_df

    # @staticmethod
    # def pnd_count_down(raw_df, sector_df, n):
    #     raw_df = raw_df.replace(0, np.nan)
    #     raw_df_mean = bt.AZ_Rolling_mean(raw_df, n) * sector_df
    #     raw_df_count_down = 1 / (raw_df_mean.round(4).replace(0, np.nan))
    #     return raw_df_count_down

    # return fun
    @staticmethod
    def pnd_return_volatility(adj_r, n):
        vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5)
        vol_df[vol_df < 0.08] = 0.08
        return vol_df

    @staticmethod
    def pnd_return_volatility_count_down(adj_r, sector_df, n):
        vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5) * sector_df
        vol_df[vol_df < 0.08] = 0.08
        return 1 / vol_df.replace(0, np.nan)

    @staticmethod
    def pnd_return_evol(adj_r, sector_df, n):
        vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5)
        vol_df[vol_df < 0.08] = 0.08
        evol_df = bt.AZ_Rolling(vol_df, 30).apply(lambda x: 1 if x[-1] > 2 * x.mean() else 0)
        return evol_df * sector_df


class SpecialClass:
    """
    某些数据使用的特殊函数
    """

    @staticmethod
    def pnd_evol(adj_r, sector_df, n):
        vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5)
        vol_df[vol_df < 0.08] = 0.08
        evol_df = bt.AZ_Rolling(vol_df, 30).apply(lambda x: 1 if x[-1] > 2 * x.mean() else 0)
        return evol_df * sector_df

    # @staticmethod
    # def ():
    #


class SectorFilter:
    def __init__(self, root_path):
        self.root_path = root_path

    def filter_market(self):
        market_df = bt.AZ_Load_csv(f'{self.root_path}/')

    def filter_vol(self):
        pass

    def filter_moment(self):
        pass


class BaseDeal(DiscreteClass, ContinueClass):
    @staticmethod
    def info_dict_fun(fun, raw_data_path, args, save_path, if_replace):
        info_dict = dict()
        info_dict['fun'] = fun
        info_dict['raw_data_path'] = raw_data_path
        info_dict['args'] = args
        info_dict['if_replace'] = if_replace
        pd.to_pickle(info_dict, save_path)

    def judge_save_fun(self, target_df, file_name, save_root_path, fun, raw_data_path, args, if_filter=True,
                       if_replace=False):
        factor_to_fun = '/mnt/mfs/dat_whs/data/factor_to_fun'
        if target_df.sum().sum() == 0:
            print('factor not enough!')
            return -1
        elif if_filter:
            print(f'{file_name}')
            print(target_df.iloc[-100:].abs().replace(0, np.nan).sum(axis=1).mean(), len(target_df.iloc[-100:].columns))
            print(
                target_df.iloc[-100:].abs().replace(0, np.nan).sum(axis=1).mean() / len(target_df.iloc[-100:].columns))
            target_df.to_pickle(os.path.join(save_root_path, file_name + '.pkl'))
            # 构建factor_to_fun的字典并存储
            self.info_dict_fun(fun, raw_data_path, args, os.path.join(factor_to_fun, file_name), if_replace)
            print(f'{file_name} success!')
            return 0
        else:
            target_df.to_pickle(os.path.join(save_root_path, file_name + '.pkl'))
            # 构建factor_to_fun的字典并存储
            self.info_dict_fun(fun, raw_data_path, args, os.path.join(factor_to_fun, file_name), if_replace)
            print(f'{file_name} success!')
            return 0

# root_path = '/mnt/mfs/DAT_EQT'
# index_name = '000905'
# sector_df = bt.AZ_Load_csv(f'{root_path}/EM_Funda/IDEX_YS_WEIGHT_A/SECURITYNAME_{index_name}.csv')
# sector_df[sector_df == sector_df] = 1
#
# table_path = f'{root_path}/EM_Funda/LICO_IM_INCHG'
# target_file_list = [x[:-4] for x in os.listdir(table_path) if 'ZhongXing_Level2' in x]
#
#
# def fun(file_name):
#     xinx = sector_df.index
#     raw_df = bt.AZ_Load_csv(f'{table_path}/{file_name}.csv')
#     xnms = sorted(list(set(sector_df.columns) & set(raw_df.columns)))
#     tmp_df = raw_df.reindex(index=xinx, columns=xnms)
#     target_df = tmp_df * sector_df.reindex(columns=xnms)
#     target_num = target_df.sum(1)
#     target_num.name = file_name
#     return target_num
#
#
# target_num_list = [fun(file_name) for file_name in target_file_list]
# data = pd.concat(target_num_list, axis=1)
