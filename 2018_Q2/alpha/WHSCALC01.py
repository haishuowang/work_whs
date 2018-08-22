import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import time
sys.path.append("/mnt/mfs/LIB_ROOT")
# import funda_data as fd
# from funda_data.funda_data_deal import SectorData
import open_lib.shared_paths.path as pt
# import warnings
# warnings.filterwarnings('ignore')
# import loc_lib.shared_tools.back_test as bt


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


class FD:
    class funda_data_deal:
        class BaseDeal:
            @staticmethod
            def signal_mean_fun(signal_df):
                return signal_df.abs().sum(axis=1).mean() > 1

            @staticmethod
            def pnd_continue_ud(raw_df, sector_df, n_list):
                all_target_df = pd.DataFrame()
                for n in n_list:
                    target_df = raw_df.rolling(window=n + 1).apply(
                        lambda x: 1 if (np.diff(x) >= 0).all() and sum(np.diff(x)) > 0
                        else (-1 if (np.diff(x) <= 0).all() and sum(np.diff(x)) < 0 else 0))

                    target_df = target_df * sector_df
                    all_target_df = all_target_df.add(target_df, fill_value=0)
                return all_target_df

            @staticmethod
            def pnd_continue_ud_pct(raw_df, sector_df, n_list):
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
            def pnd_col_extre(raw_df, sector_df, window, percent, min_periods=1):
                dn_df = raw_df.rolling(window=window, min_periods=min_periods).quantile(percent)
                up_df = raw_df.rolling(window=window, min_periods=min_periods).quantile(1 - percent)
                dn_target = -(raw_df < dn_df).astype(int)
                up_target = (raw_df > up_df).astype(int)
                target_df = dn_target + up_target
                return target_df * sector_df

            @staticmethod
            def info_dict_fun(fun, raw_data_path, args, save_path, if_replace):
                info_dict = dict()
                info_dict['fun'] = fun
                info_dict['raw_data_path'] = raw_data_path
                info_dict['args'] = args
                info_dict['if_replace'] = if_replace
                pd.to_pickle(info_dict, save_path)

            @staticmethod
            def pnd_hl(high, low, close, sector_df, n):
                high_n = high.rolling(window=n, min_periods=1).max().shift(1)
                low_n = low.rolling(window=n, min_periods=1).min().shift(1)
                h_diff = (close - high_n)
                l_diff = (close - low_n)

                h_diff[h_diff > 0] = 1
                h_diff[h_diff <= 0] = 0

                l_diff[l_diff >= 0] = 0
                l_diff[l_diff < 0] = -1

                pos = h_diff + l_diff
                return pos * sector_df

            @staticmethod
            def pnd_volume(volume, sector_df, n):
                volume_roll_mean = bt.AZ_Rolling_mean(volume, n) * sector_df
                volume_df_count_down = 1 / (volume_roll_mean.replace(0, np.nan))
                return volume_df_count_down

            @staticmethod
            def pnd_volitality(adj_r, sector_df, n):
                vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5)
                vol_df[vol_df < 0.08] = 0.08
                return vol_df * sector_df

            @staticmethod
            def pnd_volitality_count_down(adj_r, sector_df, n):
                vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5) * sector_df
                vol_df[vol_df < 0.08] = 0.08
                return 1 / vol_df.replace(0, np.nan)

            @staticmethod
            def pnd_evol(adj_r, sector_df, n):
                vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5)
                vol_df[vol_df < 0.08] = 0.08
                evol_df = bt.AZ_Rolling(vol_df, 30).apply(lambda x: 1 if x[-1] > 2 * x.mean() else 0)
                return evol_df * sector_df

            def pnd_vol_continue_ud(self, adj_r, sector_df, n):
                vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5)
                vol_df[vol_df < 0.08] = 0.08
                vol_continue_ud_df = self.pnd_continue_ud(vol_df, sector_df, n_list=[3, 4, 5])
                return vol_continue_ud_df

            @staticmethod
            def pnnd_moment(df, sector_df, n_short=10, n_long=60):
                ma_long = df.rolling(window=n_long, min_periods=1).mean()
                ma_short = df.rolling(window=n_short, min_periods=1).mean()
                ma_dif = ma_short - ma_long
                ma_dif[ma_dif == 0] = 0
                ma_dif[ma_dif > 0] = 1
                ma_dif[ma_dif < 0] = -1
                return ma_dif * sector_df

            @staticmethod
            def p1d_jump_hl(close, open, sector_df, split_float_list):
                target_df = pd.DataFrame()
                for split_float in split_float_list:
                    jump_df = open / close.shift(1) - 1
                    tmp_df = pd.DataFrame(index=jump_df.index, columns=jump_df.columns)
                    tmp_df[(jump_df > 0.101) | (jump_df < -0.101)] = 0
                    tmp_df[(split_float >= jump_df) & (jump_df >= -split_float)] = 0
                    tmp_df[jump_df > split_float] = 1
                    tmp_df[jump_df < -split_float] = -1
                    target_df = target_df.add(tmp_df, fill_value=0)
                return target_df * sector_df

            def judge_save_fun(self, target_df, file_name, save_root_path, fun, raw_data_path, args, if_filter=True,
                               if_replace=False):
                factor_to_fun = '/mnt/mfs/dat_whs/data/factor_to_fun'
                if if_filter:
                    if self.signal_mean_fun(target_df):
                        target_df.to_pickle(os.path.join(save_root_path, file_name + '.pkl'))
                        # 构建factor_to_fun的字典并存储
                        self.info_dict_fun(fun, raw_data_path, args, os.path.join(factor_to_fun, file_name), if_replace)
                        print(f'{file_name} success!')
                    else:
                        print(f'{file_name} not enough!')
                else:
                    target_df.to_pickle(os.path.join(save_root_path, file_name + '.pkl'))
                    # 构建factor_to_fun的字典并存储
                    self.info_dict_fun(fun, raw_data_path, args, os.path.join(factor_to_fun, file_name), if_replace)
                    print(f'{file_name} success!')

        class TechBaseDeal(BaseDeal):
            def __init__(self, sector_df, root_path, save_root_path):
                xnms = sector_df.columns
                xinx = sector_df.index

                self.load_path = root_path.EM_Funda.TRAD_SK_DAILY_JC
                self.part_load_path = 'EM_Funda/TRAD_SK_DAILY_JC'
                self.sector_open = bt.AZ_Load_csv(self.load_path / 'OPEN.csv').reindex(columns=xnms, index=xinx)
                self.sector_high = bt.AZ_Load_csv(self.load_path / 'HIGH.csv').reindex(columns=xnms, index=xinx)
                self.sector_low = bt.AZ_Load_csv(self.load_path / 'LOW.csv').reindex(columns=xnms, index=xinx)
                self.sector_close = bt.AZ_Load_csv(self.load_path / 'NEW.csv').reindex(columns=xnms, index=xinx)
                self.sector_volume = bt.AZ_Load_csv(self.load_path / 'TVOL.csv').reindex(columns=xnms, index=xinx)
                self.sector_amount = bt.AZ_Load_csv(self.load_path / 'TVALCNY.csv').reindex(columns=xnms, index=xinx)
                self.sector_adj_r = bt.AZ_Load_csv(root_path.EM_Funda.DERIVED_14 / 'aadj_r.csv').reindex(columns=xnms,
                                                                                                         index=xinx)
                self.sector_df = sector_df
                self.save_root_path = save_root_path
                self.factor_to_fun = '/mnt/mfs/dat_whs/data/factor_to_fun'

            def pnd_hl_(self, n_list):
                for n in n_list:
                    target_df = self.pnd_hl(self.sector_high, self.sector_low, self.sector_close, self.sector_df, n)

                    file_name = f'price_p{n}d_hl'
                    fun = 'funda_data_deal.TechBaseDeal.pnd_hl'
                    raw_data_path = (self.part_load_path + '/HIGH.csv',
                                     self.part_load_path + '/LOW.csv',
                                     self.part_load_path + '/NEW.csv',)
                    args = (n,)
                    self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)

            def pnd_volume_(self, n_list):
                for n in n_list:
                    target_df = self.pnd_volume(self.sector_volume, self.sector_df, n)
                    file_name = f'volume_count_down_p{n}d'
                    fun = 'funda_data_deal.TechBaseDeal.pnd_volume'
                    raw_data_path = (self.part_load_path + '/TVOL.csv',)
                    args = (n,)
                    self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args, if_filter=False)

            def pnd_volitality_and_more_(self, n_list):
                for n in n_list:
                    vol_df = bt.AZ_Rolling(self.sector_adj_r, n).std()*(250 ** 0.5)
                    vol_df[vol_df < 0.08] = 0.08
                    evol_df = bt.AZ_Rolling(vol_df, 30).apply(lambda x: 1 if x[-1] > 2 * x.mean() else 0) * self.sector_df
                    vol_continue_ud_df = self.pnd_continue_ud(vol_df, self.sector_df, n_list=[3, 4, 5])
                    # 剔除极值
                    vol_df_count_down = 1 / (vol_df.replace(0, np.nan) * self.sector_df)

                    file_name = f'vol_p{n}d'
                    fun = 'funda_data_deal.BaseDeal.pnd_volitality'
                    raw_data_path = ('EM_Funda/DERIVED_14/aadj_r.csv',)
                    args = (n,)
                    self.judge_save_fun(vol_df, file_name, self.save_root_path, fun, raw_data_path, args, if_filter=False)

                    file_name = f'vol_count_down_p{n}d'
                    fun = 'funda_data_deal.BaseDeal.pnd_volitality_count_down'
                    raw_data_path = ('EM_Funda/DERIVED_14/aadj_r.csv',)
                    args = (n,)
                    self.judge_save_fun(vol_df_count_down, file_name, self.save_root_path, fun, raw_data_path, args)

                    file_name = f'evol_p{n}d'
                    fun = 'funda_data_deal.BaseDeal.pnd_evol'
                    raw_data_path = ('EM_Funda/DERIVED_14/aadj_r.csv',)
                    args = (n,)
                    self.judge_save_fun(evol_df, file_name, self.save_root_path, fun, raw_data_path, args)

                    file_name = f'continue_ud_p{n}d'
                    fun = 'funda_data_deal.BaseDeal.pnd_vol_continue_ud'
                    raw_data_path = ('EM_Funda/DERIVED_14/aadj_r.csv',)
                    args = (n,)
                    self.judge_save_fun(vol_continue_ud_df, file_name, self.save_root_path, fun, raw_data_path, args)

            def pnnd_moment_(self, short_long_list):
                # short_long_list = [(5, 10), (10, 60), (10, 100), (20, 100), (20, 200), (40, 200)]
                for n_short, n_long in short_long_list:
                    target_df = self.pnnd_moment(self.sector_adj_r, self.sector_df, n_short, n_long)
                    file_name = f'moment_p{n_short}{n_long}d'
                    fun = 'funda_data_deal.TechBaseDeal.pnnd_moment'
                    raw_data_path = ('EM_Funda/DERIVED_14/aadj_r.csv',)
                    args = (n_short, n_long)
                    self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)

            def p1d_jump_hl_(self, split_float_list):
                target_df = self.p1d_jump_hl(self.sector_close, self.sector_open, self.sector_df, split_float_list)
                str_name = ''.join([str(x) for x in split_float_list])
                file_name = f'p1d_jump_hl{str_name}'
                fun = 'funda_data_deal.TechBaseDeal.p1d_jump_hl'
                raw_data_path = (self.load_path / 'NEW.csv',
                                 self.load_path / 'OPEN.csv',)
                args = (split_float_list,)
                self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)

        class FundaBaseDeal(BaseDeal):
            def __init__(self, sector_df, root_path, table_num, table_name, data_name, save_root_path, if_replace=False):
                xnms = sector_df.columns
                xinx = sector_df.index
                data = bt.AZ_Load_csv(getattr(getattr(root_path, table_num), table_name) / (data_name + '.csv'))
                if if_replace:
                    self.raw_df = data.replace(0, np.nan).reindex(columns=xnms, index=xinx)
                else:
                    self.raw_df = data.reindex(columns=xnms, index=xinx)
                self.sector_df = sector_df
                self.table_num = table_num
                self.table_name = table_name
                self.data_name = data_name
                self.save_root_path = save_root_path
                self.factor_to_fun = '/mnt/mfs/dat_whs/data/factor_to_fun'
                self.if_replace = if_replace

            def pnd_continue_ud_(self, n_list):
                target_df = self.pnd_continue_ud(self.raw_df, self.sector_df, n_list)
                file_name = self.data_name + '_p{}d_continue_ud'.format(''.join([str(x) for x in n_list]))
                fun = 'funda_data_deal.FundaBaseDeal.pnd_continue_ud'
                raw_data_path = (f'{self.table_num}/{self.table_name}/{self.data_name}.csv',)
                args = (n_list,)
                self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args,
                                    if_replace=self.if_replace)

            def pnd_continue_ud_pct_(self, n_list):
                target_df = self.pnd_continue_ud_pct(self.raw_df, self.sector_df, n_list)
                file_name = self.data_name + '_p{}d_continue_ud_pct'.format(''.join([str(x) for x in n_list]))
                fun = 'funda_data_deal.FundaBaseDeal.pnd_continue_ud_pct'
                raw_data_path = (f'{self.table_num}/{self.table_name}/{self.data_name}.csv',)
                args = (n_list,)
                self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args,
                                    if_replace=self.if_replace)

            def row_extre_(self, percent):
                target_df = self.row_extre(self.raw_df, self.sector_df, percent)
                file_name = self.data_name + '_row_extre_{}'.format(percent)
                fun = 'funda_data_deal.FundaBaseDeal.row_extre'
                raw_data_path = (f'{self.table_num}/{self.table_name}/{self.data_name}.csv',)
                args = (percent,)
                self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args,
                                    if_replace=self.if_replace)

            def pnd_col_extre_(self, n_list, percent, min_periods=1):
                for n in n_list:
                    target_df = self.pnd_col_extre(self.raw_df, self.sector_df, n, percent, min_periods=min_periods)

                    file_name = self.data_name + '_p{}d_col_extre_{}'.format(n, percent)
                    fun = 'funda_data_deal.FundaBaseDeal.pnd_col_extre'
                    raw_data_path = (f'{self.table_num}/{self.table_name}/{self.data_name}.csv',)
                    args = (n, percent)
                    self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args,
                                        if_replace=self.if_replace)


class SectorData(object):
    def __init__(self, root_path):
        self.root_path = root_path

    # 获取剔除新股的矩阵
    def get_new_stock_info(self, xnms, xinx):
        new_stock_data = bt.AZ_Load_csv(self.root_path.EM_Tab01.CDSY_SECUCODE / 'LISTSTATE.csv')
        new_stock_data.fillna(method='ffill', inplace=True)
        # 获取交易日信息
        return_df = bt.AZ_Load_csv(self.root_path.EM_Funda.DERIVED_14 / 'aadj_r.csv').astype(float)
        trade_time = return_df.index
        new_stock_data = new_stock_data.reindex(index=trade_time).fillna(method='ffill')
        target_df = new_stock_data.shift(40).notnull().astype(int)
        target_df = target_df.reindex(columns=xnms, index=xinx)
        return target_df

    # 获取剔除st股票的矩阵
    def get_st_stock_info(self, xnms, xinx):
        data = bt.AZ_Load_csv(self.root_path.EM_Tab01.CDSY_CHANGEINFO / 'CHANGEA.csv')
        data = data.reindex(columns=xnms, index=xinx)
        data.fillna(method='ffill', inplace=True)

        data = data.astype(str)
        target_df = data.applymap(lambda x: 0 if 'ST' in x or 'PT' in x else 1)
        return target_df

    # 读取 sector(行业 最大市值等)
    def load_sector_data(self, begin_date, end_date, sector_name):
        market_top_n = bt.AZ_Load_csv(self.root_path.EM_Funda.DERIVED_10 / (sector_name + '.csv'))
        market_top_n = market_top_n.shift(1)[(market_top_n.index >= begin_date) & (market_top_n.index < end_date)]
        market_top_n.dropna(how='all', axis='columns', inplace=True)
        xnms = market_top_n.columns
        xinx = market_top_n.index

        new_stock_df = self.get_new_stock_info(xnms, xinx).shift(1)
        st_stock_df = self.get_st_stock_info(xnms, xinx).shift(1)
        sector_df = market_top_n * new_stock_df * st_stock_df
        sector_df.replace(0, np.nan, inplace=True)
        return sector_df


def find_fun(fun_list):
    target_fun = FD
    for a in fun_list:
        target_fun = getattr(target_fun, a)
    return target_fun


def load_raw_data(root_path, raw_data_path, xnms, xinx, if_replace):
    raw_data_list = []
    for target_path in raw_data_path:
        tmp_data = bt.AZ_Load_csv(os.path.join('/media/hdd1/DAT_EQT', target_path)).reindex(columns=xnms, index=xinx)
        if if_replace:
            tmp_data = tmp_data.replace(0, np.nan)
        raw_data_list += [tmp_data]
    return raw_data_list


def create_data_fun(root_path, info_path, sector_df, xnms, xinx):
    info = pd.read_pickle(info_path)
    args = info['args']
    fun_list = info['fun'].split('.')
    raw_data_path = info['raw_data_path']
    if_replace = info['if_replace']
    raw_data_list = load_raw_data(root_path, raw_data_path, xnms, xinx, if_replace)

    target_fun = find_fun(fun_list)
    target_df = target_fun(*raw_data_list, sector_df, *args)
    return target_df


def main():
    mode = 'pro'

    begin_date = pd.to_datetime('20100101')
    end_date = datetime.now()

    sector_name = 'market_top_500'

    save_root_path = f'/media/hdd1/DAT_PreCalc/PreCalc_whs/{sector_name}'

    bt.AZ_Path_create(save_root_path)
    root_path = pt._BinFiles(mode)
    sector_data_class = SectorData(root_path)
    sector_df = sector_data_class.load_sector_data(begin_date, end_date, sector_name)

    xnms = sector_df.columns
    xinx = sector_df.index

    file_name_list = [x[:-4] for x in os.listdir(save_root_path)]
    # a = pd.to_datetime('20180601')
    # b = pd.to_datetime('20180801')

    config = pd.read_pickle('/mnt/mfs/alpha_whs/config01.pkl')
    factor_info = config['factor_info']
    file_name_list = list(set(factor_info[['name1', 'name2', 'name3']].values.ravel()))

    for file_name in file_name_list:
        # print(file_name)
        factor_to_fun = '/mnt/mfs/dat_whs/data/factor_to_fun'
        info_path = os.path.join(factor_to_fun, file_name)
        create_data = create_data_fun(root_path, info_path, sector_df, xnms, xinx)
        create_data.to_pickle(os.path.join(save_root_path, f'{file_name}.pkl'))


if __name__ == '__main__':
    a = time.time()
    main()
    b = time.time()
    print('pre cal cost time:{} s'.format(b-a))
