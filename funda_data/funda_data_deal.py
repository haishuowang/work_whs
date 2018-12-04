import pandas as pd
import numpy as np
import work_whs.loc_lib.shared_paths.path as pt
import work_whs.loc_lib.shared_tools.back_test as bt
import os


class BaseDeal:
    @staticmethod
    def signal_mean_fun(signal_df):
        return signal_df.abs().sum(axis=1).replace(0, np.nan).dropna() / len(signal_df) > 0.1

    @staticmethod
    def pnd_continue_ud(raw_df, sector_df, n_list):
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
        return vol_df

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
    def p1d_jump_hl(close, open_, sector_df, split_float_list):
        target_df = pd.DataFrame()
        for split_float in split_float_list:
            jump_df = open_ / close.shift(1) - 1
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
        if target_df.sum().sum() == 0:
            print('factor not enough!')
            return -1
        elif if_filter:
            print(f'{file_name}')
            print(target_df.iloc[-100:].abs().replace(0, np.nan).sum(axis=1).mean(), len(target_df.iloc[-100:].columns))
            print(target_df.iloc[-100:].abs().replace(0, np.nan).sum(axis=1).mean() / len(target_df.iloc[-100:].columns))
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
    # @staticmethod
    # def pnnd_volume_moment(volume, sector_df, n_short, n_long):
    #     volume_n_short = bt.AZ_Rolling_mean(volume, n_short)
    #     volume_n_long = bt.AZ_Rolling_mean(volume, n_long)
    #     volume_dif = volume_n_short - volume_n_long
    #     volume_dif[volume_dif == 0] = 0
    #     volume_dif[volume_dif > 0] = 1
    #     volume_dif[volume_dif < 0] = -1
    #     return volume_dif * sector_df


class TechBaseDeal(BaseDeal):
    def __init__(self, sector_df, root_path, save_root_path):
        xnms = sector_df.columns
        xinx = sector_df.index

        self.load_path = root_path.EM_Funda.DERIVED_14
        self.part_load_path1 = 'EM_Funda/TRAD_SK_DAILY_JC'
        self.part_load_path = 'EM_Funda/DERIVED_14'
        self.sector_open = bt.AZ_Load_csv(root_path.EM_Funda.DERIVED_14 / 'aadj_p_OPEN.csv')\
            .reindex(columns=xnms, index=xinx)
        self.sector_high = bt.AZ_Load_csv(root_path.EM_Funda.DERIVED_14 / 'aadj_p_HIGH.csv')\
            .reindex(columns=xnms, index=xinx)
        self.sector_low = bt.AZ_Load_csv(root_path.EM_Funda.DERIVED_14 / 'aadj_p_LOW.csv')\
            .reindex(columns=xnms, index=xinx)
        self.sector_close = bt.AZ_Load_csv(root_path.EM_Funda.DERIVED_14 / 'aadj_p.csv')\
            .reindex(columns=xnms, index=xinx)
        self.sector_volume = bt.AZ_Load_csv(root_path.EM_Funda.TRAD_SK_DAILY_JC / 'TVOL.csv')\
            .reindex(columns=xnms, index=xinx)
        self.sector_amount = bt.AZ_Load_csv(root_path.EM_Funda.TRAD_SK_DAILY_JC / 'TVALCNY.csv')\
            .reindex(columns=xnms, index=xinx)
        self.sector_adj_r = bt.AZ_Load_csv(root_path.EM_Funda.DERIVED_14 / 'aadj_r.csv')\
            .reindex(columns=xnms, index=xinx)
        self.sector_df = sector_df
        self.save_root_path = save_root_path
        self.factor_to_fun = '/mnt/mfs/dat_whs/data/factor_to_fun'

    def pnd_hl_(self, n_list):
        for n in n_list:
            target_df = self.pnd_hl(self.sector_high, self.sector_low, self.sector_close, self.sector_df, n)

            file_name = f'price_p{n}d_hl'
            fun = 'funda_data_deal.BaseDeal.pnd_hl'
            raw_data_path = (self.part_load_path + '/aadj_p_HIGH.csv',
                             self.part_load_path + '/aadj_p_LOW.csv',
                             self.part_load_path + '/aadj_p.csv',)
            args = (n,)
            self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)

    def pnd_volume_(self, n_list):
        for n in n_list:
            target_df = self.pnd_volume(self.sector_volume, self.sector_df, n)
            file_name = f'volume_count_down_p{n}d'
            fun = 'funda_data_deal.BaseDeal.pnd_volume'
            raw_data_path = (self.part_load_path1 + '/TVOL.csv',)
            args = (n,)
            self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args, if_filter=False)

    def pnd_volitality_and_more_(self, n_list):
        for n in n_list:
            vol_df = bt.AZ_Rolling(self.sector_adj_r, n).std() * (250 ** 0.5)
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
            fun = 'funda_data_deal.BaseDeal.pnnd_moment'
            raw_data_path = ('EM_Funda/DERIVED_14/aadj_r.csv',)
            args = (n_short, n_long)
            self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)

    # def p1d_jump_hl_(self, split_float_list):
    #     target_df = self.p1d_jump_hl(self.sector_close, self.sector_open, self.sector_df, split_float_list)
    #     str_name = ''.join([str(x) for x in split_float_list])
    #     file_name = f'p1d_jump_hl{str_name}'
    #     fun = 'funda_data_deal.BaseDeal.p1d_jump_hl'
    #     raw_data_path = (self.load_path / 'NEW.csv',
    #                      self.load_path / 'OPEN.csv',)
    #     args = (split_float_list,)
    #     self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)

    def pnnd_volume_moment_(self, short_long_list):
        for n_short, n_long in short_long_list:
            target_df = self.pnnd_moment(self.sector_volume, self.sector_df, n_short, n_long)
            file_name = f'volume_moment_p{n_short}{n_long}d'
            fun = 'funda_data_deal.BaseDeal.pnnd_moment'
            raw_data_path = (self.part_load_path1 + '/TVOL.csv',)
            args = (n_short, n_long)
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
        fun = 'funda_data_deal.BaseDeal.pnd_continue_ud'
        raw_data_path = (f'{self.table_num}/{self.table_name}/{self.data_name}.csv',)
        args = (n_list,)
        self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args,
                            if_replace=self.if_replace)

    def pnd_continue_ud_pct_(self, n_list):
        target_df = self.pnd_continue_ud_pct(self.raw_df, self.sector_df, n_list)
        file_name = self.data_name + '_p{}d_continue_ud_pct'.format(''.join([str(x) for x in n_list]))
        fun = 'funda_data_deal.BaseDeal.pnd_continue_ud_pct'
        raw_data_path = (f'{self.table_num}/{self.table_name}/{self.data_name}.csv',)
        args = (n_list,)
        self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args,
                            if_replace=self.if_replace)

    def row_extre_(self, percent):
        target_df = self.row_extre(self.raw_df, self.sector_df, percent)
        file_name = self.data_name + '_row_extre_{}'.format(percent)
        fun = 'funda_data_deal.BaseDeal.row_extre'
        raw_data_path = (f'{self.table_num}/{self.table_name}/{self.data_name}.csv',)
        args = (percent,)
        self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args,
                            if_replace=self.if_replace)

    def pnd_col_extre_(self, n_list, percent, min_periods=1):
        for n in n_list:
            target_df = self.pnd_col_extre(self.raw_df, self.sector_df, n, percent, min_periods=min_periods)

            file_name = self.data_name + '_p{}d_col_extre_{}'.format(n, percent)
            fun = 'funda_data_deal.BaseDeal.pnd_col_extre'
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
    def load_sector_data(self, begin_date, end_date, sector_name, sector_path=None):
        market_top_n = bt.AZ_Load_csv(self.root_path.EM_Funda.DERIVED_10 / (sector_name + '.csv'))
        market_top_n = market_top_n[(market_top_n.index >= begin_date) & (market_top_n.index < end_date)]
        market_top_n.dropna(how='all', axis='columns', inplace=True)
        xnms = market_top_n.columns
        xinx = market_top_n.index

        new_stock_df = self.get_new_stock_info(xnms, xinx)
        st_stock_df = self.get_st_stock_info(xnms, xinx)
        sector_df = market_top_n * new_stock_df * st_stock_df
        sector_df.replace(0, np.nan, inplace=True)
        return sector_df


if __name__ == '__main__':
    mode = 'bkt'
    begin_date = pd.to_datetime('20100101')
    end_date = pd.to_datetime('20180801')
    sector_name = 'market_top_500'
    table_num, table_name, data_name = ('EM_Funda', 'TRAD_MT_MARGIN', 'RQCHL')
    save_root_path = '/mnt/mfs/dat_whs/data/new_factor_data/market_top_500'
    bt.AZ_Path_create(save_root_path)
    root_path = pt._BinFiles(mode)

    sector_data_class = SectorData(root_path)
    sector_df = sector_data_class.load_sector_data(begin_date, end_date, sector_name)

    funda_base_deal = FundaBaseDeal(sector_df, root_path, table_num, table_name, data_name, save_root_path)
    funda_base_deal.row_extre_(0.2)


# def pnd_continue_ud(raw_df, n_list):
#     all_target_df = pd.DataFrame()
#     for n in n_list:
#         target_df = raw_df.rolling(window=n + 1).apply(
#             lambda x: 1 if (np.diff(x) >= 0).all() and sum(np.diff(x)) > 0
#             else (-1 if (np.diff(x) <= 0).all() and sum(np.diff(x)) < 0 else 0))
#
#         target_df = target_df
#         all_target_df = all_target_df.add(target_df, fill_value=0)
#     return all_target_df
