import funda_data.funda_data_deal as fdd
import loc_lib.shared_paths.path as pt
import loc_lib.shared_tools.back_test as bt
import pandas as pd
import numpy as np
import sklearn
from datetime import datetime, timedelta
import time

BaseDeal = fdd.BaseDeal
FundaBaseDeal = fdd.FundaBaseDeal
SectorData = fdd.SectorData
TechBaseDeal = fdd.TechBaseDeal


class EM_Tab14_Deal(BaseDeal):
    def return_pnd(self, aadj_r, sector_df, n, percent):
        return_pnd_df = bt.AZ_Rolling(aadj_r, n).sum()
        target_df = self.row_extre(return_pnd_df, sector_df, percent)
        return target_df

    def wgt_return_pnd(self, aadj_r, turnratio, sector_df, n, percent):
        aadj_r_c = (aadj_r * turnratio)
        wgt_return_pnd_df = bt.AZ_Rolling(aadj_r_c, n).sum()
        target_df = self.row_extre(wgt_return_pnd_df, sector_df, percent)
        return target_df

    def log_price(self, close, sector_df, percent):
        target_df = self.row_extre(np.log(close), sector_df, percent)
        return target_df

    def turn_pnd(self, turnratio, sector_df, n, percent):
        turnratio_mean = bt.AZ_Rolling(turnratio, n).mean()
        target_df = self.row_extre(turnratio_mean, sector_df, percent)
        return target_df * sector_df

    @staticmethod
    def bias_turn_pnd(turnratio, sector_df, n):
        bias_turnratio = bt.AZ_Rolling(turnratio, n).mean() / bt.AZ_Rolling(turnratio, 480).mean() - 1
        bias_turnratio_up = (bias_turnratio > 0).astype(int)
        bias_turnratio_dn = (bias_turnratio < 0).astype(int)
        target_df = bias_turnratio_up - bias_turnratio_dn
        return target_df * sector_df

    @staticmethod
    def MACD(close, sector_df, n_fast, n_slow):
        EMAfast = close.ewm(span=n_fast, min_periods=n_slow - 1).mean()
        EMAslow = close.ewm(span=n_slow, min_periods=n_slow - 1).mean()
        MACD = EMAfast - EMAslow
        MACDsign = MACD.ewm(span=9, min_periods=8).mean()
        MACDdiff = MACD - MACDsign
        target_df_up = (MACDdiff > 0).astype(int)
        target_df_dn = (MACDdiff < 0).astype(int)
        target_df = target_df_up - target_df_dn
        return target_df * sector_df

    @staticmethod
    def CCI(high, low, close, sector_df, n, limit_list):
        PP = (high + low + close) / 3
        CCI_signal = (PP - bt.AZ_Rolling(PP, n).mean()) / bt.AZ_Rolling(PP, n).std()
        all_target_df = pd.DataFrame()
        for limit in limit_list:
            CCI_up = (CCI_signal >= limit).astype(int)
            CCI_dn = -(CCI_signal <= -limit).astype(int)
            CCI = CCI_up + CCI_dn
            all_target_df = all_target_df.add(CCI, fill_value=0)
        return all_target_df * sector_df


class TRAD_SK_DAILY_JC_Deal(EM_Tab14_Deal):
    def __init__(self, sector_df, root_path, save_root_path):
        xnms = sector_df.columns
        xinx = sector_df.index

        self.load_path = root_path.EM_Funda.DERIVED_14
        self.part_load_path = 'EM_Funda/DERIVED_14'

        self.open_path = self.load_path / 'aadj_p_OPEN.csv'
        self.open = bt.AZ_Load_csv(self.open_path).reindex(columns=xnms, index=xinx)

        self.high_path = self.load_path / 'aadj_p_HIGH.csv'
        self.high = bt.AZ_Load_csv(self.high_path).reindex(columns=xnms, index=xinx)

        self.low_path = self.load_path / 'aadj_p_LOW.csv'
        self.low = bt.AZ_Load_csv(self.low_path).reindex(columns=xnms, index=xinx)

        self.close_path = self.load_path / 'aadj_p.csv'
        self.close = bt.AZ_Load_csv(self.close_path).reindex(columns=xnms, index=xinx)

        self.volume_path = root_path.EM_Funda.TRAD_SK_DAILY_JC / 'TVOL.csv'
        self.volume = bt.AZ_Load_csv(self.volume_path).reindex(columns=xnms, index=xinx)

        self.amount_path = root_path.EM_Funda.TRAD_SK_DAILY_JC / 'TVALCNY.csv'
        self.amount = bt.AZ_Load_csv(self.amount_path).reindex(columns=xnms, index=xinx)

        self.turnrate_path = root_path.EM_Funda.TRAD_SK_DAILY_JC / 'TURNRATE.csv'
        self.turnrate = bt.AZ_Load_csv(self.turnrate_path).reindex(columns=xnms, index=xinx)

        self.aadj_r_path = self.load_path / 'aadj_r.csv'
        self.aadj_r = bt.AZ_Load_csv(self.aadj_r_path).reindex(columns=xnms, index=xinx)

        self.sector_df = sector_df
        self.save_root_path = save_root_path
        self.factor_to_fun = '/mnt/mfs/dat_whs/data/factor_to_fun'

    def return_pnd_(self, n_list, percent):
        for n in n_list:
            target_df = self.return_pnd(self.aadj_r, self.sector_df, n, percent)
            file_name = 'return_p{}d_{}'.format(n, percent)
            fun = 'EM_Tab14.EM_Tab14_Deal.return_pnd'
            raw_data_path = (self.aadj_r_path,)
            args = (n, percent,)
            self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)

    def wgt_return_pnd_(self, n_list, percent):
        for n in n_list:
            target_df = self.wgt_return_pnd(self.aadj_r, self.turnrate, self.sector_df, n, percent)
            file_name = 'wgt_return_p{}d_{}'.format(n, percent)
            fun = 'EM_Tab14.EM_Tab14_Deal.wgt_return_pnd'
            raw_data_path = (self.aadj_r_path,
                             self.turnrate_path)
            args = (n, percent,)
            self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)

    def log_price_(self, percent):
        target_df = self.log_price(self.close, self.sector_df, percent)
        file_name = 'log_price_{}'.format(percent)
        fun = 'EM_Tab14.EM_Tab14_Deal.log_price'
        raw_data_path = (self.close_path,)
        args = (percent,)
        self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)

    def turn_pnd_(self, n_list, percent):
        for n in n_list:
            target_df = self.turn_pnd(self.turnrate, self.sector_df, n, percent)
            file_name = 'turn_p{}d_{}'.format(n, percent)
            fun = 'EM_Tab14.EM_Tab14_Deal.turn_pnd'
            raw_data_path = (self.turnrate_path,)
            args = (n, percent,)
            self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)

    def bias_turn_pnd_(self, n_list):
        for n in n_list:
            target_df = self.bias_turn_pnd(self.turnrate, self.sector_df, n)
            file_name = 'bias_turn_p{}d'.format(n)
            fun = 'EM_Tab14.EM_Tab14_Deal.bias_turn_pnd'
            raw_data_path = (self.turnrate_path,)
            args = (n,)
            self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)

    # def HAlpha(aadj_r, market_index, sector_df, window=120):
    def MACD_(self, long_short_list):
        for n_fast, n_slow in long_short_list:
            target_df = self.MACD(self.close, self.sector_df, n_fast, n_slow)
            file_name = 'MACD_{}_{}'.format(n_fast, n_slow)
            fun = 'EM_Tab14.EM_Tab14_Deal.MACD'
            raw_data_path = (self.close_path,)
            args = (n_fast, n_slow)
            self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)

    def CCI_(self, n_list, limit_list):
        for n in n_list:
            target_df = self.CCI(self.high, self.low, self.close, self.sector_df, n, limit_list)
            file_name = 'CCI_p{}d_limit_{}'.format(n, ''.join([str(x) for x in limit_list]))
            fun = 'EM_Tab14.EM_Tab14_Deal.CCI'
            raw_data_path = (self.high_path,
                             self.low_path,
                             self.close_path,)
            args = (n, limit_list)
            self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)


def common_fun(sector_df, root_path, table_num, table_name, data_name_list, save_root_path, if_replace=False):
    percent = 0.2
    n_list = [3, 4, 5]
    window_list = [10, 20, 60, 120]
    for data_name in data_name_list:
        funda_base_deal = FundaBaseDeal(sector_df, root_path, table_num, table_name, data_name, save_root_path,
                                        if_replace=if_replace)
        funda_base_deal.row_extre_(percent)
        funda_base_deal.pnd_continue_ud_(n_list)
        funda_base_deal.pnd_col_extre_(window_list, percent, min_periods=1)


def DERIVED_14_fun(sector_df, root_path, table_num, table_name, data_name_list, save_root_path, if_replace=False):
    percent = 0.2
    n_list = [3, 4, 5]
    window_list = [10, 20, 60, 120]
    for data_name in data_name_list:
        funda_base_deal = FundaBaseDeal(sector_df, root_path, table_num, table_name, data_name, save_root_path,
                                        if_replace=if_replace)
        funda_base_deal.row_extre_(percent)
        funda_base_deal.pnd_continue_ud_pct_(n_list)
        funda_base_deal.pnd_continue_ud_(n_list)
        funda_base_deal.pnd_col_extre_(window_list, percent, min_periods=1)


def base_data_fun(sector_df, root_path, save_root_path):
    window_list = [10, 20, 60, 120]
    short_long_list = [(5, 10), (10, 60), (10, 100), (20, 100), (20, 200), (40, 200)]
    split_float_list = [0.03, 0.02, 0.01]

    tech_base_deal = TechBaseDeal(sector_df, root_path, save_root_path)
    tech_base_deal.pnd_hl_(window_list)
    tech_base_deal.pnd_volume_(window_list)
    tech_base_deal.pnd_volitality_and_more_(window_list)
    tech_base_deal.pnnd_moment_(short_long_list)
    tech_base_deal.p1d_jump_hl_(split_float_list)
    tech_base_deal.pnnd_volume_moment_([(5, 30)])


def TRAD_SK_DAILY_JC_fun(sector_df, root_path, save_root_path):
    percent = 0.2
    n_list = [20, 60, 120]
    short_long_list = [(10, 30), (20, 100), (20, 200), (40, 200)]
    limit_list = [1, 2]
    TRAD_SK_DAILY_JC_deal = TRAD_SK_DAILY_JC_Deal(sector_df, root_path, save_root_path)
    TRAD_SK_DAILY_JC_deal.return_pnd_(n_list, percent)
    TRAD_SK_DAILY_JC_deal.wgt_return_pnd_(n_list, percent)
    TRAD_SK_DAILY_JC_deal.log_price_(percent)
    TRAD_SK_DAILY_JC_deal.turn_pnd_(n_list, percent)
    TRAD_SK_DAILY_JC_deal.bias_turn_pnd_(n_list)
    TRAD_SK_DAILY_JC_deal.MACD_(short_long_list)
    TRAD_SK_DAILY_JC_deal.CCI_(n_list, limit_list)


#
# if __name__ == '__main__':
#     sector_name = 'market_top_500'
#     root_path = pt._BinFiles(mode='bkt')
#     load_path = root_path.EM_Funda.TRAD_SK_DAILY_JC
#     begin_date = pd.to_datetime('20100101')
#     end_date = pd.to_datetime('20180801')
#     sector_data_class = SectorData(root_path)
#     sector_df = sector_data_class.load_sector_data(begin_date, end_date, sector_name)
#     xnms = sector_df.columns
#     xinx = sector_df.index



