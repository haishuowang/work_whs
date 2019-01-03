import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *
import work_whs.funda_data.funda_data_deal as fdd

BaseDeal = fdd.BaseDeal
FundaBaseDeal = fdd.FundaBaseDeal
SectorData = fdd.SectorData
TechBaseDeal = fdd.TechBaseDeal


class IntradayDeal(BaseDeal):
    def __init__(self, sector_df, intraday_path, para_name, save_root_path, if_replace=False):
        data = bt.AZ_Load_csv(f'{intraday_path}/{para_name}.csv').reindex(index=sector_df.index,
                                                                          columns=sector_df.columns)
        if if_replace:
            self.raw_df = data.replace(0, np.nan)
        else:
            self.raw_df = data

        self.sector_df = sector_df
        self.save_root_path = save_root_path
        self.factor_to_fun = '/mnt/mfs/dat_whs/data/factor_to_fun'
        self.if_replace = if_replace
        self.para_name = para_name
        self.intraday_path = intraday_path

    def row_extre_(self, percent):
        target_df = self.row_extre(self.raw_df, self.sector_df, percent)
        file_name = self.para_name + '_row_extre_{}'.format(percent)
        fun = 'Intraday.IntradayDeal.row_extre'
        raw_data_path = (f'EM_Funda/my_data/{self.para_name}',)
        args = (percent,)
        self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args,
                            if_replace=self.if_replace)

    def col_score_row_extre(self, raw_df, sector_df, n, percent):
        col_score_df = bt.AZ_Col_zscore(raw_df, n)
        target_df = self.row_extre(col_score_df, sector_df, percent)
        return target_df

    def col_score_row_extre_(self, n, percent):
        target_df = self.col_score_row_extre(self.raw_df, self.sector_df, n, percent)
        file_name = self.para_name + '_col_score_row_extre_{}'.format(percent)
        fun = 'Intraday.IntradayDeal.col_score_row_extre'
        raw_data_path = (f'EM_Funda/my_data/{self.para_name}',)
        args = (n, percent,)
        self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args,
                            if_replace=self.if_replace)


def intra_fun(sector_df, save_root_path):
    intraday_path = '/mnt/mfs/dat_whs/EM_Funda/my_data_test'
    intra_file_list = ['intra_dn_15_bar_div_daily',
                       'intra_dn_15_bar_vol',
                       'intra_dn_15_bar_vwap',
                       'intra_dn_div_daily',
                       'intra_dn_vol',
                       'intra_dn_vwap',
                       'intra_up_15_bar_div_daily',
                       'intra_up_15_bar_div_dn_15_bar',
                       'intra_up_15_bar_vol',
                       'intra_up_15_bar_vwap',
                       'intra_up_div_daily',
                       'intra_up_div_dn',
                       'intra_up_vol',
                       'intra_up_vwap']

    percent = 0.3
    # for para_name in intra_file_list:
    #     intra_deal = IntradayDeal(sector_df, intraday_path, para_name, save_root_path)
    #     intra_deal.row_extre_(percent)
    for para_name in intra_file_list:
        intra_deal = IntradayDeal(sector_df, intraday_path, para_name, save_root_path)
        for n in [20, 40, 100]:
            intra_deal.col_score_row_extre_(n, percent)
