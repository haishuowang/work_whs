import funda_data.funda_data_deal as fdd
import loc_lib.shared_paths.path as pt
import loc_lib.shared_tools.back_test as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

BaseDeal = fdd.BaseDeal
FundaBaseDeal = fdd.FundaBaseDeal
SectorData = fdd.SectorData
TechBaseDeal = fdd.TechBaseDeal

MCAP = '/mnt/mfs/DAT_EQT/EM_Funda/LICO_YS_STOCKVALUE/AmarketCapExStri.csv'


class EM_Funda_Deal(BaseDeal):
    def dev_row_extre(self, data1, data2, sector_df, percent):
        target_df = self.row_extre(data1 / data2, sector_df, percent)
        return target_df

    def mix_factor_mcap(self, df, denom, grow_1, grow_2, std, sector_df, persent):
        factor_1 = (df / denom) * sector_df
        factor_2 = (((grow_1 + grow_2) / 2)/std) * sector_df
        a = bt.AZ_Row_zscore(factor_1)
        b = bt.AZ_Row_zscore(factor_2)
        c = a + b
        target_df = self.row_extre(c, sector_df, persent)
        return target_df

    def mix_factor_asset(self, df, asset, grow_1, grow_2, std, sector_df, persent):
        factor_1 = (df / asset) * sector_df
        factor_2 = (((grow_1 + grow_2) / 2)/std) * sector_df
        a = bt.AZ_Row_zscore(factor_1)
        b = bt.AZ_Row_zscore(factor_2)
        c = a + b
        target_df = self.row_extre(c, sector_df, persent)
        return target_df

    def mix_factor_mcap_intdebt(self, df, mcap, intdebt, grow_1, grow_2, std, sector_df, persent):
        factor_1 = (df / (mcap + intdebt)) * sector_df
        factor_2 = (((grow_1 + grow_2) / 2)/std) * sector_df
        a = bt.AZ_Row_zscore(factor_1, 5)
        b = bt.AZ_Row_zscore(factor_2, 5)
        c = a + b
        target_df = self.row_extre(c, sector_df, persent)
        return target_df


class Daily_Deal(EM_Funda_Deal):
    def __init__(self, sector_df, root_path, save_root_path):
        xnms = sector_df.columns
        xinx = sector_df.index
        self.sector_df = sector_df
        self.load_path = root_path.EM_Funda.daily
        self.part_load_path = 'EM_Funda/daily'

        self.R_IntDebt_Y3YGR_path = self.load_path / 'R_IntDebt_Y3YGR.csv'
        self.R_IntDebt_Y3YGR = bt.AZ_Load_csv(self.R_IntDebt_Y3YGR_path).reindex(columns=xnms, index=xinx)

        self.R_SUMASSET_First_path = self.load_path / 'R_SUMASSET_First.csv'
        self.R_SUMASSET_First = bt.AZ_Load_csv(self.R_SUMASSET_First_path).reindex(columns=xnms, index=xinx)

        # self.R_EBITDA_QTTM_path = self.load_path / 'R_EBITDA_QTTM.csv'
        # self.R_EBITDA_QTTM = bt.AZ_Load_csv(self.R_EBITDA_QTTM_path).reindex(columns=xnms, index=xinx)

        self.MCAP_path = root_path.EM_Funda.LICO_YS_STOCKVALUE / 'AmarketCapExStri.csv'
        self.MCAP = bt.AZ_Load_csv(self.MCAP_path).reindex(columns=xnms, index=xinx)

        self.R_EBITDA_QYOY_path = self.load_path / 'R_IntDebt_Y3YGR.csv'
        self.R_EBITDA_QYOY = bt.AZ_Load_csv(self.R_IntDebt_Y3YGR_path).reindex(columns=xnms, index=xinx)

        self.R_EBIT2_Y3YGR_path = self.load_path / 'R_IntDebt_Y3YGR.csv'
        self.R_EBIT2_Y3YGR = bt.AZ_Load_csv(self.R_IntDebt_Y3YGR_path).reindex(columns=xnms, index=xinx)

        self.R_SalesGrossMGN_QYOY_path = self.load_path / 'R_SalesGrossMGN_QYOY.csv'
        self.R_SalesGrossMGN_QYOY = bt.AZ_Load_csv(self.R_SalesGrossMGN_QYOY_path).reindex(columns=xnms, index=xinx)

        self.R_SalesGrossMGN_QTTM_path = self.load_path / 'R_SalesGrossMGN_QTTM.csv'
        self.R_SalesGrossMGN_QTTM = bt.AZ_Load_csv(self.R_SalesGrossMGN_QTTM_path).reindex(columns=xnms, index=xinx)

        self.R_OPCF_CurrentLiab_First_path = self.load_path / 'R_OPCF_CurrentLiab_First.csv'
        self.R_OPCF_CurrentLiab_First = bt.AZ_Load_csv(self.R_OPCF_CurrentLiab_First_path) \
            .reindex(columns=xnms, index=xinx)

        self.R_CurrentRatio_QTTM_path = self.load_path / 'R_CurrentRatio_QTTM.csv'
        self.R_CurrentRatio_QTTM = bt.AZ_Load_csv(self.R_CurrentRatio_QTTM_path).reindex(columns=xnms, index=xinx)

        self.save_root_path = save_root_path

    def dev_row_extre_(self, percent):
        data_info = [[(self.R_IntDebt_Y3YGR, self.R_IntDebt_Y3YGR_path, 'R_IntDebt_Y3YGR'),
                      (self.R_SUMASSET_First, self.R_SUMASSET_First_path, 'R_SUMASSET_First')],
                     # [(self.R_EBITDA_QTTM, self.R_EBITDA_QTTM_path, 'R_EBITDA_QTTM'),
                     #  (self.R_SUMASSET_First, self.R_SUMASSET_First_path, 'R_SUMASSET_First')],
                     # [(self.R_EBITDA_QTTM, self.R_EBITDA_QTTM_path, 'R_EBITDA_QTTM'),
                     #  (self.MCAP, self.MCAP_path, 'MCAP')],
                     # [(self.R_EBITDA_QYOY, self.R_EBITDA_QYOY_path, 'R_EBITDA_QYOY'),
                     #  (self.MCAP, self.MCAP_path, 'MCAP')],
                     [(self.R_EBIT2_Y3YGR, self.R_EBIT2_Y3YGR_path, 'R_EBIT2_Y3YGR'),
                      (self.MCAP, self.MCAP_path, 'MCAP')],
                     [(self.R_SalesGrossMGN_QYOY, self.R_SalesGrossMGN_QYOY_path, 'R_SalesGrossMGN_QYOY'),
                      (self.MCAP, self.MCAP_path, 'MCAP')],
                     [(self.R_SalesGrossMGN_QTTM, self.R_SalesGrossMGN_QTTM_path, 'R_SalesGrossMGN_QTTM'),
                      (self.MCAP, self.MCAP_path, 'MCAP')],
                     [(self.R_OPCF_CurrentLiab_First, self.R_OPCF_CurrentLiab_First_path, 'R_OPCF_CurrentLiab_First'),
                      (self.MCAP, self.MCAP_path, 'MCAP')],
                     [(self.R_CurrentRatio_QTTM, self.R_CurrentRatio_QTTM, 'R_EBIT2_Y3YGR'),
                      (self.MCAP, self.MCAP_path, 'MCAP')]
                     ]

        for ((data1, data1_path, data1_name), (data2, data2_path, data2_name)) in data_info:
            target_df = self.dev_row_extre(data1, data2, self.sector_df, percent)
            file_name = '{}_and_{}_{}'.format(data1_name, data2_name, percent)
            fun = 'EM_Funda.EM_Funda_Deal.dev_row_extre'
            raw_data_path = (data1_path, data2_path)
            args = (percent,)
            self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)

    def dev_row_extre2_(self, percent):
        change_list = ['R_ACCOUNTPAY_QYOY',
                       'R_ACCOUNTREC_QYOY',
                       'R_ASSETDEVALUELOSS_s_QYOY',
                       'R_Cashflow_s_YOY_First',
                       'R_CFO_s_YOY_First',
                       'R_CostSales_QYOY',
                       'R_EBITDA2_QYOY',
                       'R_EPSDiluted_YOY_First',
                       'R_ESTATEINVEST_QYOY',
                       'R_FairVal_TotProfit_QYOY',
                       'R_FINANCEEXP_s_QYOY',
                       'R_GrossProfit_TTM_QYOY',
                       'R_GSCF_sales_QYOY',
                       'R_IntDebt_Mcap_QYOY',
                       'R_INVESTINCOME_s_QYOY',
                       'R_LTDebt_WorkCap_QYOY',
                       'R_LOANREC_s_QYOY',
                       'R_NetAssets_s_YOY_First',
                       'R_NetInc_s_QYOY',
                       'R_NETPROFIT_s_QYOY',
                       'R_OPCF_TTM_QYOY',
                       'R_OperCost_sales_QYOY',
                       'R_OperProfit_YOY_First',
                       'R_OPEX_sales_QYOY',
                       'R_ROE1_QYOY',
                       'R_SUMLIAB_QYOY',
                       'R_TangAssets_IntDebt_QYOY',
                       'R_WorkCapital_QYOY',
                       'R_OTHERLASSET_QYOY',
                       ]
        for file_name in change_list:
            data1_path = self.load_path / f'{file_name}.csv'
            data2_path = self.load_path / ('_'.join(file_name.split('_')[:-1]) + '_QTTM.csv')

            data1 = bt.AZ_Load_csv(data1_path)
            data2 = bt.AZ_Load_csv(data2_path)

            target_df = self.dev_row_extre(data1, data2, self.sector_df, percent)
            file_name = '{}_and_{}_{}'.format(file_name, 'QTTM', percent)
            fun = 'EM_Funda.EM_Funda_Deal.dev_row_extre'
            raw_data_path = (data1_path, data2_path)
            args = (percent,)
            self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)


class DailySpecielDeal(EM_Funda_Deal):
    def __init__(self, sector_df, root_path, save_root_path):
        self.xnms = sector_df.columns
        self.xinx = sector_df.index
        self.save_root_path = save_root_path
        self.root_path = '/mnt/mfs/DAT_EQT'
        self.sector_df = sector_df

    def mix_factor_mcap_(self, data_name, grow_1_name, grow_2_name, sector_df, percent):
        df = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/daily/R_{data_name}_TTM_First.csv')\
            .reindex(index=self.xinx, columns=self.xnms)
        mcap = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/LICO_YS_STOCKVALUE/AmarketCap.csv')\
            .reindex(index=self.xinx, columns=self.xnms)

        grow_1 = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/daily/R_{data_name}_TTM_{grow_1_name}.csv')\
            .reindex(index=self.xinx, columns=self.xnms)
        grow_2 = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/daily/R_{data_name}_TTM_{grow_2_name}.csv')\
            .reindex(index=self.xinx, columns=self.xnms)

        std = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/daily/R_{data_name}_TTM_QSD4Y.csv') \
            .reindex(index=self.xinx, columns=self.xnms)

        target_df = self.mix_factor_mcap(df, mcap, grow_1, grow_2, std, sector_df, percent)
        file_name = '{}_and_mcap_{}_{}_{}'.format(data_name, grow_1_name, grow_2_name, percent)
        fun = 'EM_Funda.EM_Funda_Deal.mix_factor_mcap'
        raw_data_path = (f'EM_Funda/daily/R_{data_name}_TTM_First.csv',
                         'EM_Funda/LICO_YS_STOCKVALUE/AmarketCap.csv',
                         f'EM_Funda/daily/R_{data_name}_TTM_{grow_1_name}.csv',
                         f'EM_Funda/daily/R_{data_name}_TTM_{grow_2_name}.csv',
                         f'EM_Funda/daily/R_{data_name}_TTM_QSD4Y.csv'
                         )
        args = (percent,)
        self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)

    def mix_factor_asset_(self, data_name, grow_1_name, grow_2_name, sector_df, percent):
        df = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/daily/R_{data_name}_TTM_First.csv')\
            .reindex(index=self.xinx, columns=self.xnms)
        asset = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/daily/R_SUMASSET_First.csv')\
            .reindex(index=self.xinx, columns=self.xnms)

        grow_1 = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/daily/R_{data_name}_TTM_{grow_1_name}.csv')\
            .reindex(index=self.xinx, columns=self.xnms)
        grow_2 = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/daily/R_{data_name}_TTM_{grow_2_name}.csv')\
            .reindex(index=self.xinx, columns=self.xnms)

        std = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/daily/R_{data_name}_TTM_QSD4Y.csv') \
            .reindex(index=self.xinx, columns=self.xnms)

        target_df = self.mix_factor_asset(df, asset, grow_1, grow_2, std, sector_df, percent)
        file_name = '{}_and_asset_{}_{}_{}'.format(data_name, grow_1_name, grow_2_name, percent)
        fun = 'EM_Funda.EM_Funda_Deal.mix_factor_asset'
        raw_data_path = (f'EM_Funda/daily/R_{data_name}_TTM_First.csv',
                         'EM_Funda/daily/R_SUMASSET_First.csv',
                         f'EM_Funda/daily/R_{data_name}_TTM_{grow_1_name}.csv',
                         f'EM_Funda/daily/R_{data_name}_TTM_{grow_2_name}.csv',
                         f'EM_Funda/daily/R_{data_name}_TTM_QSD4Y.csv')
        args = (percent,)
        self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)

    def mix_factor_mcap_intdebt_(self, data_name, grow_1_name, grow_2_name, sector_df, percent):
        df = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/daily/R_{data_name}_TTM_First.csv')\
            .reindex(index=self.xinx, columns=self.xnms)
        mcap = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/LICO_YS_STOCKVALUE/AmarketCap.csv')\
            .reindex(index=self.xinx, columns=self.xnms)
        intdebt = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/daily/R_IntDebt_First.csv')\
            .reindex(index=self.xinx, columns=self.xnms)
        grow_1 = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/daily/R_{data_name}_TTM_{grow_1_name}.csv')\
            .reindex(index=self.xinx, columns=self.xnms)
        grow_2 = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/daily/R_{data_name}_TTM_{grow_2_name}.csv')\
            .reindex(index=self.xinx, columns=self.xnms)

        std = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/daily/R_{data_name}_TTM_QSD4Y.csv') \
            .reindex(index=self.xinx, columns=self.xnms)

        target_df = self.mix_factor_mcap_intdebt(df, mcap, intdebt, grow_1, grow_2, std, sector_df, percent)
        file_name = '{}_and_mcap_intdebt_{}_{}_{}'.format(data_name, grow_1_name, grow_2_name, percent)
        fun = 'EM_Funda.EM_Funda_Deal.mix_factor_mcap_intdebt'
        raw_data_path = (f'EM_Funda/daily/R_{data_name}_TTM_First.csv',
                         'EM_Funda/LICO_YS_STOCKVALUE/AmarketCap.csv',
                         'EM_Funda/daily/R_IntDebt_First.csv',
                         f'EM_Funda/daily/R_{data_name}_TTM_{grow_1_name}.csv',
                         f'EM_Funda/daily/R_{data_name}_TTM_{grow_2_name}.csv',
                         f'EM_Funda/daily/R_{data_name}_TTM_QSD4Y.csv')
        args = (percent,)
        self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)
        return target_df

    def speciel_deal(self, percent):
        data_name_list = ['OPCF', 'EBIT', 'NetProfit', 'TotRev']
        # data_name_list = ['EBIT']
        grow_para_list = [('Y3YGR', 'Y5YGR'), ('QYOY', 'Y3YGR')]
        for data_name in data_name_list:
            for grow_1_name, grow_2_name in grow_para_list:
                self.mix_factor_mcap_(data_name, grow_1_name, grow_2_name, self.sector_df, percent)
                self.mix_factor_asset_(data_name, grow_1_name, grow_2_name, self.sector_df, percent)
                self.mix_factor_mcap_intdebt_(data_name, grow_1_name, grow_2_name, self.sector_df, percent)


data_name_list = ['R_BusinessCycle_First',
                  'R_CurrentAssets_TotAssets_First',
                  'R_CurrentAssetsTurnover_First',
                  'R_DebtAssets_First',
                  'R_DebtEqt_First',
                  'R_OPCF_IntDebt_QTTM',
                  'R_IntDebt_Mcap_First',
                  'R_OPEX_sales_TTM_Y3YGR',
                  'R_EBITDA_sales_TTM_First',
                  'R_EBITDA_sales_TTM_QTTM',
                  'R_ROA_TTM_Y3YGR',
                  'R_OPCF_TotDebt_QYOY',
                  'R_OPCF_TotDebt_First',
                  'R_OPCF_NetInc_s_First',
                  'R_OPCF_sales_First']


def common_fun(sector_df, root_path, data_name_list, save_root_path, if_replace=False, percent=0.3):
    table_num, table_name = ('EM_Funda', 'daily')
    for data_name in data_name_list:
        funda_base_deal = FundaBaseDeal(sector_df, root_path, table_num, table_name, data_name, save_root_path,
                                        if_replace=if_replace)
        funda_base_deal.row_extre_(percent)


def daily_fun(sector_df, root_path, save_root_path):
    percent = 0.3
    daily_deal = Daily_Deal(sector_df, root_path, save_root_path)
    # daily_deal.dev_row_extre_(percent)
    daily_deal.dev_row_extre2_(percent)


def speciel_fun(sector_df, root_path, save_root_path):
    percent = 0.3
    daily_speciel_deal = DailySpecielDeal(sector_df, root_path, save_root_path)
    daily_speciel_deal.speciel_deal(percent)
