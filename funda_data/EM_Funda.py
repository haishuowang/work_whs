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


class EM_Fanda_Deal(BaseDeal):
    def dev_row_extre(self, data1, data2, sector_df, percent):
        target_df = self.row_extre(data1 / data2, sector_df, percent)
        return target_df


class Daily_Deal(EM_Fanda_Deal):
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

        self.R_EBITDA_QTTM_path = self.load_path / 'R_EBITDA_QTTM.csv'
        self.R_EBITDA_QTTM = bt.AZ_Load_csv(self.R_EBITDA_QTTM_path).reindex(columns=xnms, index=xinx)

        self.MCAP_path = root_path.EM_Funda.LICO_YS_STOCKVALUE / 'AmarketCapExStri.csv'
        self.MCAP = bt.AZ_Load_csv(self.MCAP_path).reindex(columns=xnms, index=xinx)

        self.R_EBITDA_QYOY_path = self.load_path / 'R_IntDebt_Y3YGR.csv'
        self.R_EBITDA_QYOY = bt.AZ_Load_csv(self.R_IntDebt_Y3YGR_path).reindex(columns=xnms, index=xinx)

        self.R_EBIT2_Y3YGR_path = self.load_path / 'R_IntDebt_Y3YGR.csv'
        self.R_EBIT2_Y3YGR = bt.AZ_Load_csv(self.R_IntDebt_Y3YGR_path).reindex(columns=xnms, index=xinx)

        self.save_root_path = save_root_path

        # R_IntDebt_Y3YGR.csv / R_SUMASSET_First.csv
        # R_EBITDA_QTTM.csv / R_SUMASSET_First.csv
        # R_EBITDA_QTTM.csv / MCAP
        # R_EBITDA_QYOY.csv / MCAP
        # R_EBIT2_Y3YGR.csv / MCAP

    def dev_row_extre_(self, percent):
        data_info = [[(self.R_IntDebt_Y3YGR, self.R_IntDebt_Y3YGR_path, 'R_IntDebt_Y3YGR'),
                      (self.R_SUMASSET_First, self.R_SUMASSET_First_path, 'R_SUMASSET_First')],
                     [(self.R_EBITDA_QTTM, self.R_EBITDA_QTTM_path, 'R_EBITDA_QTTM'),
                      (self.R_SUMASSET_First, self.R_SUMASSET_First_path, 'R_SUMASSET_First')],
                     [(self.R_EBITDA_QTTM, self.R_EBITDA_QTTM_path, 'R_EBITDA_QTTM'),
                      (self.MCAP, self.MCAP_path, 'MCAP')],
                     [(self.R_EBITDA_QYOY, self.R_EBITDA_QYOY_path, 'R_EBITDA_QYOY'),
                      (self.MCAP, self.MCAP_path, 'MCAP')],
                     [(self.R_EBIT2_Y3YGR, self.R_EBIT2_Y3YGR_path, 'R_EBIT2_Y3YGR'),
                      (self.MCAP, self.MCAP_path, 'MCAP')]]

        for ((data1, data1_path, data1_name), (data2, data2_path, data2_name)) in data_info:
            target_df = self.dev_row_extre(data1, data2, self.sector_df, percent)
            file_name = '{}_and_{}_{}'.format(data1_name, data2_name, percent)
            fun = 'EM_Fanda.EM_Fanda_Deal.dev_row_extre'
            raw_data_path = (data1_path, data2_path)
            args = (percent,)
            self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)


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


def common_fun(sector_df, root_path, data_name_list, save_root_path, if_replace=False):
    percent = 0.3
    table_num, table_name = ('EM_Funda', 'daily')
    for data_name in data_name_list:
        funda_base_deal = FundaBaseDeal(sector_df, root_path, table_num, table_name, data_name, save_root_path,
                                        if_replace=if_replace)
        funda_base_deal.row_extre_(percent)


def daily_fun(sector_df, root_path, save_root_path):
    percent = 0.3
    daily_deal = Daily_Deal(sector_df, root_path, save_root_path)
    daily_deal.dev_row_extre_(percent)
