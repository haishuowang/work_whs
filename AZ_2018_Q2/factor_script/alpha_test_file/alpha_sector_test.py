import pandas as pd
from collections import OrderedDict
import numpy as np
import os
import sys

sys.path.append('/mnt/mfs/work_whs')
sys.path.append('/mnt/mfs/work_whs/AZ_2018_Q2')
import loc_lib.shared_tools.back_test as bt
from factor_script import main_file as mf


class FactorTest(mf.FactorTest):
    def __init__(self, *args):
        super(FactorTest, self).__init__(*args)

    def load_sector_data(self):
        market_top_n = bt.AZ_Load_csv(f'/mnt/mfs/dat_whs/data/sector_data/{self.sector_name}.csv')
        # print(market_top_n.iloc[-1].dropna())
        market_top_n = market_top_n[(market_top_n.index >= self.begin_date) & (market_top_n.index < self.end_date)]
        market_top_n.dropna(how='all', axis='columns', inplace=True)
        xnms = market_top_n.columns
        xinx = market_top_n.index

        new_stock_df = self.get_new_stock_info(xnms, xinx)
        st_stock_df = self.get_st_stock_info(xnms, xinx)
        sector_df = market_top_n * new_stock_df * st_stock_df
        sector_df.replace(0, np.nan, inplace=True)
        return sector_df


if __name__ == '__main__':
    root_path = '/mnt/mfs/DAT_EQT'
    if_save = True
    if_new_program = True

    begin_date = pd.to_datetime('20100101')
    cut_date = pd.to_datetime('20160401')
    end_date = pd.to_datetime('20180901')

    # sector_name = 'market_top_2000_moment_bot_1000'
    # sector_name = 'market_top_2000_moment_top_1000'
    # sector_name = 'market_top_2000_vol_bot_1000'
    # sector_name = 'market_top_2000_vol_top_1000'
    sector_name_dict = OrderedDict({'moment_bot': 'market_top_2000_moment_bot_1000',
                                    'moment_top': 'market_top_2000_moment_top_1000',
                                    'vol_bot': 'market_top_2000_vol_bot_1000',
                                    'vol_top': 'market_top_2000_vol_top_1000'})

    for key in sector_name_dict.keys():
        sector_name = sector_name_dict[key]
        # sector_name = 'market_top_2000'
        index_name = '000905'
        return_file = 'pct_p1d'
        hold_time = 20
        lag = 2
        return_file = ''

        if_hedge = True
        if_only_long = False
        time_para_dict = OrderedDict()

        time_para_dict['time_para_1'] = [pd.to_datetime('20110101'), pd.to_datetime('20150101'),
                                         pd.to_datetime('20150401'), pd.to_datetime('20150701'),
                                         pd.to_datetime('20151001'), pd.to_datetime('20160101')]
        time_para_dict['time_para_2'] = [pd.to_datetime('20120101'), pd.to_datetime('20160101'),
                                         pd.to_datetime('20160401'), pd.to_datetime('20160701'),
                                         pd.to_datetime('20161001'), pd.to_datetime('20170101')]
        time_para_dict['time_para_3'] = [pd.to_datetime('20130601'), pd.to_datetime('20170601'),
                                         pd.to_datetime('20170901'), pd.to_datetime('20171201'),
                                         pd.to_datetime('20180301'), pd.to_datetime('20180601')]
        main = FactorTest(root_path, if_save, if_new_program, begin_date, cut_date, end_date, time_para_dict,
                          sector_name, index_name, hold_time, lag, return_file, if_hedge, if_only_long)

        tech_name_list = ['CCI_p120d_limit_12',
                          'MACD_20_100',
                          'MACD_40_200',
                          'log_price_0.2',
                          'bias_turn_p20d',
                          'bias_turn_p120d',
                          'vol_p20d',
                          'vol_p60d',
                          'evol_p20d',
                          'moment_p20100d',
                          'turn_p20d_0.2',
                          'turn_p120d_0.2',
                          'vol_count_down_p60d',
                          'TVOL_p20d_col_extre_0.2',
                          'TVOL_p120d_col_extre_0.2',
                          'price_p20d_hl',
                          'price_p120d_hl',
                          'aadj_r_p345d_continue_ud_pct',
                          'volume_moment_p530d',
                          'return_p60d_0.2',
                          ]

        funda_name_list = ['R_RecurNetProft_NetProfit_s_First_row_extre_0.3',
                           'R_OPEX_sales_s_First_row_extre_0.3',
                           'R_EBITDA_QTTM_and_MCAP_0.3',
                           'R_RevenuePS_s_First_row_extre_0.3',
                           'R_EBITDA_QTTM_and_R_SUMASSET_First_0.3',
                           'R_CostSales_s_First_row_extre_0.3',
                           'R_NetProfit_sales_s_First_row_extre_0.3',
                           'R_ROE_s_First_row_extre_0.3',
                           'R_FairValChgPnL_s_First_row_extre_0.3',
                           'R_NetROA_s_First_row_extre_0.3',
                           'R_FinExp_sales_s_First_row_extre_0.3',
                           'R_CFOPS_s_First_row_extre_0.3',
                           'R_GSCF_sales_s_First_row_extre_0.3',
                           'R_EBITDA_QYOY_and_MCAP_0.3',
                           'R_NetIncRecur_s_First_row_extre_0.3',
                           'R_SalesNetMGN_s_First_row_extre_0.3',
                           'R_OperProfit_s_POP_First_row_extre_0.3',
                           'R_SalesCost_s_First_row_extre_0.3',
                           'R_AssetDepSales_s_First_row_extre_0.3',
                           'R_OPCF_sales_s_First_row_extre_0.3']
        pool_num = 20
        main.test_index_3(tech_name_list, funda_name_list, pool_num, suffix_name='alpha2')
