import pandas as pd
import numpy as np
import time
from multiprocessing import Pool
import os
from datetime import datetime, timedelta
import sys

sys.path.append('/mnt/mfs/work_whs/AZ_2018_Q2')
sys.path.append('/mnt/mfs')
sys.path.append("/mnt/mfs/LIB_ROOT")

import work_whs.funda_data as fd
from work_whs.funda_data import EM_Tab14, EM_Funda, Tech_Factor, EM_Funda_test, IntradayData
import open_lib.shared_paths.path as pt
import open_lib.shared_tools.back_test as bt
from work_whs.loc_lib.pre_load import send_email

FundaBaseDeal = fd.funda_data_deal.FundaBaseDeal
SectorData = fd.funda_data_deal.SectorData


def main_fun(sector_name):
    try:
        mode = 'bkt'

        begin_date = pd.to_datetime('20100101')
        end_date = datetime.now()
        save_root_path = '/mnt/mfs/dat_whs/data/new_factor_data/{}'.format(sector_name)
        # save_root_path = '/media/hdd2/dat_whs/data/new_factor_data/{}'.format(sector_name)
        # save_root_path = '/mnt/mfs/dat_whs/data/new_factor_data_v2'
        bt.AZ_Path_create(save_root_path)
        root_path = pt._BinFiles(mode)

        # 1 load sector
        sector_data_class = SectorData(root_path)
        sector_df = sector_data_class.load_sector_data(begin_date, end_date, sector_name)

        print('***************************table 14********************************')
        data_name_list = ['RZRQYE', 'RZMRE', 'RZYE', 'RQMCL', 'RQYE', 'RQYL', 'RQCHL', 'RZCHE']
        table_num, table_name = ('EM_Funda', 'TRAD_MT_MARGIN')
        # EM_Tab14.common_fun(sector_df, root_path, table_num, table_name, data_name_list, save_root_path, if_replace=True)

        table_num, table_name = ('EM_Funda', 'TRAD_SK_DAILY_JC')
        data_name_list = ['TVOL']
        # EM_Tab14.common_fun(sector_df, root_path, table_num, table_name, data_name_list, save_root_path)
        # EM_Tab14.common_fun(sector_df, root_path, table_num, table_name, data_name_list, save_root_path,
        #                     n_list=[3, 4, 5], window_list=[30, 90])

        table_num, table_name = ('EM_Funda', 'DERIVED_14')
        data_name_list = ['aadj_r']
        # EM_Tab14.DERIVED_14_fun(sector_df, root_path, table_num, table_name, data_name_list, save_root_path)
        #
        # EM_Tab14.base_data_fun(sector_df, root_path, save_root_path)

        # add factor
        # data_name_list = ['PE_TTM', 'PS_TTM', 'PBLast']
        # table_num, table_name = ('EM_Funda', 'TRAD_SK_REVALUATION')
        # EM_Tab14.common_fun(sector_df, root_path, table_num, table_name, data_name_list, save_root_path)

        EM_Tab14.TRAD_SK_DAILY_JC_fun(sector_df, root_path, save_root_path)

        print('***************************table funda********************************')
        data_name_list = ['R_AssetDepSales_s_First',
                          'R_CFOPS_s_First',
                          'R_CFO_TotRev_s_First',
                          'R_CFO_s_YOY_First',
                          'R_Cashflow_s_YOY_First',
                          'R_CostSales_s_First',
                          'R_CurrentAssetsTurnover_QTTM',
                          'R_DebtAssets_QTTM',
                          'R_EBITDA_IntDebt_QTTM',
                          'R_EBIT_sales_QTTM',
                          'R_EPS_s_First',
                          'R_EPS_s_YOY_First',
                          'R_FCFTot_Y3YGR',
                          'R_FairValChgPnL_s_First',
                          'R_FairValChg_TotProfit_s_First',
                          'R_FinExp_sales_s_First',
                          'R_GSCF_sales_s_First',
                          'R_LTDebt_WorkCap_QTTM',
                          'R_MgtExp_sales_s_First',
                          'R_NetAssets_s_POP_First',
                          'R_NetAssets_s_YOY_First',
                          'R_NetCashflowPS_s_First',
                          'R_NetIncRecur_s_First',
                          'R_NetInc_TotProfit_s_First',
                          'R_NetInc_s_First',
                          'R_NetMargin_s_YOY_First',
                          'R_NetProfit_sales_s_First',
                          'R_NetROA_TTM_First',
                          'R_NetROA_s_First',
                          'R_NonOperProft_TotProfit_s_First',
                          'R_OPCF_NetInc_s_First',
                          'R_OPCF_TotDebt_QTTM',
                          'R_OPCF_sales_s_First',
                          'R_OPEX_sales_TTM_First',
                          'R_OPEX_sales_s_First',
                          'R_OperCost_sales_s_First',
                          'R_OperProfit_s_POP_First',
                          'R_OperProfit_s_YOY_First',
                          'R_OperProfit_sales_s_First',
                          'R_ParentProfit_s_POP_First',
                          'R_ParentProfit_s_YOY_First',
                          'R_ROENetIncRecur_s_First',
                          'R_ROE_s_First',
                          'R_RecurNetProft_NetProfit_s_First',
                          'R_RevenuePS_s_First',
                          'R_RevenueTotPS_s_First',
                          'R_Revenue_s_POP_First',
                          'R_Revenue_s_YOY_First',
                          'R_SUMLIAB_Y3YGR',
                          'R_SalesCost_s_First',
                          'R_SalesGrossMGN_QTTM',
                          'R_SalesGrossMGN_s_First',
                          'R_SalesNetMGN_s_First',
                          'R_TangAssets_TotLiab_QTTM',
                          'R_Tax_TotProfit_QTTM',
                          'R_Tax_TotProfit_s_First',
                          'R_Tax_TotProfit_s_First',
                          'R_TotAssets_s_YOY_First',
                          'R_TotLiab_s_YOY_First',
                          'R_TotRev_TTM_Y3YGR',
                          'R_TotRev_s_POP_First',
                          'R_TotRev_s_YOY_First']

        # EM_Funda.common_fun(sector_df, root_path, data_name_list, save_root_path, if_replace=False, percent=0.2)
        # EM_Funda.daily_fun(sector_df, root_path, save_root_path)
        # EM_Funda.speciel_fun(sector_df, root_path, save_root_path)
        # EM_Funda.remy_fun(sector_df, root_path, save_root_path)
        print('***************************tech factor********************************')
        # Tech_Factor.main(sector_df, root_path, save_root_path)
        # Tech_Factor.main_alpha(sector_df, root_path, save_root_path)
        print('***************************self factor********************************')
        # EM_Funda_test.common_deal(sector_df, save_root_path)
        print('***************************intra factor********************************')
        # IntradayData.intra_fun(sector_df, save_root_path)
    except Exception as error:
        send_email.send_email(sector_name+error, ['whs@yingpei.com'], [], '[BAK]data error')


if __name__ == '__main__':

    sector_name_list = [
        # 'index_000300',
        # 'index_000905',
        # 'market_top_300plus',
        # 'market_top_300plus_industry_10_15',
        # 'market_top_300plus_industry_20_25_30_35',
        # 'market_top_300plus_industry_40',
        # 'market_top_300plus_industry_45_50',
        # 'market_top_300plus_industry_55',
        #
        # 'market_top_300to800plus',
        # 'market_top_300to800plus_industry_10_15',
        'market_top_300to800plus_industry_20_25_30_35',
        # 'market_top_300to800plus_industry_40',
        # 'market_top_300to800plus_industry_45_50',
        # 'market_top_300to800plus_industry_55',

        # 'market_top_800plus',
        # 'market_top_800plus_industry_10_15',
        # 'market_top_800plus_industry_20_25_30_35',
        # 'market_top_800plus_industry_40',
        # 'market_top_800plus_industry_45_50',
        # 'market_top_800plus_industry_55',
    ]

    t1 = time.time()
    # pool = Pool(20)
    for sector_name in sector_name_list:
        print('_________________________________________________________________________________________')
        print(sector_name)
        a = time.time()
        main_fun(sector_name)
        # pool.apply_async(main_fun, args=(sector_name,))
        b = time.time()
        print(f'{sector_name} Data Updated!, cost time {b-a}\'s!')
    # pool.close()
    # pool.join()
    t2 = time.time()
    print(f'totle cost time: {t2-t1}!')
