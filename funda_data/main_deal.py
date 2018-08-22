import pandas as pd
import numpy as np
import funda_data as fd
from funda_data import EM_Tab14, EM_Funda

import open_lib.shared_paths.path as pt
import loc_lib.shared_tools.back_test as bt
import os
from datetime import datetime, timedelta

FundaBaseDeal = fd.funda_data_deal.FundaBaseDeal
SectorData = fd.funda_data_deal.SectorData


def main_fun():
    mode = 'bkt'

    begin_date = pd.to_datetime('20100101')
    end_date = datetime.now()

    sector_name = 'market_top_2000'

    save_root_path = '/mnt/mfs/dat_whs/data/new_factor_data/{}'.format(sector_name)
    bt.AZ_Path_create(save_root_path)
    root_path = pt._BinFiles(mode)

    sector_data_class = SectorData(root_path)
    sector_df = sector_data_class.load_sector_data(begin_date, end_date, sector_name)

    # print('***************************table14********************************')
    # data_name_list = ['RZRQYE', 'RZMRE', 'RZYE', 'RQMCL', 'RQYE', 'RQYL', 'RQCHL', 'RZCHE']
    # table_num, table_name = ('EM_Funda', 'TRAD_MT_MARGIN')
    # EM_Tab14.common_fun(sector_df, root_path, table_num, table_name, data_name_list, save_root_path, if_replace=True)
    #
    # table_num, table_name = ('EM_Funda', 'TRAD_SK_DAILY_JC')
    # data_name_list = ['TVOL']
    # EM_Tab14.common_fun(sector_df, root_path, table_num, table_name, data_name_list, save_root_path)
    #
    # table_num, table_name = ('EM_Funda', 'DERIVED_14')
    # data_name_list = ['aadj_r']
    # EM_Tab14.DERIVED_14_fun(sector_df, root_path, table_num, table_name, data_name_list, save_root_path)
    #
    # EM_Tab14.base_data_fun(sector_df, root_path, save_root_path)
    #
    # # add factor
    # data_name_list = ['PE_TTM', 'PS_TTM', 'PBLast']
    # table_num, table_name = ('EM_Funda', 'TRAD_SK_REVALUATION')
    # EM_Tab14.common_fun(sector_df, root_path, table_num, table_name, data_name_list, save_root_path)
    #
    # EM_Tab14.TRAD_SK_DAILY_JC_fun(sector_df, root_path, save_root_path)

    print('***************************table funda********************************')
    data_name_list = ['R_RevenuePS_s_First',
                      'R_OPCF_sales_s_First',
                      'R_NetProfit_sales_s_First',
                      'R_Revenue_s_POP_First',
                      'R_NetInc_TotProfit_s_First',
                      'R_SalesNetMGN_s_First',
                      'R_OperProfit_s_POP_First',
                      'R_TotRev_s_POP_First',
                      'R_AssetDepSales_s_First',
                      'R_EPS_s_YOY_First',
                      'R_EPS_s_First',
                      'R_MgtExp_sales_s_First',
                      'R_CostSales_s_First',
                      'R_RevenueTotPS_s_First',
                      'R_NonOperProft_TotProfit_s_First',
                      'R_NetIncRecur_s_First',
                      'R_FinExp_sales_s_First',
                      'R_OperProfit_s_YOY_First',
                      'R_FairValChg_TotProfit_s_First',
                      'R_CFO_TotRev_s_First',
                      'R_COMPANYCODE_First',
                      'R_ROENetIncRecur_s_First',
                      'R_Cashflow_s_YOY_First',
                      'R_ParentProfit_s_POP_First',
                      'R_TotAssets_s_YOY_First',
                      'R_NetAssets_s_YOY_First',
                      'R_NetInc_s_First',
                      'R_Revenue_s_YOY_First',
                      'R_NetROA_s_First',
                      'R_NOTICEDATE_First',
                      'R_CFO_s_YOY_First',
                      'R_OPCF_NetInc_s_First',
                      'R_NetMargin_s_YOY_First',
                      'R_SalesGrossMGN_s_First',
                      'R_NetCashflowPS_s_First',
                      'R_ROE_s_First',
                      'R_NetAssets_s_POP_First',
                      'R_ParentProfit_s_YOY_First',
                      'R_OPEX_sales_s_First',
                      'R_GSCF_sales_s_First',
                      'R_Tax_TotProfit_s_First',
                      'R_TotLiab_s_YOY_First',
                      'R_CFOPS_s_First',
                      'R_OperCost_sales_s_First',
                      'R_SalesCost_s_First',
                      'R_RecurNetProft_NetProfit_s_First',
                      'R_OperProfit_sales_s_First',
                      'R_TotRev_s_YOY_First',
                      'R_FairValChgPnL_s_First']
    EM_Funda.common_fun(sector_df, root_path, data_name_list, save_root_path, if_replace=False)
    # EM_Funda.daily_fun(sector_df, root_path, save_root_path)


if __name__ == '__main__':
    main_fun()
