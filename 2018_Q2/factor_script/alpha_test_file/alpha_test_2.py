import pandas as pd
from collections import OrderedDict
import random
import sys

sys.path.append('/mnt/mfs/work_whs')
sys.path.append('/mnt/mfs/work_whs/2018_Q2')
from factor_script.main_file import main_file_return_hedge as mf
from loc_lib.shared_tools import send_email

if __name__ == '__main__':
    root_path = '/mnt/mfs/DAT_EQT'
    if_save = True
    if_new_program = True
    # old_file_name = 'market_top_2000_True_20180903_1018_hold_20_aadj_r.txt'

    begin_date = pd.to_datetime('20110101')
    cut_date = pd.to_datetime('20160401')
    end_date = pd.to_datetime('20180901')

    sector_name = 'market_top_2000'
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

    time_para_dict['time_para_4'] = [pd.to_datetime('20140601'), pd.to_datetime('20180601'),
                                     pd.to_datetime('20180901'), pd.to_datetime('20180901'),
                                     pd.to_datetime('20180901'), pd.to_datetime('20180901')]

    time_para_dict['time_para_5'] = [pd.to_datetime('20140701'), pd.to_datetime('20180701'),
                                     pd.to_datetime('20180901'), pd.to_datetime('20180901'),
                                     pd.to_datetime('20180901'), pd.to_datetime('20180901')]

    time_para_dict['time_para_6'] = [pd.to_datetime('20140801'), pd.to_datetime('20180801'),
                                     pd.to_datetime('20180901'), pd.to_datetime('20180901'),
                                     pd.to_datetime('20180901'), pd.to_datetime('20180901')]

    main = mf.FactorTest(root_path, if_save, if_new_program, begin_date, cut_date, end_date, time_para_dict,
                         sector_name, hold_time, lag, return_file, if_hedge, if_only_long)

    # tech_name_list_list = ['CCI_p120d_limit_12',
    #                        'MACD_20_100',
    #                        'MACD_40_200',
    #                        'log_price_0.2',
    #                        'bias_turn_p20d',
    #                        'bias_turn_p120d',
    #                        'vol_p20d',
    #                        'vol_p60d',
    #                        'evol_p20d',
    #                        'evol_p60d',
    #                        'moment_p20100d',
    #                        'moment_p40200d',
    #                        'moment_p1060d',
    #                        'turn_p20d_0.2',
    #                        'turn_p120d_0.2',
    #                        'vol_count_down_p60d',
    #                        'TVOL_p20d_col_extre_0.2',
    #                        'TVOL_p60d_col_extre_0.2',
    #                        'TVOL_p120d_col_extre_0.2',
    #                        'TVOL_row_extre_0.2',
    #                        'price_p20d_hl',
    #                        'price_p120d_hl',
    #                        'aadj_r_p20d_col_extre_0.2',
    #                        'aadj_r_p345d_continue_ud_pct',
    #                        'aadj_r_p345d_continue_ud',
    #                        'volume_moment_p530d',
    #                        # 'return_p20d_0.2',
    #                        # 'return_p60d_0.2',
    #                        'wgt_return_p120d_0.2',
    #                        'wgt_return_p20d_0.2']
    #
    # funda_name_list_list = [['R_RecurNetProft_NetProfit_s_First_row_extre_0.3',
    #                          'R_OPEX_sales_s_First_row_extre_0.3',
    #                          'R_EBITDA_QTTM_and_MCAP_0.3',
    #                          'R_RevenuePS_s_First_row_extre_0.3',
    #                          'R_EBITDA_QTTM_and_R_SUMASSET_First_0.3',
    #                          'R_CostSales_s_First_row_extre_0.3',
    #                          'R_NetProfit_sales_s_First_row_extre_0.3',
    #                          'R_ROE_s_First_row_extre_0.3',
    #                          'R_FairValChgPnL_s_First_row_extre_0.3',
    #                          'R_NetROA_s_First_row_extre_0.3',
    #                          'R_FinExp_sales_s_First_row_extre_0.3',
    #                          'R_CFOPS_s_First_row_extre_0.3',
    #                          'R_GSCF_sales_s_First_row_extre_0.3',
    #                          'R_EBITDA_QYOY_and_MCAP_0.3',
    #                          'R_NetIncRecur_s_First_row_extre_0.3',
    #                          'R_SalesNetMGN_s_First_row_extre_0.3',
    #                          'R_OperProfit_s_POP_First_row_extre_0.3',
    #                          'R_SalesCost_s_First_row_extre_0.3',
    #                          'R_AssetDepSales_s_First_row_extre_0.3',
    #                          'R_OPCF_sales_s_First_row_extre_0.3'],
    #                         ['R_SalesGrossMGN_s_First_row_extre_0.3',
    #                          'R_MgtExp_sales_s_First_row_extre_0.3',
    #                          'R_EBITDA_QTTM_and_MCAP_0.3.pkl',
    #                          'R_NetAssets_s_POP_First_row_extre_0.3',
    #                          'R_EBITDA_QYOY_and_MCAP_0.3.pkl',
    #                          'R_NetCashflowPS_s_First_row_extre_0.3',
    #                          'R_TotLiab_s_YOY_First_row_extre_0.3',
    #                          'R_FairValChg_TotProfit_s_First_row_extre_0.3',
    #                          'R_NonOperProft_TotProfit_s_First_row_extre_0.3',
    #                          'R_EBITDA_QTTM_and_R_SUMASSET_First_0.3',
    #                          'R_Revenue_s_POP_First_row_extre_0.3',
    #                          'R_NetMargin_s_YOY_First_row_extre_0.3',
    #                          'R_OperCost_sales_s_First_row_extre_0.3',
    #                          'R_IntDebt_Y3YGR_and_R_SUMASSET_First_0.3',
    #                          'R_EBIT2_Y3YGR_and_MCAP_0.3.pkl',
    #                          'R_OperProfit_sales_s_First_row_extre_0.3',
    #                          'R_TotRev_TTM_Y3YGR',
    #                          'R_FCFTot_Y3YGR',
    #                          'R_SUMLIAB_Y3YGR',
    #                          'R_EBIT_Y3YGR',
    #                          'R_Tax_TotProfit_QTTM'],
    #                         ['R_CFO_TotRev_s_First_row_extre_0.3',
    #                          'R_CFO_s_YOY_First_row_extre_0.3',
    #                          'R_COMPANYCODE_First_row_extre_0.3',
    #                          'R_Cashflow_s_YOY_First_row_extre_0.3',
    #                          'R_EPS_s_First_row_extre_0.3',
    #                          'R_EPS_s_YOY_First_row_extre_0.3',
    #                          'R_NetAssets_s_YOY_First_row_extre_0.3',
    #                          'R_NetInc_TotProfit_s_First_row_extre_0.3',
    #                          'R_NetInc_s_First_row_extre_0.3',
    #                          'R_OPCF_NetInc_s_First_row_extre_0.3',
    #                          'R_OperProfit_s_YOY_First_row_extre_0.3',
    #                          'R_ParentProfit_s_POP_First_row_extre_0.3',
    #                          'R_ParentProfit_s_YOY_First_row_extre_0.3',
    #                          'R_ROENetIncRecur_s_First_row_extre_0.3',
    #                          'R_RevenueTotPS_s_First_row_extre_0.3',
    #                          'R_Revenue_s_YOY_First_row_extre_0.3',
    #                          'R_Tax_TotProfit_s_First_row_extre_0.3',
    #                          'R_TotAssets_s_YOY_First_row_extre_0.3',
    #                          'R_TotRev_s_POP_First_row_extre_0.3',
    #                          'R_TotRev_s_YOY_First_row_extre_0.3']]
    # for i in range(3):
    #     tech_name_list = random.sample(tech_name_list_list, 20)
    #     funda_name_list = funda_name_list_list[i]
    #     print(tech_name_list, funda_name_list)
    #     pool_num = 20
    #     main.test_index_3(tech_name_list, funda_name_list, pool_num)
    #     send_email.send_email('i={}'.format(i), ['whs@yingpei.com'], [], 'alpha_test_2')
