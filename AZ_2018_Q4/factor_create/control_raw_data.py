def daily_path(x, root_path='/mnt/mfs/DAT_EQT'):
    return [f'{root_path}/daily/{x}.csv']


daily_list = [
    'R_ACCOUNTPAY_QYOY',
    'R_ACCOUNTREC_QYOY',
    'R_ASSETDEVALUELOSS_s_QYOY',
    'R_Cashflow_s_YOY_First',
    'R_CFO_s_YOY_First',
    'R_CostSales_QYOY',
    'R_EBITDA2_QYOY',
    'R_ESTATEINVEST_QYOY',
    'R_FairVal_TotProfit_QYOY',
    'R_FINANCEEXP_s_QYOY',
    'R_GrossProfit_TTM_QYOY',
    'R_INVESTINCOME_s_QYOY',
    'R_NetAssets_s_YOY_First',
    'R_NetInc_s_QYOY',
    'R_NETPROFIT_s_QYOY',
    'R_OPCF_TTM_QYOY',
    'R_OperProfit_YOY_First',
    'R_SUMLIAB_QYOY',
    'R_WorkCapital_QYOY',
    'R_OTHERLASSET_QYOY',
    'R_NetIncRecur_QYOY',
    'R_DebtAssets_QTTM',
    'R_EBITDA_IntDebt_QTTM',
    'R_EBITDA_sales_TTM_First',
    'R_BusinessCycle_First',
    'R_DaysReceivable_First',
    'R_DebtEqt_First',
    'R_FairVal_TotProfit_TTM_First',
    'R_LTDebt_WorkCap_QTTM',
    'R_OPCF_TotDebt_QTTM',
    'R_OPEX_sales_TTM_First',
    'R_SalesGrossMGN_QTTM',
    'R_CurrentAssetsTurnover_QTTM',
    'R_TangAssets_TotLiab_QTTM',
    'R_NetROA_TTM_First',
    'R_ROE_s_First',
    'R_EBIT_sales_QTTM',
]

DataPath = dict(zip(*[daily_list, [daily_path(x) for x in daily_list]]))
print(DataPath)