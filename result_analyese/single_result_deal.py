import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *
import work_whs.main_file.main_file_single_test as mfst


def mul_fun(a, b):
    a_l = a.where(a > 0, 0)
    a_s = a.where(a < 0, 0)

    b_l = b.where(b > 0, 0)
    b_s = b.where(b < 0, 0)

    pos_l = a_l.mul(b_l)
    pos_s = a_s.mul(b_s)

    pos = pos_l.sub(pos_s)
    return pos


def load_tech_factor(self, file_name):
    load_path = os.path.join('/mnt/mfs/dat_whs/data/new_factor_data/' + self.sector_name)
    target_df = pd.read_pickle(os.path.join(load_path, file_name + '.pkl')) \
        .reindex(index=self.xinx, columns=self.xnms)
    if self.if_only_long:
        target_df = target_df[target_df > 0]
    return target_df


def load_daily_factor(self, file_name):
    load_path = '/mnt/mfs/DAT_EQT/EM_Funda/daily/'
    tmp_df = bt.AZ_Load_csv(os.path.join(load_path, file_name + '.csv')) \
        .reindex(index=self.xinx, columns=self.xnms)

    target_df = self.row_extre(tmp_df, self.sector_df, 0.3)
    if self.if_only_long:
        target_df = target_df[target_df > 0]
    return target_df


def load_whs_factor(self, file_name):
    load_path = '/mnt/mfs/DAT_EQT/EM_Funda/dat_whs/'
    tmp_df = bt.AZ_Load_csv(os.path.join(load_path, file_name + '.csv')) \
        .reindex(index=self.xinx, columns=self.xnms)

    target_df = self.row_extre(tmp_df, self.sector_df, 0.3)
    if self.if_only_long:
        target_df = target_df[target_df > 0]
    return target_df


def get_result_data(root_path, file_name):
    data = pd.read_csv(f'{root_path}/{file_name}', sep='|', index_col=0, header=None)
    print(data)
    data.columns = ['name_1', 'fun_name', 'sector_name', 'in_condition', 'out_condition', 'ic', 'sp_d', 'sp_m', 'sp_u',
                    'pot_in', 'fit_ratio', 'leve_ratio', 'sp_in', 'sp_q_out']
    data_sort = data.sort_values(by='sp_in')
    return data_sort


# root_path = '/mnt/mfs/dat_whs/result/result'
# # def main():
# file_name_list = [x for x in os.listdir(root_path) if 'single_test' in x and
#                   os.path.getsize(os.path.join(root_path, x)) != 0]
# print(file_name_list)
# for file_name in file_name_list:
#     result_df = get_result_data(root_path, file_name)

def get_file_name(sector_name):
    tmp_file_list = os.listdir(f'/mnt/mfs/dat_whs/data/single_factor_pnl/{sector_name}')
    target_file_list = [x.split('|') for x in tmp_file_list if '|' in x]
    data = pd.DataFrame(target_file_list, columns=['factor_name', 'sector_name', 'hold_time', 'if_only_long'])
    return data


def get_pnl_table(part_df, sector_name):
    file_name_df = part_df.apply(lambda x: '|'.join(x), axis=1)
    all_pnl_df = pd.DataFrame()
    for file_name in file_name_df:
        # print(file_name)
        pnl_df = pd.read_csv(f'/mnt/mfs/dat_whs/data/single_factor_pnl/{sector_name}/{file_name}',
                             index_col=0, parse_dates=True)
        pnl_df.columns = [file_name]
        all_pnl_df = pd.concat([all_pnl_df, pnl_df], axis=1)
    all_pnl_df = all_pnl_df.loc[pd.to_datetime('20130101'):]
    return all_pnl_df


my_factor_dict = dict({
    'RZCHE_p120d_col_extre_0.2': 'load_tech_factor',
    'RZCHE_p60d_col_extre_0.2': 'load_tech_factor',
    'RZCHE_p20d_col_extre_0.2': 'load_tech_factor',
    'RZCHE_p10d_col_extre_0.2': 'load_tech_factor',
    'RZCHE_p345d_continue_ud': 'load_tech_factor',
    'RZCHE_row_extre_0.2': 'load_tech_factor',
    'RQCHL_p120d_col_extre_0.2': 'load_tech_factor',
    'RQCHL_p60d_col_extre_0.2': 'load_tech_factor',
    'RQCHL_p20d_col_extre_0.2': 'load_tech_factor',
    'RQCHL_p10d_col_extre_0.2': 'load_tech_factor',
    'RQCHL_p345d_continue_ud': 'load_tech_factor',
    'RQCHL_row_extre_0.2': 'load_tech_factor',
    'RQYL_p120d_col_extre_0.2': 'load_tech_factor',
    'RQYL_p60d_col_extre_0.2': 'load_tech_factor',
    'RQYL_p20d_col_extre_0.2': 'load_tech_factor',
    'RQYL_p10d_col_extre_0.2': 'load_tech_factor',
    'RQYL_p345d_continue_ud': 'load_tech_factor',
    'RQYL_row_extre_0.2': 'load_tech_factor',
    'RQYE_p120d_col_extre_0.2': 'load_tech_factor',
    'RQYE_p60d_col_extre_0.2': 'load_tech_factor',
    'RQYE_p20d_col_extre_0.2': 'load_tech_factor',
    'RQYE_p10d_col_extre_0.2': 'load_tech_factor',
    'RQYE_p345d_continue_ud': 'load_tech_factor',
    'RQYE_row_extre_0.2': 'load_tech_factor',
    'RQMCL_p120d_col_extre_0.2': 'load_tech_factor',
    'RQMCL_p60d_col_extre_0.2': 'load_tech_factor',
    'RQMCL_p20d_col_extre_0.2': 'load_tech_factor',
    'RQMCL_p10d_col_extre_0.2': 'load_tech_factor',
    'RQMCL_p345d_continue_ud': 'load_tech_factor',
    'RQMCL_row_extre_0.2': 'load_tech_factor',
    'RZYE_p120d_col_extre_0.2': 'load_tech_factor',
    'RZYE_p60d_col_extre_0.2': 'load_tech_factor',
    'RZYE_p20d_col_extre_0.2': 'load_tech_factor',
    'RZYE_p10d_col_extre_0.2': 'load_tech_factor',
    'RZYE_p345d_continue_ud': 'load_tech_factor',
    'RZYE_row_extre_0.2': 'load_tech_factor',
    'RZMRE_p120d_col_extre_0.2': 'load_tech_factor',
    'RZMRE_p60d_col_extre_0.2': 'load_tech_factor',
    'RZMRE_p20d_col_extre_0.2': 'load_tech_factor',
    'RZMRE_p10d_col_extre_0.2': 'load_tech_factor',
    'RZMRE_p345d_continue_ud': 'load_tech_factor',
    'RZMRE_row_extre_0.2': 'load_tech_factor',
    'RZRQYE_p120d_col_extre_0.2': 'load_tech_factor',
    'RZRQYE_p60d_col_extre_0.2': 'load_tech_factor',
    'RZRQYE_p20d_col_extre_0.2': 'load_tech_factor',
    'RZRQYE_p10d_col_extre_0.2': 'load_tech_factor',
    'RZRQYE_p345d_continue_ud': 'load_tech_factor',
    'RZRQYE_row_extre_0.2': 'load_tech_factor',
    'WILLR_200_40': 'load_tech_factor',
    'WILLR_200_30': 'load_tech_factor',
    'WILLR_200_20': 'load_tech_factor',
    'WILLR_140_40': 'load_tech_factor',
    'WILLR_140_30': 'load_tech_factor',
    'WILLR_140_20': 'load_tech_factor',
    'WILLR_100_40': 'load_tech_factor',
    'WILLR_100_30': 'load_tech_factor',
    'WILLR_100_20': 'load_tech_factor',
    'WILLR_40_40': 'load_tech_factor',
    'WILLR_40_30': 'load_tech_factor',
    'WILLR_40_20': 'load_tech_factor',
    'WILLR_20_40': 'load_tech_factor',
    'WILLR_20_30': 'load_tech_factor',
    'WILLR_20_20': 'load_tech_factor',
    'WILLR_10_40': 'load_tech_factor',
    'WILLR_10_30': 'load_tech_factor',
    'WILLR_10_20': 'load_tech_factor',
    'BBANDS_10_2': 'load_tech_factor',
    'BBANDS_10_1.5': 'load_tech_factor',
    'BBANDS_10_1': 'load_tech_factor',
    'MACD_20_60_18': 'load_tech_factor',
    'BBANDS_200_2': 'load_tech_factor',
    'BBANDS_200_1.5': 'load_tech_factor',
    'BBANDS_200_1': 'load_tech_factor',
    'BBANDS_140_2': 'load_tech_factor',
    'BBANDS_140_1.5': 'load_tech_factor',
    'BBANDS_140_1': 'load_tech_factor',
    'BBANDS_100_2': 'load_tech_factor',
    'BBANDS_100_1.5': 'load_tech_factor',
    'BBANDS_100_1': 'load_tech_factor',
    'BBANDS_40_2': 'load_tech_factor',
    'BBANDS_40_1.5': 'load_tech_factor',
    'BBANDS_40_1': 'load_tech_factor',
    'BBANDS_20_2': 'load_tech_factor',
    'BBANDS_20_1.5': 'load_tech_factor',
    'BBANDS_20_1': 'load_tech_factor',
    'MA_LINE_160_60': 'load_tech_factor',
    'MA_LINE_120_60': 'load_tech_factor',
    'MA_LINE_100_40': 'load_tech_factor',
    'MA_LINE_60_20': 'load_tech_factor',
    'MA_LINE_10_5': 'load_tech_factor',
    'MACD_12_26_9': 'load_tech_factor',
    'intra_up_vwap_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_up_vol_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_up_div_dn_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_up_div_daily_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_up_15_bar_vwap_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_up_15_bar_vol_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_up_15_bar_div_dn_15_bar_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_up_15_bar_div_daily_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_dn_vwap_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_dn_vol_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_dn_div_daily_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_dn_15_bar_vwap_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_dn_15_bar_vol_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_dn_15_bar_div_daily_col_score_row_extre_0.3': 'load_tech_factor',
    'intra_up_vwap_row_extre_0.3': 'load_tech_factor',
    'intra_up_vol_row_extre_0.3': 'load_tech_factor',
    'intra_up_div_dn_row_extre_0.3': 'load_tech_factor',
    'intra_up_div_daily_row_extre_0.3': 'load_tech_factor',
    'intra_up_15_bar_vwap_row_extre_0.3': 'load_tech_factor',
    'intra_up_15_bar_vol_row_extre_0.3': 'load_tech_factor',
    'intra_up_15_bar_div_dn_15_bar_row_extre_0.3': 'load_tech_factor',
    'intra_up_15_bar_div_daily_row_extre_0.3': 'load_tech_factor',
    'intra_dn_vwap_row_extre_0.3': 'load_tech_factor',
    'intra_dn_vol_row_extre_0.3': 'load_tech_factor',
    'intra_dn_div_daily_row_extre_0.3': 'load_tech_factor',
    'intra_dn_15_bar_vwap_row_extre_0.3': 'load_tech_factor',
    'intra_dn_15_bar_vol_row_extre_0.3': 'load_tech_factor',
    'intra_dn_15_bar_div_daily_row_extre_0.3': 'load_tech_factor',
    'tab5_15_row_extre_0.3': 'load_tech_factor',
    'tab5_14_row_extre_0.3': 'load_tech_factor',
    'tab5_13_row_extre_0.3': 'load_tech_factor',
    'tab4_5_row_extre_0.3': 'load_tech_factor',
    'tab4_2_row_extre_0.3': 'load_tech_factor',
    'tab4_1_row_extre_0.3': 'load_tech_factor',
    'tab2_11_row_extre_0.3': 'load_tech_factor',
    'tab2_9_row_extre_0.3': 'load_tech_factor',
    'tab2_8_row_extre_0.3': 'load_tech_factor',
    'tab2_7_row_extre_0.3': 'load_tech_factor',
    'tab2_4_row_extre_0.3': 'load_tech_factor',
    'tab2_1_row_extre_0.3': 'load_tech_factor',
    'tab1_9_row_extre_0.3': 'load_tech_factor',
    'tab1_8_row_extre_0.3': 'load_tech_factor',
    'tab1_7_row_extre_0.3': 'load_tech_factor',
    'tab1_5_row_extre_0.3': 'load_tech_factor',
    'tab1_2_row_extre_0.3': 'load_tech_factor',
    'tab1_1_row_extre_0.3': 'load_tech_factor',
    'RSI_200_30': 'load_tech_factor',
    'RSI_140_30': 'load_tech_factor',
    'RSI_100_30': 'load_tech_factor',
    'RSI_40_30': 'load_tech_factor',
    'RSI_200_10': 'load_tech_factor',
    'RSI_140_10': 'load_tech_factor',
    'RSI_100_10': 'load_tech_factor',
    'RSI_40_10': 'load_tech_factor',
    'ATR_200_0.2': 'load_tech_factor',
    'ATR_140_0.2': 'load_tech_factor',
    'ATR_100_0.2': 'load_tech_factor',
    'ATR_40_0.2': 'load_tech_factor',
    'ADOSC_60_160_0': 'load_tech_factor',
    'ADOSC_60_120_0': 'load_tech_factor',
    'ADOSC_40_100_0': 'load_tech_factor',
    'ADOSC_20_60_0': 'load_tech_factor',
    'MFI_200_70_30': 'load_tech_factor',
    'MFI_140_70_30': 'load_tech_factor',
    'MFI_100_70_30': 'load_tech_factor',
    'MFI_40_70_30': 'load_tech_factor',
    'CMO_200_0': 'load_tech_factor',
    'CMO_140_0': 'load_tech_factor',
    'CMO_100_0': 'load_tech_factor',
    'CMO_40_0': 'load_tech_factor',
    'AROON_200_80': 'load_tech_factor',
    'AROON_140_80': 'load_tech_factor',
    'AROON_100_80': 'load_tech_factor',
    'AROON_40_80': 'load_tech_factor',
    'ADX_200_20_10': 'load_tech_factor',
    'ADX_140_20_10': 'load_tech_factor',
    'ADX_100_20_10': 'load_tech_factor',
    'ADX_40_20_10': 'load_tech_factor',
    'TotRev_and_mcap_intdebt_QYOY_Y3YGR_0.3': 'load_tech_factor',
    'TotRev_and_asset_QYOY_Y3YGR_0.3': 'load_tech_factor',
    'TotRev_and_mcap_QYOY_Y3YGR_0.3': 'load_tech_factor',
    'TotRev_and_mcap_intdebt_Y3YGR_Y5YGR_0.3': 'load_tech_factor',
    'TotRev_and_asset_Y3YGR_Y5YGR_0.3': 'load_tech_factor',
    'TotRev_and_mcap_Y3YGR_Y5YGR_0.3': 'load_tech_factor',
    'NetProfit_and_mcap_intdebt_QYOY_Y3YGR_0.3': 'load_tech_factor',
    'NetProfit_and_asset_QYOY_Y3YGR_0.3': 'load_tech_factor',
    'NetProfit_and_mcap_QYOY_Y3YGR_0.3': 'load_tech_factor',
    'NetProfit_and_mcap_intdebt_Y3YGR_Y5YGR_0.3': 'load_tech_factor',
    'NetProfit_and_asset_Y3YGR_Y5YGR_0.3': 'load_tech_factor',
    'NetProfit_and_mcap_Y3YGR_Y5YGR_0.3': 'load_tech_factor',
    'EBIT_and_mcap_intdebt_QYOY_Y3YGR_0.3': 'load_tech_factor',
    'EBIT_and_asset_QYOY_Y3YGR_0.3': 'load_tech_factor',
    'EBIT_and_mcap_QYOY_Y3YGR_0.3': 'load_tech_factor',
    'EBIT_and_mcap_intdebt_Y3YGR_Y5YGR_0.3': 'load_tech_factor',
    'EBIT_and_asset_Y3YGR_Y5YGR_0.3': 'load_tech_factor',
    'EBIT_and_mcap_Y3YGR_Y5YGR_0.3': 'load_tech_factor',
    'OPCF_and_mcap_intdebt_QYOY_Y3YGR_0.3': 'load_tech_factor',
    'OPCF_and_asset_QYOY_Y3YGR_0.3': 'load_tech_factor',
    'OPCF_and_mcap_QYOY_Y3YGR_0.3': 'load_tech_factor',
    'OPCF_and_mcap_intdebt_Y3YGR_Y5YGR_0.3': 'load_tech_factor',
    'OPCF_and_asset_Y3YGR_Y5YGR_0.3': 'load_tech_factor',
    'OPCF_and_mcap_Y3YGR_Y5YGR_0.3': 'load_tech_factor',
    'R_OTHERLASSET_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_WorkCapital_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_TangAssets_IntDebt_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_SUMLIAB_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_ROE1_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_OPEX_sales_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_OperProfit_YOY_First_and_QTTM_0.3': 'load_tech_factor',
    'R_OperCost_sales_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_OPCF_TTM_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_NETPROFIT_s_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_NetInc_s_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_NetAssets_s_YOY_First_and_QTTM_0.3': 'load_tech_factor',
    'R_LOANREC_s_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_LTDebt_WorkCap_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_INVESTINCOME_s_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_IntDebt_Mcap_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_GSCF_sales_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_GrossProfit_TTM_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_FINANCEEXP_s_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_FairVal_TotProfit_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_ESTATEINVEST_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_EPSDiluted_YOY_First_and_QTTM_0.3': 'load_tech_factor',
    'R_EBITDA2_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_CostSales_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_CFO_s_YOY_First_and_QTTM_0.3': 'load_tech_factor',
    'R_Cashflow_s_YOY_First_and_QTTM_0.3': 'load_tech_factor',
    'R_ASSETDEVALUELOSS_s_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_ACCOUNTREC_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'R_ACCOUNTPAY_QYOY_and_QTTM_0.3': 'load_tech_factor',
    'CCI_p150d_limit_12': 'load_tech_factor',
    'CCI_p120d_limit_12': 'load_tech_factor',
    'CCI_p60d_limit_12': 'load_tech_factor',
    'CCI_p20d_limit_12': 'load_tech_factor',
    'MACD_40_160': 'load_tech_factor',
    'MACD_40_200': 'load_tech_factor',
    'MACD_20_200': 'load_tech_factor',
    'MACD_20_100': 'load_tech_factor',
    'MACD_10_30': 'load_tech_factor',
    'bias_turn_p120d': 'load_tech_factor',
    'bias_turn_p60d': 'load_tech_factor',
    'bias_turn_p20d': 'load_tech_factor',
    'turn_p150d_0.18': 'load_tech_factor',
    'turn_p30d_0.24': 'load_tech_factor',
    'turn_p120d_0.2': 'load_tech_factor',
    'turn_p60d_0.2': 'load_tech_factor',
    'turn_p20d_0.2': 'load_tech_factor',
    'log_price_0.2': 'load_tech_factor',
    'wgt_return_p120d_0.2': 'load_tech_factor',
    'wgt_return_p60d_0.2': 'load_tech_factor',
    'wgt_return_p20d_0.2': 'load_tech_factor',
    'return_p90d_0.2': 'load_tech_factor',
    'return_p30d_0.2': 'load_tech_factor',
    'return_p120d_0.2': 'load_tech_factor',
    'return_p60d_0.2': 'load_tech_factor',
    'return_p20d_0.2': 'load_tech_factor',
    'PBLast_p120d_col_extre_0.2': 'load_tech_factor',
    'PBLast_p60d_col_extre_0.2': 'load_tech_factor',
    'PBLast_p20d_col_extre_0.2': 'load_tech_factor',
    'PBLast_p10d_col_extre_0.2': 'load_tech_factor',
    'PBLast_p345d_continue_ud': 'load_tech_factor',
    'PBLast_row_extre_0.2': 'load_tech_factor',
    'PS_TTM_p120d_col_extre_0.2': 'load_tech_factor',
    'PS_TTM_p60d_col_extre_0.2': 'load_tech_factor',
    'PS_TTM_p20d_col_extre_0.2': 'load_tech_factor',
    'PS_TTM_p10d_col_extre_0.2': 'load_tech_factor',
    'PS_TTM_p345d_continue_ud': 'load_tech_factor',
    'PS_TTM_row_extre_0.2': 'load_tech_factor',
    'PE_TTM_p120d_col_extre_0.2': 'load_tech_factor',
    'PE_TTM_p60d_col_extre_0.2': 'load_tech_factor',
    'PE_TTM_p20d_col_extre_0.2': 'load_tech_factor',
    'PE_TTM_p10d_col_extre_0.2': 'load_tech_factor',
    'PE_TTM_p345d_continue_ud': 'load_tech_factor',
    'PE_TTM_row_extre_0.2': 'load_tech_factor',
    'volume_moment_p20120d': 'load_tech_factor',
    'volume_moment_p1040d': 'load_tech_factor',
    'volume_moment_p530d': 'load_tech_factor',
    'moment_p50300d': 'load_tech_factor',
    'moment_p30200d': 'load_tech_factor',
    'moment_p40200d': 'load_tech_factor',
    'moment_p20200d': 'load_tech_factor',
    'moment_p20100d': 'load_tech_factor',
    'moment_p10100d': 'load_tech_factor',
    'moment_p1060d': 'load_tech_factor',
    'moment_p510d': 'load_tech_factor',
    'continue_ud_p200d': 'load_tech_factor',
    'evol_p200d': 'load_tech_factor',
    'vol_count_down_p200d': 'load_tech_factor',
    'vol_p200d': 'load_tech_factor',
    'continue_ud_p100d': 'load_tech_factor',
    'evol_p100d': 'load_tech_factor',
    'vol_count_down_p100d': 'load_tech_factor',
    'vol_p100d': 'load_tech_factor',
    'continue_ud_p90d': 'load_tech_factor',
    'evol_p90d': 'load_tech_factor',
    'vol_count_down_p90d': 'load_tech_factor',
    'vol_p90d': 'load_tech_factor',
    'continue_ud_p50d': 'load_tech_factor',
    'evol_p50d': 'load_tech_factor',
    'vol_count_down_p50d': 'load_tech_factor',
    'vol_p50d': 'load_tech_factor',
    'continue_ud_p30d': 'load_tech_factor',
    'evol_p30d': 'load_tech_factor',
    'vol_count_down_p30d': 'load_tech_factor',
    'vol_p30d': 'load_tech_factor',
    'continue_ud_p120d': 'load_tech_factor',
    'evol_p120d': 'load_tech_factor',
    'vol_count_down_p120d': 'load_tech_factor',
    'vol_p120d': 'load_tech_factor',
    'continue_ud_p60d': 'load_tech_factor',
    'evol_p60d': 'load_tech_factor',
    'vol_count_down_p60d': 'load_tech_factor',
    'vol_p60d': 'load_tech_factor',
    'continue_ud_p20d': 'load_tech_factor',
    'evol_p20d': 'load_tech_factor',
    'vol_count_down_p20d': 'load_tech_factor',
    'vol_p20d': 'load_tech_factor',
    'continue_ud_p10d': 'load_tech_factor',
    'evol_p10d': 'load_tech_factor',
    'vol_count_down_p10d': 'load_tech_factor',
    'vol_p10d': 'load_tech_factor',
    'volume_count_down_p120d': 'load_tech_factor',
    'volume_count_down_p60d': 'load_tech_factor',
    'volume_count_down_p20d': 'load_tech_factor',
    'volume_count_down_p10d': 'load_tech_factor',
    'price_p120d_hl': 'load_tech_factor',
    'price_p60d_hl': 'load_tech_factor',
    'price_p20d_hl': 'load_tech_factor',
    'price_p10d_hl': 'load_tech_factor',
    'aadj_r_p120d_col_extre_0.2': 'load_tech_factor',
    'aadj_r_p60d_col_extre_0.2': 'load_tech_factor',
    'aadj_r_p20d_col_extre_0.2': 'load_tech_factor',
    'aadj_r_p10d_col_extre_0.2': 'load_tech_factor',
    'aadj_r_p345d_continue_ud': 'load_tech_factor',
    'aadj_r_p345d_continue_ud_pct': 'load_tech_factor',
    'aadj_r_row_extre_0.2': 'load_tech_factor',
    'TVOL_p90d_col_extre_0.2': 'load_tech_factor',
    'TVOL_p30d_col_extre_0.2': 'load_tech_factor',
    'TVOL_p120d_col_extre_0.2': 'load_tech_factor',
    'TVOL_p60d_col_extre_0.2': 'load_tech_factor',
    'TVOL_p20d_col_extre_0.2': 'load_tech_factor',
    'TVOL_p10d_col_extre_0.2': 'load_tech_factor',
    'TVOL_p345d_continue_ud': 'load_tech_factor',
    'TVOL_row_extre_0.2': 'load_tech_factor',

    'R_ACCOUNTPAY_QYOY': 'load_daily_factor',
    'R_ACCOUNTREC_QYOY': 'load_daily_factor',
    'R_ASSETDEVALUELOSS_s_QYOY': 'load_daily_factor',
    'R_AssetDepSales_s_First': 'load_daily_factor',
    'R_BusinessCycle_First': 'load_daily_factor',
    'R_CFOPS_s_First': 'load_daily_factor',
    'R_CFO_TotRev_s_First': 'load_daily_factor',
    'R_CFO_s_YOY_First': 'load_daily_factor',
    'R_Cashflow_s_YOY_First': 'load_daily_factor',
    'R_CostSales_QYOY': 'load_daily_factor',
    'R_CostSales_s_First': 'load_daily_factor',
    'R_CurrentAssetsTurnover_QTTM': 'load_daily_factor',
    'R_DaysReceivable_First': 'load_daily_factor',
    'R_DebtAssets_QTTM': 'load_daily_factor',
    'R_DebtEqt_First': 'load_daily_factor',
    'R_EBITDA2_QYOY': 'load_daily_factor',
    'R_EBITDA_IntDebt_QTTM': 'load_daily_factor',
    'R_EBITDA_sales_TTM_First': 'load_daily_factor',
    'R_EBIT_sales_QTTM': 'load_daily_factor',
    'R_EPS_s_First': 'load_daily_factor',
    'R_EPS_s_YOY_First': 'load_daily_factor',
    'R_ESTATEINVEST_QYOY': 'load_daily_factor',
    'R_FCFTot_Y3YGR': 'load_daily_factor',
    'R_FINANCEEXP_s_QYOY': 'load_daily_factor',
    'R_FairValChgPnL_s_First': 'load_daily_factor',
    'R_FairValChg_TotProfit_s_First': 'load_daily_factor',
    'R_FairVal_TotProfit_QYOY': 'load_daily_factor',
    'R_FairVal_TotProfit_TTM_First': 'load_daily_factor',
    'R_FinExp_sales_s_First': 'load_daily_factor',
    'R_GSCF_sales_s_First': 'load_daily_factor',
    'R_GrossProfit_TTM_QYOY': 'load_daily_factor',
    'R_INVESTINCOME_s_QYOY': 'load_daily_factor',
    'R_LTDebt_WorkCap_QTTM': 'load_daily_factor',
    'R_MgtExp_sales_s_First': 'load_daily_factor',
    'R_NETPROFIT_s_QYOY': 'load_daily_factor',
    'R_NOTICEDATE_First': 'load_daily_factor',
    'R_NetAssets_s_POP_First': 'load_daily_factor',
    'R_NetAssets_s_YOY_First': 'load_daily_factor',
    'R_NetCashflowPS_s_First': 'load_daily_factor',
    'R_NetIncRecur_QYOY': 'load_daily_factor',
    'R_NetIncRecur_s_First': 'load_daily_factor',
    'R_NetInc_TotProfit_s_First': 'load_daily_factor',
    'R_NetInc_s_First': 'load_daily_factor',
    'R_NetInc_s_QYOY': 'load_daily_factor',
    'R_NetMargin_s_YOY_First': 'load_daily_factor',
    'R_NetProfit_sales_s_First': 'load_daily_factor',
    'R_NetROA_TTM_First': 'load_daily_factor',
    'R_NetROA_s_First': 'load_daily_factor',
    'R_NonOperProft_TotProfit_s_First': 'load_daily_factor',
    'R_OPCF_NetInc_s_First': 'load_daily_factor',
    'R_OPCF_TTM_QYOY': 'load_daily_factor',
    'R_OPCF_TotDebt_QTTM': 'load_daily_factor',
    'R_OPCF_sales_s_First': 'load_daily_factor',
    'R_OPEX_sales_TTM_First': 'load_daily_factor',
    'R_OPEX_sales_s_First': 'load_daily_factor',
    'R_OTHERLASSET_QYOY': 'load_daily_factor',
    'R_OperCost_sales_s_First': 'load_daily_factor',
    'R_OperProfit_YOY_First': 'load_daily_factor',
    'R_OperProfit_s_POP_First': 'load_daily_factor',
    'R_OperProfit_s_YOY_First': 'load_daily_factor',
    'R_OperProfit_sales_s_First': 'load_daily_factor',
    'R_ParentProfit_s_POP_First': 'load_daily_factor',
    'R_ParentProfit_s_YOY_First': 'load_daily_factor',
    'R_ROENetIncRecur_s_First': 'load_daily_factor',
    'R_ROE_s_First': 'load_daily_factor',
    'R_RecurNetProft_NetProfit_s_First': 'load_daily_factor',
    'R_RevenuePS_s_First': 'load_daily_factor',
    'R_RevenueTotPS_s_First': 'load_daily_factor',
    'R_Revenue_s_POP_First': 'load_daily_factor',
    'R_Revenue_s_YOY_First': 'load_daily_factor',
    'R_SUMLIAB_QYOY': 'load_daily_factor',
    'R_SUMLIAB_Y3YGR': 'load_daily_factor',
    'R_SalesCost_s_First': 'load_daily_factor',
    'R_SalesGrossMGN_QTTM': 'load_daily_factor',
    'R_SalesGrossMGN_s_First': 'load_daily_factor',
    'R_SalesNetMGN_s_First': 'load_daily_factor',
    'R_TangAssets_TotLiab_QTTM': 'load_daily_factor',
    'R_Tax_TotProfit_QTTM': 'load_daily_factor',
    'R_Tax_TotProfit_s_First': 'load_daily_factor',
    'R_TotAssets_s_YOY_First': 'load_daily_factor',
    'R_TotLiab_s_YOY_First': 'load_daily_factor',
    'R_TotRev_TTM_Y3YGR': 'load_daily_factor',
    'R_TotRev_s_POP_First': 'load_daily_factor',
    'R_TotRev_s_YOY_First': 'load_daily_factor',
    'R_WorkCapital_QYOY': 'load_daily_factor',

    'bar_num_7_df': 'load_whs_factor',
    'bar_num_12_df': 'load_whs_factor',
    'repurchase': 'load_whs_factor',
    'dividend': 'load_whs_factor',
    'repurchase_news_title': 'load_whs_factor',
    'repurchase_news_summary': 'load_whs_factor',
    'dividend_news_title': 'load_whs_factor',
    'dividend_news_summary': 'load_whs_factor',
    'staff_changes_news_title': 'load_whs_factor',
    'staff_changes_news_summary': 'load_whs_factor',
    'funds_news_title': 'load_whs_factor',
    'funds_news_summary': 'load_whs_factor',
    'meeting_decide_news_title': 'load_whs_factor',
    'meeting_decide_news_summary': 'load_whs_factor',
    'restricted_shares_news_title': 'load_whs_factor',
    'restricted_shares_news_summary': 'load_whs_factor',
    'son_company_news_title': 'load_whs_factor',
    'son_company_news_summary': 'load_whs_factor',
    'suspend_news_title': 'load_whs_factor',
    'suspend_news_summary': 'load_whs_factor',
    'shares_news_title': 'load_whs_factor',
    '': 'load_whs_factor',
    'shares_news_summary': 'load_whs_factor',
    'ab_inventory': 'load_whs_factor',
    'ab_rec': 'load_whs_factor',
    'ab_others_rec': 'load_whs_factor',
    'ab_ab_pre_rec': 'load_whs_factor',
    'ab_sale_mng_exp': 'load_whs_factor',
    'ab_grossprofit': 'load_whs_factor',
    'lsgg_num_df_5': 'load_whs_factor',
    'lsgg_num_df_20': 'load_whs_factor',
    'lsgg_num_df_60': 'load_whs_factor',
    'bulletin_num_df': 'load_whs_factor',
    'bulletin_num_df_5': 'load_whs_factor',
    'bulletin_num_df_20': 'load_whs_factor',
    'bulletin_num_df_60': 'load_whs_factor',
    'news_num_df_5': 'load_whs_factor',
    'news_num_df_20': 'load_whs_factor',
    'news_num_df_60': 'load_whs_factor',
    'staff_changes': 'load_whs_factor',
    'funds': 'load_whs_factor',
    'meeting_decide': 'load_whs_factor',
    'restricted_shares': 'load_whs_factor',
    'son_company': 'load_whs_factor',
    'suspend': 'load_whs_factor',
    'shares': 'load_whs_factor',
    'buy_key_title__word': 'load_whs_factor',
    'sell_key_title_word': 'load_whs_factor',
    'buy_summary_key_word': 'load_whs_factor',
    'sell_summary_key_word': 'load_whs_factor',

})
my_factor_dict_2 = dict({
    'REMTK.40': 'load_remy_factor',
    'REMTK.39': 'load_remy_factor',
    'REMTK.38': 'load_remy_factor',
    'REMTK.37': 'load_remy_factor',
    'REMTK.36': 'load_remy_factor',
    'REMTK.35': 'load_remy_factor',
    'REMTK.34': 'load_remy_factor',
    'REMTK.33': 'load_remy_factor',
    'REMTK.32': 'load_remy_factor',
    'REMTK.31': 'load_remy_factor',
    'REMFF.40': 'load_remy_factor',
    'REMFF.39': 'load_remy_factor',
    'REMFF.38': 'load_remy_factor',
    'REMFF.37': 'load_remy_factor',
    'REMFF.36': 'load_remy_factor',
    'REMFF.35': 'load_remy_factor',
    'REMFF.34': 'load_remy_factor',
    'REMFF.33': 'load_remy_factor',
    'REMFF.32': 'load_remy_factor',
    'REMFF.31': 'load_remy_factor',
    'REMWB.12': 'load_remy_factor',
    'REMWB.11': 'load_remy_factor',
    'REMWB.10': 'load_remy_factor',
    'REMWB.09': 'load_remy_factor',
    'REMWB.08': 'load_remy_factor',
    'REMWB.07': 'load_remy_factor',
    'REMWB.06': 'load_remy_factor',
    'REMWB.05': 'load_remy_factor',
    'REMWB.04': 'load_remy_factor',
    'REMWB.03': 'load_remy_factor',
    'REMWB.02': 'load_remy_factor',
    'REMWB.01': 'load_remy_factor',
    'REMTK.30': 'load_remy_factor',
    'REMTK.29': 'load_remy_factor',
    'REMTK.28': 'load_remy_factor',
    'REMTK.27': 'load_remy_factor',
    'REMTK.26': 'load_remy_factor',
    'REMTK.25': 'load_remy_factor',
    'REMTK.24': 'load_remy_factor',
    'REMTK.23': 'load_remy_factor',
    'REMTK.22': 'load_remy_factor',
    'REMTK.21': 'load_remy_factor',
    'REMTK.20': 'load_remy_factor',
    'REMTK.19': 'load_remy_factor',
    'REMTK.18': 'load_remy_factor',
    'REMTK.17': 'load_remy_factor',
    'REMTK.16': 'load_remy_factor',
    'REMTK.15': 'load_remy_factor',
    'REMTK.14': 'load_remy_factor',
    'REMTK.13': 'load_remy_factor',
    'REMTK.12': 'load_remy_factor',
    'REMTK.11': 'load_remy_factor',
    'REMTK.10': 'load_remy_factor',
    'REMTK.09': 'load_remy_factor',
    'REMTK.08': 'load_remy_factor',
    'REMTK.07': 'load_remy_factor',
    'REMTK.06': 'load_remy_factor',
    'REMTK.05': 'load_remy_factor',
    'REMTK.04': 'load_remy_factor',
    'REMTK.03': 'load_remy_factor',
    'REMTK.02': 'load_remy_factor',
    'REMTK.01': 'load_remy_factor',
    'REMFF.30': 'load_remy_factor',
    'REMFF.29': 'load_remy_factor',
    'REMFF.28': 'load_remy_factor',
    'REMFF.27': 'load_remy_factor',
    'REMFF.26': 'load_remy_factor',
    'REMFF.25': 'load_remy_factor',
    'REMFF.24': 'load_remy_factor',
    'REMFF.23': 'load_remy_factor',
    'REMFF.22': 'load_remy_factor',
    'REMFF.21': 'load_remy_factor',
    'REMFF.20': 'load_remy_factor',
    'REMFF.19': 'load_remy_factor',
    'REMFF.18': 'load_remy_factor',
    'REMFF.17': 'load_remy_factor',
    'REMFF.16': 'load_remy_factor',
    'REMFF.15': 'load_remy_factor',
    'REMFF.14': 'load_remy_factor',
    'REMFF.13': 'load_remy_factor',
    'REMFF.12': 'load_remy_factor',
    'REMFF.11': 'load_remy_factor',
    'REMFF.10': 'load_remy_factor',
    'REMFF.09': 'load_remy_factor',
    'REMFF.08': 'load_remy_factor',
    'REMFF.07': 'load_remy_factor',
    'REMFF.06': 'load_remy_factor',
    'REMFF.05': 'load_remy_factor',
    'REMFF.04': 'load_remy_factor',
    'REMFF.03': 'load_remy_factor',
    'REMFF.02': 'load_remy_factor',
    'REMFF.01': 'load_remy_factor'
})
jerry_factor_dict = dict({
    'LIQ_all_original.csv': 'load_jerry_factor',
    'LIQ_all_pure.csv': 'load_jerry_factor',
    'LIQ_mix.csv': 'load_jerry_factor',
    'LIQ_p1_original.csv': 'load_jerry_factor',
    'LIQ_p1_pure.csv': 'load_jerry_factor',
    'LIQ_p2_original.csv': 'load_jerry_factor',
    'LIQ_p2_pure.csv': 'load_jerry_factor',
    'LIQ_p3_original.csv': 'load_jerry_factor',
    'LIQ_p3_pure.csv': 'load_jerry_factor',
    'LIQ_p4_original.csv': 'load_jerry_factor',
    'LIQ_p4_pure.csv': 'load_jerry_factor',
    'M0': 'load_jerry_factor',
    'M1': 'load_jerry_factor',
    'M1_p1': 'load_jerry_factor',
    'M1_p2': 'load_jerry_factor',
    'M1_p3': 'load_jerry_factor',
    'M1_p4': 'load_jerry_factor',
    'vr_afternoon_10min_20days': 'load_jerry_factor',
    'vr_afternoon_last10min_20days.csv': 'load_jerry_factor',
    'vr_original_20days.csv': 'load_jerry_factor',
    'vr_original_45days.csv': 'load_jerry_factor',
    'vr_original_75days.csv': 'load_jerry_factor',
})
my_factor_dict.update(my_factor_dict_2)

sector_name_list = [
    'market_top_300plus',
    'market_top_300plus_industry_10_15',
    'market_top_300plus_industry_20_25_30_35',
    'market_top_300plus_industry_40',
    'market_top_300plus_industry_45_50',
    'market_top_300plus_industry_55',

    'market_top_300to800plus',
    'market_top_300to800plus_industry_10_15',
    'market_top_300to800plus_industry_20_25_30_35',
    'market_top_300to800plus_industry_40',
    'market_top_300to800plus_industry_45_50',
    'market_top_300to800plus_industry_55',
]


def deal_fun(sector_name, hold_time, if_only_long, name_list, buy_sell_way_list):
    if if_only_long == 'True':
        if_only_long = True
    else:
        if_only_long = False

    root_path = '/mnt/mfs/DAT_EQT'
    if_save = True
    if_new_program = True

    begin_date = pd.to_datetime('20130101')
    cut_date = pd.to_datetime('20160401')
    end_date = pd.to_datetime('20181201')
    lag = 2
    return_file = ''

    if_hedge = True

    if sector_name.startswith('market_top_300plus'):
        if_weight = 1
        ic_weight = 0

    elif sector_name.startswith('market_top_300to800plus'):
        if_weight = 0
        ic_weight = 1

    else:
        if_weight = 0.5
        ic_weight = 0.5

    main = mfst.FactorTestSector(root_path, if_save, if_new_program, begin_date, cut_date, end_date, [], sector_name,
                                 int(hold_time), lag, return_file, if_hedge, bool(if_only_long), if_weight, ic_weight)
    # 因子相加
    # mix_factor, in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, \
    # pot_in, fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df = main.single_test_c(name_list, buy_sell_way_list)
    # 因子相乘
    mix_factor, in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, \
    pot_in, fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df = main.mul_test_c(name_list, buy_sell_way_list)
    return pot_in, fit_ratio, leve_ratio, bt.AZ_Sharpe_y(pnl_df), pnl_df


def plot_send_all(all_pnl_df):
    for file_name in all_pnl_df.index:
        plot_send_result(all_pnl_df[file_name], bt.AZ_Sharpe_y(all_pnl_df), file_name, '')


def select_fun(file_name, pnl_table_c, num=4):
    i = 0
    target_pnl = pd.DataFrame(pnl_table_c[file_name])
    target_pnl.columns = ['target_pnl']
    target_sp = bt.AZ_Sharpe_y(target_pnl)
    if target_sp.values[0] < 0:
        target_pnl = -target_pnl

    select_list = []
    while i < num:
        pnl_corr = pd.concat([pnl_table_c, target_pnl], axis=1).corr()['target_pnl']
        select_name = pnl_corr.abs().sort_values().index[0]
        select_list.append(select_name)
        select_pnl = pnl_table_c[select_name]
        select_sp = bt.AZ_Sharpe_y(select_pnl)
        if select_sp > 0:
            target_pnl = target_pnl.add(select_pnl, axis=0)
        else:
            target_pnl = target_pnl.sub(select_pnl, axis=0)
        print(bt.AZ_Sharpe_y(target_pnl))
        pnl_table_c = pnl_table_c.drop(columns=select_name)
        i += 1
    return select_list


def single_test():
    for sector_name in sector_name_list:
        print(sector_name)
        data = get_file_name(sector_name)
        for (hold_time, if_only_long), part_df in data.groupby(['hold_time', 'if_only_long']):
            print(hold_time)
            print(if_only_long)
            # print(part_df)
            pnl_table = get_pnl_table(part_df, sector_name)
            drop_list = list(my_factor_dict_2.keys())
            pnl_table = pnl_table[[x for x in pnl_table.columns if 'intra' not in x]]
            # remy pnl
            # pnl_table_remy = pnl_table[[x for x in pnl_table.columns if x.split('|')[0] in drop_list]]
            remy_list = [x for x in pnl_table.columns if x.split('|')[0] in drop_list]
            sharpe_df = pnl_table.apply(bt.AZ_Sharpe_y)
            a = sharpe_df.sort_values()
            corr_df = pnl_table.corr()

            sharpe_up = a[remy_list][a[remy_list].abs() > 0.8]
            print(a[remy_list])
            print(sharpe_up)
            sharpe_mid = a[a.abs() > 0.7]
            for file_name in sharpe_up.index:
                corr_series = corr_df.reindex(index=sharpe_mid.index, columns=sharpe_mid.index)[file_name]
                modify_index = corr_series.loc[corr_series.abs().sort_values().index].index[:3]
                portfolio_index = [file_name] + list(modify_index)

                buy_sell_way_df = sharpe_df[portfolio_index]
                select_pnl_df = pnl_table[portfolio_index]
                buy_sell_way_df[buy_sell_way_df > 0] = 1
                buy_sell_way_df[buy_sell_way_df < 0] = -1
                buy_sell_way_list = buy_sell_way_df.values
                name_list = [x.split('|')[0] for x in portfolio_index]

                target_pnl = (select_pnl_df * buy_sell_way_df).sum(1)
                target_sharpe = bt.AZ_Sharpe_y(target_pnl)
                target_lvr = bt.AZ_Leverage_ratio(target_pnl.cumsum())
                print(target_sharpe)
                if target_sharpe > 2:
                    pot_in, fit_ratio, leve_ratio, sp, pnl_df = \
                        deal_fun(sector_name, hold_time, if_only_long, name_list, buy_sell_way_list)
                    # plot_send_result(target_pnl, target_sharpe, file_name, '\n'.join(portfolio_index))
                    # if sp > 2:
                    print(name_list, buy_sell_way_list)
                    print('|'.join([x.split('|')[0] for x in portfolio_index]))
                    print('|'.join([str(x) for x in [pot_in, fit_ratio, leve_ratio]]))
                    plot_send_result(pnl_df, sp, '|'.join([x.split('|')[0] for x in portfolio_index]) + '$'
                                     + '|'.join([sector_name, hold_time, if_only_long]),
                                     '|'.join([str(x) for x in [pot_in, fit_ratio, leve_ratio]]))
                    print(target_sharpe)
                    print(sp)


def mul_test():
    for sector_name in sector_name_list:
        data = get_file_name(sector_name)
        for (hold_time, if_only_long), part_df in data.groupby(['hold_time', 'if_only_long']):
            print(sector_name, hold_time, if_only_long)

            # print(part_df)
            pnl_table = get_pnl_table(part_df, sector_name)
            drop_list = list(jerry_factor_dict.keys())
            pnl_table = pnl_table[[x for x in pnl_table.columns if 'intra' not in x]]
            # remy pnl
            # pnl_table_remy = pnl_table[[x for x in pnl_table.columns if x.split('|')[0] in drop_list]]
            # remy_list = [x for x in pnl_table.columns if x.split('|')[0] in drop_list]
            sharpe_df = pnl_table.apply(bt.AZ_Sharpe_y)
            a = sharpe_df.sort_values()
            corr_df = pnl_table.corr()

            sharpe_up = a[a.abs() > 0.8]
            print(a)
            print(sharpe_up)
            sharpe_mid = a[a.abs() > 0.7]
            for file_name in sharpe_up.index:
                corr_series = corr_df.reindex(index=sharpe_mid.index, columns=sharpe_mid.index)[file_name]
                modify_index = corr_series.loc[corr_series.abs().sort_values().index].index[:3]
                portfolio_index = [file_name] + list(modify_index)

                buy_sell_way_df = sharpe_df[portfolio_index]

                buy_sell_way_df[buy_sell_way_df > 0] = 1
                buy_sell_way_df[buy_sell_way_df < 0] = -1
                buy_sell_way_list = buy_sell_way_df.values
                name_list = [x.split('|')[0] for x in portfolio_index]
                pot_in, fit_ratio, leve_ratio, sp, pnl_df = \
                    deal_fun(sector_name, hold_time, if_only_long, name_list, buy_sell_way_list)
                print(sp)
                if sp > 2:
                    print(name_list, buy_sell_way_list)
                    print('|'.join([x.split('|')[0] for x in portfolio_index]))
                    print('|'.join([str(x) for x in [pot_in, fit_ratio, leve_ratio]]))
                    plot_send_result(pnl_df, sp, '|'.join([x.split('|')[0] for x in portfolio_index]) + '$'
                                     + '|'.join([sector_name, hold_time, if_only_long]),
                                     '|'.join([str(x) for x in [pot_in, fit_ratio, leve_ratio]]))


if __name__ == '__main__':
    single_test()
