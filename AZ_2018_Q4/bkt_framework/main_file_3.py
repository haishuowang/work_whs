import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *
import work_whs.AZ_2018_Q4.bkt_framework.bkt_base_import_script as bkt_base


class KeyFun:
    @staticmethod
    def load_daily_data(file_name, xinx, xnms, sector_df):
        load_path = '/mnt/mfs/DAT_EQT/EM_Funda/daily/'
        tmp_df = bt.AZ_Load_csv(os.path.join(load_path, file_name + '.csv'))
        tmp_df = tmp_df.reindex(index=xinx, columns=xnms) * sector_df
        target_df = bt.AZ_Row_zscore(tmp_df, cap=5)
        return target_df

    @staticmethod
    def load_filter_data(filter_name, xinx, xnms, sector_df, if_only_long):
        load_path = '/mnt/mfs/dat_whs/data/new_factor_data_v2/'
        target_df = pd.read_pickle(os.path.join(load_path, filter_name + '.pkl')).reindex(index=xinx, columns=xnms)
        if if_only_long:
            target_df = target_df[target_df > 0]
        return target_df

    @staticmethod
    def row_extre(raw_df, sector_df, percent):
        raw_df = raw_df * sector_df
        target_df = raw_df.rank(axis=1, pct=True)
        target_df[target_df >= 1 - percent] = 1
        target_df[target_df <= percent] = -1
        target_df[(target_df > percent) & (target_df < 1 - percent)] = 0
        return target_df

    def create_mix_factor(self, name_1, name_2, xinx, xnms, sector_df, if_only_long, percent):
        factor_1 = self.load_daily_data(name_1, xinx, xnms, sector_df)
        factor_2 = self.load_daily_data(name_2, xinx, xnms, sector_df)
        score_df_1 = bt.AZ_Row_zscore(factor_1, cap=5)
        score_df_2 = bt.AZ_Row_zscore(factor_2, cap=5)
        mix_df = score_df_1 + score_df_2
        target_df = self.row_extre(mix_df, sector_df, percent)
        if if_only_long:
            target_df = target_df[target_df > 0]
        return target_df


def main_fun(time_para_dict, sector_name, hold_time, if_only_long):
    root_path = '/mnt/mfs/DAT_EQT'
    if_save = True
    if_new_program = True

    begin_date = pd.to_datetime('20100101')
    cut_date = pd.to_datetime('20160401')
    end_date = pd.to_datetime('20180901')
    lag = 2
    return_file = ''

    if_hedge = True
    # if_only_long = False

    if sector_name.startswith('market_top_300plus'):
        if_weight = 1
        ic_weight = 0

    elif sector_name.startswith('market_top_300to800plus'):
        if_weight = 0
        ic_weight = 1

    else:
        if_weight = 0.5
        ic_weight = 0.5

    para_set = [root_path, if_save, if_new_program, begin_date, cut_date, end_date, time_para_dict,
                sector_name, hold_time, lag, return_file, if_hedge, if_only_long, if_weight, ic_weight]

    key_fun = KeyFun()

    main_model = bkt_base.FactorTest(key_fun, *para_set)

    filter_list = [
        'ADOSC_5_10_0',
        'ADOSC_60_120_0',
        'ADX_10_20_10',
        'ADX_140_20_10',
        'ADX_40_20_10',
        'MFI_10_70_30',
        'MFI_40_70_30',
        'MFI_140_70_30',
        'AROON_10_80',
        'AROON_140_80',
        'BBANDS_10_1.5',
        'BBANDS_20_1.5',
        'MACD_20_60_18',
        'MACD_12_26_9',
        'MA_LINE_10_5',
        'MA_LINE_160_60',
        'MFI_100_70_30',
        'MFI_20_70_30',
        'RSI_100_10',
        'RSI_200_30',
        'WILLR_100_40',
        'WILLR_100_30',
        'WILLR_10_30',
        'WILLR_140_30',
        'WILLR_200_30',
        'WILLR_20_30',
    ]

    ratio_list = [
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

    change_list = [
        'R_ACCOUNTPAY_QYOY',
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
        'R_NetIncRecur_QYOY.csv'
    ]

    pool_num = 10
    suffix_name = os.path.basename(__file__).split('.')[0][-1]
    main_model.main_test_fun(filter_list, ratio_list, change_list,
                             pool_num=pool_num, suffix_name=suffix_name, old_file_name='')


time_para_dict = OrderedDict()

time_para_dict['time_para_1'] = [pd.to_datetime('20100101'), pd.to_datetime('20150101'),
                                 pd.to_datetime('20150401'), pd.to_datetime('20150701'),
                                 pd.to_datetime('20151001'), pd.to_datetime('20160101')]

time_para_dict['time_para_2'] = [pd.to_datetime('20110101'), pd.to_datetime('20160101'),
                                 pd.to_datetime('20160401'), pd.to_datetime('20160701'),
                                 pd.to_datetime('20161001'), pd.to_datetime('20170101')]

time_para_dict['time_para_3'] = [pd.to_datetime('20130101'), pd.to_datetime('20180101'),
                                 pd.to_datetime('20180401'), pd.to_datetime('20180701'),
                                 pd.to_datetime('20181001'), pd.to_datetime('20181001')]

time_para_dict['time_para_4'] = [pd.to_datetime('20130601'), pd.to_datetime('20180601'),
                                 pd.to_datetime('20181001'), pd.to_datetime('20181001'),
                                 pd.to_datetime('20181001'), pd.to_datetime('20181001')]

time_para_dict['time_para_5'] = [pd.to_datetime('20130701'), pd.to_datetime('20180701'),
                                 pd.to_datetime('20181001'), pd.to_datetime('20181001'),
                                 pd.to_datetime('20181001'), pd.to_datetime('20181001')]

time_para_dict['time_para_6'] = [pd.to_datetime('20130801'), pd.to_datetime('20180801'),
                                 pd.to_datetime('20181001'), pd.to_datetime('20181001'),
                                 pd.to_datetime('20181001'), pd.to_datetime('20181001')]


if __name__ == '__main__':

    if_only_long_list = [False, True]
    hold_time_list = [5, 20]
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

    for if_only_long, hold_time, sector_name in list(product(if_only_long_list, hold_time_list, sector_name_list)):
        # print(sector_name, hold_time, if_only_long)
        main_fun(time_para_dict, sector_name, hold_time, if_only_long)
