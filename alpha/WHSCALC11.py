import pandas as pd
import numpy as np
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt
from open_lib.shared_tools import send_email
from itertools import combinations
from datetime import datetime
from collections import OrderedDict

base_data_dict = OrderedDict({
    'TVOL': 'EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv',
    'TVALCNY': 'EM_Funda/TRAD_SK_DAILY_JC/TVALCNY.csv',
    'aadj_r': 'EM_Funda/DERIVED_14/aadj_r.csv',
    'TURNRATE': 'EM_Funda/TRAD_SK_DAILY_JC/TURNRATE.csv',

    'aadj_p': 'EM_Funda/DERIVED_14/aadj_p.csv',
    'aadj_p_OPEN': 'EM_Funda/DERIVED_14/aadj_p_OPEN.csv',
    'aadj_p_HIGH': 'EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
    'aadj_p_LOW': 'EM_Funda/DERIVED_14/aadj_p_OPEN.csv',

    'PE_TTM': 'EM_Funda/TRAD_SK_REVALUATION/PE_TTM.csv',
    'PS_TTM': 'EM_Funda/TRAD_SK_REVALUATION/PS_TTM.csv',
    'PBLast': 'EM_Funda/TRAD_SK_REVALUATION/PBLast.csv',

    'RZMRE': 'EM_Funda/TRAD_MT_MARGIN/RZMRE.csv',
    'RZYE': 'EM_Funda/TRAD_MT_MARGIN/RZYE.csv',
    'RQMCL': 'EM_Funda/TRAD_MT_MARGIN/RQMCL.csv',
    'RQYE': 'EM_Funda/TRAD_MT_MARGIN/RQYE.csv',
    'RQYL': 'EM_Funda/TRAD_MT_MARGIN/RQYL.csv',
    'RQCHL': 'EM_Funda/TRAD_MT_MARGIN/RQCHL.csv',
    'RZCHE': 'EM_Funda/TRAD_MT_MARGIN/RZCHE.csv',

    'R_AssetDepSales_s_First': '',
    'R_CFOPS_s_First': '',
    'R_CFO_TotRev_s_First': '',
    'R_CFO_s_YOY_First': '',
    'R_Cashflow_s_YOY_First': '',
    'R_CostSales_s_First': '',
    'R_CurrentAssetsTurnover_QTTM': '',
    'R_DebtAssets_QTTM': '',
    'R_EBITDA_IntDebt_QTTM': '',
    'R_EBIT_sales_QTTM': '',
    'R_EPS_s_First': '',
    'R_EPS_s_YOY_First': '',
    'R_FCFTot_Y3YGR': '',
    'R_FairValChgPnL_s_First': '',
    'R_FairValChg_TotProfit_s_First': '',
    'R_FinExp_sales_s_First': '',
    'R_GSCF_sales_s_First': '',
    'R_LTDebt_WorkCap_QTTM': '',
    'R_MgtExp_sales_s_First': '',
    'R_NetAssets_s_POP_First': '',
    'R_NetAssets_s_YOY_First': '',
    'R_NetCashflowPS_s_First': '',
    'R_NetIncRecur_s_First': '',
    'R_NetInc_TotProfit_s_First': '',
    'R_NetInc_s_First': '',
    'R_NetMargin_s_YOY_First': '',
    'R_NetProfit_sales_s_First': '',
    'R_NetROA_TTM_First': '',
    'R_NetROA_s_First': '',
    'R_NonOperProft_TotProfit_s_First': '',
    'R_OPCF_NetInc_s_First': '',
    'R_OPCF_TotDebt_QTTM': '',
    'R_OPCF_sales_s_First': '',
    'R_OPEX_sales_TTM_First': '',
    'R_OPEX_sales_s_First': '',
    'R_OperCost_sales_s_First': '',
    'R_OperProfit_s_POP_First': '',
    'R_OperProfit_s_YOY_First': '',
    'R_OperProfit_sales_s_First': '',
    'R_ParentProfit_s_POP_First': '',
    'R_ParentProfit_s_YOY_First': '',
    'R_ROENetIncRecur_s_First': '',
    'R_ROE_s_First': '',
    'R_RecurNetProft_NetProfit_s_First': '',
    'R_RevenuePS_s_First': '',
    'R_RevenueTotPS_s_First': '',
    'R_Revenue_s_POP_First': '',
    'R_Revenue_s_YOY_First': '',
    'R_SUMLIAB_Y3YGR': '',
    'R_SalesCost_s_First': '',
    'R_SalesGrossMGN_QTTM': '',
    'R_SalesGrossMGN_s_First': '',
    'R_SalesNetMGN_s_First': '',
    'R_TangAssets_TotLiab_QTTM': '',
    'R_Tax_TotProfit_QTTM': '',
    'R_Tax_TotProfit_s_First': '',
    'R_TotAssets_s_YOY_First': '',
    'R_TotLiab_s_YOY_First': '',
    'R_TotRev_TTM_Y3YGR': '',
    'R_TotRev_s_POP_First': '',
    'R_TotRev_s_YOY_First': '',

    'R_TotRev_TTM_QTTM': '',
    'R_FinCf_TTM_QTTM': '',
    'R_GSCF_TTM_QTTM': '',
    'R_NetCf_TTM_QTTM': '',
    'R_OPCF_TTM_QTTM': '',
    'R_EBIT_TTM_QTTM': '',
    'R_WorkCapital_QTTM': '',
    'R_NetWorkCapital_QTTM': '',
    'R_TotRev_TTM_QSD4Y': '',
    'R_FinCf_TTM_QSD4Y': '',
    'R_GSCF_TTM_QSD4Y': '',
    'R_NetCf_TTM_QSD4Y': '',
    'R_OPCF_TTM_QSD4Y': '',
    'R_EBIT_TTM_QSD4Y': '',
    'R_WorkCapital_QSD4Y': '',
    'R_NetWorkCapital_QSD4Y': '',
    'R_NetIncRecur_QTTM': '',
    'R_NETPROFIT_QTTM': '',
    'R_OTHERCINCOME_QTTM': '',
    'R_AssetDepSales_QTTM': '',
    'R_ACCEPTINVREC_QTTM': '',
    'R_COMMPAY_QTTM': '',
    'R_CurrentLiabInt0_QTTM': '',
    'R_DEFERTAX_QTTM': '',
    'R_DIVIPAY_QTTM': '',
    'R_EMPLOYEEPAY_QTTM': '',
    'R_INCOMETAX_QTTM': '',
    'R_INVENTORY_QTTM': '',
    'R_TangAssets_QTTM': '',
    'R_TangAssets_First': '',
    'R_TotAssets_NBY_First': '',
    'R_ASSETOTHER_First': '',
    'R_SUMASSET_First': '',
    'R_IntDebt_First': '',
    'R_NetDebt_First': '',
    'R_TotCapital_First': '',
    'R_WorkCapital_First': '',
    'R_ROETrig_First': '',
    'R_ROEWRecur_First': '',
    'R_NetROA1_First': '',
    'R_TotProfit_EBIT_First': '',
    'R_OperProfit_sales_Y3YGR': '',
    'R_SalesGrossMGN_First': '',
    'R_SalesGrossMGN_s_Y3YGR': '',

    'stock_tab1_1': '/EM_Funda/dat_whs/stock_code_df_tab1_1',
    'stock_tab1_2': '/EM_Funda/dat_whs/stock_code_df_tab1_2',
    'stock_tab1_5': '/EM_Funda/dat_whs/stock_code_df_tab1_5',
    'stock_tab1_7': '/EM_Funda/dat_whs/stock_code_df_tab1_7',
    'stock_tab1_8': '/EM_Funda/dat_whs/stock_code_df_tab1_8',
    'stock_tab1_9': '/EM_Funda/dat_whs/stock_code_df_tab1_9',
    'stock_tab2_1': '/EM_Funda/dat_whs/stock_code_df_tab2_1',
    'stock_tab2_10': '/EM_Funda/dat_whs/stock_code_df_tab2_10',
    'stock_tab2_4': '/EM_Funda/dat_whs/stock_code_df_tab2_4',
    'stock_tab2_5': '/EM_Funda/dat_whs/stock_code_df_tab2_5',
    'stock_tab2_7': '/EM_Funda/dat_whs/stock_code_df_tab2_7',
    'stock_tab2_8': '/EM_Funda/dat_whs/stock_code_df_tab2_8',
    'stock_tab2_9': '/EM_Funda/dat_whs/stock_code_df_tab2_9',
    'stock_tab4_1': '/EM_Funda/dat_whs/stock_code_df_tab4_1',
    'stock_tab4_2': '/EM_Funda/dat_whs/stock_code_df_tab4_2',
    'stock_tab4_3': '/EM_Funda/dat_whs/stock_code_df_tab4_3',

    'lsgg_num_df_5': '/EM_Funda/dat_whs/lsgg_num_df_5.csv',
    'lsgg_num_df_20': '/EM_Funda/dat_whs/lsgg_num_df_20.csv',
    'lsgg_num_df_60': '/EM_Funda/dat_whs/lsgg_num_df_60.csv',
    'bulletin_num_df_5': '/EM_Funda/dat_whs/bulletin_num_df_5.csv',
    'bulletin_num_df_20': '/EM_Funda/dat_whs/bulletin_num_df_20.csv',
    'bulletin_num_df_60': '/EM_Funda/dat_whs/bulletin_num_df_60.csv',
    'news_num_df_5': '/EM_Funda/dat_whs/news_num_df_5.csv',
    'news_num_df_20': '/EM_Funda/dat_whs/news_num_df_20.csv',
    'news_num_df_60': '/EM_Funda/dat_whs/news_num_df_60.csv',
    'staff_changes': '/EM_Funda/dat_whs/staff_changes.csv',
    'funds': '/EM_Funda/dat_whs/funds.csv',
    'meeting_decide': '/EM_Funda/dat_whs/meeting_decide.csv',
    'restricted_shares': '/EM_Funda/dat_whs/restricted_shares.csv',
    'son_company': '/EM_Funda/dat_whs/son_company.csv',
    'suspend': '/EM_Funda/dat_whs/suspend.csv',
    'shares': '/EM_Funda/dat_whs/shares.csv',
    'bar_num_7_df': '/EM_Funda/dat_whs/bar_num_7_df.csv',
    'bar_num_12_df': '/EM_Funda/dat_whs/bar_num_12_df.csv',
    'buy_key_title__word': '/EM_Funda/dat_whs/buy_key_title__word.csv',
    'sell_key_title_word': '/EM_Funda/dat_whs/sell_key_title_word.csv',
    'buy_summary_key_word': '/EM_Funda/dat_whs/buy_summary_key_word.csv',
    'sell_summary_key_word': '/EM_Funda/dat_whs/sell_summary_key_word.csv',

    'ab_inventory': '/EM_Funda/dat_whs/ab_inventory.csv',
    'ab_rec': '/EM_Funda/dat_whs/ab_rec.csv',
    'ab_others_rec': '/EM_Funda/dat_whs/ab_others_rec.csv',
    'ab_ab_pre_rec': '/EM_Funda/dat_whs/ab_ab_pre_rec.csv',
    'ab_sale_mng_exp': '/EM_Funda/dat_whs/ab_sale_mng_exp.csv',
    'ab_grossprofit': '/EM_Funda/dat_whs/ab_grossprofit.csv',
})


class bt:
    @staticmethod
    def AZ_Path_create(target_path):
        """
        添加新路径
        :param target_path:
        :return:
        """
        if not os.path.exists(target_path):
            os.makedirs(target_path)

    @staticmethod
    def AZ_Load_csv(target_path, index_time_type=True):
        if index_time_type:
            target_df = pd.read_table(target_path, sep='|', index_col=0, low_memory=False, parse_dates=True)
        else:
            target_df = pd.read_table(target_path, sep='|', index_col=0, low_memory=False)
        return target_df

    @staticmethod
    def AZ_Rolling_mean(df, window, min_periods=0):
        target = df.rolling(window=window, min_periods=min_periods).mean()
        target.iloc[:window - 1] = np.nan
        return target

    @staticmethod
    def AZ_Sharpe_y(pnl_df):
        return round((np.sqrt(250) * pnl_df.mean()) / pnl_df.std(), 4)

    def AZ_Col_zscore(self, df, n, cap=None, min_periods=1):
        df_mean = self.AZ_Rolling_mean(df, n, min_periods=min_periods)
        df_std = df.rolling(window=n, min_periods=min_periods).std()
        target = (df - df_mean) / df_std
        if cap is not None:
            target[target > cap] = cap
            target[target < -cap] = -cap
        return target

    @staticmethod
    def AZ_Row_zscore(df, cap=None):
        df_mean = df.mean(axis=1)
        df_std = df.std(axis=1)
        target = df.sub(df_mean, axis=0).div(df_std, axis=0)
        if cap is not None:
            target[target > cap] = cap
            target[target < -cap] = -cap
        return target.replace(np.nan, 0)

    @staticmethod
    def AZ_Rolling(df, n, min_periods=0):
        return df.rolling(window=n, min_periods=min_periods)

    @staticmethod
    def AZ_Pot(pos_df, asset_last):
        """
        计算 pnl/turover*10000的值,衡量cost的影响
        :param pos_df: 仓位信息
        :param asset_last: 最后一天的收益
        :return:
        """
        pos_df = pos_df.fillna(0)
        trade_times = pos_df.diff().abs().sum().sum()
        if trade_times == 0:
            return 0
        else:
            pot = asset_last / trade_times * 10000
            return round(pot, 2)

    @staticmethod
    def AZ_Leverage_ratio(asset_df):
        """
        返回250天的return/(负的 一个月的return)
        :param asset_df:
        :return:
        """
        asset_20 = asset_df - asset_df.shift(20)
        asset_250 = asset_df - asset_df.shift(250)
        if asset_250.mean() > 0:
            return round(asset_250.mean() / (-asset_20.min()), 2)
        else:
            return round(asset_250.mean() / (-asset_20.max()), 2)

    def commit_check(self, pnl_df, mod='o'):
        """
        pnl_df
        :param pnl_df:要求DataFrame格式,其中index为时间格式,columns为pnl的名称
        :param mod: 'o':多空,'h':对冲
        :return:result_df包含corr,sp5,sp2,lv5,lv2,其中0表示不满足,1表示满足,
                info_df为具体数值
        """
        assert type(pnl_df) == pd.DataFrame
        all_pnl_df = pd.read_csv('/mnt/mfs/AATST/corr_tst_pnls', sep='|', index_col=0, parse_dates=True)
        all_pnl_df_c = pd.concat([all_pnl_df, pnl_df], axis=1)
        all_pnl_df_c_ma3 = self.AZ_Rolling(all_pnl_df_c, 3).mean().iloc[-1250:]
        matrix_corr_o = all_pnl_df_c_ma3.corr()[pnl_df.columns].drop(index=pnl_df.columns)

        matrix_sp5 = pnl_df.iloc[-1250:].apply(self.AZ_Sharpe_y)
        matrix_lv5 = pnl_df.iloc[-1250:].cumsum().apply(self.AZ_Leverage_ratio)

        matrix_sp2 = pnl_df.iloc[-500:].apply(self.AZ_Sharpe_y)
        matrix_lv2 = pnl_df.iloc[-500:].cumsum().apply(self.AZ_Leverage_ratio)

        info_df = pd.concat([matrix_corr_o.max(), matrix_sp5, matrix_sp2, matrix_lv5, matrix_lv2], axis=1)
        info_df.columns = ['corr', 'sp5', 'sp2', 'lv5', 'lv2']
        info_df = info_df.T

        if mod == 'h':
            cond_matrix = pd.DataFrame([[0.49, 1.90, 1.66, 1.70, 1.70],
                                        [0.59, 2.00, 1.75, 1.75, 1.75],
                                        [0.69, 2.10, 1.80, 1.80, 1.80]])
        else:
            cond_matrix = pd.DataFrame([[0.49, 2.00, 1.75, 2.00, 2.00],
                                        [0.59, 2.10, 1.85, 2.10, 2.10],
                                        [0.69, 2.25, 1.95, 2.20, 2.20]])

        def result_deal(x):
            for i in range(len(cond_matrix)):
                if x[0] <= cond_matrix.iloc[i, 0]:
                    corr, sp_5, sp_2, lv_5, lv_2 = cond_matrix.iloc[i]
                    res = x > [-1, sp_5, sp_2, lv_5, lv_2]
                    return res.astype(int)
            return [0, 0, 0, 0, 0]

        result_df = info_df.apply(result_deal)
        print('*******info_df*******')
        print(info_df)

        print('*******result_df*******')
        print(result_df)

        return result_df, info_df


bt = bt()


class DiscreteClass:
    """
    生成离散数据的公用函数
    """

    @staticmethod
    def pnd_con_ud(raw_df, sector_df, n_list):
        def fun(df, n):
            df_pct = df.diff()
            up_df = (df_pct > 0)
            dn_df = (df_pct < 0)
            target_up_df = up_df.copy()
            target_dn_df = dn_df.copy()

            for i in range(n - 1):
                target_up_df = target_up_df * up_df.shift(i + 1)
                target_dn_df = target_dn_df * dn_df.shift(i + 1)
            target_df = target_up_df.fillna(0).astype(int) - target_dn_df.fillna(0).astype(int)
            return target_df

        all_target_df = pd.DataFrame()
        for n in n_list:
            target_df = fun(raw_df, n)
            target_df = target_df * sector_df
            all_target_df = all_target_df.add(target_df, fill_value=0)
        return all_target_df

    @staticmethod
    def pnd_con_ud_pct(raw_df, sector_df, n_list):
        all_target_df = pd.DataFrame()
        for n in n_list:
            target_df = raw_df.rolling(window=n).apply(lambda x: 1 if (x >= 0).all() and sum(x) > 0
            else (-1 if (x <= 0).all() and sum(x) < 0 else 0))
            target_df = target_df * sector_df
            all_target_df = all_target_df.add(target_df, fill_value=0)
        return all_target_df

    @staticmethod
    def row_extre(raw_df, sector_df, percent):
        raw_df = raw_df * sector_df
        target_df = raw_df.rank(axis=1, pct=True)
        target_df[target_df >= 1 - percent] = 1
        target_df[target_df <= percent] = -1
        target_df[(target_df > percent) & (target_df < 1 - percent)] = 0
        return target_df

    @staticmethod
    def col_extre(raw_df, sector_df, window, percent, min_periods=1):
        dn_df = raw_df.rolling(window=window, min_periods=min_periods).quantile(percent)
        up_df = raw_df.rolling(window=window, min_periods=min_periods).quantile(1 - percent)
        dn_target = -(raw_df < dn_df).astype(int)
        up_target = (raw_df > up_df).astype(int)
        target_df = dn_target + up_target
        return target_df * sector_df

    @staticmethod
    def signal_fun(zscore_df, sector_df, limit):
        zscore_df[(zscore_df < limit) & (zscore_df > -limit)] = 0
        zscore_df[zscore_df >= limit] = 1
        zscore_df[zscore_df <= -limit] = -1
        return zscore_df * sector_df


class ContinueClass:
    """
    生成连续数据的公用函数
    """

    @staticmethod
    def roll_fun_20(raw_df, sector_df):
        return bt.AZ_Rolling_mean(raw_df, 20)

    @staticmethod
    def roll_fun_40(raw_df, sector_df):
        return bt.AZ_Rolling_mean(raw_df, 40)

    @staticmethod
    def roll_fun_100(raw_df, sector_df):
        return bt.AZ_Rolling_mean(raw_df, 100)

    @staticmethod
    def col_zscore(raw_df, sector_df, n, cap=5, min_periods=1):
        return bt.AZ_Col_zscore(raw_df, n, cap, min_periods)

    @staticmethod
    def row_zscore(raw_df, sector_df, cap=5):
        return bt.AZ_Row_zscore(raw_df * sector_df, cap)

    @staticmethod
    def pnd_vol(raw_df, sector_df, n):
        vol_df = bt.AZ_Rolling(raw_df, n).std() * (250 ** 0.5)
        return vol_df * sector_df

    @staticmethod
    def pnd_count_down(raw_df, sector_df, n):
        raw_df_mean = bt.AZ_Rolling_mean(raw_df, n) * sector_df
        raw_df_count_down = 1 / (raw_df_mean.replace(0, np.nan))
        return raw_df_count_down

    # return fun
    @staticmethod
    def pnd_return_volatility(adj_r, n):
        vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5)
        vol_df[vol_df < 0.08] = 0.08
        return vol_df

    @staticmethod
    def pnd_return_volatility_count_down(adj_r, sector_df, n):
        vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5) * sector_df
        vol_df[vol_df < 0.08] = 0.08
        return 1 / vol_df.replace(0, np.nan)

    @staticmethod
    def pnd_return_evol(adj_r, sector_df, n):
        vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5)
        vol_df[vol_df < 0.08] = 0.08
        evol_df = bt.AZ_Rolling(vol_df, 30).apply(lambda x: 1 if x[-1] > 2 * x.mean() else 0)
        return evol_df * sector_df


class SpecialClass:
    """
    某些数据使用的特殊函数
    """

    @staticmethod
    def pnd_evol(adj_r, sector_df, n):
        vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5)
        vol_df[vol_df < 0.08] = 0.08
        evol_df = bt.AZ_Rolling(vol_df, 30).apply(lambda x: 1 if x[-1] > 2 * x.mean() else 0)
        return evol_df * sector_df

    # @staticmethod
    # def ():
    #


class SectorFilter:
    def __init__(self, root_path):
        self.root_path = root_path

    def filter_market(self):
        market_df = bt.AZ_Load_csv(f'{self.root_path}/')

    def filter_vol(self):
        pass

    def filter_moment(self):
        pass


class BaseDeal(DiscreteClass, ContinueClass):
    @staticmethod
    def info_dict_fun(fun, raw_data_path, args, save_path, if_replace):
        info_dict = dict()
        info_dict['fun'] = fun
        info_dict['raw_data_path'] = raw_data_path
        info_dict['args'] = args
        info_dict['if_replace'] = if_replace
        pd.to_pickle(info_dict, save_path)

    def judge_save_fun(self, target_df, file_name, save_root_path, fun, raw_data_path, args, if_filter=True,
                       if_replace=False):
        factor_to_fun = '/mnt/mfs/dat_whs/data/factor_to_fun'
        if target_df.sum().sum() == 0:
            print('factor not enough!')
            return -1
        elif if_filter:
            print(f'{file_name}')
            print(target_df.iloc[-100:].abs().replace(0, np.nan).sum(axis=1).mean(), len(target_df.iloc[-100:].columns))
            print(
                target_df.iloc[-100:].abs().replace(0, np.nan).sum(axis=1).mean() / len(target_df.iloc[-100:].columns))
            target_df.to_pickle(os.path.join(save_root_path, file_name + '.pkl'))
            # 构建factor_to_fun的字典并存储
            self.info_dict_fun(fun, raw_data_path, args, os.path.join(factor_to_fun, file_name), if_replace)
            print(f'{file_name} success!')
            return 0
        else:
            target_df.to_pickle(os.path.join(save_root_path, file_name + '.pkl'))
            # 构建factor_to_fun的字典并存储
            self.info_dict_fun(fun, raw_data_path, args, os.path.join(factor_to_fun, file_name), if_replace)
            print(f'{file_name} success!')
            return 0


class SectorData:
    def __init__(self, root_path):
        self.root_path = root_path

    # 获取剔除新股的矩阵
    def get_new_stock_info(self, xnms, xinx):
        new_stock_data = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/CDSY_SECUCODE/LISTSTATE.csv')
        new_stock_data.fillna(method='ffill', inplace=True)
        # 获取交易日信息
        return_df = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/DERIVED_14/aadj_r.csv').astype(float)
        trade_time = return_df.index
        new_stock_data = new_stock_data.reindex(index=trade_time).fillna(method='ffill')
        target_df = new_stock_data.shift(40).notnull().astype(int)
        target_df = target_df.reindex(columns=xnms, index=xinx)
        return target_df

    # 获取剔除st股票的矩阵
    def get_st_stock_info(self, xnms, xinx):
        data = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/CDSY_CHANGEINFO/CHANGEA.csv')
        data = data.reindex(columns=xnms, index=xinx)
        data.fillna(method='ffill', inplace=True)

        data = data.astype(str)
        target_df = data.applymap(lambda x: 0 if 'ST' in x or 'PT' in x else 1)
        return target_df

    # 读取 sector(行业 最大市值等)
    def load_sector_data(self, begin_date, end_date, sector_name):
        if sector_name.startswith('index'):
            index_name = sector_name.split('_')[-1]
            market_top_n = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/IDEX_YS_WEIGHT_A/SECURITYNAME_{index_name}.csv')
            market_top_n = market_top_n.mask(market_top_n == market_top_n, other=1)
            print(market_top_n)
        else:
            market_top_n = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/DERIVED_10/{sector_name}.csv')

        market_top_n = market_top_n[(market_top_n.index >= begin_date) & (market_top_n.index < end_date)]
        market_top_n.dropna(how='all', axis='columns', inplace=True)

        xnms = market_top_n.columns
        xinx = market_top_n.index

        new_stock_df = self.get_new_stock_info(xnms, xinx)
        st_stock_df = self.get_st_stock_info(xnms, xinx)
        sector_df = market_top_n * new_stock_df * st_stock_df
        sector_df.replace(0, np.nan, inplace=True)
        return sector_df


class DataDeal(SectorData, DiscreteClass, ContinueClass):
    def __init__(self, begin_date, end_date, root_path, sector_name):
        super(DataDeal, self).__init__(root_path=root_path)
        # self.root_path = root_path
        self.sector_name = sector_name
        self.sector_df = self.load_sector_data(begin_date, end_date, sector_name)

        self.xinx = self.sector_df.index
        self.xnms = self.sector_df.columns

        self.save_root_path = '/mnt/mfs/dat_whs/data/factor_data'
        self.save_sector_path = f'{self.save_root_path}/{self.sector_name}'
        bt.AZ_Path_create(self.save_sector_path)

    def load_raw_data(self, file_name):
        data_path = base_data_dict[file_name]
        if len(data_path) != 0:
            raw_df = bt.AZ_Load_csv(f'{self.root_path}/{data_path}') \
                .reindex(index=self.xinx, columns=self.xnms).round(10)
        else:
            raw_df = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/daily/{file_name}.csv') \
                .reindex(index=self.xinx, columns=self.xnms).round(10)
        return raw_df

    def count_return_data(self, factor_name):
        file_name, fun_name, para_str = factor_name.split('|')
        str_to_num = lambda x: float(x) if '.' in x else int(x)
        para = [str_to_num(x) for x in para_str.split('_')]
        raw_df = self.load_raw_data(file_name)
        fun = getattr(self, fun_name)
        target_df = fun(raw_df, self.sector_df, *para)
        if len(para) != 0:
            para_str = '_'.join([str(x) for x in para])
            save_file = f'{file_name}|{fun_name}|{para_str}'
        else:
            save_file = f'{file_name}|{fun_name}'

        save_path = f'{self.save_sector_path}/{save_file}.pkl'
        return target_df


if __name__ == '__main__':
    sector_name, hold_time_str, if_only_long, percent_str = 'index_000300|20|False|0.1'.split('|')
    exe_str = 'R_AssetDepSales_QTTM|pnd_count_down|20_1.0@add_fun@' \
              'R_NetAssets_s_YOY_First|col_zscore|120_1.0@add_fun@' \
              'PE_TTM|col_zscore|20_-1.0@add_fun@' \
              'R_TotRev_s_POP_First|col_zscore|120_1.0@add_fun@' \
              'bulletin_num_df_60|pnd_count_down|20_1.0@add_fun@' \
              'R_MgtExp_sales_s_First|col_zscore|5_1.0@add_fun@' \
              'R_OperProfit_s_POP_First|row_zscore_1.0@add_fun@' \
              'lsgg_num_df_60|col_zscore|60_-1.0@add_fun@' \
              'R_EMPLOYEEPAY_QTTM|col_zscore|5_1.0@add_fun@' \
              'lsgg_num_df_60|row_zscore_-1.0@add_fun@' \
              'R_SUMLIAB_Y3YGR|pnd_count_down|60_1.0'
    root_path = '/media/hdd1/DAT_EQT'
    begin_date = pd.to_datetime('20120101')
    end_date = pd.to_datetime('20190411')
    end_date = datetime.now()

    data_deal = DataDeal(begin_date, end_date, root_path, sector_name)

    target_df = data_deal.count_return_data('R_EMPLOYEEPAY_QTTM|col_zscore|5')
