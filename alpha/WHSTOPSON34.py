import pandas as pd
import numpy as np
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt
import sys

sys.path.append("/mnt/mfs/LIB_ROOT")
from open_lib.shared_tools import send_email
from itertools import combinations
from datetime import datetime
from collections import OrderedDict
import time
import random
import string


def plot_send_result(pnl_df, sharpe_ratio, subject, text=''):
    figure_save_path = os.path.join('/mnt/mfs/dat_whs', 'tmp_figure')
    plt.figure(figsize=[16, 8])
    plt.plot(pnl_df.index, pnl_df.cumsum(), label='sharpe_ratio={}'.format(sharpe_ratio))
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(figure_save_path, '{}.png'.format(subject)))
    plt.close()
    to = ['whs@yingpei.com']
    filepath = [os.path.join(figure_save_path, '{}.png'.format(subject))]
    send_email.send_email(text, to, filepath, subject)


def get_code():
    return ''.join(random.sample(string.ascii_letters + string.digits, 32))


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
    def AZ_Load_csv(target_path, parse_dates=True, index_col=0, sep='|', **kwargs):
        target_df = pd.read_table(target_path, sep=sep, index_col=index_col, low_memory=False,
                                  parse_dates=parse_dates, **kwargs)
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
        # df_mean = self.AZ_Rolling_mean(df, n, min_periods=min_periods).round(4)
        # df_std = df.rolling(window=n, min_periods=min_periods).std().round(4).replace(0, np.nan)
        df_mean = self.AZ_Rolling_mean(df, n, min_periods=min_periods).round(4)
        df_std = df.rolling(window=n, min_periods=min_periods).std().round(4).replace(0, np.nan)
        target = (df - df_mean) / df_std
        if cap is not None:
            target[target > cap] = cap
            target[target < -cap] = -cap
        return target

    @staticmethod
    def AZ_Row_zscore(df, cap=None):
        df_mean = df.mean(axis=1)
        df_std = df.std(axis=1).replace(0, np.nan)
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
        # print('*******info_df*******')
        print(info_df)

        # print('*******result_df*******')
        print(result_df)

        return result_df, info_df


bt = bt()
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

    'PEG_EBIT_3Y': '/EM_Funda/DERIVED_EVA/PEG_precast/PEG_EBIT_3Y.csv',
    'PEG_EBIT_5Y': '/EM_Funda/DERIVED_EVA/PEG_precast/PEG_EBIT_5Y.csv',
    'PEG_OPCF_3Y': '/EM_Funda/DERIVED_EVA/PEG_precast/PEG_OPCF_3Y.csv',
    'PEG_OPCF_5Y': '/EM_Funda/DERIVED_EVA/PEG_precast/PEG_OPCF_5Y.csv',
    'PEG_OPERATEREVE_3Y': '/EM_Funda/DERIVED_EVA/PEG_precast/PEG_OPERATEREVE_3Y.csv',
    'PEG_OPERATEREVE_5Y': '/EM_Funda/DERIVED_EVA/PEG_precast/PEG_OPERATEREVE_5Y.csv',
    'PEG_PARENTNETPROFIT_3Y': '/EM_Funda/DERIVED_EVA/PEG_precast/PEG_PARENTNETPROFIT_3Y.csv',
    'PEG_PARENTNETPROFIT_5Y': '/EM_Funda/DERIVED_EVA/PEG_precast/PEG_PARENTNETPROFIT_5Y.csv',
})


class FunSet:
    @staticmethod
    def div(a, b):
        """
        ratio -～|+~
        :param a:
        :param b:
        :return:
        """
        return a.div(b)

    @staticmethod
    def div_2(a, b):
        """
        ratio -～|+~
        :param a:
        :param b:
        :return:
        """
        return a.div(b.sub(b.min(axis=1) - 1, axis=0))

    @staticmethod
    def diff(a, n):
        """
        diff
        :param a:
        :param n:
        :return:
        """
        return a.diff(n)

    @staticmethod
    def pct_change(a, n):
        """
        ratio
        :param a:
        :param n:
        :return:
        """
        return a.pct_change(n)

    @staticmethod
    def add(a, b):
        """
        no mul +1
        :param a:
        :param b:
        :return:
        """
        return a.add(b)

    @staticmethod
    def mul(a, b):
        """
        no
        :param a:
        :param b:
        :return:
        """
        return a.mul(b)

    @staticmethod
    def sub(a, b):
        """
        diff
        :param a:
        :param b:
        :return:
        """
        return a.sub(b)

    @staticmethod
    def std(a, n):
        """
        std
        :param a:
        :param n:
        :return:
        """
        return bt.AZ_Rolling(a, n).std()

    @staticmethod
    def max_fun(a, n):
        """
        no
        :param a:
        :param n:
        :return:
        """
        return bt.AZ_Rolling(a, n).max()

    @staticmethod
    def min_fun(a, n):
        """
        no
        :param a:
        :param n:
        :return:
        """
        return bt.AZ_Rolling(a, n).min()

    @staticmethod
    def rank_pct(a):
        """
        continue 0|1
        :param a:
        :return:
        """
        return a.rank(axis=1, pct=True)

    @staticmethod
    def corr(a, b, n):
        """
        线性相关性
        continue -1|1
        :param a:
        :param b:
        :param n:
        :return:
        """

        return bt.AZ_Rolling(a, n).corr(b)

    def corr_rank(self, a, b, n):
        """
        线性相关性
        continue -1|1
        :param a:
        :param b:
        :param n:
        :return:
        """
        a = self.rank_pct(a)
        b = self.rank_pct(b)
        return bt.AZ_Rolling(a, n).corr(b)

    @staticmethod
    def ma(a, n):
        """
        no
        :param a:
        :param n:
        :return:
        """
        return bt.AZ_Rolling(a, n).mean()

    @staticmethod
    def shift(a, n):
        """
        no
        :param a:
        :param n:
        :return:
        """
        return a.shift(n)

    @staticmethod
    def abs_fun(a):
        """
        no 0|
        :param a:
        :return:
        """
        return np.abs(a)


class FactorTestBase(FunSet):
    def __init__(self, root_path, if_save, if_new_program, begin_date, end_date, sector_name,
                 hold_time, lag, return_file, if_hedge, if_only_long):
        self.root_path = root_path
        self.if_save = if_save
        self.if_new_program = if_new_program
        self.begin_date = begin_date
        self.end_date = end_date
        self.sector_name = sector_name
        self.hold_time = hold_time
        self.lag = lag
        self.return_file = return_file
        self.if_hedge = if_hedge
        self.if_only_long = if_only_long

        if sector_name.startswith('market_top_300plus') \
                or sector_name.startswith('index_000300'):
            if_weight = 1
            ic_weight = 0

        elif sector_name.startswith('market_top_300to800plus') \
                or sector_name.startswith('index_000905'):
            if_weight = 0
            ic_weight = 1

        else:
            if_weight = 0.5
            ic_weight = 0.5

        self.if_weight = if_weight
        self.ic_weight = ic_weight
        return_df = self.load_return_data()
        self.xinx = return_df.index
        sector_df = self.load_sector_data()
        self.xnms = sector_df.columns

        return_df = return_df.reindex(columns=self.xnms)
        self.sector_df = sector_df.reindex(index=self.xinx)
        # print('Loaded sector DataFrame!')
        if if_hedge:
            if ic_weight + if_weight != 1:
                exit(-1)
        else:
            if_weight = 0
            ic_weight = 0

        index_df_1 = self.load_index_data('000300').fillna(0)
        index_df_2 = self.load_index_data('000905').fillna(0)
        hedge_df = if_weight * index_df_1 + ic_weight * index_df_2
        self.return_df = return_df.sub(hedge_df, axis=0)
        # print('Loaded return DataFrame!')

        suspendday_df, limit_buy_sell_df = self.load_locked_data()
        limit_buy_sell_df_c = limit_buy_sell_df.shift(-1)
        limit_buy_sell_df_c.iloc[-1] = 1

        suspendday_df_c = suspendday_df.shift(-1)
        suspendday_df_c.iloc[-1] = 1
        self.suspendday_df_c = suspendday_df_c
        self.limit_buy_sell_df_c = limit_buy_sell_df_c
        # print('Loaded suspendday_df and limit_buy_sell DataFrame!')
        alpha_name = os.path.basename(__file__).split('.')[0]
        self.tmp_mix_path = f'/media/hdd1/DAT_PreCalc/PreCalc_whs/tmp/{alpha_name}'
        self.mix_factor_name_dict = {}
        self.factor_way_dict = {}

    @staticmethod
    def row_zscore(raw_df, sector_df, cap=5):
        return bt.AZ_Row_zscore(raw_df * sector_df, cap)

    def reindex_fun(self, df):
        return df.reindex(index=self.xinx, columns=self.xnms)

    @staticmethod
    def create_log_save_path(target_path):
        top_path = os.path.split(target_path)[0]
        if not os.path.exists(top_path):
            os.mkdir(top_path)
        if not os.path.exists(target_path):
            os.mknod(target_path)

    @staticmethod
    def row_extre(raw_df, sector_df, percent):
        raw_df = raw_df * sector_df
        target_df = raw_df.rank(axis=1, pct=True)
        target_df[target_df >= 1 - percent] = 1
        target_df[target_df <= percent] = -1
        target_df[(target_df > percent) & (target_df < 1 - percent)] = 0
        return target_df

    @staticmethod
    def pos_daily_fun(df, n=5):
        return df.rolling(window=n, min_periods=1).sum()

    def check_factor(self, name_list, file_name, check_path=None):
        if check_path is None:
            load_path = os.path.join('/mnt/mfs/dat_whs/data/new_factor_data/' + self.sector_name)
        else:
            load_path = check_path
        exist_factor = set([x[:-4] for x in os.listdir(load_path)])
        use_factor = set(name_list)
        a = use_factor - exist_factor
        if len(a) != 0:
            # print('factor not enough!')
            send_email.send_email(f'{file_name} factor not enough!', ['whs@yingpei.com'], [], 'Factor Test Warning!')

    @staticmethod
    def create_all_para(tech_name_list, funda_name_list):

        target_list_1 = []
        for tech_name in tech_name_list:
            for value in combinations(funda_name_list, 2):
                target_list_1 += [[tech_name] + list(value)]

        target_list_2 = []
        for funda_name in funda_name_list:
            for value in combinations(tech_name_list, 2):
                target_list_2 += [[funda_name] + list(value)]

        target_list = target_list_1 + target_list_2
        return target_list

    # 获取剔除新股的矩阵
    def get_new_stock_info(self, xnms, xinx):
        target_df = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/DERIVED_01/NewStock.csv')
        target_df = target_df.reindex(columns=xnms, index=xinx)
        return target_df

    # 获取剔除st股票的矩阵
    def get_st_stock_info(self, xnms, xinx):
        target_df = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/DERIVED_01/StAndPtStock.csv')
        target_df = target_df.reindex(columns=xnms, index=xinx)
        return target_df

    def load_return_data(self):
        return_df = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Funda/DERIVED_14/aadj_r.csv'))
        return_df = return_df[(return_df.index >= self.begin_date) & (return_df.index < self.end_date)]
        return return_df

    # 获取sector data
    def load_sector_data(self):
        if self.sector_name.startswith('index'):
            index_name = self.sector_name.split('_')[-1]
            market_top_n = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/IDEX_YS_WEIGHT_A/SECURITYNAME_{index_name}.csv')
            market_top_n = market_top_n.where(market_top_n != market_top_n, other=1)
        else:
            market_top_n = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/DERIVED_10/{self.sector_name}.csv')

        market_top_n = market_top_n.reindex(index=self.xinx)
        market_top_n.dropna(how='all', axis='columns', inplace=True)

        xnms = market_top_n.columns
        xinx = market_top_n.index

        new_stock_df = self.get_new_stock_info(xnms, xinx)
        st_stock_df = self.get_st_stock_info(xnms, xinx)
        sector_df = market_top_n * new_stock_df * st_stock_df
        sector_df.replace(0, np.nan, inplace=True)
        return sector_df

    def load_index_weight_data(self, index_name):
        index_info = bt.AZ_Load_csv(self.root_path + f'/EM_Funda/IDEX_YS_WEIGHT_A/SECURITYNAME_{index_name}.csv')
        index_info = self.reindex_fun(index_info)
        index_mask = (index_info.notnull() * 1).replace(0, np.nan)

        mkt_cap = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Funda/LICO_YS_STOCKVALUE/AmarketCapExStri.csv'))
        mkt_roll = mkt_cap.rolling(250, min_periods=0).mean()
        mkt_roll = self.reindex_fun(mkt_roll)

        mkt_roll_qrt = np.sqrt(mkt_roll)
        mkt_roll_qrt_index = mkt_roll_qrt * index_mask
        index_weight = mkt_roll_qrt_index.div(mkt_roll_qrt_index.sum(axis=1), axis=0)
        return index_weight

    # 涨跌停都不可交易
    def load_locked_data(self):
        suspendday_df = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/DERIVED_01/SuspendedStock.csv') \
            .reindex(columns=self.xnms, index=self.xinx)
        limit_buy_sell_df = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/DERIVED_01/LimitedBuySellStock.csv') \
            .reindex(columns=self.xnms, index=self.xinx)
        return suspendday_df, limit_buy_sell_df

    # 获取index data
    def load_index_data(self, index_name):
        data = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/DERIVED_WHS/CHG_{index_name}.csv', header=None)
        target_df = data.iloc[:, 0].reindex(index=self.xinx)
        return target_df

    def signal_to_pos(self, signal_df):
        # 下单日期pos
        order_df = signal_df.replace(np.nan, 0)
        # 排除入场场涨跌停的影响
        order_df = order_df * self.sector_df * self.limit_buy_sell_df_c * self.suspendday_df_c
        order_df = order_df.div(order_df.abs().sum(axis=1).replace(0, np.nan), axis=0)
        order_df[order_df > 0.05] = 0.05
        order_df[order_df < -0.05] = -0.05
        daily_pos = bt.AZ_Rolling_mean(order_df, self.hold_time)
        daily_pos.fillna(0, inplace=True)
        # 排除出场涨跌停的影响
        daily_pos = daily_pos * self.limit_buy_sell_df_c * self.suspendday_df_c
        daily_pos.fillna(method='ffill', inplace=True)
        return daily_pos

    def signal_to_pos_ls(self, signal_df, ls_para):
        if ls_para == 'l':
            signal_df_up = signal_df[signal_df > 0]
            daily_pos = self.signal_to_pos(signal_df_up)
        elif ls_para == 's':
            signal_df_dn = signal_df[signal_df < 0].abs()
            daily_pos = self.signal_to_pos(signal_df_dn)
        elif ls_para == 'ls':
            daily_pos = self.signal_to_pos(signal_df)
        else:
            daily_pos = self.signal_to_pos(signal_df)
        return daily_pos

    @staticmethod
    def judge_way(sharpe):
        if sharpe > 0:
            return 1
        elif sharpe < 0:
            return -1
        else:
            return 0

    def load_raw_data(self, file_name):
        data_path = base_data_dict[file_name]
        if len(data_path) != 0:
            raw_df = bt.AZ_Load_csv(f'{self.root_path}/{data_path}') \
                .reindex(index=self.xinx, columns=self.xnms).round(4)
        else:
            raw_df = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/daily/{file_name}.csv') \
                .reindex(index=self.xinx, columns=self.xnms).round(4)
        return raw_df

    def load_mix_data(self, file_name):
        file_name_str = str(file_name)
        if type(file_name) is str:
            target_df = self.load_raw_data(file_name)
        elif file_name_str in self.mix_factor_name_dict.keys():
            mix_name = self.mix_factor_name_dict[file_name_str]
            target_df = pd.read_pickle(f'{self.tmp_mix_path}/{self.sector_name}/{mix_name}')
        else:
            mix_name = get_code()

            fun_name, factor_name_list, para_list = file_name
            factor_df_list = [self.load_mix_data(factor_name) for factor_name in factor_name_list]
            target_df = getattr(self, fun_name)(*factor_df_list, *para_list)
            # zcore 根据用到的数据增加权重
            target_df = self.row_zscore(target_df, self.sector_df) * np.sqrt(len(factor_name_list))

            self.mix_factor_name_dict[file_name_str] = mix_name
            bt.AZ_Path_create(f'{self.tmp_mix_path}/{self.sector_name}')
            pd.to_pickle(target_df, f'{self.tmp_mix_path}/{self.sector_name}/{mix_name}')
        return target_df

    def back_test(self, data_df, cut_date, percent, return_pos=False, ls_para='ls'):
        cut_time = pd.to_datetime(cut_date)
        signal_df = self.row_extre(data_df, self.sector_df, percent)
        if len(signal_df.abs().sum(1).replace(0, np.nan).dropna()) / len(self.xinx) > 0.7:
            pos_df = self.signal_to_pos_ls(signal_df, ls_para)
            pnl_table = pos_df.shift(self.lag) * self.return_df
            pnl_df = pnl_table.sum(1)
            sample_in_index = (pnl_df.index < cut_time)
            sample_out_index = (pnl_df.index >= cut_time)

            pnl_df_in = pnl_df[sample_in_index]
            pnl_df_out = pnl_df[sample_out_index]

            pos_df_in = pos_df[sample_in_index]
            pos_df_out = pos_df[sample_out_index]

            sp_in = bt.AZ_Sharpe_y(pnl_df_in)
            sp_out = bt.AZ_Sharpe_y(pnl_df_out)

            pot_in = bt.AZ_Pot(pos_df_in, pnl_df_in.sum())
            pot_out = bt.AZ_Pot(pos_df_out, pnl_df_out.sum())

            sp = bt.AZ_Sharpe_y(pnl_df)
            pot = bt.AZ_Pot(pos_df, pnl_df.sum())
            if self.if_only_long:
                if ls_para == 'l':
                    way_in, way_out, way = 1, 1, 1
                elif ls_para == 's':
                    way_in, way_out, way = -1, -1, -1
                else:
                    way_in, way_out, way = self.judge_way(sp_in), self.judge_way(sp_out), self.judge_way(sp)
            else:
                way_in, way_out, way = self.judge_way(sp_in), self.judge_way(sp_out), self.judge_way(sp)
            result_list = [sp_in, sp_out, sp, pot_in, pot_out, pot, way_in, way_out, way]
            info_df = pd.Series(result_list, index=['sp_in', 'sp_out', 'sp',
                                                    'pot_in', 'pot_out', 'pot',
                                                    'way_in', 'way_out', 'way'])
        else:
            info_df = pd.Series([0] * 9, index=['sp_in', 'sp_out', 'sp',
                                                'pot_in', 'pot_out', 'pot',
                                                'way_in', 'way_out', 'way'])
            pnl_df = pd.Series([0] * len(self.xinx), index=self.xinx)
            pos_df = pd.DataFrame(columns=data_df.columns, index=data_df.index)
        if return_pos:
            return info_df, pnl_df, pos_df
        else:
            return info_df, pnl_df


class FactorTestResult(FactorTestBase):
    def __init__(self, *args):
        super(FactorTestResult, self).__init__(*args)

    def load_mix_data(self, file_name):
        if type(file_name) is str:
            target_df = self.load_raw_data(file_name)
        else:
            fun_name, factor_name_list, para_list = file_name
            factor_df_list = [self.load_mix_data(factor_name) for factor_name in factor_name_list]
            target_df = getattr(self, fun_name)(*factor_df_list, *para_list)
            # zcore 根据用到的数据增加权重
            target_df = self.row_zscore(target_df, self.sector_df)  # * np.sqrt(len(factor_name_list))
        return target_df

    def get_pnl_df(self, file_name, cut_date, percent):
        data_df = self.load_mix_data(file_name)
        if self.if_only_long:
            info_df_l, pnl_df_l, pos_df_l = self.back_test(data_df, cut_date, percent, ls_para='l', return_pos=True)
            pnl_df_l.name = str(file_name)
            info_df_l.name = str(file_name)

            info_df_s, pnl_df_s, pos_df_s = self.back_test(data_df, cut_date, percent, ls_para='s', return_pos=True)
            pnl_df_s.name = str(file_name)
            info_df_s.name = str(file_name)
            if info_df_l['sp_in'] > info_df_s['sp_in']:
                info_df = info_df_l
                pnl_df = pnl_df_l
                pos_df = pos_df_l
            else:
                info_df = info_df_s
                pnl_df = pnl_df_s
                pos_df = pos_df_s
        else:
            info_df, pnl_df, pos_df = self.back_test(data_df, cut_date, percent, return_pos=True)
            pnl_df.name = str(file_name)
            info_df.name = str(file_name)
        self.factor_way_dict[str(file_name)] = info_df['way_in']
        return info_df, pnl_df, pos_df


def main_fun(base_info_str, exe_list):
    root_path = '/media/hdd1/DAT_EQT'

    sector_name, hold_time_str, if_only_long, percent_str = base_info_str.split('|')

    hold_time = int(hold_time_str)
    percent = float(percent_str)
    if if_only_long == 'False':
        if_only_long = False
    else:
        if_only_long = True
    alpha_name = os.path.basename(__file__).split('.')[0]

    if_save = True
    if_new_program = True

    begin_date = pd.to_datetime('20130101')
    end_date = datetime.now()
    cut_date = pd.to_datetime('20180101')
    lag = 2
    return_file = ''

    if_hedge = True

    factor_test = FactorTestResult(root_path, if_save, if_new_program, begin_date, end_date, sector_name, hold_time,
                                   lag, return_file, if_hedge, if_only_long)

    info_df, pnl_df, pos_df = factor_test.get_pnl_df(exe_list, cut_date, percent)

    # 相关性测试
    bt.commit_check(pd.DataFrame(pnl_df), mod='h')
    annual_r = pnl_df.sum() / pos_df.abs().sum(1).sum() * 250
    print(annual_r)
    plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df),
                     f'[new framewor result]{sector_name}_{if_only_long} {round(annual_r, 4)}',
                     )

    if factor_test.if_weight != 0:
        pos_df['IF01'] = -factor_test.if_weight * pos_df.sum(axis=1)
    if factor_test.ic_weight != 0:
        pos_df['IC01'] = -factor_test.ic_weight * pos_df.sum(axis=1)

    pos_df.round(5).fillna(0).to_csv(f'/mnt/mfs/AAPOS/{alpha_name}.pos', sep='|', index_label='Date')
    return info_df, pnl_df, pos_df


if __name__ == '__main__':
    a = time.time()
    base_info_str = 'index_000300|20|True|0.1'
    exe_list = ['add', (['add', (['sub', (['add', (['add', (['add', (['add', (['add', (['add', (['add', (['sub', (['sub', (['add', (['sub', (['add', ('stock_tab2_9', ['max_fun', ('stock_tab4_2',), (127,)]), ()], ['min_fun', ('aadj_p',), (41,)]), ()], ['pct_change', ('R_OperProfit_s_YOY_First',), (127,)]), ()], ['ma', ('PEG_PARENTNETPROFIT_5Y',), (41,)]), ()], ['abs_fun', ('PEG_PARENTNETPROFIT_3Y',), ()]), ()], ['mul', (['min_fun', ('aadj_p',), (41,)], 'R_GSCF_TTM_QSD4Y'), ()]), ()], ['add', ('stock_tab2_9', ['abs_fun', ('R_EPS_s_First',), ()]), ()]), ()], ['div', ('stock_tab2_9', 'PEG_PARENTNETPROFIT_3Y'), ()]), ()], ['sub', (['sub', (['add', (['sub', (['add', ('stock_tab2_9', ['max_fun', ('stock_tab4_2',), (127,)]), ()], ['min_fun', ('aadj_p',), (41,)]), ()], ['pct_change', ('R_OperProfit_s_YOY_First',), (127,)]), ()], ['ma', ('PEG_PARENTNETPROFIT_5Y',), (41,)]), ()], ['abs_fun', ('PEG_PARENTNETPROFIT_3Y',), ()]), ()]), ()], ['mul', (['sub', (['max_fun', ('PEG_PARENTNETPROFIT_3Y',), (127,)], ['abs_fun', ('R_EPS_s_First',), ()]), ()], ['mul', (['max_fun', ('aadj_p_HIGH',), (5,)], ['min_fun', ('PEG_PARENTNETPROFIT_3Y',), (41,)]), ()]), ()]), ()], ['add', (['sub', (['sub', (['add', (['sub', (['add', ('stock_tab2_9', ['max_fun', ('stock_tab4_2',), (127,)]), ()], ['min_fun', ('aadj_p',), (41,)]), ()], ['pct_change', ('R_OperProfit_s_YOY_First',), (127,)]), ()], ['ma', ('PEG_PARENTNETPROFIT_5Y',), (41,)]), ()], ['abs_fun', ('PEG_PARENTNETPROFIT_3Y',), ()]), ()], ['div', ('stock_tab2_9', 'PEG_PARENTNETPROFIT_3Y'), ()]), ()]), ()], ['diff', (['min_fun', (['sub', (['pct_change', ('stock_tab2_9',), (127,)], 'stock_tab2_9'), ()],), (17,)],), (5,)]), ()], ['mul', (['sub', (['sub', (['max_fun', ('PEG_PARENTNETPROFIT_3Y',), (127,)], ['abs_fun', ('R_EPS_s_First',), ()]), ()], ['add', ('stock_tab2_9', ['abs_fun', ('R_EPS_s_First',), ()]), ()]), ()], ['min_fun', (['abs_fun', ('stock_tab2_9',), ()],), (41,)]), ()]), ()], ['add', (['add', (['add', (['add', (['add', (['sub', (['sub', (['add', (['sub', (['add', ('stock_tab2_9', ['max_fun', ('stock_tab4_2',), (127,)]), ()], ['min_fun', ('aadj_p',), (41,)]), ()], ['pct_change', ('R_OperProfit_s_YOY_First',), (127,)]), ()], ['ma', ('PEG_PARENTNETPROFIT_5Y',), (41,)]), ()], ['abs_fun', ('PEG_PARENTNETPROFIT_3Y',), ()]), ()], ['mul', (['min_fun', ('aadj_p',), (41,)], 'R_GSCF_TTM_QSD4Y'), ()]), ()], ['add', ('stock_tab2_9', ['abs_fun', ('R_EPS_s_First',), ()]), ()]), ()], ['div', ('stock_tab2_9', 'PEG_PARENTNETPROFIT_3Y'), ()]), ()], ['sub', (['sub', (['add', (['sub', (['add', ('stock_tab2_9', ['max_fun', ('stock_tab4_2',), (127,)]), ()], ['min_fun', ('aadj_p',), (41,)]), ()], ['pct_change', ('R_OperProfit_s_YOY_First',), (127,)]), ()], ['ma', ('PEG_PARENTNETPROFIT_5Y',), (41,)]), ()], ['abs_fun', ('PEG_PARENTNETPROFIT_3Y',), ()]), ()]), ()], ['min_fun', (['add', ('stock_tab2_9', ['abs_fun', ('R_EPS_s_First',), ()]), ()],), (127,)]), ()]), ()], ['mul', (['add', (['sub', (['sub', (['add', (['sub', (['add', ('stock_tab2_9', ['max_fun', ('stock_tab4_2',), (127,)]), ()], ['min_fun', ('aadj_p',), (41,)]), ()], ['pct_change', ('R_OperProfit_s_YOY_First',), (127,)]), ()], ['ma', ('PEG_PARENTNETPROFIT_5Y',), (41,)]), ()], ['abs_fun', ('PEG_PARENTNETPROFIT_3Y',), ()]), ()], ['div', ('stock_tab2_9', 'PEG_PARENTNETPROFIT_3Y'), ()]), ()], ['sub', (['sub', (['pct_change', ('stock_tab2_9',), (127,)], 'stock_tab2_9'), ()], ['add', ('stock_tab2_9', ['abs_fun', ('R_EPS_s_First',), ()]), ()]), ()]), ()]), ()]
    info_df, pnl_df, pos_df = main_fun(base_info_str, exe_list)
    b = time.time()
    print(b-a)
