import pandas as pd
from itertools import product, permutations, combinations
import random
from collections import OrderedDict
from datetime import datetime
import sys

sys.path.append('/mnt/mfs/work_whs')
sys.path.append('/mnt/mfs/work_whs/AZ_2018_Q2')
import loc_lib.shared_tools.back_test as bt
# 读取数据的函数 以及
from factor_script.script_load_data import load_index_data, load_sector_data, load_locked_data, load_pct, \
    load_part_factor, create_log_save_path, deal_mix_factor, deal_mix_factor_both, load_locked_data_both

from factor_script.script_filter_fun import pos_daily_fun, out_sample_perf, filter_all, filter_time_para_fun

data = pd.read_csv('/mnt/mfs/dat_whs/result/result/market_top_2000_True_20180822_1514_hold_20_aadj_r.txt',
                   sep='|', header=None)

data.columns = ['time_para', 'key', 'fun_name', 'name1', 'name2', 'name3', 'filter_fun_name', 'sector_name',
                'con_in', 'con_out_1', 'con_out_2', 'con_out_3', 'con_out_4', 'ic', 'sp_u', 'sp_m', 'sp_d',
                'pot_in', 'fit_ratio', 'leve_ratio', 'sp_out_1', 'sp_out_2', 'sp_out_3', 'sp_out_4']


def create_all_para():
    factor_list = ['CCI_p120d_limit_12',
                   'MACD_20_100',
                   'MACD_40_200',
                   'R_CFO_TotRev_s_First_row_extre_0.3',
                   'R_CFO_s_YOY_First_row_extre_0.3',
                   'R_COMPANYCODE_First_row_extre_0.3',
                   'R_Cashflow_s_YOY_First_row_extre_0.3',
                   'R_EPS_s_First_row_extre_0.3',
                   'R_EPS_s_YOY_First_row_extre_0.3',
                   'R_NetAssets_s_YOY_First_row_extre_0.3',
                   'R_NetInc_TotProfit_s_First_row_extre_0.3',
                   'R_NetInc_s_First_row_extre_0.3',
                   'R_OPCF_NetInc_s_First_row_extre_0.3',
                   'R_OperProfit_s_YOY_First_row_extre_0.3',
                   'R_ParentProfit_s_POP_First_row_extre_0.3',
                   'R_ParentProfit_s_YOY_First_row_extre_0.3',
                   'R_ROENetIncRecur_s_First_row_extre_0.3',
                   'R_RevenueTotPS_s_First_row_extre_0.3',
                   'R_Revenue_s_YOY_First_row_extre_0.3',
                   'R_Tax_TotProfit_s_First_row_extre_0.3',
                   'R_TotAssets_s_YOY_First_row_extre_0.3',
                   'R_TotRev_s_POP_First_row_extre_0.3',
                   'R_TotRev_s_YOY_First_row_extre_0.3',
                   'TVOL_p120d_col_extre_0.2',
                   'TVOL_p20d_col_extre_0.2',
                   'aadj_r_p345d_continue_ud_pct',
                   'bias_turn_p120d',
                   'bias_turn_p20d',
                   'evol_p20d',
                   'log_price_0.2',
                   'moment_p20100d',
                   'price_p120d_hl',
                   'price_p20d_hl',
                   'return_p60d_0.2',
                   'turn_p120d_0.2',
                   'turn_p20d_0.2',
                   'vol_count_down_p60d',
                   'vol_p20d',
                   'vol_p60d',
                   'volume_moment_p530d']
    target_list = list(combinations(sorted(factor_list), 3))
    return target_list


month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

begin_date = pd.to_datetime('20100101')
cut_date = pd.to_datetime('20160401')
end_date = pd.to_datetime('20180401')

sector_name = 'market_top_2000'

para_list = create_all_para()
total_para_num = len(para_list)
# sector
sector_df = load_sector_data(begin_date, end_date, sector_name)

xnms = sector_df.columns
xinx = sector_df.index

# suspend or limit up_dn
suspendday_df, limit_buy_sell_df = load_locked_data_both(xnms, xinx)

# return
return_choose = pd.read_table('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv', sep='|', index_col=0) \
    .astype(float)
return_choose.index = pd.to_datetime(return_choose.index)
return_choose = return_choose.reindex(columns=xnms, index=xinx, fill_value=0)

# index data
index_df = load_index_data(xinx, index_name)
