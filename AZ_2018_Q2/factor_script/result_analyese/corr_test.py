import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *


def get_all_pnl_corr(pnl_df, col_name):
    all_pnl_df = pd.read_csv('/mnt/mfs/AATST/corr_tst_pnls', sep='|', index_col=0, parse_dates=True)
    all_pnl_df_c = pd.concat([all_pnl_df, pnl_df], axis=1)
    a = all_pnl_df_c.iloc[-600:].corr()[col_name]
    print(a[a > 0.6])
    return a


result_file_name_list = [
    'market_top_300to800plus_True_20181228_1404_hold_5__16'
]
for result_file_name in result_file_name_list:
    pnl_df = pd.read_csv(f'/mnt/mfs/dat_whs/tmp_pnl_file/{result_file_name}.csv', index_col=0, parse_dates=True)
    pnl_df.columns = [result_file_name]
    print(get_all_pnl_corr(pnl_df, result_file_name))