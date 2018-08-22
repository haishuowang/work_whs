import pandas as pd
import numpy as np
import time


def pnd_continue_ud_c(raw_df, n_list):
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
        all_target_df = all_target_df.add(target_df, fill_value=0)
    return all_target_df


# if __name__ == '__main__':
#     data = pd.read_csv('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv', sep='|', index_col=0, parse_dates=True) \
#         .round(4)
#     begin_1 = time.time()
#     a = pnd_continue_ud(data, [3])
#     end_1 = time.time()
#     print(end_1 - begin_1)
#     begin_2 = time.time()
#     b = pnd_continue_ud_c(data, [3])
#     end_2 = time.time()
#     print(end_2 - begin_2)

# aadj_r_vwap = pd.read_table('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r_vwap.csv', sep='|', parse_dates=True,
#                             index_col=0)
# aadj_r = pd.read_table('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv', sep='|', parse_dates=True, index_col=0)
