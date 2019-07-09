import sys
import os
import pandas as pd

sys.path.append('/mnt/mfs')
import work_whs.loc_lib.shared_tools.back_test as bt
from work_whs.loc_lib.pre_load.plt import savfig_send

import numpy as np
import matplotlib.pyplot as plt


class SignalAnalysis:
    @staticmethod
    def CDF(raw_signal_df, raw_return_df, hold_time, title='CDF Figure', lag=2, zero_drop=False):
        raw_signal_df = raw_signal_df.shift(lag).values
        if zero_drop:
            filter_mask = pd.notna(raw_signal_df) & (raw_signal_df != 0)
        else:
            filter_mask = pd.notna(raw_signal_df)

        signal_df = raw_signal_df[filter_mask]
        raw_return_df_sum = bt.AZ_Rolling_sum(raw_return_df, hold_time).shift(-hold_time + 1).values
        return_df = raw_return_df_sum[filter_mask]

        f_return_m = return_df - np.nanmean(return_df)
        a = np.argsort(signal_df)
        plt.figure(figsize=(10, 6))
        p1 = plt.subplot(221)
        p2 = plt.subplot(222)
        p3 = plt.subplot(223)
        p4 = plt.subplot(224)
        p1.plot(np.cumsum(return_df[a]))
        p1.set_title('cumsum return')
        p1.grid(1)

        p2.plot(signal_df[a], np.cumsum(return_df[a]))
        p2.set_title('signal and cumsum return')
        p2.grid(1)

        p3.plot(np.cumsum(f_return_m[a]))
        p3.set_title('cumsum mean return')
        p3.grid(1)

        p4.plot(signal_df[a], np.cumsum(f_return_m[a]))
        p4.set_title('signal and cumsum mean return')
        p4.grid(1)

        plt.suptitle(title)
        savfig_send(subject=title, text='')

    @staticmethod
    def CDF_c(signal_df, raw_return_df, hold_time, title='CDF Figure', lag=2):
        signal_df = signal_df.shift(lag).replace('np.nan')
        a = np.argsort(signal_df.values)
        return_df = bt.AZ_Rolling_sum(raw_return_df, hold_time).shift(-hold_time + 1)
        f_return_m = return_df - np.nanmean(return_df)
        plt.figure(figsize=(20, 6))
        p1 = plt.subplot(121)
        p2 = plt.subplot(122)

        p1.plot(signal_df.iloc[a].values, np.cumsum(return_df.iloc[a].values))
        # p1.plot(np.cumsum(return_df[a]))

        # p1.set_xticklabels(signal_df[a].values)
        ticks_list = signal_df.iloc[a].values[[int(len(signal_df[a]) / 14) * i - 1 for i in range(1, 15)]]
        print(ticks_list)

        p1.set_xticks(ticks_list)
        # p1.vlines(x='13:30', ymin=np.inf, ymax=np.inf)
        p1.set_title('cumsum return')
        p1.grid(1)

        p2.plot(signal_df.iloc[a].values, np.cumsum(f_return_m.iloc[a]))
        # p2.plot(np.cumsum(f_return_m[a]))
        # p2.set_xticklabels(signal_df[a].values)
        p2.set_xticks(ticks_list)
        p2.set_title('cumsum mean return')
        p2.grid(1)

        plt.suptitle(title)
        savfig_send(subject=title, text='')

    @staticmethod
    def corr_plot(x, y):
        pass
