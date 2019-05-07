import sys
import os
sys.path.append('/mnt/mfs')
import work_whs.loc_lib.shared_tools.back_test as bt
from work_whs.loc_lib.pre_load.plt import savfig_send

import numpy as np
import matplotlib.pyplot as plt


class SignalAnalysis:
    @staticmethod
    def CDF(signal_df, raw_return_df, hold_time, title='CDF Figure', lag=2):
        signal_df = signal_df.shift(lag).values
        return_df = bt.AZ_Rolling_sum(raw_return_df, hold_time).shift(-hold_time + 1).values
        f_return_m = return_df - return_df.mean()
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
