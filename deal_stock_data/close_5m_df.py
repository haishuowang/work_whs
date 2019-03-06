import pandas as pd
import numpy as np

a = pd.read_pickle('/mnt/mfs/DAT_PUBLIC/intraday/special/close_5m_2010-2017.pkl')
target_df = pd.DataFrame()
for key in sorted(list(a.keys())):
    print(key)
    target_df = pd.concat([target_df, a[key]], axis=1)
    a.pop(key)

