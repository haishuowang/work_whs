import os
import pandas as pd
import open_lib.shared_tools.back_test as bt


sector_name = 'market_top_500'
use_factor = os.listdir('/mnt/mfs/dat_whs/data/factor_data/' + sector_name)
# bt.AZ_Path_create(use_factor)
# pd.to_pickle(use_factor, '/mnt/mfs/dat_whs/data/use_factor_set/' + sector_name + '')
