import pandas as pd
import numpy as np
import time
from multiprocessing import Pool
import os
from datetime import datetime, timedelta
import sys

sys.path.append('/mnt/mfs/work_whs/AZ_2018_Q2')
sys.path.append('/mnt/mfs')
sys.path.append("/mnt/mfs/LIB_ROOT")

import work_whs.funda_data as fd
from work_whs.funda_data import EM_Tab14, EM_Funda, Tech_Factor, EM_Funda_test, IntradayData
import open_lib.shared_paths.path as pt
import open_lib.shared_tools.back_test as bt

FundaBaseDeal = fd.funda_data_deal.FundaBaseDeal
SectorData = fd.funda_data_deal.SectorData

# class TechFactorDeal:
#     def __init__(self):
