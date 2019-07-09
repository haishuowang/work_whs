import numpy as np
import pandas as pd
import os
import sys
from sqlalchemy import create_engine
from itertools import product, permutations, combinations
from datetime import datetime, timedelta
import time
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
import random
import matplotlib
from collections import OrderedDict
from multiprocessing import Pool, Lock
from multiprocessing.dummy import Pool as ThreadPool
import re
import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.shared_tools import send_email
import work_whs.loc_lib.shared_tools.back_test as bt
from collections import Counter


i = pd.date_range('2018-04-09', periods=4, freq='1D20min')
ts = pd.DataFrame({'A': [1,2,3,4]}, index=i)

# 分钟
begin_time = '0:15'
end_time = '0:45'

ts.between_time(begin_time, end_time)

# b = pd.Series([2] * 6)
# a = pd.DataFrame([[1, 2, 3, 4, 5, 6]]*10).T
# a.le(b, axis='index')
