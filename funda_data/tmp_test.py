import pandas as pd
import numpy as np
import funda_data.Tech_Factor_intra
from funda_data.funda_data_deal import SectorData
import loc_lib.shared_paths.path as pt
import loc_lib.shared_tools.back_test as bt
import os
from datetime import datetime, timedelta
from multiprocessing import Pool