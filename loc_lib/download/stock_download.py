import pandas as pd
from EmQuantAPI import *
from datetime import timedelta, datetime
import time

loginResult = c.start("ForceLogin=1")

# data = c.csd("000016.SH", "OPEN,CLOSE,HIGH,LOW,VOLUME,TAFACTOR,FRONTTAFACTOR", "2018-06-08", "2018-06-29",
#              "period=1,adjustflag=1,curtype=1,pricetype=1,order=1,market=CNSESH,BaseDate=2018-06-29")
data = c.csd("159005.OF",
             "UNITNAV,ACCUMULATEDNAV,ADJUSTEDNAV,UNITNAVRATE,ACCUMULATEDNAVRATE,OPEN,CLOSE,HIGH,LOW,PRECLOSE"
             , "2018-06-12", "2018-07-03", "period=1,adjustflag=1,curtype=1,pricetype=1,order=1,market=CNSESH")
target_df = pd.DataFrame(data.Data["159005.OF"], index=data.Indicators, columns=data.Dates).T
outResult = c.stop()
