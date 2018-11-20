import pandas as pd
import numpy as np
import datetime as dt
import calendar as cl
from datetime import datetime
from pandas import Series, DataFrame

ps = pd.read_pickle('/mnt/mfs/DAT_PUBLIC/bara_data/raw_data/PS_TTM.pkl')  # 用ps的timelist
datetest = ps.index.strftime('%Y-%m-%d')  # 将行名转化为%Y-%m-%d日期格式
dttest = np.array([dt.datetime.strptime(x, '%Y-%m-%d') for x in datetest])
S = datetime.strptime('2005-1-4', '%Y-%m-%d')  # 从05年到１８年4月底
E = datetime.strptime('2018-4-27', '%Y-%m-%d')
dtlist = dttest[(dttest <= E) & (dttest >= S)]


def AZ_filter_stock(stock_list):  # 筛选相应股票池
    target_list = [x for x in stock_list if x[:2] == 'SH' and x[2] == '6' or
                   x[:2] == 'SZ' and x[2] in ['0', '3']]
    return target_list


# 首先初始化一个dataframe用于之后concat
# Up_factor=Series(np.repeat(0,len(AZ_filter_stock(Close0.columns))),index=AZ_filter_stock(Close0.columns)).to_frame()
Up_factor = Down_factor = Up_3bars_factor = Down_3bars_factor = Daily_vwap = Up_t3vol_vwap = Down_t3vol_vwap = Up_div_down = Up_div_daily = Down_div_daily = Up_t3rtn_vwap = Down_t3rtn_vwap = pd.DataFrame()

for date in dtlist:
    date = date.strftime('%Y%m%d')
    Close = pd.read_table(f'/mnt/mfs/DAT_PUBLIC/intraday/eqt_1mbar/{date[:4]}/{date[:6]}/{date}/Close.csv', sep=',',
                          index_col=0)
    Volume = pd.read_table(f'/mnt/mfs/DAT_PUBLIC/intraday/eqt_1mbar/{date[:4]}/{date[:6]}/{date}/Volume.csv', sep=',',
                           index_col=0)
    # mindex=[datetime.strptime(date+x,'%Y%m%d%H:%M') for x in Open.index]

    Eclose = Close[AZ_filter_stock(Close.columns)]
    Evolume = Volume[AZ_filter_stock(Volume.columns)]

    # 涨跌bar的close*volume之和
    Eclose1 = Eclose.shift(1, axis=0)
    Updown = Eclose.sub(Eclose1)
    up_table = (Updown > 0) * 1  # 所有涨的数据点１
    down_table = (Updown < 0) * 1  # 所有跌的数据点１
    up_sum = Eclose.mul(up_table).mul(Evolume)
    down_sum = Eclose.mul(down_table).mul(Evolume)
    up_factor = up_sum.sum(axis=0)
    down_factor = down_sum.sum(axis=0)  # 涨跌close*volume 两因子  里面无涨跌的就直接是０,无NA值

    # 连涨连跌3个bar的close price
    up_bars = up_table.rolling(window=3, axis=0).sum()  # 每隔3个取和
    up_3bar = (up_bars == 3) * 1  # 是否连涨3个bars的0-1矩阵
    up_3bar_num = up_3bar.sum(axis=0)  # 连涨3个bars的数量
    up_3bars_factor = Eclose.mul(up_3bar).sum(axis=0) / up_3bar_num  # 无连涨的直接变成NA

    down_bars = down_table.rolling(window=3, axis=0).sum()
    down_3bar = (down_bars == 3) * 1  # 是否连跌3个bars的0-1矩阵
    down_3bar_num = down_3bar.sum(axis=0)  # 连跌3个bars的数量
    down_3bars_factor = Eclose.mul(down_3bar).sum(axis=0) / down_3bar_num  # 无连跌的直接变成NA

    # 算daily vwap:
    # daily_vwap=Eclose.mul(Evolume).sum()/Evolume.sum()

    # 后期最好还是用1mbar的数据算daily vwap:
    Eclose_1m = Close[AZ_filter_stock(Close)]
    Evolume_1m = Volume[AZ_filter_stock(Volume)]
    daily_vwap = Eclose.mul(Evolume).sum() / Evolume_1m.sum()

    # 涨跌　最大三个volume的vwap　与一天vwap比值
    up_volume = Evolume.mul(up_table)
    down_volume = Evolume.mul(down_table)  # 所有跌的volume

    up_volume_rank = up_volume.rank(ascending=False, na_option='bottom', axis=0)  # 对所有涨的Volume Rank
    up_volume_01top3 = (up_volume_rank < 4) * 1  # 并列第三的也要, 0-1 matrix
    up_volume_top3 = up_volume.mul(up_volume_01top3)  # 0-1矩阵诚意volume
    up_top3_vwap = Eclose.mul(up_volume_top3).sum() / up_volume_top3.sum()  # 存在NA值
    up_t3vol_vwap = up_top3_vwap / daily_vwap  # len(up_t3vol_vwap[up_t3vol_vwap>1])     明显比小于１的数量多

    down_volume_rank = down_volume.rank(ascending=False, na_option='bottom', axis=0)
    down_volume_01top3 = (down_volume_rank < 4) * 1  # 并列第三的也要, 0-1 matrix
    down_volume_top3 = down_volume.mul(down_volume_01top3)  # 0-1矩阵诚意volume
    down_top3_vwap = Eclose.mul(down_volume_top3).sum() / down_volume_top3.sum()  # 存在NA值
    down_t3vol_vwap = down_top3_vwap / daily_vwap  # len(down_t3vol_vwap[down_t3vol_vwap<1])   明显比大于１的数量多

    # 涨Vwap/跌Vwap -1
    up_all_vwap = up_factor / up_volume.sum()  # 存在十几个NA值
    down_all_vwap = down_factor / down_volume.sum()
    up_div_down = (up_all_vwap / down_all_vwap) - 1  # 存在二十个NA值

    # 涨跌Vwap/当天Vwap -1
    up_div_daily = (up_all_vwap / daily_vwap) - 1  # 绝大多数>0
    down_div_daily = (down_all_vwap / daily_vwap) - 1  # 绝大多数<0

    # 每5分钟return中最大３个和最小三个对应的VWAP/daily_vwap
    Updown_return = Updown.div(Eclose)
    up_return_rank = Updown_return.rank(ascending=False, na_option='bottom', axis=0)
    up_return_top3 = Evolume.mul((up_return_rank < 4) * 1)
    up_return_top3_vwap = Eclose.mul(up_return_top3).sum() / up_return_top3.sum()
    up_t3rtn_vwap = up_return_top3_vwap / daily_vwap  # len(up_t3rtn_vwap[up_t3rtn_vwap>1])   绝大多数ｕ

    down_return_rank = Updown_return.rank(ascending=True, na_option='bottom', axis=0)
    down_return_top3 = Evolume.mul((down_return_rank < 4) * 1)
    down_return_top3_vwap = Eclose.mul(down_return_top3).sum() / down_return_top3.sum()
    down_t3rtn_vwap = down_return_top3_vwap / daily_vwap  # len(down_t3rtn_vwap[down_t3rtn_vwap<1]) 绝大多数

    # 批量修改名字，赋值给该series
    up_factor.name = down_factor.name = up_3bars_factor.name = down_3bars_factor.name = daily_vwap.name = \
        up_t3vol_vwap.name = down_t3vol_vwap.name = up_div_down.name = up_div_daily.name = \
        down_div_daily.name = up_t3rtn_vwap.name = down_t3rtn_vwap.name = date

    Up_factor = pd.concat([Up_factor, up_factor], axis=1)
    Down_factor = pd.concat([Down_factor, down_factor], axis=1)
    Up_3bars_factor = pd.concat([Up_3bars_factor, up_3bars_factor], axis=1)
    Down_3bars_factor = pd.concat([Down_3bars_factor, down_3bars_factor], axis=1)
    Daily_vwap = pd.concat([Daily_vwap, daily_vwap], axis=1)
    Up_t3vol_vwap = pd.concat([Up_t3vol_vwap, up_t3vol_vwap], axis=1)
    Down_t3vol_vwap = pd.concat([Down_t3vol_vwap, down_t3vol_vwap], axis=1)
    Up_div_down = pd.concat([Up_div_down, up_div_down], axis=1)
    Up_div_daily = pd.concat([Up_div_daily, up_div_daily], axis=1)
    Down_div_daily = pd.concat([Down_div_daily, down_div_daily], axis=1)
    Up_t3rtn_vwap = pd.concat([Up_t3rtn_vwap, up_t3rtn_vwap], axis=1)
    Down_t3rtn_vwap = pd.concat([Down_t3rtn_vwap, down_t3rtn_vwap], axis=1)
    print(date)

route5m = '/mnt/mfs/DAT_PUBLIC/bara_data/result_1mbar/'
Up_factor.T.to_csv(route5m + 'Up_factor.csv')
Down_factor.T.to_csv(route5m + 'Down_factor.csv')
Up_3bars_factor.T.to_csv(route5m + 'Up_3bars_factor.csv')
Down_3bars_factor.T.to_csv(route5m + 'Down_3bars_factor.csv')
Daily_vwap.T.to_csv(route5m + 'Daily_vwap.csv')
Up_t3vol_vwap.T.to_csv(route5m + 'Up_t3vol_vwap.csv')
Down_t3vol_vwap.T.to_csv(route5m + 'Down_t3vol_vwap.csv')
Up_div_down.T.to_csv(route5m + 'Up_div_down.csv')
Up_div_daily.T.to_csv(route5m + 'Up_div_daily.csv')
Down_div_daily.T.to_csv(route5m + 'Down_div_daily.csv')
Up_t3rtn_vwap.T.to_csv(route5m + 'Up_t3rtn_vwap.csv')
Down_t3rtn_vwap.T.to_csv(route5m + 'Down_t3rtn_vwap.csv')

uf = pd.read_table(route5m + 'Up_t3rtn_vwap.csv', sep=',', index_col=0)

'''
def up01(col):           #too slow
    col[col>0]=1
    col[col<0]=0
    return col
up_table=up_table.apply(up01,axis=0)
'''

'''
EQT_list = AZ_filter_stock(Turnover.columns)
Turnover_EQT =Turnover[EQT_list]
a=(Turnover_EQT<0).sum(axis=1)
a.sum()
windex=a[a>=1].index
b=Turnover_EQT.loc[windex[0]]
b[b<0]

mzeros=[np.nan]
for ld in lastday[:-3]:    #检查Volume里面有无负数
    date=ld.strftime('%Y%m%d')
    Volume0=pd.read_table(f'/mnt/mfs/DAT_PUBLIC/intraday/eqt_1mbar/{date[:4]}/{date[:6]}/{date}/Volume.csv',sep=',',index_col=0)
    mzeros+=[(Volume0<0).sum(axis=1).sum()]
'''
