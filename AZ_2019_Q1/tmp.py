import os, sys
import pandas as pd
import numpy as np
from pathlib import Path

BASE_PATH = Path("/mnt/mfs/DAT_FUT/intraday/fut_1mbar")

HAND_PATH = Path("/mnt/mfs/temp/test3")
deal_file_name = os.listdir(HAND_PATH)


# BS = pd.read_csv("/mnt/mfs/DAT_FUT/intraday/fut_1mbar/IC/IC1905.csv", index_col=0, parse_dates=True, sep="|")

def add_data(file_name):
    a = pd.read_excel(HAND_PATH / file_name, encoding="gbk").rename(
        columns={'证券代码': "code", '证券名称': "name", '交易时间': "Time", '开盘价': "Open", '最高价': "High", '最低价': "Low",
                 '收盘价': "Close", '涨跌': "diff", '涨跌幅%': "differ", '成交量': "Volume", '成交额': "Turnover"})
    a.index = pd.to_datetime(a.Time.values)
    a = a.dropna(axis=0)
    a["Date"] = [x.strftime("%Y-%m-%d") for x in a.index]
    a["Time"] = [x.strftime('%H:%M') for x in a.index]

    a["OpenInterest"] = [np.nan for x in a.index]
    code = a.name[0]
    del (a["code"])
    del (a["name"])
    del (a["diff"])
    del (a["differ"])
    a = a.drop(index=[a.index[-1]])
    future_type = code[:2]
    ori_df = pd.read_csv(BASE_PATH / future_type / "{}.CFE".format(code), index_col=0, sep="|", parse_dates=True)
    f_df = ori_df.append(a, sort=False)
    f_df = f_df.fillna(method="ffill")
    return f_df


# for file_name  in deal_file_name:
#     add_data(file_name)
#
# file_path = "/mnt/mfs/temp/ic1906_04_24.xls"
#
# df = pd.read_excel(file_path, encoding="gbk").rename(
#     columns={'证券代码': "code", '证券名称': "name", '交易时间': "Time",
#              '开盘价': "Open", '最高价': "High", '最低价': "Low", '收盘价': "Close",
#              '涨跌': "diff", '涨跌幅%': "differ", '成交量': "Volume", '成交额': "Turnover"})
#
# df.index = pd.to_datetime(df.Time.values)
# df = df.dropna(axis=0)
# df["Date"] = [x.strftime("%Y-%m-%d") for x in df.index]
# df["Time"] = [x.strftime('%H:%M') for x in df.index]
#
# df["OpenInterest"] = [np.nan for x in df.index]
# code = df.name[0]
# del (df["code"])
# del (df["name"])
# del (df["diff"])
# del (df["differ"])
#
# ori_df = pd.read_csv(BASE_PATH / "IC/IC1906.CFE", index_col=0, sep="|", parse_dates=True)
# cut_ori_df_up = ori_df.loc[:df.index[0]]
# cut_ori_df_down = ori_df.loc[df.index[-1]:]
# final_df = cut_ori_df_up.append(df)
# final_df = final_df.append(cut_ori_df_down)
# final_df = final_df.fillna(method="ffill")
# final_df.to_csv(BASE_PATH / "IC/IC1906.CFE", index_label='TradeDate')
