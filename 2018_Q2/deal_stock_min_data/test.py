from EmQuantAPI import *
from datetime import timedelta, datetime
import time
import copy
import numpy as np
import pandas as pd
from multiprocessing import Lock


def mainCallback(quantdata):
    """
    mainCallback 是主回调函数，可捕捉如下错误
    在start函数第三个参数位传入，该函数只有一个为c.EmQuantData类型的参数quantdata
    :param quantdata:c.EmQuantData
    :return:
    """
    print("mainCallback", str(quantdata))

    # 登录掉线或者 登陆数达到上线（即登录被踢下线） 这时所有的服务都会停止
    if str(quantdata.ErrorCode) == "10001011" or str(quantdata.ErrorCode) == "10001009":
        print("Your account is disconnect. You can force login automatically here if you need.")

    # 行情登录验证失败（每次连接行情服务器时需要登录验证）或者行情流量验证失败时，会取消所有订阅，用户需根据具体情况处理
    elif str(quantdata.ErrorCode) == "10001021" or str(quantdata.ErrorCode) == "10001022":
        print("Your all csq subscribe have stopped.")

    # 行情服务器断线自动重连连续6次失败（1分钟左右）不过重连尝试还会继续进行直到成功为止，遇到这种情况需要确认两边的网络状况
    elif str(quantdata.ErrorCode) == "10002009":
        print("Your all csq subscribe have stopped.")
    else:
        pass


# def cstCallBack(quantdata):
#     stock_name = list(quantdata.Data.keys())[0]
#     data_array = np.array(quantdata.Data[stock_name])
#     print(data_array)
#     save_data[stock_name] = data_array
#     print(save_data)

def cstCallBack(quantdata):
    stock_name = list(quantdata.Data.keys())[0]
    # save_data['name'] = stock_name
    save_data.add(stock_name)
    print(stock_name)
    data_array = np.array(quantdata.Data[stock_name])
    tmp_df = pd.DataFrame(data_array.reshape([6, int(len(data_array) / 6)]).T,
                          columns=['Time', 'Now', 'HIGH', 'LOW', 'Volume', 'Amount'])

    Close_df.loc[end_time, stock_name] = tmp_df['Now'].iloc[-1]
    Volume_df.loc[end_time, stock_name] = tmp_df['Volume'].iloc[-1]
    Amount_df.loc[end_time, stock_name] = tmp_df['Amount'].iloc[-1]
    Vwap_df.loc[end_time, stock_name] = (tmp_df['Now'] * tmp_df['Volume'].diff()).sum() / tmp_df['Volume'].diff().sum()


def update_time(begin, end, stock_codes):
    global Close_df, High_df, Low_df, Volume_df, Amount_df, Vwap_df, save_data
    save_data = set()
    Close_df = pd.DataFrame(index=[end_time], columns=stock_codes)
    High_df = pd.DataFrame(index=[end_time], columns=stock_codes)
    Low_df = pd.DataFrame(index=[end_time], columns=stock_codes)
    Volume_df = pd.DataFrame(index=[end_time], columns=stock_codes)
    Amount_df = pd.DataFrame(index=[end_time], columns=stock_codes)
    Vwap_df = pd.DataFrame(index=[end_time], columns=stock_codes)
    data = c.cst(','.join(stock_codes), 'Time, Now, HIGH, LOW, Volume, Amount', begin, end, "", cstCallBack)
    # 等待时间
    time.sleep(5)
    while True:
        print(set(stock_codes) - save_data)
        if set(stock_codes) - save_data == set():
            print(save_data)
            break
    print(Close_df)
    return Close_df, High_df, Low_df, Volume_df, Amount_df, Vwap_df


# begin_time = "100000"
# end_time = "100010"
# loginResult = c.start("ForceLogin=1", '', mainCallback)
# if loginResult.ErrorCode != 0:
#     print("login in fail")
#     exit()
#
# now_date = datetime.now()
# stock_codes = c.sector("001004", now_date.strftime('%Y%m%d')).Codes[:100]
# Close_df, High_df, Low_df, Volume_df, Amount_df, Vwap_df = update_time(begin_time, end_time, stock_codes)
# print(Close_df)


# Close_df = pd.DataFrame(index=[end_time], columns=stock_codes)
# Volume_df = pd.DataFrame(index=[end_time], columns=stock_codes)
# Amount_df = pd.DataFrame(index=[end_time], columns=stock_codes)
# Vwap_df = pd.DataFrame(index=[end_time], columns=stock_codes)
# Close_df = pd.DataFrame(columns=stock_codes)
# Close_df.to_csv('/mnt/mfs/dat_whs/data/intra_close.csv')
# Close_df.to_csv('/mnt/mfs/dat_whs/data/intra_volume.csv')
# Close_df.to_csv('/mnt/mfs/dat_whs/data/intra_amount.csv')
# Close_df.to_csv('/mnt/mfs/dat_whs/data/intra_vwap.csv')
