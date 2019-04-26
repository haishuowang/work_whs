__author__ = 'jerry'

import pandas as pd
import datetime as datetime
import re
import os
from sqlalchemy import create_engine
from multiprocessing import Pool


def connect_sql():
    # Step.1 åˆ›å»ºæ•°æ®åº“é“¾æŽ¥
    cfg = {
        "db_type": "mysql",
        "db_driver": "pymysql",
        "db_ip": "192.168.16.28",
        "db_port": 7777,
        "account": "yp_intern_ydb",
        "password": "yp_intern_ydb",
    }
    cfg_as_str = "{db_type}+{db_driver}://{account}:{password}@{db_ip}:{db_port}".format(**cfg)
    engine = create_engine(cfg_as_str, connect_args={"charset": "utf8"})  # ä½¿ç”¨utf-8å­—ç¬¦é›†
    return engine


def read_contract_info_today(engine, today_date) -> pd.DataFrame:
    sql = f"SELECT ContractId, StartTradeDate, LastTradeDate " \
          f"FROM crawl.FutureInfo_em t " \
          f"WHERE t.StartTradeDate <= '{today_date}' " \
          f"AND t.LastTradeDate >= '{today_date}'"
    return pd.read_sql(sql, engine)


def today_loadfromsql(contract_id, start_time, end_time) -> pd.DataFrame:
    engine = connect_sql()
    year = end_time.year
    sql = f"SELECT ContractId, TradeDate, Open, High, Low, Close, Volume, Turnover, OpenInterest, Settlement " \
          f"FROM crawl_intraday.FutureQuote1_{year}_em t " \
          f"WHERE t.ContractId = '{contract_id}' " \
          f"AND t.TradeDate BETWEEN '{start_time}' AND '{end_time}'"

    tmp_df = pd.read_sql(sql, engine)
    if len(tmp_df) == 0:
        return None
    else:
        tmp_df_time = tmp_df.TradeDate.astype(str).str.split(' ', expand=True)
        tmp_df.TradeDate = tmp_df.TradeDate.astype(str).str.slice(stop=16)
        tmp_df = tmp_df.assign(Date=tmp_df_time.iloc[:, 0], Time=tmp_df_time.iloc[:, 1].str.slice(stop=5))
        new_columns = ['TradeDate', 'Date', 'Time', 'Open', 'High', 'Low',
                       'Close', 'Volume', 'Turnover', 'OpenInterest', 'Settlement']
        return tmp_df[new_columns]


def SaveSafe(new_data, now_path):
    path_split = os.path.split(now_path)
    new_path = path_split[0] + '/.' + path_split[1] + '.temp'
    new_data.to_csv(new_path, sep='|', index=False, encoding='utf-8')
    os.rename(new_path, now_path)


def updateIntraday(contract_id, start_time, end_time, save_path):
    try:
        category_tmp = re.search(r'\D{1,2}', contract_id).group()

        new_data = today_loadfromsql(contract_id, start_time, end_time)

        if new_data is None:
            print('{} not updated since no data collected from Mysql..\n'.format(contract_id))
        else:
            new_data = new_data.sort_values(['TradeDate'])
            now_path = save_path + '/' + category_tmp + '/' + contract_id
            if os.path.exists(now_path):
                old_data = pd.read_csv(now_path, sep='|')
                if (min(new_data.TradeDate) <= min(old_data.TradeDate)) or (
                        max(new_data.TradeDate) <= max(old_data.TradeDate)):
                    print('{} : No new data colleted from Mysql....\n'.format(contract_id))
                    return -1
                old_data = old_data.loc[old_data.TradeDate < min(new_data.TradeDate)]
                new_data = old_data.append(new_data)
                # return new_data
            SaveSafe(new_data, now_path)
    except Exception as e:
        print(e)


def main(today_date):
    sql_engine = connect_sql()
    today_contract_info = read_contract_info_today(sql_engine, today_date)
    today_contract_info = today_contract_info.assign(category=list(map(lambda x: re.search(r'\D{1,2}', x).group(), today_contract_info.ContractId)))
    today_contract_info = today_contract_info.loc[~today_contract_info.category.isin(['IC', 'IH', 'IF', 'T', 'TS', 'TF']),]

    save_path = '/mnt/mfs/DAT_FUT/intraday/fut_1mbar'

    assumed_start_time = today_date - datetime.timedelta(hours=4)
    assumed_end_time = today_date + datetime.timedelta(hours=16)

    print('Daily Contract all collected, update begins...... \n')

    # updateIntraday('AG2004.SHF', assumed_start_time, assumed_end_time, save_path)

    pool = Pool(10)
    for ii in range(len(today_contract_info)):
        pool.apply_async(updateIntraday, args=(today_contract_info.ContractId.iloc[ii],
                                               assumed_start_time,
                                               assumed_end_time,
                                               save_path,))
    pool.close()
    pool.join()

    # missing SC LC1907 ??????, sth happen
    print('All contracts processed, intraday data has been updated. \n')


if __name__ == '__main__':
    today_date = pd.to_datetime(datetime.datetime.today().date())
    # 6点
    main(today_date)
