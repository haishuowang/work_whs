import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *

mysql_name = 'crawl'
usr_name = 'yp_labman_whs'
pass_word = 'whs300742'


def fill_index(raw_df, root_path, window):
    raw_df = raw_df[bt.AZ_split_stock(raw_df.columns)]
    return_df = pd.read_csv(f'{root_path}/EM_Funda/DERIVED_14/aadj_r.csv',
                            sep='|', index_col=0, parse_dates=True)
    date_index = return_df.index

    xinx = sorted(list(set(raw_df.index) | set(date_index)))
    raw_df_c = raw_df.reindex(index=xinx)
    raw_df_c_mean = raw_df_c.rolling(window=window, min_periods=0).sum()
    target_df = raw_df_c_mean.reindex(date_index)
    return target_df


def class_bulletintype(StockId):
    engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.23:7777/{mysql_name}?charset=utf8')
    conn = engine.connect()
    try:
        exe_str = f'SELECT * FROM crawl.StockBulletin_sina WHERE StockId="{StockId}"'
        print(exe_str)
        data = pd.read_sql(exe_str, conn)
        # data = pd.read_sql('SELECT * FROM crawl.StockBulletin_sina', conn)
        data = data.sort_values(by=['ReportDate'])
        notice_type_list = sorted(list(set(data['BulletinType'])))
        # notice_type = notice_type_list[0]
        return_df = pd.Series(name=StockId, index=notice_type_list)
        for notice_type in notice_type_list:
            part_data = data[data['BulletinType'] == notice_type][['BulletinTitle', 'BulletinType']]
            return_df[notice_type] = len(part_data.index)
        return return_df
    except Exception as e:
        print(e)
        send_email.send_email('', ['whs@yingpei.com'], [], 'mysql_error')
        return None


def get_bulletin_num_table():
    engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.23:7777/{mysql_name}?charset=utf8')
    conn = engine.connect()
    StockId_df = pd.read_sql('SELECT StockId FROM crawl.StockBulletin_sina', conn)
    StockId_list = sorted(list(set(StockId_df.values.ravel())))[:3]
    pool = Pool(10)
    result_list = []
    for StockId in StockId_list:
        print(StockId)
        result_list.append(pool.apply_async(class_bulletintype, args=(StockId,)))
        # result_list.append(class_bulletintype(StockId))
    pool.close()
    pool.join()

    target_df = pd.DataFrame()
    for x in result_list:
        target_df = pd.concat([target_df, x.get()], axis=1)
        # target_df = pd.concat([target_df, x], axis=1)
    target_df.to_pickle('/mnt/mfs/dat_whs/notice_mysql.pkl')


def get_lsgg_table(root_path):
    engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.23:7777/{mysql_name}?charset=utf8')
    conn = engine.connect()
    lsgg_df = pd.read_sql('SELECT StockId, ReportDate FROM crawl.StockBulletin_sina WHERE BulletinType="LSGG"', conn)
    # LSGG_df.to_pickle('/mnt/mfs/dat_whs/tmp/LSGG.pkl')
    lsgg_df['mark'] = 1
    lsgg_num_df = lsgg_df.groupby(['ReportDate', 'StockId'])['mark'].sum().unstack()
    lsgg_num_df.index = pd.to_datetime(lsgg_num_df.index)
    lsgg_num_df.columns = bt.AZ_add_stock_suffix(lsgg_num_df.columns)
    window_list = [5, 20, 60]
    for window in window_list:
        lsgg_num_df_c = fill_index(lsgg_num_df, root_path, window)
        lsgg_num_df_c.to_csv(f'/mnt/mfs/dat_whs/EM_Funda/my_data_test/lsgg_num_df_{window}.csv', sep='|')


def get_bulletin_table(root_path):
    engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.23:7777/{mysql_name}?charset=utf8')
    conn = engine.connect()
    lsgg_df = pd.read_sql('SELECT StockId, ReportDate FROM crawl.StockBulletin_sina', conn)
    # LSGG_df.to_pickle('/mnt/mfs/dat_whs/tmp/LSGG.pkl')
    lsgg_df['mark'] = 1
    lsgg_num_df = lsgg_df.groupby(['ReportDate', 'StockId'])['mark'].sum().unstack()
    lsgg_num_df.index = pd.to_datetime(lsgg_num_df.index)
    lsgg_num_df.columns = bt.AZ_add_stock_suffix(lsgg_num_df.columns)
    lsgg_num_df.to_csv('/mnt/mfs/dat_whs/EM_Funda/my_data_test/bulletin_num_df.csv', sep='|')
    # window_list = [5, 20, 60]
    # for window in window_list:
    #     lsgg_num_df_c = fill_index(lsgg_num_df, root_path, window)
    #     lsgg_num_df_c.to_csv(f'/mnt/mfs/dat_whs/EM_Funda/my_data_test/bulletin_num_df_{window}.csv', sep='|')


def date_deal_fun(x):
    cut_date = datetime(year=x.year, month=x.month, day=x.day, hour=7, minute=00)
    if cut_date>x:
        # print(datetime(year=x.year, month=x.month, day=x.day) - timedelta(days=1))
        return datetime(year=x.year, month=x.month, day=x.day) - timedelta(days=1)
    else:
        # print(datetime(year=x.year, month=x.month, day=x.day))
        return datetime(year=x.year, month=x.month, day=x.day)


def get_news_table(root_path):
    engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.23:7777/{mysql_name}?charset=utf8')
    conn = engine.connect()
    news_df = pd.read_sql('SELECT StockId, NewsDate FROM crawl.StockNews_xueqiu', conn)
    news_df['mark'] = 1
    news_df['NewsDate'] = [date_deal_fun(x) for x in news_df['NewsDate']]
    news_num_df = news_df.groupby(['NewsDate', 'StockId'])['mark'].sum().unstack()
    news_num_df.index = pd.to_datetime(news_num_df.index)
    news_num_df.columns = bt.AZ_add_stock_suffix(news_num_df.columns)

    news_num_df.to_csv('/mnt/mfs/dat_whs/EM_Funda/my_data_test/news_num_df.csv', sep='|')

    # window_list = [5, 20, 60]
    # for window in window_list:
    #     news_num_df_c = fill_index(news_num_df, root_path, window)
    #     news_num_df_c.to_csv(f'/mnt/mfs/dat_whs/EM_Funda/my_data_test/news_num_df_{window}.csv', sep='|')


if __name__ == '__main__':
    root_path = '/mnt/mfs/DAT_EQT'
    print('process begin')
    a = time.time()
    # get_lsgg_table(root_path)
    get_bulletin_table(root_path)
    get_news_table(root_path)
    b = time.time()
    print(b - a)
    print('get all stock id')
