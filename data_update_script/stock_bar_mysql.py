import sys
from multiprocessing import Pool

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *

mysql_name = 'crawl'
usr_name = 'yp_labman_whs'
pass_word = 'whs300742'


def date_deal_fun(x, cut_hour=7):
    cut_date = datetime(year=x.year, month=x.month, day=x.day, hour=cut_hour, minute=00)
    if cut_date > x:
        # print(datetime(year=x.year, month=x.month, day=x.day) - timedelta(days=1))
        return datetime(year=x.year, month=x.month, day=x.day) - timedelta(days=1)
    else:
        # print(datetime(year=x.year, month=x.month, day=x.day))
        return datetime(year=x.year, month=x.month, day=x.day)


def part_deal_fun(stock_id, target_df):
    print(stock_id)
    engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.10:7777/{mysql_name}?charset=utf8')
    conn = engine.connect()
    stock_df = pd.read_sql(f"SELECT FirstPostTime FROM StockBar_em WHERE StockId='{stock_id}'", conn)
    stock_df['FirstPostTime'] = [date_deal_fun(x) for x in stock_df['FirstPostTime']]
    stock_df['StockId'] = stock_id
    stock_df['mark'] = 1
    part_df = stock_df.groupby(['FirstPostTime', 'StockId'])['mark'].sum().unstack()
    # print(part_df)
    target_df = pd.concat([target_df, part_df], axis=1)
    return target_df


def part_deal_fun_mul(stock_id, cut_hour):
    print(stock_id)
    engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.10:7777/{mysql_name}?charset=utf8')
    conn = engine.connect()
    stock_df = pd.read_sql(f"SELECT FirstPostTime FROM StockBar_em WHERE StockId='{stock_id}'", conn)
    stock_df['FirstPostTime'] = [date_deal_fun(x, cut_hour) for x in stock_df['FirstPostTime']]
    stock_df['StockId'] = stock_id
    stock_df['mark'] = 1
    part_df = stock_df.groupby(['FirstPostTime', 'StockId'])['mark'].sum().unstack()
    # print(part_df)
    # target_df = pd.concat([target_df, part_df], axis=1)
    return part_df


def get_bar_num(save_root_path, cut_hour):
    engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.10:7777/{mysql_name}?charset=utf8')
    conn = engine.connect()
    all_stock_id = pd.read_sql('SELECT DISTINCT StockId '
                               'FROM StockBar_em', conn)

    # target_df = pd.DataFrame()
    # for stock_id in all_stock_id.values.ravel():
    #     target_df = part_deal_fun(stock_id, target_df)

    result_list = []
    pool = Pool(5)
    for stock_id in all_stock_id.values.ravel():
        result_list.append(pool.apply_async(part_deal_fun_mul, (stock_id, cut_hour)))
    pool.close()
    pool.join()

    target_df = pd.concat([x.get() for x in result_list], axis=1)
    target_df = target_df[target_df.index >= pd.to_datetime('20050101')]
    target_df.columns = bt.AZ_add_stock_suffix(target_df.columns)
    target_df = target_df[sorted(target_df.columns)]
    target_df.to_csv(os.path.join(save_root_path, f'bar_num_{cut_hour}_df.csv'), sep='|')
    # return target_df


def main(mod):
    try:
        if mod == 'pro':
            save_root_path = '/media/hdd1/DAT_EQT/EM_Funda/dat_whs'
        else:
            save_root_path = '/mnt/mfs/DAT_EQT/EM_Funda/dat_whs'
        a1 = time.time()
        get_bar_num(save_root_path, 7)
        b1 = time.time()
        print(b1 - a1)

        a2 = time.time()
        get_bar_num(save_root_path, 12)
        b2 = time.time()
        print(b2 - a2)
    except Exception as error:
        send_email.send_email(error, ['whs@yingpei.com'], [], '[pro_update]error stock_bar_mysql')


if __name__ == '__main__':
    mod = 'pro'
    main(mod)
