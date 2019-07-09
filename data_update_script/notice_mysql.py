import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *

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


def save_fun(df, save_path, sep='|'):
    print(1)
    print(save_path)
    df.to_csv(save_path, sep=sep)
    test_save_path = '/mnt/mfs/dat_whs/EM_Funda/{}'.format(datetime.now().strftime('%Y%m%d'))
    bt.AZ_Path_create(test_save_path)
    df.to_csv(os.path.join(test_save_path, os.path.split(save_path)[-1]))


def class_bulletintype(StockId):
    engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.10:7777/{mysql_name}?charset=utf8')
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
    engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.10:7777/{mysql_name}?charset=utf8')
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


def get_lsgg_table(root_path, save_path):
    engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.10:7777/{mysql_name}?charset=utf8')
    conn = engine.connect()
    print(0)
    print(datetime.now())
    lsgg_df = pd.read_sql('SELECT StockId, ReportDate FROM crawl.StockBulletin_sina WHERE BulletinType="LSGG"', conn)
    print(datetime.now())
    print(1)
    lsgg_df['mark'] = 1
    lsgg_num_df = lsgg_df.groupby(['ReportDate', 'StockId'])['mark'].sum().unstack()
    lsgg_num_df.index = pd.to_datetime(lsgg_num_df.index)
    lsgg_num_df.columns = bt.AZ_add_stock_suffix(lsgg_num_df.columns)
    window_list = [5, 20, 60]
    for window in window_list:
        lsgg_num_df_c = fill_index(lsgg_num_df, root_path, window)
        # lsgg_num_df_c.to_csv(f'{save_path}/lsgg_num_df_{window}.csv', sep='|')
        save_fun(lsgg_num_df_c, f'{save_path}/lsgg_num_df_{window}.csv', sep='|')


def get_bulletin_table(root_path, save_path):
    engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.10:7777/{mysql_name}?charset=utf8')
    conn = engine.connect()
    lsgg_df = pd.read_sql('SELECT StockId, ReportDate FROM crawl.StockBulletin_sina', conn)
    # LSGG_df.to_pickle('/mnt/mfs/dat_whs/tmp/LSGG.pkl')
    lsgg_df['mark'] = 1
    lsgg_num_df = lsgg_df.groupby(['ReportDate', 'StockId'])['mark'].sum().unstack()
    lsgg_num_df.index = pd.to_datetime(lsgg_num_df.index)
    lsgg_num_df.columns = bt.AZ_add_stock_suffix(lsgg_num_df.columns)
    # lsgg_num_df.to_csv(f'{save_path}/bulletin_num_df.csv', sep='|')
    save_fun(lsgg_num_df, f'{save_path}/bulletin_num_df.csv', sep='|')
    window_list = [5, 20, 60]
    for window in window_list:
        lsgg_num_df_c = fill_index(lsgg_num_df, root_path, window)
        # lsgg_num_df_c.to_csv(f'{save_path}/bulletin_num_df_{window}.csv', sep='|')
        save_fun(lsgg_num_df_c, f'{save_path}/bulletin_num_df_{window}.csv', sep='|')


def date_deal_fun(x):
    cut_date = datetime(year=x.year, month=x.month, day=x.day, hour=7, minute=00)
    if cut_date > x:
        # print(datetime(year=x.year, month=x.month, day=x.day) - timedelta(days=1))
        return datetime(year=x.year, month=x.month, day=x.day) - timedelta(days=1)
    else:
        # print(datetime(year=x.year, month=x.month, day=x.day))
        return datetime(year=x.year, month=x.month, day=x.day)


def get_news_table(root_path, save_path):
    engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.10:7777/{mysql_name}?charset=utf8')
    conn = engine.connect()
    news_df = pd.read_sql('SELECT StockId, NewsDate FROM crawl.StockNews_xueqiu', conn)
    news_df['mark'] = 1
    news_df['NewsDate'] = [date_deal_fun(x) for x in news_df['NewsDate']]
    news_num_df = news_df.groupby(['NewsDate', 'StockId'])['mark'].sum().unstack()
    news_num_df.index = pd.to_datetime(news_num_df.index)
    news_num_df.columns = bt.AZ_add_stock_suffix(news_num_df.columns)
    # news_num_df.to_csv(f'{save_path}/news_num_df.csv', sep='|')
    window_list = [5, 20, 60]
    for window in window_list:
        news_num_df_c = fill_index(news_num_df, root_path, window)
        news_num_df_zsorce = bt.AZ_Col_zscore(news_num_df_c, window * 2, cap=5)
        news_num_df_zsorce.to_csv(f'{save_path}/news_num_df_{window}.csv', sep='|')


class ClassifyBulletin:
    def __init__(self, save_path):
        engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.10:7777/{mysql_name}?charset=utf8')
        conn = engine.connect()
        news_df = pd.read_sql('SELECT StockId, BulletinTitle, ReportDate FROM crawl.StockBulletin_sina', conn)
        news_df['mark'] = 1
        self.news_df = news_df
        self.save_path = save_path

    def common_deal(self, key_word_list, save_file_name):
        print(save_file_name)
        regx = re.compile('|'.join(key_word_list))
        a = [regx.search(x) for x in self.news_df['BulletinTitle']]
        part_news_df = self.news_df[pd.notna(a)]
        tmp_df = part_news_df.groupby(['ReportDate', 'StockId'])['mark'].sum().unstack()
        target_df = tmp_df.notna().astype(int)
        target_df.columns = bt.AZ_add_stock_suffix(target_df.columns)
        # target_df.to_csv(f'{self.save_path}/{save_file_name}.csv', sep='|')
        save_fun(target_df, f'{self.save_path}/{save_file_name}.csv', sep='|')
        return target_df

    def staff_changes_fun(self):
        key_word_list = ['任职', '聘任', '担任', '辞任', '候选人', '辞职']
        save_file_name = 'staff_changes'
        target_df = self.common_deal(key_word_list, save_file_name)
        return target_df

    def funds_fun(self):
        key_word_list = ['资金']
        save_file_name = 'funds'
        target_df = self.common_deal(key_word_list, save_file_name)
        return target_df

    def meeting_decide_fun(self):
        key_word_list = ['董事会决议', '董事会', '股东大会', '监事会']
        save_file_name = 'meeting_decide'
        target_df = self.common_deal(key_word_list, save_file_name)
        return target_df

    def restricted_shares_fun(self):
        key_word_list = ['限售股解禁', '解除限购']
        save_file_name = 'restricted_shares'
        target_df = self.common_deal(key_word_list, save_file_name)
        return target_df

    def son_company_fun(self):
        key_word_list = ['子公司']
        save_file_name = 'son_company'
        target_df = self.common_deal(key_word_list, save_file_name)
        return target_df

    def suspend_fun(self):
        key_word_list = ['停牌']
        save_file_name = 'suspend'
        target_df = self.common_deal(key_word_list, save_file_name)
        return target_df

    def shares_fun(self):
        key_word_list = ['股权分置', '股票激励', '股权激励', '分红', '派息', '质押']
        save_file_name = 'shares'
        target_df = self.common_deal(key_word_list, save_file_name)
        return target_df


class ClassifyNews:
    def __init__(self, save_path):
        engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.10:7777/{mysql_name}?charset=utf8')
        conn = engine.connect()
        news_df = pd.read_sql('SELECT StockId, NewsDate, NewsTitle, Summary FROM crawl.StockNews_xueqiu', conn)
        news_df['mark'] = 1
        news_df['NewsDate'] = [date_deal_fun(x) for x in news_df['NewsDate']]
        self.news_df = news_df
        self.save_path = save_path
        self.key_word_data = pd.read_csv('/mnt/mfs/work_whs/AZ_2018_Q4/news_key_words.csv')

    def title_key_word_fun(self, key_word_list):
        # for regx in ['增持', '买入', '分红']:
        for regx in key_word_list:
            a = np.array([len(regx.findall(x)) for x in self.news_df['NewsTitle']])
            part_news_df = self.news_df[pd.notna(a)][['StockId', 'NewsDate']]
            part_news_df['key_word_num'] = a[pd.notna(a)]
            tmp_df = part_news_df.groupby(['NewsDate', 'StockId'])['key_word_num'].sum().unstack()
            tmp_df.columns = bt.AZ_add_stock_suffix(tmp_df.columns)
            # tmp_df.to_csv(f'{self.save_path}/buy_key_title__word.csv', sep='|')
            save_fun(tmp_df, f'{self.save_path}/key_word_{regx}_title.csv', sep='|')

    def summary_key_word_fun(self, key_word_list):
        for regx in key_word_list:
            a = np.array([len(regx.findall(x)) for x in self.news_df['Summary']])
            part_news_df = self.news_df[pd.notna(a)][['StockId', 'NewsDate']]
            part_news_df['key_word_num'] = a[pd.notna(a)]
            tmp_df = part_news_df.groupby(['NewsDate', 'StockId'])['key_word_num'].sum().unstack()
            tmp_df.columns = bt.AZ_add_stock_suffix(tmp_df.columns)
            # tmp_df.to_csv(f'{self.save_path}/buy_key_title__word.csv', sep='|')
            save_fun(tmp_df, f'{self.save_path}/key_word_{regx}_summary.csv', sep='|')

    def title_key_word_buy_fun(self):
        key_word_list = self.key_word_data['buy']
        print('|'.join(key_word_list.dropna()))
        regx = re.compile('|'.join(key_word_list.dropna()))
        a = np.array([len(regx.findall(x)) for x in self.news_df['NewsTitle']])
        part_news_df = self.news_df[pd.notna(a)][['StockId', 'NewsDate']]
        part_news_df['key_word_num'] = a[pd.notna(a)]
        tmp_df = part_news_df.groupby(['NewsDate', 'StockId'])['key_word_num'].sum().unstack()
        tmp_df.columns = bt.AZ_add_stock_suffix(tmp_df.columns)
        # tmp_df.to_csv(f'{self.save_path}/buy_key_title__word.csv', sep='|')
        save_fun(tmp_df, f'{self.save_path}/buy_key_title__word.csv', sep='|')

    def title_key_word_sell_fun(self):
        key_word_list = self.key_word_data['sell']
        print('|'.join(key_word_list.dropna()))
        regx = re.compile('|'.join(key_word_list.dropna()))
        a = np.array([len(regx.findall(x)) for x in self.news_df['NewsTitle']])
        part_news_df = self.news_df[pd.notna(a)][['StockId', 'NewsDate']]
        part_news_df['key_word_num'] = a[pd.notna(a)]
        tmp_df = part_news_df.groupby(['NewsDate', 'StockId'])['key_word_num'].sum().unstack()
        tmp_df.columns = bt.AZ_add_stock_suffix(tmp_df.columns)
        # tmp_df.to_csv(f'{self.save_path}/sell_key_title_word.csv', sep='|')
        save_fun(tmp_df, f'{self.save_path}/sell_key_title_word.csv', sep='|')

    def summary_key_word_buy_fun(self):
        key_word_list = self.key_word_data['buy']
        print('|'.join(key_word_list.dropna()))
        regx = re.compile('|'.join(key_word_list.dropna()))
        a = np.array([len(regx.findall(x)) for x in self.news_df['Summary']])
        part_news_df = self.news_df[pd.notna(a)][['StockId', 'NewsDate']]
        part_news_df['key_word_num'] = a[pd.notna(a)]
        tmp_df = part_news_df.groupby(['NewsDate', 'StockId'])['key_word_num'].sum().unstack()
        tmp_df.columns = bt.AZ_add_stock_suffix(tmp_df.columns)
        # tmp_df.to_csv(f'{self.save_path}/buy_summary_key_word.csv', sep='|')
        save_fun(tmp_df, f'{self.save_path}/buy_summary_key_word.csv', sep='|')

    def summary_key_word_sell_fun(self):
        key_word_list = self.key_word_data['sell']
        print('|'.join(key_word_list.dropna()))
        regx = re.compile('|'.join(key_word_list.dropna()))
        a = np.array([len(regx.findall(x)) for x in self.news_df['Summary']])
        part_news_df = self.news_df[pd.notna(a)][['StockId', 'NewsDate']]
        part_news_df['key_word_num'] = a[pd.notna(a)]
        tmp_df = part_news_df.groupby(['NewsDate', 'StockId'])['key_word_num'].sum().unstack()
        tmp_df.columns = bt.AZ_add_stock_suffix(tmp_df.columns)
        # tmp_df.to_csv(f'{self.save_path}/sell_summary_key_word.csv', sep='|')
        save_fun(tmp_df, f'{self.save_path}/sell_summary_key_word.csv', sep='|')


def test_fun():
    engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.10:7777/{mysql_name}?charset=utf8')
    conn = engine.connect()
    news_df = pd.read_sql('SELECT StockId, NewsDate, SourceDomain FROM crawl.StockNews_xueqiu', conn)
    news_df['mark'] = 1
    news_df['NewsDate'] = [date_deal_fun(x) for x in news_df['NewsDate']]
    return list(set(news_df['SourceDomain']))


def main(mod):
    if mod == 'pro':
        root_path = '/media/hdd1/DAT_EQT'
    elif mod == 'bkt':
        root_path = '/mnt/mfs/DAT_EQT'
    else:
        root_path = '/mnt/mfs/DAT_EQT'
    save_path = f'{root_path}/EM_Funda/dat_whs'
    print('process begin')
    a = time.time()
    get_lsgg_table(root_path, save_path)
    get_bulletin_table(root_path, save_path)
    get_news_table(root_path, save_path)

    classify_bullitin = ClassifyBulletin(save_path)
    classify_bullitin.staff_changes_fun()
    classify_bullitin.funds_fun()
    classify_bullitin.meeting_decide_fun()
    classify_bullitin.restricted_shares_fun()
    classify_bullitin.son_company_fun()
    classify_bullitin.suspend_fun()
    classify_bullitin.shares_fun()

    classify_news = ClassifyNews(save_path)
    # key_word_list = ['增持', '买入', '分红']

    # classify_news.title_key_word_fun(key_word_list)
    # classify_news.summary_key_word_fun(key_word_list)

    classify_news.title_key_word_buy_fun()
    classify_news.title_key_word_sell_fun()

    classify_news.summary_key_word_buy_fun()
    classify_news.summary_key_word_sell_fun()

    b = time.time()
    print(b - a)
    print('get all stock id')


mysql_name = 'crawl'
usr_name = 'yp_labman_whs'
pass_word = 'whs300742'


if __name__ == '__main__':
    mod = 'pro'
    main(mod)
