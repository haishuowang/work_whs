import os
from datetime import datetime, timedelta
from multiprocessing import Pool
from sqlalchemy import create_engine

import numpy as np
import pandas as pd
import time
import re
from collections import OrderedDict
from open_lib.shared_utils.config.dbconfig import load_engine
from open_lib.shared_utils.io import serialize, parser

__author__ = "whs<whs@yingpei.com>"
__use_multi__ = False

mysql_name = 'crawl'
usr_name = 'yp_labman_whs'
pass_word = 'whs300742'


class bt:
    @staticmethod
    def AZ_Load_csv(target_path, index_time_type=True):
        target_df = pd.read_table(target_path, sep='|', index_col=0, low_memory=False).round(8)
        if index_time_type:
            target_df.index = pd.to_datetime(target_df.index)
        return target_df

    @staticmethod
    def AZ_Rolling(df, n, min_periods=1):
        return df.rolling(window=n, min_periods=min_periods)

    @staticmethod
    def AZ_Rolling_mean(df, n, min_periods=1):
        target = df.rolling(window=n, min_periods=min_periods).mean()
        target.iloc[:n - 1] = np.nan
        return target

    @staticmethod
    def AZ_Path_create(target_path):
        """
        添加新路径
        :param target_path:
        :return:
        """
        if not os.path.exists(target_path):
            os.makedirs(target_path)

    @staticmethod
    def AZ_split_stock(stock_list):
        """
        在stock_list中寻找A股代码
        :param stock_list:
        :return:
        """
        eqa = [x for x in stock_list if (x.startswith('0') or x.startswith('3')) and x.endswith('SZ')
               or x.startswith('6') and x.endswith('SH')]
        return eqa

    @staticmethod
    def AZ_add_stock_suffix(stock_list):
        """
        whs
        给stock_list只有数字的 A股代码 添加后缀
        如 000001 运行后 000001.SZ
        :param stock_list:
        :return:　　
        """
        return list(map(lambda x: x + '.SH' if x.startswith('6') else x + '.SZ', stock_list))

    def AZ_Col_zscore(self, df, n, cap=None, min_periods=1):
        df_mean = self.AZ_Rolling_mean(df, n, min_periods=min_periods)
        df_std = df.rolling(window=n, min_periods=min_periods).std()
        target = (df - df_mean) / df_std
        if cap is not None:
            target[target > cap] = cap
            target[target < -cap] = -cap
        return target


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


def save_fun(df, save_path):
    print(df)
    # serialize.to_csv_(df, save_path, stream=print)


def class_bulletintype(StockId):
    engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.28:7777/{mysql_name}?charset=utf8')
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
        # send_email.send_email('', ['whs@yingpei.com'], [], 'mysql_error')
        return None


def get_bulletin_num_table():
    engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.28:7777/{mysql_name}?charset=utf8')
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
    engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.28:7777/{mysql_name}?charset=utf8')
    conn = engine.connect()
    lsgg_df = pd.read_sql('SELECT StockId, ReportDate FROM crawl.StockBulletin_sina WHERE BulletinType="LSGG"', conn)
    lsgg_df['mark'] = 1
    lsgg_num_df = lsgg_df.groupby(['ReportDate', 'StockId'])['mark'].sum().unstack()
    lsgg_num_df.index = pd.to_datetime(lsgg_num_df.index)
    lsgg_num_df.columns = bt.AZ_add_stock_suffix(lsgg_num_df.columns)
    window_list = [5, 20, 60]
    for window in window_list:
        lsgg_num_df_c = fill_index(lsgg_num_df, root_path, window)
        save_fun(lsgg_num_df_c, save_path / f'lsgg_num_df_{window}.csv')


def get_bulletin_table(root_path, save_path):
    engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.28:7777/{mysql_name}?charset=utf8')
    conn = engine.connect()
    lsgg_df = pd.read_sql('SELECT StockId, ReportDate FROM crawl.StockBulletin_sina', conn)
    # LSGG_df.to_pickle('/mnt/mfs/dat_whs/tmp/LSGG.pkl')
    lsgg_df['mark'] = 1
    lsgg_num_df = lsgg_df.groupby(['ReportDate', 'StockId'])['mark'].sum().unstack()
    lsgg_num_df.index = pd.to_datetime(lsgg_num_df.index)
    lsgg_num_df.columns = bt.AZ_add_stock_suffix(lsgg_num_df.columns)
    # lsgg_num_df.to_csv(f'{save_path}/bulletin_num_df.csv', sep='|')
    save_fun(lsgg_num_df, save_path / 'bulletin_num_df.csv')
    window_list = [5, 20, 60]
    for window in window_list:
        lsgg_num_df_c = fill_index(lsgg_num_df, root_path, window)
        save_fun(lsgg_num_df_c, save_path / f'bulletin_num_df_{window}.csv')


def date_deal_fun(x):
    cut_date = datetime(year=x.year, month=x.month, day=x.day, hour=7, minute=00)
    if cut_date > x:
        # print(datetime(year=x.year, month=x.month, day=x.day) - timedelta(days=1))
        return datetime(year=x.year, month=x.month, day=x.day) - timedelta(days=1)
    else:
        # print(datetime(year=x.year, month=x.month, day=x.day))
        return datetime(year=x.year, month=x.month, day=x.day)


def get_news_table(root_path, save_path):
    engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.28:7777/{mysql_name}?charset=utf8')
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
        save_fun(news_num_df_zsorce, save_path / f'news_num_df_{window}.csv')


class ClassifyBulletin:
    def __init__(self, save_path):
        engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.28:7777/{mysql_name}?charset=utf8')
        conn = engine.connect()
        print(1)
        news_df = pd.read_sql('SELECT StockId, BulletinTitle, ReportDate FROM crawl.StockBulletin_sina', conn)
        print(2)
        news_df['mark'] = 1
        self.news_df = news_df
        bt.AZ_Path_create(save_path)
        self.save_path = save_path

    def key_word_fun(self, key_word_list, save_file_name):
        print(save_file_name)
        regx = re.compile('|'.join(key_word_list))
        a = [regx.search(x) for x in self.news_df['BulletinTitle']]
        part_news_df = self.news_df[pd.notna(a)]
        tmp_df = part_news_df.groupby(['ReportDate', 'StockId'])['mark'].sum().unstack()
        target_df = tmp_df.notna().astype(int)
        target_df.columns = bt.AZ_add_stock_suffix(target_df.columns)
        save_fun(target_df, self.save_path/f'{save_file_name}.csv')
        return target_df


class ClassifyNews:
    def __init__(self, save_path):
        engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.28:7777/{mysql_name}?charset=utf8')
        conn = engine.connect()
        news_df = pd.read_sql('SELECT StockId, NewsDate, NewsTitle, Summary '
                              'FROM crawl.StockNews_xueqiu '
                              'WHERE BulletinType != lsgg', conn)
        news_df['mark'] = 1
        news_df['NewsDate'] = [date_deal_fun(x) for x in news_df['NewsDate']]
        self.news_df = news_df
        bt.AZ_Path_create(save_path)
        self.save_path = save_path
        key_word_data = pd.DataFrame()
        key_word_data['buy'] = ['利润', '分红', '推动', '升', '赚', '涨', '大', '牛', '增长', '加速', '亮点', '青睐', '信心',
                                '强',  '质量', '合作', '买入', '增持', '看多', '回购', '奖', '高', '上线', '推',  '倍',
                                '启动', '发展', '优异', '稳', '批准', '超', '确认', '乐观', '景气', '完善', '充足', '见底',]
        key_word_data['sell'] = ['罚单', '处罚', '罚款', '罚', '不良', 'ST', '欠缺', '跌', '质押', '回落', '压力', '危机',
                                 '被查', '立案', '减持', '贷款', '违规', '减少']
        self.key_word_data = key_word_data

    def title_key_word_fun(self, key_word_list, save_file):
        regx = re.compile('|'.join(key_word_list))
        a = np.array([len(regx.findall(x)) for x in self.news_df['Summary']])
        part_news_df = self.news_df[pd.notna(a)][['StockId', 'NewsDate']]
        part_news_df['key_word_num'] = a[pd.notna(a)]
        tmp_df = part_news_df.groupby(['NewsDate', 'StockId'])['key_word_num'].sum().unstack()
        tmp_df.columns = bt.AZ_add_stock_suffix(tmp_df.columns)
        save_fun(tmp_df, self.save_path/f'{save_file}_news_title.csv')

    def summary_key_word_fun(self, key_word_list, save_file):
        regx = re.compile('|'.join(key_word_list))
        a = np.array([len(regx.findall(x)) for x in self.news_df['Summary']])
        part_news_df = self.news_df[pd.notna(a)][['StockId', 'NewsDate']]
        part_news_df['key_word_num'] = a[pd.notna(a)]
        tmp_df = part_news_df.groupby(['NewsDate', 'StockId'])['key_word_num'].sum().unstack()
        tmp_df.columns = bt.AZ_add_stock_suffix(tmp_df.columns)
        save_fun(tmp_df, self.save_path/f'{save_file}_news_summary.csv')

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
        save_fun(tmp_df, self.save_path/'buy_key_title__word.csv')

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
        save_fun(tmp_df, self.save_path/f'sell_key_title_word.csv')

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
        save_fun(tmp_df, self.save_path/f'buy_summary_key_word.csv')

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
        save_fun(tmp_df, self.save_path/f'sell_summary_key_word.csv')


def test_fun():
    engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.28:7777/{mysql_name}?charset=utf8')
    conn = engine.connect()
    news_df = pd.read_sql('SELECT StockId, NewsDate, SourceDomain FROM crawl.StockNews_xueqiu', conn)
    news_df['mark'] = 1
    news_df['NewsDate'] = [date_deal_fun(x) for x in news_df['NewsDate']]
    return list(set(news_df['SourceDomain']))


def single_fun(root_path):
    save_path = root_path.EM_Funda
    key_word_dict = OrderedDict({
        'repurchase': ['回购'],
        'dividend': ['分红', '派息'],
        'staff_changes': ['任职', '聘任', '担任', '辞任', '候选人', '辞职'],
        'funds': ['资金'],
        'meeting_decide': ['董事会决议', '董事会', '股东大会', '监事会'],
        'restricted_shares': ['限售股解禁', '解除限购'],
        'son_company': ['子公司'],
        'suspend': ['停牌'],
        'shares': ['股权分置', '股票激励', '股权激励', '质押'],
        'bonus_share': ['股票激励', '激励股票',
                        '股权激励', '激励股权',
                        ]
    })

    a = time.time()
    get_lsgg_table(root_path, save_path)
    get_bulletin_table(root_path, save_path)
    get_news_table(root_path, save_path)

    classify_bullitin = ClassifyBulletin(save_path)

    for file_name in key_word_dict.keys():
        classify_bullitin.key_word_fun(key_word_dict[file_name], file_name)

    classify_news = ClassifyNews(save_path)

    classify_news.title_key_word_buy_fun()
    classify_news.title_key_word_sell_fun()

    classify_news.summary_key_word_buy_fun()
    classify_news.summary_key_word_sell_fun()

    for file_name in key_word_dict.keys():
        print(file_name)
        classify_news.title_key_word_fun(key_word_dict[file_name], file_name)
        classify_news.summary_key_word_fun(key_word_dict[file_name], file_name)

    b = time.time()
    print(b - a)
    print('get all stock id')


def main(env):
    root_path = env.BinFiles
    single_fun(root_path)


if __name__ == '__main__':
    from work_whs.open_lib.shared_utils.config.config_global import Env
    main(Env('bkt'))
