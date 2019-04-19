import sys
from sqlalchemy import create_engine

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *
from multiprocessing import Pool, Manager
import open_lib.shared_tools.back_test as bt


def f(x):
    return x.iloc[-1]


def multi_list_to_df(target_list, target_df):
    for x in target_list:
        x_result = x.get()
        target_df[x_result.name] = x_result
    return target_df


def multi_proccess_fun(data, target_fun, window):
    def target_fun_c(series, window):
        return series.rolling(window, min_periods=0).apply(target_fun)

    target_list = []
    pool = Pool(10)
    target_df = pd.DataFrame(index=data.index, columns=data.columns)

    for col in data.columns:
        target_list.append(pool.apply_async(target_fun_c, args=(data[col], window)))
    pool.close()
    pool.join()
    target_df = multi_list_to_df(target_list, target_df)
    return target_df


def fun(x):
    # if '董事' in x:
    if x.startswith('010'):
        return True
    else:
        return False


def select_astock(x):
    if len(x) == 6:
        if x[:1] in ['0', '3', '6']:
            return True
        else:
            return False
    else:
        return False


def add_suffix(x):
    if x[0] in ['0', '3']:
        return x + '.SZ'
    elif x[0] in ['6']:
        return x + '.SH'
    else:
        print('error')


def time_format(data, col):
    def f(x):
        if not pd.isna(x):
            return x.strftime('%Y%m%d')
        else:
            return None

    data.loc[:, col] = pd.to_datetime([f(x) for x in data[col]])
    return data


def create_date_list(ei_time, target_time):
    target_time_add = target_time + timedelta(days=10)
    target_mask = ei_time >= target_time_add
    ei_time.loc[target_mask] = target_time_add[target_mask]
    return ei_time


def data_deal(date_index,  data):
    df_start = data.groupby(['STARTDATE', 'COMPANYCODE']).apply(lambda x: len(x)).unstack()
    df_end = data.groupby(['ENDDATE', 'COMPANYCODE']).apply(lambda x: len(x)).unstack()

    index_list = sorted(list(set(df_start.index) | set(df_end.index)))

    df_start = df_start.reindex(index=index_list, fill_value=0)
    df_end = df_end.reindex(index=index_list, fill_value=0)

    df_start_cum = df_start.fillna(0).cumsum()
    df_end_cum = df_end.fillna(0).cumsum()

    company_code_df = df_start_cum - df_end_cum
    company_code_df.index = pd.to_datetime(company_code_df.index)
    xinx = company_code_df.index
    company_code_df = company_code_df.reindex(index=xinx).fillna(method='ffill').fillna(0)
    company_code_df = company_code_df.reindex(index=date_index)
    return company_code_df


def fill_index(map_data_c, date_index, df, limit=None):
    xinx = sorted(list(set(df.index) | set(date_index)))
    tmp_df = df.reindex(index=xinx, columns=map_data_c.index).fillna(method='ffill', limit=limit)
    target_df = tmp_df.reindex(index=date_index)
    return target_df


def company_to_stock(map_data_c, company_code_df):
    print('company_to_stock')
    # stock_code_df = pd.DataFrame(columns=sorted(self.map_data_c.values))
    company_code_df = company_code_df.reindex(columns=map_data_c.index)
    company_code_df.columns = map_data_c.values
    company_code_df = company_code_df[sorted(company_code_df.columns)]
    company_code_df.columns = [add_suffix(x) for x in company_code_df.columns]
    company_code_df.dropna(how='all', inplace=True, axis='columns')
    return company_code_df


def save_fun(df, save_path, sep='|'):
    print(1)
    print(save_path)
    df.to_csv(save_path, sep=sep)
    test_save_path = '/mnt/mfs/dat_whs/tmp/{}'.format(datetime.now().strftime('%Y%m%d'))
    bt.AZ_Path_create(test_save_path)
    df.to_csv(os.path.join(test_save_path, os.path.split(save_path)[-1]))


class LICO_MO_DSHJS_deal:
    def __init__(self, conn, date_index, map_data_c, save_path):
        # super(LICO_MO_DSHJS_deal, self).__init__(conn, root_path)
        select_col = ','.join(['EITIME', 'STARTDATE', 'ENDDATE', 'PERSONCODE', 'COMPANYCODE', 'POSTCODE'])
        raw_df = pd.read_sql('SELECT {} FROM choice_fndb.LICO_MO_DSHJS'.format(select_col), conn)

        raw_df = time_format(raw_df, 'EITIME')
        raw_df = time_format(raw_df, 'STARTDATE')
        raw_df = time_format(raw_df, 'ENDDATE')
        raw_df['STARTDATE'] = create_date_list(raw_df['EITIME'], raw_df['STARTDATE'])
        raw_df = raw_df[raw_df['STARTDATE'] < raw_df['ENDDATE']]
        self.raw_df = raw_df
        self.date_index = date_index
        self.map_data_c = map_data_c
        self.save_path = save_path

    def index_tab1_1(self):
        """
        董事会规模
        :return:
        """
        data_ds = self.raw_df[[fun(x) for x in self.raw_df['POSTCODE']]] \
            .dropna(how='any', subset=['STARTDATE', 'ENDDATE'])
        company_code_df_ds = data_deal(self.date_index, data_ds)
        stock_code_df_tab1_1 = company_to_stock(self.map_data_c, company_code_df_ds)
        save_fun(stock_code_df_tab1_1, os.path.join(self.save_path, 'stock_code_df_tab1_1'), sep='|')
        return stock_code_df_tab1_1

    def index_tab1_8(self):
        """
        3 年内董事长变动(不含退休、任期届满)次数
        :return:
        """
        def f(df):
            date_series = df['STARTDATE']
            target_df = pd.DataFrame(index=date_series, columns=['3Y_DSZ_num'])
            for end_date in date_series:
                begin_date = end_date - timedelta(days=750)
                ceo_name_set = set(df[(date_series > begin_date) & (date_series <= end_date)]['PERSONCODE'])
                # print(begin_date, end_date)
                # print(ceo_name_set)
                target_df.loc[end_date, '3Y_CEO_num'] = len(ceo_name_set) - 1
            return target_df

        DSZ_list = np.array(list(map(lambda x: False if type(x) is not str else
        (True if '010101' in x.split(',') else False), self.raw_df['POSTCODE'])))
        data_dsz = self.raw_df[DSZ_list]
        start_date_df = data_dsz.groupby(by=['COMPANYCODE'])[['STARTDATE', 'PERSONCODE']].apply(f).reset_index()

        raw_df = start_date_df.reset_index().groupby(['STARTDATE', 'COMPANYCODE'])['3Y_DSZ_num'].apply(
            lambda x: x.iloc[-1]).unstack()
        raw_df = fill_index(self.map_data_c, self.date_index, raw_df)
        stock_code_df_tab1_8 = company_to_stock(self.map_data_c, raw_df)
        save_fun(stock_code_df_tab1_8, os.path.join(self.save_path, 'stock_code_df_tab1_8'), sep='|')
        return stock_code_df_tab1_8

    def index_tab1_9(self):
        """
        独立董事 占所有董事比率
        :return:
        """
        data_dlds = self.raw_df[self.raw_df['POSTCODE'] == '010401'] \
            .dropna(how='any', subset=['STARTDATE', 'ENDDATE'])

        company_code_df_dlds = data_deal(self.date_index, data_dlds)

        data_ds = self.raw_df[[fun(x) for x in self.raw_df['POSTCODE']]] \
            .dropna(how='any', subset=['STARTDATE', 'ENDDATE'])
        company_code_df_ds = data_deal(self.date_index, data_ds)
        dlds_ratio = company_code_df_dlds / company_code_df_ds
        dlds_ratio = fill_index(self.map_data_c, self.date_index, dlds_ratio)
        stock_code_df_tab1_9 = company_to_stock(self.map_data_c, dlds_ratio)
        save_fun(stock_code_df_tab1_9, os.path.join(self.save_path, 'stock_code_df_tab1_9'), sep='|')
        return stock_code_df_tab1_9


class LICO_MO_MANHOLDRPAY_deal:
    def __init__(self, conn, date_index, map_data_c, save_path):
        select_col = ','.join(['EITIME', 'NOTICEDATE', 'COMPANYCODE', 'PERSONCODE', 'POSTCODE', 'EHN', 'ANUALWAGE'])
        raw_df = pd.read_sql('SELECT {} FROM choice_fndb.LICO_MO_MANHOLDRPAY'.format(select_col), conn)

        raw_df = time_format(raw_df, 'EITIME')
        raw_df = time_format(raw_df, 'NOTICEDATE')

        raw_df['NOTICEDATE'] = create_date_list(raw_df['EITIME'], raw_df['NOTICEDATE'])

        self.raw_df = raw_df
        self.date_index = date_index
        self.map_data_c = map_data_c
        self.save_path = save_path

    def index_tab1_2(self):
        """
        2 : 董事长 CEO 是否是同一人
        :return:
        """
        CEO_list = np.array(list(map(lambda x: False if type(x) is not str else
        (True if ('110101' in x.split(',') or
                  '110102' in x.split(',')) and
                 len(x.split(',')) != 1 else False), self.raw_df['POSTCODE'])))

        DSZ_list = np.array(list(map(lambda x: False if type(x) is not str else
        (True if '010101' in x.split(',') and
                 len(x.split(',')) != 1 else False), self.raw_df['POSTCODE'])))

        data_part_ceo = self.raw_df[CEO_list].groupby(['NOTICEDATE', 'COMPANYCODE'])['PERSONCODE'] \
            .apply(f).unstack()
        data_part_dsz = self.raw_df[DSZ_list].groupby(['NOTICEDATE', 'COMPANYCODE'])['PERSONCODE'] \
            .apply(f).unstack()

        xinx = np.array(sorted(list(set(data_part_ceo.index) | set(data_part_dsz.index) | set(self.date_index))))
        xnms = sorted(list(set(data_part_ceo.columns) | set(data_part_dsz.columns)))

        data_part_ceo_fill = data_part_ceo.reindex(index=xinx, columns=xnms).fillna(method='ffill')
        data_part_dsz_fill = data_part_dsz.reindex(index=xinx, columns=xnms).fillna(method='ffill')

        data_part_ceo_fill = data_part_ceo_fill.reindex(index=self.date_index)
        data_part_dsz_fill = data_part_dsz_fill.reindex(index=self.date_index)
        data_dsz_ceo = (data_part_ceo_fill == data_part_dsz_fill).astype(int)
        stock_code_df_tab1_2 = company_to_stock(self.map_data_c, data_dsz_ceo)
        save_fun(stock_code_df_tab1_2, os.path.join(self.save_path, 'stock_code_df_tab1_2'), sep='|')
        return stock_code_df_tab1_2

    def index_tab2_4(self):
        """
        监事持股比例
        :return:
        """

        def f_1(x):
            if x is not None and x.startswith('02'):
                return True
            else:
                return False

        def f_2(x):
            x = x.replace(0, np.nan)
            js_share_ratio = x.notnull().sum() / len(x)
            return js_share_ratio

        data_js = self.raw_df[[f_1(x) for x in self.raw_df['POSTCODE']]]
        raw_df = data_js.groupby(['NOTICEDATE', 'COMPANYCODE'])['EHN'].apply(f_2).unstack()
        raw_df = fill_index(self.map_data_c, self.date_index, raw_df)
        stock_code_df_tab2_4 = company_to_stock(self.map_data_c, raw_df)
        save_fun(stock_code_df_tab2_4, os.path.join(self.save_path, 'stock_code_df_tab2_4'), sep='|')
        return stock_code_df_tab2_4

    def index_tab2_5(self):
        """
        高管持股比例
        :return:
        """

        def f_1(x):
            if x is not None and x.startswith('11'):
                return True
            else:
                return False

        def f_2(x):
            # print(x)
            x = x.replace(0, np.nan)
            js_share_ratio = x.notnull().sum() / len(x)
            # print(js_share_ratio)
            return js_share_ratio

        data_gg = self.raw_df[[f_1(x) for x in self.raw_df['POSTCODE']]]
        raw_df = data_gg.groupby(['NOTICEDATE', 'COMPANYCODE'])['EHN'].apply(f_2).unstack()
        raw_df = fill_index(self.map_data_c, self.date_index, raw_df)
        stock_code_df_tab2_5 = company_to_stock(self.map_data_c, raw_df)
        save_fun(stock_code_df_tab2_5, os.path.join(self.save_path, 'stock_code_df_tab2_5'), sep='|')
        return stock_code_df_tab2_5

    def index_tab4_1(self):
        """
        log(董事年薪前三名)
        :return:
        """

        def f_1(x):
            if x is not None and x.startswith('010'):
                return True
            else:
                return False

        def fun(x):
            x = x.sort_values(ascending=False)
            # print(x)
            exe_pay = sum(x.iloc[:3])

            if exe_pay < 1:
                log_exe_pay = np.nan
            else:
                log_exe_pay = np.log(exe_pay)
            return log_exe_pay

        data_ds = self.raw_df[[f_1(x) for x in self.raw_df['POSTCODE']]]

        raw_df = data_ds.groupby(['NOTICEDATE', 'COMPANYCODE'])['ANUALWAGE'].apply(fun).unstack()
        raw_df = fill_index(self.map_data_c, self.date_index, raw_df, limit=300)
        stock_code_df_tab4_1 = company_to_stock(self.map_data_c, raw_df)
        save_fun(stock_code_df_tab4_1, os.path.join(self.save_path, 'stock_code_df_tab4_1'), sep='|')
        return stock_code_df_tab4_1

    def index_tab4_2(self):
        """
        log(高管年薪前三名)
        :return:
        """

        def f_1(x):
            if x is not None and x.startswith('110'):
                return True
            else:
                return False

        def fun(x):
            exe_pay = sum(x.sort_values(ascending=False).iloc[:3])
            if exe_pay < 1:
                log_exe_pay = np.nan
            else:
                log_exe_pay = np.log(exe_pay)
            return log_exe_pay

        data_ds = self.raw_df[[f_1(x) for x in self.raw_df['POSTCODE']]]

        raw_df = data_ds.groupby(['NOTICEDATE', 'COMPANYCODE'])['ANUALWAGE'].apply(fun).unstack()
        raw_df = raw_df.fillna(method='ffill', limit=300)
        stock_code_df_tab4_2 = company_to_stock(self.map_data_c, raw_df)
        save_fun(stock_code_df_tab4_2, os.path.join(self.save_path, 'stock_code_df_tab4_2'), sep='|')
        return stock_code_df_tab4_2


class LICO_MO_MANS_deal:
    def __init__(self, conn, date_index, map_data_c, save_path):
        select_col = ','.join(['EITIME', 'PASSNOTICEDATE', 'COMPANYCODE', 'PERSONTYPE'])
        raw_df = pd.read_sql('SELECT {} FROM choice_fndb.LICO_MO_MANS'.format(select_col), conn)

        raw_df = time_format(raw_df, 'EITIME')
        raw_df = time_format(raw_df, 'PASSNOTICEDATE')
        raw_df['PASSNOTICEDATE'] = create_date_list(raw_df['EITIME'], raw_df['PASSNOTICEDATE'])
        self.raw_df = raw_df
        self.date_index = date_index
        self.map_data_c = map_data_c
        self.save_path = save_path

    def index_tab1_5(self, time_period=250):
        """
        监事会会议次数
        :return:
        """
        part_data = self.raw_df[self.raw_df['PERSONTYPE'] == '02']
        company_meet = part_data.groupby(['PASSNOTICEDATE', 'COMPANYCODE'])['PERSONTYPE'].sum().unstack()
        company_meet = company_meet.reindex(sorted(set(company_meet.index) | set(self.date_index)))
        company_meet_num = company_meet.notna().astype(int)
        company_meet_num_s = company_meet_num.rolling(window=time_period).sum()
        stock_code_df_tab1_5 = company_to_stock(self.map_data_c, company_meet_num_s)
        save_fun(stock_code_df_tab1_5, os.path.join(self.save_path, 'stock_code_df_tab1_5'), sep='|')
        return stock_code_df_tab1_5


class LICO_MO_BUSILEVEL_deal:
    def __init__(self, conn, date_index, map_data_c, save_path):
        select_col = ','.join(['EITIME', 'STARTDATE', 'COMPANYCODE', 'PERSONCODE', 'POSTCODE'])
        raw_df = pd.read_sql('SELECT {} FROM choice_fndb.LICO_MO_BUSILEVEL'.format(select_col), conn)

        raw_df = time_format(raw_df, 'EITIME')
        raw_df = time_format(raw_df, 'STARTDATE')
        raw_df['STARTDATE'] = create_date_list(raw_df['EITIME'], raw_df['STARTDATE'])
        self.raw_df = raw_df
        self.date_index = date_index
        self.map_data_c = map_data_c
        self.save_path = save_path

    def index_tab1_7(self):
        """
        3 年内 CEO 变动(不含退休、任期届满)次数
        :return:
        """

        def f(df):
            date_series = df['STARTDATE']
            target_df = pd.DataFrame(index=date_series, columns=['3Y_CEO_num'])
            for end_date in date_series:
                begin_date = end_date - timedelta(days=750)
                # print(begin_date, end_date)
                ceo_name_set = set(df[(date_series > begin_date) & (date_series <= end_date)]['PERSONCODE'])

                target_df.loc[end_date, '3Y_CEO_num'] = len(ceo_name_set) - 1
            return target_df

        CEO_list = np.array(list(map(lambda x: False if type(x) is not str else
        (True if ('110101' in x.split(',') or
                  '110102' in x.split(',')) else False), self.raw_df['POSTCODE'])))
        data_ceo = self.raw_df[CEO_list]
        data_ceo = data_ceo[data_ceo['STARTDATE'] == data_ceo['STARTDATE']]

        start_date_df = data_ceo.groupby(by=['COMPANYCODE'])[['STARTDATE', 'PERSONCODE']].apply(f).reset_index()

        raw_df = start_date_df.reset_index().groupby(['STARTDATE', 'COMPANYCODE'])['3Y_CEO_num'].apply(
            lambda x: x.iloc[-1]).unstack()
        raw_df = fill_index(self.map_data_c, self.date_index, raw_df)
        stock_code_df_tab1_7 = company_to_stock(self.map_data_c, raw_df)
        save_fun(stock_code_df_tab1_7, os.path.join(self.save_path, 'stock_code_df_tab1_7'), sep='|')
        return stock_code_df_tab1_7


class LICO_ES_LISHOLD_deal:
    def __init__(self, conn, date_index, map_data_c, save_path):
        select_col = ','.join(['EITIME', 'NOTICEDATE', 'COMPANYCODE', 'SHAREHDRATIO', 'SHAREHDTYPE', 'SHAREHDCODE'])
        raw_df = pd.read_sql('SELECT {} FROM choice_fndb.LICO_ES_LISHOLD'.format(select_col), conn)

        raw_df = time_format(raw_df, 'EITIME')
        raw_df = time_format(raw_df, 'NOTICEDATE')
        raw_df['NOTICEDATE'] = create_date_list(raw_df['EITIME'], raw_df['NOTICEDATE'])
        self.raw_df = raw_df
        self.date_index = date_index
        self.map_data_c = map_data_c
        self.save_path = save_path

    def index_tab2_1(self):
        """
        股权集中度指标(前 8 大股东股票比例)
        :return:
        """

        def f(df):
            df = df.drop_duplicates(subset=['SHAREHDCODE'], keep='last')
            df = df.sort_values(by=['SHAREHDRATIO'], ascending=False)
            # print(df)
            if sum(df['SHAREHDRATIO']) > 60:
                return sum(df['SHAREHDRATIO'].iloc[:8])
            else:
                return np.nan

        raw_df = self.raw_df.groupby(['NOTICEDATE', 'COMPANYCODE'])[['SHAREHDRATIO', 'SHAREHDCODE']].apply(f).unstack()
        raw_df = fill_index(self.map_data_c, self.date_index, raw_df)
        stock_code_df_tab2_1 = company_to_stock(self.map_data_c, raw_df)
        save_fun(stock_code_df_tab2_1, os.path.join(self.save_path, 'stock_code_df_tab2_1'), sep='|')
        return stock_code_df_tab2_1

    def index_tab2_7(self):
        """
        基金持股比例
        :return:
        """

        data_jj = self.raw_df[(self.raw_df['SHAREHDTYPE'] == '007') |
                              (self.raw_df['SHAREHDTYPE'] == '008') |
                              (self.raw_df['SHAREHDTYPE'] == '018')]
        raw_df = data_jj.groupby(['NOTICEDATE', 'COMPANYCODE'])['SHAREHDRATIO'].sum().unstack()
        raw_df = fill_index(self.map_data_c, self.date_index, raw_df)
        stock_code_df_tab2_7 = company_to_stock(self.map_data_c, raw_df)
        save_fun(stock_code_df_tab2_7, os.path.join(self.save_path, 'stock_code_df_tab2_7'), sep='|')
        return stock_code_df_tab2_7

    def index_tab2_8(self):
        """
        社保基金持股比例
        :return:
        """
        # self.raw_df = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_ES_LISHOLD.pkl')
        data_sbjj = self.raw_df[self.raw_df['SHAREHDTYPE'] == '012']
        raw_df = data_sbjj.groupby(['NOTICEDATE', 'COMPANYCODE'])['SHAREHDRATIO'].sum().unstack()
        raw_df = fill_index(self.map_data_c, self.date_index, raw_df)
        stock_code_df_tab2_8 = company_to_stock(self.map_data_c, raw_df)
        save_fun(stock_code_df_tab2_8, os.path.join(self.save_path, 'stock_code_df_tab2_8'), sep='|')
        return stock_code_df_tab2_8

    def index_tab2_9(self):
        """
        QFII 持股比例
        :return:
        """
        # self.raw_df = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_ES_LISHOLD.pkl')
        data_qfii = self.raw_df[self.raw_df['SHAREHDTYPE'] == '001']
        raw_df = data_qfii.groupby(['NOTICEDATE', 'COMPANYCODE'])['SHAREHDRATIO'].sum().unstack()
        raw_df = fill_index(self.map_data_c, self.date_index, raw_df)
        stock_code_df_tab2_9 = company_to_stock(self.map_data_c, raw_df)
        save_fun(stock_code_df_tab2_9, os.path.join(self.save_path, 'stock_code_df_tab2_9'), sep='|')
        return stock_code_df_tab2_9

    def index_tab2_10(self):
        """
        非金融类上市公司持股比例
        :return:
        """
        # self.raw_df = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_ES_LISHOLD.pkl')
        data_fjr = self.raw_df[self.raw_df['SHAREHDTYPE'] == '013']
        raw_df = data_fjr.groupby(['NOTICEDATE', 'COMPANYCODE'])['SHAREHDRATIO'].sum().unstack()
        raw_df = fill_index(self.map_data_c, self.date_index, raw_df)
        stock_code_df_tab2_10 = company_to_stock(self.map_data_c, raw_df)
        save_fun(stock_code_df_tab2_10, os.path.join(self.save_path, 'stock_code_df_tab2_10'), sep='|')
        return stock_code_df_tab2_10


class LICO_ES_EMSHAREDE_deal:
    def __init__(self, conn, date_index, map_data_c, save_path):
        select_col = ','.join(['EITIME', 'FIRSTNOTICEDATE', 'SECINNERCODE', 'RATIO'])
        raw_df = pd.read_sql('SELECT {} FROM choice_fndb.LICO_ES_EMSHAREDE'.format(select_col), conn)

        raw_df = time_format(raw_df, 'EITIME')
        raw_df = time_format(raw_df, 'FIRSTNOTICEDATE')
        raw_df['FIRSTNOTICEDATE'] = create_date_list(raw_df['EITIME'], raw_df['FIRSTNOTICEDATE'])
        self.raw_df = raw_df
        self.date_index = date_index
        self.map_data_c = map_data_c
        self.save_path = save_path

    def index_tab4_3(self):
        """
        员工持股计划/总股本
        :return:
        """
        # self.raw_df = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_ES_EMSHAREDE.pkl')
        raw_df = self.raw_df.groupby(['FIRSTNOTICEDATE', 'SECINNERCODE'])['RATIO'] \
            .sum().unstack().fillna(method='ffill', limit=300)

        stock_code_df_tab4_3 = company_to_stock(self.map_data_c, raw_df)
        save_fun(stock_code_df_tab4_3, os.path.join(self.save_path, 'stock_code_df_tab4_3'), sep='|')
        return stock_code_df_tab4_3


class LICO_MO_GQJLJBZL_deal:
    def __init__(self, conn, date_index, map_data_c, save_path):
        select_col = ','.join(['EITIME', 'NOTICEDATE', 'COMPANYCODE', 'EXCISHARERATIO'])
        raw_df = pd.read_sql('SELECT {} FROM choice_fndb.LICO_MO_GQJLJBZL'.format(select_col), conn)

        raw_df = time_format(raw_df, 'EITIME')
        raw_df = time_format(raw_df, 'NOTICEDATE')
        raw_df['NOTICEDATE'] = create_date_list(raw_df['EITIME'], raw_df['NOTICEDATE'])
        self.raw_df = raw_df
        self.date_index = date_index
        self.map_data_c = map_data_c
        self.save_path = save_path

    def index_tab4_5(self):
        """
        股权激励权益/总股本
        :return:
        """
        # self.raw_df = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_MO_GQJLJBZL.pkl')
        raw_df = self.raw_df.groupby(['NOTICEDATE', 'COMPANYCODE'])['EXCISHARERATIO'] \
            .sum().unstack().fillna(method='ffill', limit=300)

        stock_code_df_tab4_5 = company_to_stock(self.map_data_c, raw_df)
        save_fun(stock_code_df_tab4_5, os.path.join(self.save_path, 'stock_code_df_tab4_5'), sep='|')
        return stock_code_df_tab4_5


class LICO_CM_ILLEGAL_deal:
    def __init__(self, conn, date_index, map_data_c, save_path):
        select_col = ','.join(['EITIME', 'NOTICEDATE', 'COMPANYCODE', 'GOOLACTION'])
        raw_df = pd.read_sql('SELECT {} FROM choice_fndb.LICO_CM_ILLEGAL'.format(select_col), conn)

        raw_df = time_format(raw_df, 'EITIME')
        raw_df = time_format(raw_df, 'NOTICEDATE')
        raw_df['NOTICEDATE'] = create_date_list(raw_df['EITIME'], raw_df['NOTICEDATE'])
        self.raw_df = raw_df
        self.date_index = date_index
        self.map_data_c = map_data_c
        self.save_path = save_path

    def index_tab5_13(self):
        """
        近 1 年来违规次数
        :return:
        """
        violation_df = self.raw_df.groupby(['NOTICEDATE', 'COMPANYCODE'])['GOOLACTION'].sum().unstack()
        xinx = sorted(list(set(violation_df.index) | set(self.date_index)))
        violation_df = violation_df.reindex(index=xinx)
        violation_df_mask = violation_df.notna()
        raw_df = violation_df_mask.rolling(250).sum()
        raw_df = fill_index(self.map_data_c, self.date_index, raw_df)
        stock_code_df_tab5_13 = company_to_stock(self.map_data_c, raw_df)
        save_fun(stock_code_df_tab5_13, os.path.join(self.save_path, 'stock_code_df_tab5_13'), sep='|')
        return stock_code_df_tab5_13


class LICO_CM_LAWARBI_deal:
    def __init__(self, conn, date_index, map_data_c, save_path):
        select_col = ','.join(['EITIME', 'NOTICEDATE', 'COMPANYCODE', 'SHEANJINE'])
        raw_df = pd.read_sql('SELECT {} FROM choice_fndb.LICO_CM_LAWARBI'.format(select_col), conn)

        raw_df = time_format(raw_df, 'EITIME')
        raw_df = time_format(raw_df, 'NOTICEDATE')
        raw_df['NOTICEDATE'] = create_date_list(raw_df['EITIME'], raw_df['NOTICEDATE'])
        self.raw_df = raw_df
        self.date_index = date_index
        self.map_data_c = map_data_c
        self.save_path = save_path

    def index_tab5_14(self):
        """
        近半年来诉讼仲裁涉案金额数/营业收入
        :return:
        """
        # self.raw_df = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_CM_LAWARBI.pkl')
        amount_involved = self.raw_df.groupby(['NOTICEDATE', 'COMPANYCODE'])['SHEANJINE'].sum().unstack()
        xinx = sorted(list(set(amount_involved.index) | set(self.date_index)))
        amount_involved = amount_involved.reindex(index=xinx)
        raw_df = amount_involved.rolling(125, min_periods=0).sum()
        raw_df = raw_df.reindex(self.date_index)
        stock_code_df = company_to_stock(self.map_data_c, raw_df)
        Revenue_TTM = pd.read_csv('/mnt/mfs/DAT_EQT/EM_Funda/daily/R_Revenue_TTM_First.csv', sep='|', index_col=0,
                                  parse_dates=True).reindex(index=self.date_index, columns=stock_code_df.columns)
        stock_code_df_tab5_14 = (stock_code_df / Revenue_TTM * 10000).round(8).replace(0, np.nan)
        save_fun(stock_code_df_tab5_14, os.path.join(self.save_path, 'stock_code_df_tab5_14'), sep='|')
        return stock_code_df_tab5_14


class LICO_ES_SHHDFROZEN_deal:
    def __init__(self, conn, date_index, map_data_c, save_path):
        select_col = ','.join(['EITIME', 'ENDDATE', 'FREEDATE', 'NOTICEDATE', 'COMPANYCODE', 'FROZENINTOTAL'])
        raw_df = pd.read_sql('SELECT {} FROM choice_fndb.LICO_ES_SHHDFROZEN'.format(select_col), conn)
        raw_df.loc[raw_df['ENDDATE'] > datetime(3018, 12, 14, 0, 0), 'ENDDATE'] = None
        raw_df = time_format(raw_df, 'EITIME')
        raw_df = time_format(raw_df, 'ENDDATE')
        raw_df = time_format(raw_df, 'FREEDATE')
        raw_df = time_format(raw_df, 'NOTICEDATE')
        raw_df['NOTICEDATE'] = create_date_list(raw_df['EITIME'], raw_df['NOTICEDATE'])

        self.raw_df = raw_df
        self.date_index = date_index
        self.map_data_c = map_data_c
        self.save_path = save_path

    def index_tab5_15(self):
        """
        冻结股票比例
        :return:
        """
        # self.raw_df = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_ES_SHHDFROZEN.pkl')

        self.raw_df['ENDDATE'] = pd.to_datetime(self.raw_df['ENDDATE'].values)
        self.raw_df['FREEDATE'] = pd.to_datetime(self.raw_df['FREEDATE'].values)
        self.raw_df['NOTICEDATE'] = pd.to_datetime(self.raw_df['NOTICEDATE'].values)

        FREEDATE_mask = self.raw_df['FREEDATE'].notnull()
        self.raw_df.loc[FREEDATE_mask, 'ENDDATE'] = self.raw_df.loc[FREEDATE_mask, 'FREEDATE']
        self.raw_df = self.raw_df.dropna(how='any', subset=['NOTICEDATE', 'ENDDATE'])
        self.raw_df = self.raw_df[self.raw_df['ENDDATE'] > self.raw_df['NOTICEDATE']]

        df_start = self.raw_df.groupby(['NOTICEDATE', 'COMPANYCODE'])['FROZENINTOTAL'].sum().unstack()
        df_end = self.raw_df.groupby(['ENDDATE', 'COMPANYCODE'])['FROZENINTOTAL'].sum().unstack()

        xinx = sorted(list(set(df_start.index) | set(df_end.index)))

        df_start_cum = df_start.reindex(xinx).fillna(0).cumsum()
        df_end_cum = df_end.reindex(xinx).fillna(0).cumsum()

        raw_df = (df_start_cum - df_end_cum).round(5)
        raw_df = fill_index(self.map_data_c, self.date_index, raw_df)
        stock_code_df_tab5_15 = company_to_stock(self.map_data_c, raw_df)
        save_fun(stock_code_df_tab5_15, os.path.join(self.save_path, 'stock_code_df_tab5_15'), sep='|')
        return stock_code_df_tab5_15


def singleprocessing_fun():
    usr_name = 'whs'
    pass_word = 'kj23#12!^3weghWhjqQ2rjj197'
    engine = create_engine('mysql+pymysql://{}:{}@192.168.16.33:3306/choice_fndb?charset=utf8'
                           .format(usr_name, pass_word))
    # root_path = '/mnt/mfs/DAT_EQT'
    root_path = '/media/hdd1/DAT_EQT'
    conn = engine.connect()
    a = time.time()

    return_df = pd.read_csv(f'{root_path}/EM_Funda/DERIVED_14/aadj_r.csv', sep='|', index_col=0, parse_dates=True)
    today_date = pd.to_datetime(datetime.now().strftime('%Y%m%d'))
    date_index = list(return_df.index) + [today_date]
    map_data = pd.read_sql('SELECT * FROM choice_fndb.CDSY_SECUCODE', conn)
    map_data.index = map_data['COMPANYCODE']
    map_data_c = map_data['SECURITYCODE'][map_data['SECURITYTYPE'] == 'A股']
    map_data_c = map_data_c[map_data_c.apply(select_astock)]
    save_path = f'{root_path}/EM_Funda/dat_whs'

    cls_0 = LICO_MO_DSHJS_deal(conn, date_index, map_data_c, save_path)
    print('load LICO_MO_DSHJS_deal')
    cls_1 = LICO_MO_MANHOLDRPAY_deal(conn, date_index, map_data_c, save_path)
    print('load LICO_MO_MANHOLDRPAY_deal')
    cls_2 = LICO_MO_MANS_deal(conn, date_index, map_data_c, save_path)
    print('load LICO_MO_MANS_deal')
    cls_3 = LICO_MO_BUSILEVEL_deal(conn, date_index, map_data_c, save_path)
    print('load LICO_MO_BUSILEVEL_deal')
    cls_4 = LICO_ES_LISHOLD_deal(conn, date_index, map_data_c, save_path)
    print('load LICO_ES_LISHOLD_deal')
    cls_5 = LICO_ES_EMSHAREDE_deal(conn, date_index, map_data_c, save_path)
    print('load LICO_ES_EMSHAREDE_deal')
    cls_6 = LICO_MO_GQJLJBZL_deal(conn, date_index, map_data_c, save_path)
    print('load LICO_MO_GQJLJBZL_deal')
    cls_7 = LICO_CM_ILLEGAL_deal(conn, date_index, map_data_c, save_path)
    print('load LICO_CM_ILLEGAL_deal')
    cls_8 = LICO_CM_LAWARBI_deal(conn, date_index, map_data_c, save_path)
    print('load LICO_CM_LAWARBI_deal')
    cls_9 = LICO_ES_SHHDFROZEN_deal(conn, date_index, map_data_c, save_path)
    print('load LICO_ES_SHHDFROZEN_deal')
    print(0)
    cls_0.index_tab1_1()
    cls_0.index_tab1_8()
    cls_0.index_tab1_9()
    print(1)
    cls_1.index_tab1_2()
    cls_1.index_tab2_4()
    cls_1.index_tab2_5()
    cls_1.index_tab4_1()
    cls_1.index_tab4_2()
    print(2)
    cls_2.index_tab1_5()
    print(3)
    cls_3.index_tab1_7()
    print(4)
    cls_4.index_tab2_1()
    cls_4.index_tab2_7()
    cls_4.index_tab2_8()
    cls_4.index_tab2_9()
    cls_4.index_tab2_10()
    print(5)
    cls_5.index_tab4_3()
    print(6)
    cls_6.index_tab4_5()
    print(7)
    cls_7.index_tab5_13()
    print(8)
    cls_8.index_tab5_14()
    print(9)
    cls_9.index_tab5_15()

    b = time.time()
    print('cost time:{}'.format(b - a))


def multiprocessing_fun(cpu_num):
    usr_name = 'whs'
    pass_word = 'kj23#12!^3weghWhjqQ2rjj197'
    engine = create_engine('mysql+pymysql://{}:{}@192.168.16.33:3306/choice_fndb?charset=utf8'
                           .format(usr_name, pass_word))
    # root_path = '/mnt/mfs/DAT_EQT'
    root_path = '/media/hdd1/DAT_EQT'
    conn = engine.connect()
    a = time.time()

    return_df = pd.read_csv(f'{root_path}/EM_Funda/DERIVED_14/aadj_r.csv',
                            sep='|', index_col=0, parse_dates=True)
    date_index = return_df.index
    map_data = pd.read_sql('SELECT * FROM choice_fndb.CDSY_SECUCODE', conn)
    map_data.index = map_data['COMPANYCODE']
    map_data_c = map_data['SECURITYCODE'][map_data['SECURITYTYPE'] == 'A股']
    map_data_c = map_data_c[map_data_c.apply(select_astock)]
    save_path = f'{root_path}/EM_Funda/dat_whs'

    pool = Pool(cpu_num)
    cls_0 = LICO_MO_DSHJS_deal(conn, date_index, map_data_c, save_path)
    print('load LICO_MO_DSHJS_deal')
    cls_1 = LICO_MO_MANHOLDRPAY_deal(conn, date_index, map_data_c, save_path)
    print('load LICO_MO_MANHOLDRPAY_deal')
    cls_2 = LICO_MO_MANS_deal(conn, date_index, map_data_c, save_path)
    print('load LICO_MO_MANS_deal')
    cls_3 = LICO_MO_BUSILEVEL_deal(conn, date_index, map_data_c, save_path)
    print('load LICO_MO_BUSILEVEL_deal')
    cls_4 = LICO_ES_LISHOLD_deal(conn, date_index, map_data_c, save_path)
    print('load LICO_ES_LISHOLD_deal')
    cls_5 = LICO_ES_EMSHAREDE_deal(conn, date_index, map_data_c, save_path)
    print('load LICO_ES_EMSHAREDE_deal')
    cls_6 = LICO_MO_GQJLJBZL_deal(conn, date_index, map_data_c, save_path)
    print('load LICO_MO_GQJLJBZL_deal')
    cls_7 = LICO_CM_ILLEGAL_deal(conn, date_index, map_data_c, save_path)
    print('load LICO_CM_ILLEGAL_deal')
    cls_8 = LICO_CM_LAWARBI_deal(conn, date_index, map_data_c, save_path)
    print('load LICO_CM_LAWARBI_deal')
    cls_9 = LICO_ES_SHHDFROZEN_deal(conn, date_index, map_data_c, save_path)
    print('load LICO_ES_SHHDFROZEN_deal')
    print(0)
    pool.apply_async(cls_0.index_tab1_1)
    pool.apply_async(cls_0.index_tab1_8)
    pool.apply_async(cls_0.index_tab1_9)
    print(1)
    pool.apply_async(cls_1.index_tab1_2)
    pool.apply_async(cls_1.index_tab2_4)
    pool.apply_async(cls_1.index_tab2_5)
    pool.apply_async(cls_1.index_tab4_1)
    pool.apply_async(cls_1.index_tab4_2)
    print(2)
    pool.apply_async(cls_2.index_tab1_5)
    print(3)
    pool.apply_async(cls_3.index_tab1_7)
    print(4)
    pool.apply_async(cls_4.index_tab2_1)
    pool.apply_async(cls_4.index_tab2_7)
    pool.apply_async(cls_4.index_tab2_8)
    pool.apply_async(cls_4.index_tab2_9)
    pool.apply_async(cls_4.index_tab2_10)
    print(5)
    pool.apply_async(cls_5.index_tab4_3)
    print(6)
    pool.apply_async(cls_6.index_tab4_5)
    print(7)
    pool.apply_async(cls_7.index_tab5_13)
    print(8)
    pool.apply_async(cls_8.index_tab5_14)
    print(9)
    pool.apply_async(cls_9.index_tab5_15)
    pool.close()
    pool.join()
    b = time.time()
    print('cost time:{}'.format(b - a))


if __name__ == '__main__':
    singleprocessing_fun()

