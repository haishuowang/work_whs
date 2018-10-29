import pandas as pd
import numpy as np
import time
from functools import reduce
from datetime import datetime
import matplotlib


def f(x):
    return x.iloc[-1]


class ESG:
    def __init__(self):
        return_df = pd.read_csv('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                                sep='|', index_col=0, parse_dates=True)
        self.company_code_df = None
        self.date_index = return_df.index
        LICO_MO_DSHJS = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_MO_DSHJS.pkl')
        LICO_MO_DSHJS = LICO_MO_DSHJS[LICO_MO_DSHJS['STARTDATE'] < LICO_MO_DSHJS['ENDDATE']]
        self.LICO_MO_DSHJS = LICO_MO_DSHJS

        map_data = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/CDSY_SECUCODE.pkl')
        map_data.index = map_data['COMPANYCODE']
        map_data_c = map_data['SECURITYCODE'][map_data['SECURITYTYPE'] == 'A股']
        self.map_data_c = map_data_c[map_data_c.apply(self.select_astock)]

    @staticmethod
    def fun(x):
        # if '董事' in x:
        if x.startswith('010'):
            return True
        else:
            return False

    @staticmethod
    def select_astock(x):
        if len(x) == 6:
            if x[:1] in ['0', '3', '6']:
                return True
            else:
                return False
        else:
            return False

    @staticmethod
    def add_suffix(x):
        if x[0] in ['0', '3']:
            return x + '.SZ'
        elif x[0] in ['6']:
            return x + '.SH'
        else:
            print('error')

    def data_deal(self, data):
        df_start = data.groupby(['STARTDATE', 'COMPANYCODE']).apply(lambda x: len(x)).unstack()
        df_end = data.groupby(['ENDDATE', 'COMPANYCODE']).apply(lambda x: len(x)).unstack()

        index_list = sorted(list(set(df_start.index) | set(df_end.index)))

        df_start = df_start.reindex(index=index_list, fill_value=0)
        df_end = df_end.reindex(index=index_list, fill_value=0)

        df_start_cum = df_start.fillna(0).cumsum()
        df_end_cum = df_end.fillna(0).cumsum()

        company_code_df = df_start_cum - df_end_cum
        company_code_df.index = pd.to_datetime(company_code_df.index)
        xinx = company_code_df.index | self.date_index
        company_code_df = company_code_df.reindex(index=xinx).fillna(method='ffill').fillna(0)
        company_code_df = company_code_df.reindex(index=self.date_index)
        return company_code_df

    def company_to_stock(self, company_code_df):
        print('company_to_stock')
        stock_code_df = pd.DataFrame(columns=sorted(self.map_data_c.values))
        for company_code in sorted(list(set(company_code_df.columns) & set(self.map_data_c.index))):
            # print(company_code)
            if company_code in self.map_data_c.index:
                part_stock_code_df = company_code_df[company_code]
                stock_code = self.map_data_c.loc[company_code]
                if type(stock_code) is str:
                    stock_code_df[stock_code] = part_stock_code_df
                else:
                    print(type(self.map_data_c.loc[company_code]))
                    for stock_code in self.map_data_c.loc[company_code].values:
                        stock_code_df[stock_code] = part_stock_code_df
            else:
                pass
        stock_code_df.columns = [self.add_suffix(x) for x in stock_code_df.columns]
        stock_code_df.dropna(how='all', inplace=True, axis='columns')
        return stock_code_df

    def index_tab1_1(self):
        """
        董事会规模
        :return:
        """
        data_ds = self.LICO_MO_DSHJS[[self.fun(x) for x in self.LICO_MO_DSHJS['POSTCODE']]] \
            .dropna(how='any', subset=['STARTDATE', 'ENDDATE'])
        company_code_df_ds = self.data_deal(data_ds)
        stock_code_df = self.company_to_stock(company_code_df_ds)
        return stock_code_df

    def index_tab1_2(self):
        """
        2 : 董事长 CEO 是否是同一人
        :return:
        """
        data = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_MO_MANHOLDRPAY.pkl')
        CEO_list = np.array(list(map(lambda x: False if type(x) is not str else
        (True if ('110101' in x.split(',') or
                  '110102' in x.split(',')) and
                 len(x.split(',')) != 1 else False), data['POSTCODE'])))

        DSZ_list = np.array(list(map(lambda x: False if type(x) is not str else
        (True if '010101' in x.split(',') and
                 len(x.split(',')) != 1 else False), data['POSTCODE'])))

        data_part_ceo = data[CEO_list].groupby(['NOTICEDATE', 'COMPANYCODE'])['PERSONCODE'] \
            .apply(f).unstack()
        data_part_dsz = data[DSZ_list].groupby(['NOTICEDATE', 'COMPANYCODE'])['PERSONCODE'] \
            .apply(f).unstack()

        xinx = np.array(sorted(list(set(data_part_ceo.index) | set(data_part_dsz.index) | set(self.date_index))))
        xnms = sorted(list(set(data_part_ceo.columns) | set(data_part_dsz.columns)))

        data_part_ceo_fill = data_part_ceo.reindex(index=xinx, columns=xnms).fillna(method='ffill')
        data_part_dsz_fill = data_part_dsz.reindex(index=xinx, columns=xnms).fillna(method='ffill')

        data_part_ceo_fill = data_part_ceo_fill.reindex(index=self.date_index)
        data_part_dsz_fill = data_part_dsz_fill.reindex(index=self.date_index)
        data_dsz_ceo = (data_part_ceo_fill == data_part_dsz_fill).astype(int)
        stock_code_df_2 = self.company_to_stock(data_dsz_ceo)
        return stock_code_df_2

    def index_tab1_3(self):
        pass

    def index_tab1_4(self):
        pass

    def index_tab1_5(self, time_period=250):
        """
        监事会会议次数
        :return:
        """
        data = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_MO_MANS.pkl')
        part_data = data[data['PERSONTYPE'] == '02']
        company_meet = part_data.groupby(['PASSNOTICEDATE', 'COMPANYCODE'])['PERSONTYPE'].sum().unstack()
        company_meet = company_meet.reindex(set(company_meet.index) & set(self.date_index))
        company_meet_num = company_meet.notna().astype(int)
        company_meet_num_s = company_meet_num.rolling(window=time_period).sum()
        stock_code_df_tab1_5 = self.company_to_stock(company_meet_num_s)
        return stock_code_df_tab1_5

    def index_tab1_7(self):
        """
        3 年内 CEO 变动(不含退休、任期届满)次数
        :return:
        """
        data = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_MO_MANHOLDRPAY.pkl')
        CEO_list = np.array(list(map(lambda x: False if type(x) is not str else
        (True if ('110101' in x.split(',') or
                  '110102' in x.split(',')) and
                 len(x.split(',')) != 1 else False), data['POSTCODE'])))
        data_part_ceo = data[CEO_list].groupby(['NOTICEDATE', 'COMPANYCODE'])['PERSONCODE'] \
            .apply(f).unstack()

        xinx = np.array(sorted(list(set(data_part_ceo.index) | set(self.date_index))))

        data_part_ceo_fill = data_part_ceo.reindex(index=xinx).fillna(method='ffill')
        data_part_ceo_fill = data_part_ceo_fill.reindex(index=self.date_index)
        data_part_ceo = data_part_ceo_fill.rolling(window=250, min_periods=0).apply(lambda x: len(list(set(x))))
        stock_code_df_tab1_7 = self.company_to_stock(data_part_ceo)
        return stock_code_df_tab1_7

    def index_tab1_8(self):
        """
        3 年内董事长变动(不含退休、任期届满)次数
        :return:
        """
        data = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_MO_MANHOLDRPAY.pkl')
        DSZ_list = np.array(list(map(lambda x: False if type(x) is not str else
        (True if '010101' in x.split(',') and
                 len(x.split(',')) != 1 else False), data['POSTCODE'])))
        data_part_dsz = data[DSZ_list].groupby(['NOTICEDATE', 'COMPANYCODE'])['PERSONCODE'] \
            .apply(f).unstack()

        xinx = np.array(sorted(list(set(data_part_dsz.index) | set(self.date_index))))
        xinx = xinx[xinx > pd.to_datetime('20100101')]

        data_part_dsz_fill = data_part_dsz.reindex(index=xinx).fillna(method='ffill')
        data_part_dsz_fill = data_part_dsz_fill.reindex(index=self.date_index)
        data_part_dsz = data_part_dsz_fill.rolling(window=250, min_periods=0).apply(lambda x: len(list(set(x))))
        stock_code_df_8 = self.company_to_stock(data_part_dsz)
        return stock_code_df_8

    def index_tab1_9(self):
        """
        独立董事 占所有董事比率
        :return:
        """
        data_dlds = self.LICO_MO_DSHJS[self.LICO_MO_DSHJS['POSTCODE'] == '010401'] \
            .dropna(how='any', subset=['STARTDATE', 'ENDDATE'])

        company_code_df_dlds = self.data_deal(data_dlds)

        data_ds = self.LICO_MO_DSHJS[[self.fun(x) for x in self.LICO_MO_DSHJS['POSTCODE']]] \
            .dropna(how='any', subset=['STARTDATE', 'ENDDATE'])
        company_code_df_ds = self.data_deal(data_ds)
        dlds_ratio = company_code_df_dlds / company_code_df_ds
        stock_code_df = self.company_to_stock(dlds_ratio)
        return stock_code_df

    def index_tab2_1(self):
        """
        股权集中度指标(前 8 大股东股票比例)
        :return:
        """

        def f_1(x):
            x = x.sort_values(ascending=False)
            print(x)
            if sum(x) > 60:
                return sum(x.iloc[:8])
            else:
                return np.nan

        data = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_ES_LISHOLD.pkl')
        raw_df = data.groupby(['NOTICEDATE', 'COMPANYCODE'])['SHAREHDRATIO'].apply(f_1).unstack()
        stock_code_df = self.company_to_stock(raw_df)
        return stock_code_df

    def index_tab2_2(self):
        """
        公司是否有实际控制人*
        :return:
        """

    def index_tab2_3(self):
        """
        两权分离度(控制权/所有权)
        :return:
        """

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
            # print(x)
            x = x.replace(0, np.nan)
            js_share_ratio = x.notnull().sum() / len(x)
            # print(js_share_ratio)
            return js_share_ratio

        LICO_MO_MANHOLDRPAY = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_MO_MANHOLDRPAY.pkl')
        data_js = LICO_MO_MANHOLDRPAY[[f_1(x) for x in LICO_MO_MANHOLDRPAY['POSTCODE']]]
        raw_df = data_js.groupby(['NOTICEDATE', 'COMPANYCODE'])['EHN'].apply(f_2).unstack()
        stock_code_df = self.company_to_stock(raw_df)
        return stock_code_df

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

        LICO_MO_MANHOLDRPAY = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_MO_MANHOLDRPAY.pkl')
        data_gg = LICO_MO_MANHOLDRPAY[[f_1(x) for x in LICO_MO_MANHOLDRPAY['POSTCODE']]]
        raw_df = data_gg.groupby(['NOTICEDATE', 'COMPANYCODE'])['EHN'].apply(f_2).unstack()
        stock_code_df = self.company_to_stock(raw_df)
        return stock_code_df

    def index_tab2_6(self):
        """
        管理层(除高管外)持股比例
        :return:
        """

    def index_tab2_7(self):
        """
        基金持股比例
        :return:
        """

        data = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_ES_LISHOLD.pkl')
        data_jj = data[(data['SHAREHDTYPE'] == '007') | (data['SHAREHDTYPE'] == '008') | (data['SHAREHDTYPE'] == '018')]
        raw_df = data_jj.groupby(['NOTICEDATE', 'COMPANYCODE'])['SHAREHDRATIO'].sum().unstack()
        stock_code_df = self.company_to_stock(raw_df)
        return stock_code_df

    def index_tab2_8(self):
        """
        社保基金持股比例
        :return:
        """
        data = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_ES_LISHOLD.pkl')
        data_sbjj = data[data['SHAREHDTYPE'] == '012']
        raw_df = data_sbjj.groupby(['NOTICEDATE', 'COMPANYCODE'])['SHAREHDRATIO'].sum().unstack()
        stock_code_df = self.company_to_stock(raw_df)
        return stock_code_df

    def index_tab2_9(self):
        """
        QFII 持股比例
        :return:
        """
        data = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_ES_LISHOLD.pkl')
        data_qfii = data[data['SHAREHDTYPE'] == '001']
        raw_df = data_qfii.groupby(['NOTICEDATE', 'COMPANYCODE'])['SHAREHDRATIO'].sum().unstack()
        stock_code_df = self.company_to_stock(raw_df)
        return stock_code_df

    def index_tab2_10(self):
        """
        非金融类上市公司持股比例
        :return:
        """
        data = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_ES_LISHOLD.pkl')
        data_fjr = data[data['SHAREHDTYPE'] == '013']
        raw_df = data_fjr.groupby(['NOTICEDATE', 'COMPANYCODE'])['SHAREHDRATIO'].sum().unstack()
        stock_code_df = self.company_to_stock(raw_df)
        return stock_code_df

    def index_tab2_11(self):
        """
        是否发行 H 股
        :return:
        """
        data = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_ES_CPHSSTRUCT.pkl')
        a = data.groupby(['NOTICEDATE', 'COMPANYCODE'])['HSHARE'].sum().unstack()
        h_share_df = a.replace(0, np.nan).dropna(how='all', axis='columns')
        raw_df = h_share_df.isnan()
        stock_code_df = self.company_to_stock(raw_df)
        return stock_code_df

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
            exe_pay = sum(x.sort_values(ascending=False).iloc[:3])
            if exe_pay < 1:
                log_exe_pay = np.nan
            else:
                log_exe_pay = np.log(exe_pay)
            return log_exe_pay

        data = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_MO_MANHOLDRPAY.pkl')
        data_ds = data[[f_1(x) for x in data['POSTCODE']]]

        raw_df = data_ds.groupby(['NOTICEDATE', 'COMPANYCODE'])['ANUALWAGE'].apply(fun)
        stock_code_df = self.company_to_stock(raw_df)
        return stock_code_df

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

        data = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_MO_MANHOLDRPAY.pkl')
        data_ds = data[[f_1(x) for x in data['POSTCODE']]]

        raw_df = data_ds.groupby(['NOTICEDATE', 'COMPANYCODE'])['ANUALWAGE'].apply(fun).unstack()
        raw_df = raw_df.fillna(method='ffill', limit=300)
        stock_code_df = self.company_to_stock(raw_df)
        return stock_code_df

    def index_tab4_3(self):
        """
        员工持股计划/总股本
        :return:
        """
        data = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_ES_EMSHAREDE.pkl')
        raw_df = data.groupby(['FIRSTNOTICEDATE', 'SECINNERCODE'])['RATIO'] \
            .sum().unstack().fillna(method='ffill', limit=300)
        stock_code_df = self.company_to_stock(raw_df)
        return stock_code_df

    def index_tab4_4(self):
        """
        高管员工持股计划/总股本
        :return:
        """

    def index_tab4_5(self):
        """
        股权激励权益/总股本
        :return:
        """
        data = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_MO_GQJLJBZL.pkl')
        raw_df = data.groupby(['NOTICEDATE', 'COMPANYCODE'])['EXCISHARERATIO'] \
            .sum().unstack().fillna(method='ffill', limit=300)
        stock_code_df = self.company_to_stock(raw_df)
        return stock_code_df

    def index_tab4_6(self):
        """
        高管激励权益/总股本
        :return:
        """

    def index_tab5_13(self):
        """
        近 1 年来违规次数
        :return:
        """
        data = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_CM_ILLEGAL.pkl')
        violation_df = data.groupby(['NOTICEDATE', 'COMPANYCODE'])['GOOLACTION'].sum().unstack()
        xinx = sorted(list(set(violation_df.index) | set(self.date_index)))
        violation_df = violation_df.reindex(index=xinx)
        violation_df_mask = violation_df.notna()
        raw_df = violation_df_mask.rolling(250).sum()
        stock_code_df = self.company_to_stock(raw_df)
        return stock_code_df

    def index_tab5_14(self):
        """
        近半年来诉讼仲裁涉案金额数/营业收入
        :return: 
        """
        data = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_CM_LAWARBI.pkl')
        amount_involved = data.groupby(['NOTICEDATE', 'COMPANYCODE'])['SHEANJINE'].sum().unstack()
        xinx = sorted(list(set(amount_involved.index) | set(self.date_index)))
        amount_involved = amount_involved.reindex(index=xinx)
        raw_df = amount_involved.rolling(250, min_periods=0).sum()
        raw_df = raw_df.reindex(self.date_index)
        stock_code_df = self.company_to_stock(raw_df)
        Revenue_TTM = pd.read_csv('/mnt/mfs/DAT_EQT/EM_Funda/daily/R_Revenue_TTM_First.csv', sep='|', index_col=0,
                                  parse_dates=True).reindex(index=self.date_index, columns=stock_code_df.columns)
        target_df = (stock_code_df / Revenue_TTM).repalce(0, np.nan)
        return target_df

    def index_tab5_15(self):
        """
        冻结股票比例
        :return:
        """
        data = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_ES_SHHDFROZEN.pkl')

        data.loc[data['ENDDATE'] > datetime(3018, 12, 14, 0, 0), 'ENDDATE'] = None

        data['ENDDATE'] = pd.to_datetime(data['ENDDATE'].values)
        data['FREEDATE'] = pd.to_datetime(data['FREEDATE'].values)
        data['NOTICEDATE'] = pd.to_datetime(data['NOTICEDATE'].values)

        FREEDATE_mask = data['FREEDATE'].notnull()
        data.loc[FREEDATE_mask, 'ENDDATE'] = data.loc[FREEDATE_mask, 'FREEDATE']
        data = data.dropna(how='any', subset=['NOTICEDATE', 'ENDDATE'])
        data = data[data['ENDDATE'] > data['NOTICEDATE']]

        df_start = data.groupby(['NOTICEDATE', 'COMPANYCODE'])['FROZENINTOTAL'].sum().unstack()
        df_end = data.groupby(['ENDDATE', 'COMPANYCODE'])['FROZENINTOTAL'].sum().unstack()

        xinx = sorted(list(set(df_start.index) | set(df_end.index)))

        df_start_cum = df_start.reindex(xinx).fillna(0).cumsum()
        df_end_cum = df_end.reindex(xinx).fillna(0).cumsum()

        raw_df = (df_start_cum - df_end_cum).round(5)
        stock_code_df = self.company_to_stock(raw_df)
        return stock_code_df


# def select_astock(x):
#     if len(x) == 6:
#         if x[:1] in ['0', '3', '6']:
#             return True
#         else:
#             return False
#     else:
#         return False
#
#
# def add_suffix(x):
#     if x[0] in ['0', '3']:
#         return x + '.SZ'
#     elif x[0] in ['6']:
#         return x + '.SH'
#     else:
#         print('error')
#
#
# def func():
#     def f(x):
#         return 1
#
#     data = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_IM_GDDHDATE.pkl')
#     data = data[data['EISDEL'] == '0']
#
#     part_data = data[data['DATETYPECODE'] == '006']
#
#     # part_data.loc[:, 'STARTDATE'] = [pd.to_datetime(x.strftime('%Y%m%d')) for x in part_data['STARTDATE']]
#     part_data.loc[:, ['STARTDATE']] = pd.to_datetime([x.strftime('%Y%m%d') for x in part_data['STARTDATE']])
#
#     target_df = part_data.groupby(['STARTDATE', 'SECURITYCODE']).apply(f).unstack()
#     target_df = target_df[[x for x in target_df.columns if select_astock(x)]]
#     target_df.columns = [add_suffix(x) for x in target_df.columns]
#     target_df.to_csv('/mnt/mfs/DAT_EQT/EM_Funda/LICO_IM_GDDHDATE/STARTDATE_006.csv', sep='|')


if __name__ == '__main__':
    esg = ESG()
    a = time.time()
    # stock_code_df_tab1_1 = esg.index_tab1_1()
    # stock_code_df_tab1_2 = esg.index_tab1_2()
    # stock_code_df_tab1_5 = esg.index_tab1_5()
    stock_code_df_tab1_7 = esg.index_tab1_7()
    # stock_code_df_tab1_8 = esg.index_tab1_8()
    b = time.time()
    print(b - a)
    return_df = pd.read_csv('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                            sep='|', index_col=0, parse_dates=True)
    date_index = return_df.index

# ZIWHEN
# OPERATEREVE
# OPERATEREVE_S
# Revenue_TTM
