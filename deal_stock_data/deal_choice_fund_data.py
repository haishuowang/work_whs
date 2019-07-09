import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load.sql import conn


def add_suffix(x):
    if x[0] in ['0', '3']:
        return x + '.SZ'
    elif x[0] in ['6']:
        return x + '.SH'
    else:
        print('error')


def select_astock(x):
    if len(x) == 6:
        if x[:1] in ['0', '3', '6']:
            return True
        else:
            return False
    else:
        return False


def company_to_stock(map_data_c, company_code_df):
    # stock_code_df = pd.DataFrame(columns=sorted(self.map_data_c.values))
    company_code_df = company_code_df.reindex(columns=map_data_c.index)
    company_code_df.columns = map_data_c.values
    company_code_df = company_code_df[sorted(company_code_df.columns)]
    company_code_df.columns = [add_suffix(x) for x in company_code_df.columns]
    company_code_df.dropna(how='all', inplace=True, axis='columns')
    return company_code_df


table_name = 'LICO_FN_ACCOUNTPAY5'

# table_name = LICO_FN_CUSTOMER5

# table_name = LICO_FN_SUPPLIER5


class ChoiceDataDeal:
    def __init__(self, root_path):
        map_data = pd.read_sql('SELECT COMPANYCODE, SECURITYCODE, SECURITYTYPE '
                               'FROM choice_fndb.CDSY_SECUCODE '
                               'WHERE EISDEL=0', conn)
        map_data.index = map_data['COMPANYCODE']
        map_data_c = map_data['SECURITYCODE'][map_data['SECURITYTYPE'] == 'A股']
        self.map_data_c = map_data_c[map_data_c.apply(select_astock)]
        self.root_path = root_path

    def deal_LICO_FN_CUSTOMER5(self):
        """
        # 前五大客户
        :return:
        """
        def fun(x):
            return x.sort_values(by=['REPORTDATE'])['PCTREVENUESUM'].iloc[-1]

        raw_df = pd.read_sql('SELECT FIRSTNOTICEDATE, COMPANYCODE, PCTREVENUESUM, ACTUALITEM, REPORTDATE '
                             'FROM choice_fndb.LICO_FN_CUSTOMER5 '
                             'WHERE EISDEL=0', conn)

        raw_df_c = raw_df[raw_df['ACTUALITEM'] == '合计']
        tmp_df = raw_df_c.groupby(['FIRSTNOTICEDATE', 'COMPANYCODE'])[['PCTREVENUESUM', 'REPORTDATE']].apply(fun).unstack()
        target_df = company_to_stock(self.map_data_c, tmp_df)
        return target_df

    def deal_LICO_FN_SUPPLIER5(self):
        """
        # 前五名供应商
        :return:
        """
        def fun(x):
            return x.sort_values(by=['REPORTDATE'])['PCTSUPPLYSUM'].iloc[-1]

        raw_df = pd.read_sql('SELECT FIRSTNOTICEDATE, COMPANYCODE, PCTSUPPLYSUM, ACTUALITEM, REPORTDATE '
                             'FROM choice_fndb.LICO_FN_SUPPLIER5 '
                             'WHERE EISDEL=0', conn)

        raw_df_c = raw_df[raw_df['ACTUALITEM'] == '合计']
        tmp_df = raw_df_c.groupby(['FIRSTNOTICEDATE', 'COMPANYCODE'])[['PCTSUPPLYSUM', 'REPORTDATE']].apply(
            fun).unstack()
        target_df = company_to_stock(self.map_data_c, tmp_df)
        return target_df


# choice_data_deal = ChoiceDataDeal('/mnt/mfs/DAT_EQT')
# # 客户集中度
# target_df_1 = choice_data_deal.deal_LICO_FN_CUSTOMER5()
# # 供应商集中度
# target_df_2 = choice_data_deal.deal_LICO_FN_SUPPLIER5()
#
# target_df_1.to_pickle('/mnt/mfs/temp/客户集中度.pkl')
# target_df_2.to_pickle('/mnt/mfs/temp/供应商集中度.pkl')
