import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *
import work_whs.loc_lib.pre_load.sql as sql


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


# class AAA:
#     def __init__(self, save_path):
#         # 高管关联人持股
#
if __name__ == '__main__':
    print(1)
    raw_df = pd.read_sql('SELECT * FROM choice_fndb.LICO_MO_MANRPHOLD', sql.conn)
    print(2)
    # raw_df[['NOTICEDATE', 'PERSONCODE', 'PERSONNAME', 'POSITIONCODE', 'CHANNUM', 'AVGPRICE']]
    raw_df['EITIME'] = [sql.date_deal_fun(x) for x in raw_df['EITIME']]
    raw_df['NOTICEDATE'] = sql.create_date_list(raw_df['EITIME'], raw_df['NOTICEDATE'])

    raw_df = time_format(raw_df, 'EITIME')
    map_data_c = sql.get_map_data()
    raw_df_c = raw_df[list(map(lambda x: True if x in map_data_c.index else False, raw_df['COMPANYCODE']))]
    raw_df_c['CHANCASH'] = raw_df_c['CHANNUM'] * raw_df_c['AVGPRICE']
    tmp_df = raw_df_c.groupby(['NOTICEDATE', 'COMPANYCODE'])['CHANCASH'].sum().unstack()
    target_df = company_to_stock(map_data_c, tmp_df)
    target_df_1 = (target_df > 100000).astype(int)
    target_df_2 = (target_df > 200000).astype(int)
    target_df_3 = (target_df > 300000).astype(int)
    data_raw = (target_df_1 + target_df_2 + target_df_3)

    a = pd.DataFrame.from_dict(Counter(data_raw.columns), orient='index')
    b = a[a > 1].dropna()
    c = a[a == 1].dropna().index
    data = data_raw[c]
    for code in b.index:
        print(code)
        data[code] = data_raw[code].sum(1)
    data.to_pickle('/mnt/mfs/dat_whs/LICO_MO_MANRPHOLD.pkl')
