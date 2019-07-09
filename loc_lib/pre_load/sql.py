import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *
from sqlalchemy import create_engine

usr_name = 'whs'
pass_word = 'kj23#12!^3weghWhjqQ2rjj197'
engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.33:3306/choice_fndb?charset=utf8')
conn = engine.connect()


def mysql_select(select_col_list, table_name, conn, key_col=None, cond=None, cpu_num=20, step=100):
    key_col_list = pd.read_sql(f'SELECT DISTINCT {key_col} '
                               f'FROM {table_name}', conn).values.ravel()
    select_col_str = ', '.join(select_col_list)

    def fetch_data(sids):
        print(f"SELECT {select_col_str} "
              f"FROM {table_name} "
              f"WHERE {key_col} in {str(tuple(sids))} AND BulletinType = 'lsgg'")
        lsgg_df = pd.read_sql(f"SELECT {select_col_str} "
                              f"FROM {table_name} "
                              f"WHERE {key_col} in {str(tuple(sids))} AND BulletinType = 'lsgg'", conn)
        return lsgg_df

    p = ThreadPool(cpu_num)
    res = pd.concat(p.map(fetch_data, [key_col_list[i: i + step] for i in range(0, len(key_col_list), step)]))
    return res


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


def fill_index(map_data_c, date_index, df, limit=None):
    """

    :param map_data_c:
    :param date_index:
    :param df:
    :param limit:
    :return:
    """
    xinx = sorted(list(set(df.index) | set(date_index)))
    tmp_df = df.reindex(index=xinx, columns=map_data_c.index).fillna(method='ffill', limit=limit)
    target_df = tmp_df.reindex(index=date_index)
    return target_df


def company_to_stock(map_data_c, company_code_df):
    """
    把table columns company code替换成stock code
    :param map_data_c:
    :param company_code_df:
    :return:
    """
    print('company_to_stock')
    # stock_code_df = pd.DataFrame(columns=sorted(self.map_data_c.values))
    company_code_df = company_code_df.reindex(columns=map_data_c.index)
    company_code_df.columns = map_data_c.values
    company_code_df = company_code_df[sorted(company_code_df.columns)]
    company_code_df.columns = [add_suffix(x) for x in company_code_df.columns]
    company_code_df.dropna(how='all', inplace=True, axis='columns')
    return company_code_df


def get_map_data():
    map_data = pd.read_sql('SELECT COMPANYCODE,SECURITYCODE,SECURITYTYPE FROM choice_fndb.CDSY_SECUCODE', conn)
    map_data.index = map_data['COMPANYCODE']
    map_data_c = map_data['SECURITYCODE'][map_data['SECURITYTYPE'] == 'A股']
    map_data_c = map_data_c[map_data_c.apply(select_astock)]
    return map_data_c


def save_fun(df, save_path, sep='|'):
    print(1)
    print(save_path)
    df.to_csv(save_path, sep=sep)
    test_save_path = '/mnt/mfs/dat_whs/EM_Funda/{}'.format(datetime.now().strftime('%Y%m%d'))
    bt.AZ_Path_create(test_save_path)
    df.to_csv(os.path.join(test_save_path, os.path.split(save_path)[-1]))


def date_deal_fun(x):
    cut_date = datetime(year=x.year, month=x.month, day=x.day, hour=7, minute=00)
    if cut_date > x:
        # print(datetime(year=x.year, month=x.month, day=x.day) - timedelta(days=1))
        return datetime(year=x.year, month=x.month, day=x.day) - timedelta(days=1)
    else:
        # print(datetime(year=x.year, month=x.month, day=x.day))
        return datetime(year=x.year, month=x.month, day=x.day)


if __name__ == '__main__':
    raw_df = pd.read_sql(f'SELECT REPORTDATE, FIRSTNOTICEDATE, LATESTNOTICEDATE, STARTDATE '
                         f'FROM LICO_FN_RGINCOME', conn)
    print(raw_df)
