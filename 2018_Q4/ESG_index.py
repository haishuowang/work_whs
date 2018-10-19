import pandas as pd


def fun(x):
    if '董事' in x:
        return True
    else:
        return False


def fun1(x):
    if len(x) == 6:
        if x[:1] in ['0', '3', '6']:
            return True
        else:
            return False
    else:
        return False


def fun2(x):
    if x == '上交所主板' or x == '深交所主板':
        return True
    else:
        return False


map_data = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/CDSY_SECUCODE.pkl')
map_data.index = map_data['COMPANYCODE']

map_data_c = map_data['SECURITYCODE'][map_data['SECURITYTYPE'] == 'A股']
map_data_c = map_data_c[map_data_c.apply(fun1)]


# ZIWHEN

def director_num(map_data_c):
    data = pd.read_pickle('/mnt/mfs/dat_whs/EM_Funda/LICO_MO_DSHJS.pkl')

    # part_df = df[['PERSONNAME', 'POST', 'STATUS', 'SESS', 'STARTDATE', 'ENDDATE']]

    data = data[[fun(x) for x in data['POST']]].dropna(how='any', subset=['STARTDATE', 'ENDDATE'])

    df_start = data.groupby(['STARTDATE', 'COMPANYCODE']).apply(lambda x: len(x)).unstack()
    df_end = data.groupby(['ENDDATE', 'COMPANYCODE']).apply(lambda x: len(x)).unstack()

    index_list = sorted(list(set(df_start.index) | set(df_start.index)))

    df_start = df_start.reindex(index=index_list)
    df_end = df_end.reindex(index=index_list)

    df_start_cum = df_start.fillna(0).cumsum()
    df_end_cum = df_end.fillna(0).cumsum()

    company_num_df = df_start_cum - df_end_cum
    company_num_df = company_num_df[company_num_df > 0]

    map_df = map_data_c.loc[company_num_df.columns].dropna(how='all', axis=0)

    company_num_df = company_num_df[map_df.index]
    company_num_df.columns = map_df.values.ravel()

    map_data_c.loc[company_num_df.columns]
    # b = map_data['SECURITYCODE'][]
