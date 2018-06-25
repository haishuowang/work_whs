import pandas as pd


def add_stock_suffix(stock_list):
    return list(map(lambda x: x + '.SH' if x.startswith('6') else x + '.SZ', stock_list))

# info = pd.read_pickle('/mnt/mfs/DAT_EQT/EM_Tab1/CDSY_SECUCODE.pkl')
# universe_EQA2 = info[(info['SECURITYTYPECODE'] == '058001001') & (info['LISTSTATE'] != '9')]['SECURITYCODE'].values
# data_path = '/mnt/mfs/DAT_EQT/EM_Tab14/TRAD_TD_SUSPENDDAY.pkl'
# data = pd.read_pickle(data_path)
# universe_EQA = sorted(list(set(universe_EQA2) & set(data['SECURITYCODE'])))
# data.index = data['SECURITYCODE']
# data_path = '/mnt/mfs/DAT_EQT/EM_Tab14/EQA/TRAD_TD_SUSPEND.pkl'


data_1_path = '/mnt/mfs/DAT_EQT/EM_Tab14/EQA/TRAD_TD_SUSPENDDAY.pkl'
# data = pd.read_pickle(data_path)
data_1 = pd.read_pickle(data_1_path)

a = data_1.groupby(['TDATE', 'SECURITYCODE'])['SUSPENDREASON'].apply(lambda x: x.values[0]).unstack()
a.columns = add_stock_suffix(a.columns)
# a.to_pickle('/mnt/mfs/DAT_EQT/EM_Tab14/adj_data/TRAD_TD_SUSPENDDAY/SUSPENDREASON.pkl')
