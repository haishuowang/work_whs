import pandas as pd

config_set_1 = pd.read_pickle(f'/mnt/mfs/alpha_whs/CRTSECJUN04.pkl')
config_data_1 = config_set_1['factor_info']
a1 = sorted(list(set(config_data_1[['name1', 'name2', 'name3']].values.ravel())))
b1 = list(config_data_1[['name1', 'name2', 'name3']].values.ravel())
c1 = dict()
for i in a1:
    c1[i] = b1.count(i)

config_set_2 = pd.read_pickle(f'/mnt/mfs/alpha_whs/CRTJUN01.pkl')
config_data_2 = config_set_2['factor_info']
a2 = sorted(list(set(config_data_2[['name1', 'name2', 'name3']].values.ravel())))
b2 = list(config_data_2[['name1', 'name2', 'name3']].values.ravel())
c2 = dict()
for i in a2:
    c2[i] = b2.count(i)
