import pandas as pd
import numpy as np
import os


def data_test():
    sector_name = 'market_top_2000'

    root_path_1 = os.path.join('/media/hdd1/DAT_PreCalc/PreCalc_whs/tmp', sector_name)
    root_path_2 = os.path.join('/media/hdd1/DAT_PreCalc/PreCalc_whs', sector_name)

    file_name_list = os.listdir(root_path_1)

    for file_name in file_name_list:
        print(file_name)
        load_path_1 = os.path.join(root_path_1, file_name)
        load_path_2 = os.path.join(root_path_2, file_name)

        data1 = pd.read_pickle(load_path_1)
        data2 = pd.read_pickle(load_path_2)
        a = data1.loc[pd.to_datetime('20180918')].dropna()
        b = data2.loc[pd.to_datetime('20180918')].dropna()
        common_index = sorted(list(set(a.index) & set(b.index)))
        a = a.loc[common_index]
        b = b.loc[common_index]
        c = (a != b)
        d = c.sum()

        if d != 0:

            print((a[c] - b[c]).sum())
            print(d)
            print(pd.concat([a[c], b[c]], axis=1))
        #     break


def find_diff(file_path):
    p1 = '/media/hdd1/DAT_EQT'
    p2 = '/media/hdd1/B01/DAT_EQT'

    data1 = pd.read_pickle(os.path.join(p1, file_path)).loc[:pd.to_datetime('20180919')]
    data2 = pd.read_pickle(os.path.join(p2, file_path)).loc[:pd.to_datetime('20180919')]


# if __name__ == '__main__':
#     data_test()
    # file_path = 'EM_Funda/DERIVED_14/aadj_p.csv'
    # p1 = '/media/hdd1/DAT_EQT'
    # p2 = '/media/hdd1/B01/DAT_EQT'
    #
    # data1 = pd.read_csv(os.path.join(p1, file_path), sep='|', index_col=0, parse_dates=True).loc[
    #         pd.to_datetime('20170701'):pd.to_datetime('20170706'), '300273.SZ'].replace(np.nan, 0)
    # data2 = pd.read_csv(os.path.join(p2, file_path), sep='|', index_col=0, parse_dates=True).loc[
    #         pd.to_datetime('20170701'):pd.to_datetime('20170706'), '300273.SZ'].replace(np.nan, 0)
    # common_index = sorted(list(set(data1.index) & set(data2.index)))
    # a = data1.loc[common_index].dropna()
    # b = data2.loc[common_index].dropna()
    # print((a != b).sum())
    # print(pd.concat([a[a != b], b[a != b]], axis=1))
