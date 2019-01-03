import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *


def data_test():
    sector_name = 'market_top_2000'

    # root_path_1 = '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_10'
    root_path_1 = '/media/hdd1/DAT_EQT/EM_Funda/DERIVED_F1'
    # root_path_2 = '/media/hdd1/DAT_EQT/EM_Funda/DERIVED_10'
    root_path_2 = '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_F1'

    file_name_list = ['REMFF.04', 'REMFF.06', 'REMFF.08', 'REMFF.11', 'REMFF.12', 'REMFF.13', 'REMFF.15', 'REMFF.20',
                      'REMFF.21', 'REMFF.30', 'REMFF.33', 'REMFF.34', 'REMFF.37', 'REMTK.06', 'REMTK.07', 'REMTK.13',
                      'REMTK.17', 'REMTK.25', 'REMTK.26', 'REMTK.29', 'REMTK.30', 'REMTK.35', 'REMWB.08', 'REMWB.09',
                      'REMWB.12'
                      ]

    # file_name_list = ['market_top_300to800plus_industry_10_15.csv']

    begin_date = pd.to_datetime('20130101')
    end_date = pd.to_datetime('20190101')
    for file_name in file_name_list:
        print(file_name)
        load_path_1 = os.path.join(root_path_1, file_name)
        load_path_2 = os.path.join(root_path_2, file_name)

        # data1 = pd.read_pickle(load_path_1)
        # data2 = pd.read_pickle(load_path_2)
        #
        data1 = bt.AZ_Load_csv(load_path_1)
        data2 = bt.AZ_Load_csv(load_path_2)

        a = data1.loc[begin_date: end_date].dropna()
        b = data2.loc[begin_date: end_date].dropna()
        xinx = sorted(list(set(a.index) & set(b.index)))
        xnms = sorted(list(set(a.columns) & set(b.columns)))

        a = a.reindex(index=xinx, columns=xnms)
        b = b.reindex(index=xinx, columns=xnms)

        c = (a != b)

        d = c.sum().sum()
        print(d)
        if d != 0:
            # print((a[c] - b[c]).sum())
            # print(pd.concat([a[c], b[c]], axis=1))
            # break
            pass


def find_diff(file_path):
    p1 = '/media/hdd1/DAT_EQT/EM_Funda/DERIVED_F1'
    p2 = '/media/hdd1/DAT_EQT/EM_Funda/DERIVED_F1T'

    data1 = pd.read_pickle(os.path.join(p1, file_path)).loc[:pd.to_datetime('20180919')]
    data2 = pd.read_pickle(os.path.join(p2, file_path)).loc[:pd.to_datetime('20180919')]


if __name__ == '__main__':
    data_test()
