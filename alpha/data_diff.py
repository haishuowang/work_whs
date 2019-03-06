import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *


def data_test():
    sector_name = 'market_top_2000'

    # root_path_1 = '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_10'
    # root_path_2 = '/media/hdd1/DAT_EQT/EM_Funda/DERIVED_10'

    # root_path_1 = '/media/hdd1/DAT_EQT/EM_Funda/DERIVED_F1'
    # root_path_2 = '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_F1'

    root_path_1 = '/mnt/mfs/dat_whs/data/new_factor_data/market_top_300to800plus_industry_20_25_30_35'
    root_path_2 = '/media/hdd1/DAT_PreCalc/PreCalc_whs/market_top_300to800plus_industry_20_25_30_35'
    # price_p10d_hl 1 turn_p150d_0.18 0 moment_p510d 很多 aadj_r_p345d_continue_ud * volume_moment_p530d 0
    #  continue_ud_p200d 3 TVOL_row_extre_0.2|evol_p10d|volume_count_down_p10d pass BBANDS_10_2 75738
    factor_str = 'bias_turn_p120d'
    file_name_list = factor_str.split('|')
    # file_name_list = ['continue_ud_p20d', 'continue_ud_p100d', 'TVOL_p345d_continue_ud', 'BBANDS_10_1',
    #                   'R_OTHERLASSET_QYOY_and_QTTM_0.3', 'tab2_9_row_extre_0.3']

    begin_date = pd.to_datetime('20130101')
    end_date = pd.to_datetime('20190305')
    for file_name in file_name_list:
        print(file_name)
        load_path_1 = os.path.join(root_path_1, file_name + '.pkl')
        load_path_2 = os.path.join(root_path_2, file_name + '.pkl')

        data1 = pd.read_pickle(load_path_1)
        data2 = pd.read_pickle(load_path_2)
        #
        # data1 = bt.AZ_Load_csv(load_path_1)
        # data2 = bt.AZ_Load_csv(load_path_2)

        a = data1.loc[begin_date: end_date].dropna(how='all', axis='columns')
        b = data2.loc[begin_date: end_date].dropna(how='all', axis='columns')
        xinx = sorted(list(set(a.index) & set(b.index)))
        xnms = sorted(list(set(a.columns) & set(b.columns)))

        a = a.reindex(index=xinx, columns=xnms).replace(np.nan, 0)
        b = b.reindex(index=xinx, columns=xnms).replace(np.nan, 0)

        c = (a != b)

        d = c.sum().sum()
        print(d)
        if d != 0:
            print((a[c] - b[c]).sum())
            a_c = a[c].dropna(how='all', axis=1).dropna(how='all', axis=0)
            b_c = b[c].dropna(how='all', axis=1).dropna(how='all', axis=0)
            # print(pd.concat([a[c], b[c]], axis=1))
            # print(a[c], b[c])
            # break
            # pass
            return file_name, a, b, c
    return None, None, None


def find_diff(file_path):
    p1 = '/media/hdd1/DAT_EQT/EM_Funda/DERIVED_F1'
    p2 = '/media/hdd1/DAT_EQT/EM_Funda/DERIVED_F1T'

    # data1 = pd.read_pickle(os.path.join(p1, file_path)).loc[:pd.to_datetime('20180919')]
    # data2 = pd.read_pickle(os.path.join(p2, file_path)).loc[:pd.to_datetime('20180919')]


if __name__ == '__main__':
    file_name, a, b, c = data_test()
    a_c = a[c].dropna(how='all', axis=1).dropna(how='all', axis=0)
    b_c = b[c].dropna(how='all', axis=1).dropna(how='all', axis=0)
    print(a_c)
    print(b_c)
