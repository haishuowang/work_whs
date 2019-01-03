import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *


# root_path = '/mnt/mfs/dat_whs/tmp'
# # date_path_list = os.listdir(root_path)
#
# date_list = ['201811211648', '201811231800']
#
# data_path_1 = f'{root_path}/{date_list[0]}'
# data_path_2 = f'{root_path}/{date_list[1]}'
#
# sector_list = sorted(os.listdir(data_path_1))
# for sector_name in sector_list:
#     sector_path_1 = f'{data_path_1}/{sector_name}'
#     sector_path_2 = f'{data_path_2}/{sector_name}'
#     index_list = sorted(os.listdir(sector_path_1))
#     for index_name in index_list:
#         print(index_name)
#         index_df_1 = pd.read_pickle(f'{sector_path_1}/{index_name}').dropna(how='all', axis=1).dropna(how='all', axis=0)
#         index_df_2 = pd.read_pickle(f'{sector_path_2}/{index_name}').dropna(how='all', axis=1).dropna(how='all', axis=0)
#         xinx = sorted(list(set(index_df_1.index) & set(index_df_2.index)))
#         xnms = sorted(list(set(index_df_1.columns) | set(index_df_2.columns)))
#         a = index_df_1.reindex(index=xinx, columns=xnms).replace(np.nan, 0)
#         b = index_df_2.reindex(index=xinx, columns=xnms).replace(np.nan, 0)
#
#         c = a != b
#         d = c.sum().sum()
#         print(d)
#         if d != 0:
#             exit(-1)


def date_check(path_1, path_2, file_type='csv'):
    file_list = sorted(os.listdir(path_1))
    cut_date = pd.to_datetime('20180101')
    for file_name in file_list:
        if file_name in ['bulletin_num_df.csv', 'lsgg_num_df_20.csv', 'funds.csv', 'buy_summary_key_word.csv',
                         'buy_key_title__word.csv', 'lsgg_num_df_5.csv', 'lsgg_num_df_60.csv', 'meeting_decide.csv',
                         'news_num_df_20.csv', 'news_num_df_5.csv', 'news_num_df_60.csv', 'sell_summary_key_word.csv',
                         'sell_key_title_word.csv', 'restricted_shares.csv', 'shares.csv', 'son_company.csv',
                         'staff_changes.csv', 'suspend.csv', '', '', '']:
            continue

        print(file_name)
        if file_type == 'csv':
            data1 = bt.AZ_Load_csv(os.path.join(path_1, file_name))
            data2 = bt.AZ_Load_csv(os.path.join(path_2, file_name))
        elif file_type == 'pkl':
            data1 = pd.read_pickle(os.path.join(path_1, file_name))
            data2 = pd.read_pickle(os.path.join(path_2, file_name))
        xinx = np.array(sorted(list(set(data1.index) & set(data2.index))))
        xnms = sorted(list(set(data1.columns) | set(data2.columns)))
        xinx = xinx[xinx > cut_date][:-1]
        part_1 = data1.reindex(index=xinx, columns=xnms).replace(np.nan, 0)
        part_2 = data2.reindex(index=xinx, columns=xnms).replace(np.nan, 0)

        c = part_1 != part_2
        d = c.sum().sum()
        print(d)
        if d != 0:
            return part_1, part_2, c
    print('Perfect')
    return 0, 0, 0


if __name__ == '__main__':
    path_1 = '/mnt/mfs/dat_whs/EM_Funda/my_data_test_TEST'
    path_2 = '/mnt/mfs/dat_whs/EM_Funda/my_data_test'
    part_1, part_2, c = date_check(path_1, path_2, file_type='csv')
    if c != 0:
        a = part_1[c].dropna(how='all', axis=1).dropna(how='all', axis=0)
        b = part_2[c].dropna(how='all', axis=1).dropna(how='all', axis=0)
