from work_whs.loc_lib.pre_load import *

# path_1 = '/media/hdd1/DAT_PreCalc/PreCalc_whs/tmp/20/market_top_1000_industry_45_50'
# path_2 = '/media/hdd1/DAT_PreCalc/PreCalc_whs/tmp/21/market_top_1000_industry_45_50'
# file_list = os.listdir(path_1)
# # file_list = ['ADX_200_20_10.pkl']
# for file_name in file_list:
#     print(file_name)
#     data_1 = pd.read_pickle(os.path.join(path_1, file_name))
#     data_2 = pd.read_pickle(os.path.join(path_2, file_name))
#
#     xnms = list(sorted(set(data_1.columns) & set(data_2.columns)))
#     # xinx = [pd.to_datetime('20181119')]
#     xinx = list(sorted(set(data_1.index) & set(data_2.index)))
#     data_1 = data_1.reindex(index=xinx, columns=xnms).replace(np.nan, 0)
#     data_2 = data_2.reindex(index=xinx, columns=xnms).replace(np.nan, 0)
#
#     c = (data_1 != data_2)
#     print(c.sum().sum())
#     if c.sum().sum() != 0:
#         print('error!')
#         print(c.replace(False, np.nan).dropna(how='all', axis=0).dropna(how='all', axis=1))
#     else:
#         print('success!')

path_0 = '/mnt/mfs/dat_whs/EM_Funda/20181130'
path_1 = '/mnt/mfs/dat_whs/EM_Funda/20181128'
path_2 = '/mnt/mfs/dat_whs/EM_Funda/20181129'

# file_list = sorted(os.listdir(path_1))
# file_list = ['ADX_200_20_10.pkl']
# file_list = [
#     'stock_code_df_tab5_15',
#     'stock_code_df_tab5_14',
#     'stock_code_df_tab5_13',
#     'stock_code_df_tab2_7',
#     'stock_code_df_tab2_1',
#     'stock_code_df_tab2_5',
#     'stock_code_df_tab1_9',
#     'stock_code_df_tab1_5',
#     'stock_code_df_tab1_1'
# ]

file_list = [
    'stock_code_df_tab1_1',
    'stock_code_df_tab1_5',
    'stock_code_df_tab1_7',
    'stock_code_df_tab1_9',
    'stock_code_df_tab5_15',
]
for file_name in file_list:
    print(file_name)
    # data_1 = pd.read_pickle(os.path.join(path_1, file_name))
    # data_2 = pd.read_pickle(os.path.join(path_2, file_name))
    data_0 = pd.read_csv(os.path.join(path_0, file_name), index_col=0, parse_dates=True)
    data_1 = pd.read_csv(os.path.join(path_1, file_name), index_col=0, parse_dates=True)
    data_2 = pd.read_csv(os.path.join(path_2, file_name), index_col=0, parse_dates=True)

    xnms = list(sorted(set(data_0.columns) & set(data_1.columns) & set(data_2.columns)))
    # xinx = [pd.to_datetime('20181119')]
    begin_date = pd.to_datetime('20180119')

    xinx = np.array(list(sorted(set(data_0.index) & set(data_1.index) & set(data_2.index))))
    xinx = xinx[xinx > begin_date]

    data_0 = data_0.reindex(index=xinx, columns=xnms).replace(np.nan, 0)
    data_1 = data_1.reindex(index=xinx, columns=xnms).replace(np.nan, 0)
    data_2 = data_2.reindex(index=xinx, columns=xnms).replace(np.nan, 0)
    # print(data_1, data_2)
    c_1 = (data_1 != data_2)
    c_2 = (data_1 != data_0)
    c_3 = (data_2 != data_0)
    print(c_3.sum().sum())
    if c_3.sum().sum() != 0:
        print('error!')
        d = c_3.replace(False, np.nan).dropna(how='all', axis=0).dropna(how='all', axis=1)
        # print(d)
        # break
    else:
        print('success!')

# data_1 = pd.read_csv('/mnt/mfs/dat_whs/EM_Funda/20181128', index_col=0, sep='|', parse_dates=True)
# data_2 = pd.read_csv('/mnt/mfs/dat_whs/EM_Funda/20181129', index_col=0, sep='|', parse_dates=True)
# a_1 = data_1[c].dropna(how='all', axis=0).dropna(how='all', axis=1)
# a_2 = data_2[c].dropna(how='all', axis=0).dropna(how='all', axis=1)

# part_data_0 = data_0.loc[:begin_date]
# part_data_1 = data_1.loc[:begin_date]
# part_data_2 = data_2.loc[:begin_date]
