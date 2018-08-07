import pandas as pd
import numpy as np
import os
from factor_script.script_load_data import load_index_data, load_sector_data, load_locked_data, load_pct, \
    load_part_factor, create_log_save_path, deal_mix_factor
from factor_script.script_filter_fun import pos_daily_fun, out_sample_perf, \
    filter_ic, filter_ic_sharpe, filter_ic_leve, filter_pot_sharpe, filter_all
import open_lib.shared_tools.back_test as bt


def AZ_Cut_window(df, begin_date, end_date=None, column=None):
    if column is None:
        if end_date is None:
            return df[df.index > begin_date]
        else:
            return df[(df.index > begin_date) & (df.index < end_date)]
    else:
        if end_date is None:
            return df[df[column] > begin_date]
        else:
            return df[(df[column] > begin_date) & (df[column] < end_date)]


def factor_analysis(factor_path):
    file_list = os.listdir(factor_path)
    for file_name in file_list:
        factor_df = pd.read_pickle(os.path.join(factor_path, file_name))
        factor_df.sum(axis=1)
        print(factor_df)


def sector_factor_len(factor_root_path, sector_name_list):
    for sector_name in sector_name_list:
        factor_path = os.listdir(factor_root_path + '/' + sector_name)
        print(sector_name, len(factor_path))


def create_pnl_file_and_delete_factor(factor_root_path, sector_name_list):
    begin_date = pd.to_datetime('20100101')
    cut_date = pd.to_datetime('20160401')
    end_date = pd.to_datetime('20180401')

    # sector_name = 'market_top_100'
    index_name = '000016'
    for sector_name in sector_name_list:
        # sector_name = 'market_top_500'
        # index_name = '000905'
        # sector
        sector_df = load_sector_data(begin_date, end_date, sector_name)
        xnms = sector_df.columns
        xinx = sector_df.index
        # suspend or limit up_dn
        locked_df = load_locked_data(begin_date, end_date, xnms, xinx)
        # return
        return_choose = pd.read_pickle('/mnt/mfs/DAT_EQT/EM_Tab14/DERIVED/aadj_r.pkl')
        return_choose = AZ_Cut_window(return_choose, begin_date, end_date, column=None)
        return_choose = return_choose.reindex(columns=xnms, index=xinx, fill_value=0)

        factor_path = factor_root_path + '/' + sector_name
        # index data
        index_df = load_index_data(begin_date, end_date, xinx, index_name)
        factor_name_list = [x for x in os.listdir(factor_path) if 'pkl' in x]
        save_pnl_path = '/mnt/mfs/dat_whs/data/single_factor_pnl/' + sector_name
        # bt.AZ_Delete_file(save_pnl_path)
        for factor_name in factor_name_list:
            factor_load_path = os.path.join(factor_path, factor_name)
            print(factor_load_path)
            factor_df = pd.read_pickle(factor_load_path)
            factor_df = factor_df.reindex(columns=xnms, index=xinx, fill_value=0)
            daily_pos = deal_mix_factor(factor_df, sector_df, locked_df, hold_time=5, lag=2, if_only_long=False)
            pnl_df = (daily_pos * return_choose).sum(axis=1)

            bt.AZ_Path_create(save_pnl_path)
            pnl_df = pd.DataFrame(pnl_df, columns=[factor_name[:-4]])

            if len(pnl_df.replace(0, np.nan).dropna())/len(pnl_df) < 0.3:
                print(factor_name+' is delete')
                os.remove(factor_load_path)
            else:
                pnl_df.to_csv(os.path.join(save_pnl_path, factor_name[:-4] + '.csv'))


def create_corr_matrix(sector_name_list):
    for sector_name in sector_name_list:
        save_pnl_path = '/mnt/mfs/dat_whs/data/single_factor_pnl/' + sector_name
        all_pnl_df = pd.DataFrame()
        for file_name in os.listdir(save_pnl_path):
            pnl_df = pd.read_csv(os.path.join(save_pnl_path, file_name), index_col=0)
            all_pnl_df = pd.concat([all_pnl_df, pnl_df], axis=1)
        corr_df = all_pnl_df.replace(0, np.nan).corr()
        corr_df.to_csv('/mnt/mfs/dat_whs/data/single_factor_pnl/corr_matrix/' + sector_name + '.csv')


def analyse_corr_matrix_and_delete_factor(factor_root_path, sector_name_list, if_delete_factor=False):
    for sector_name in sector_name_list:
        print(sector_name)
        protect_list = os.listdir(factor_root_path + '/' + sector_name)
        corr_df = pd.read_csv('/mnt/mfs/dat_whs/data/single_factor_pnl/corr_matrix/' + sector_name + '.csv', index_col=0)
        # corr_df = corr_df.drop('a', axis=1).drop('a', axis=0)
        corr_df[(corr_df > 0.8) | (corr_df < -0.8)] = 1
        corr_df[(corr_df < 0.8) & (corr_df > -0.8)] = 0
        all_factor_set = set(corr_df.index)
        # print(corr_num_sort)
        while True:
            corr_num_sort = corr_df.sum().sort_values()
            # print(corr_num_sort)
            b = corr_num_sort.iloc[-1]

            factor_name = corr_num_sort.index[-1]
            if b > 1:
                corr_df = corr_df.drop(factor_name, axis=1).drop(factor_name, axis=0)
                # print(corr_df)
            else:
                break
        corr_num_sort = corr_df.sum().sort_values()
        use_factor_set = set(corr_df.index)
        print(len(corr_df.index))
        delete_set = all_factor_set - use_factor_set
        delete_set_c = delete_set - set(protect_list)
        protected_list = delete_set & set(protect_list)
        print(delete_set)
        print('protected_list: ' + ','.join(protected_list))

        for file_name in list(delete_set_c):
            factor_path = factor_root_path + '/{}/{}.pkl'.format(sector_name, file_name)
            save_pnl_path = '/mnt/mfs/dat_whs/data/single_factor_pnl/{}/{}.csv'.format(sector_name, file_name)

            try:
                if if_delete_factor:
                    os.remove(factor_path)
                os.remove(save_pnl_path)
                print(sector_name, file_name, 'DELETING')
            except:
                print(sector_name, file_name, 'Already DELETED')


if __name__ == '__main__':
    factor_root_path = '/mnt/mfs/dat_whs/data/new_factor_data/'
    sector_name_list = ['market_top_500']

    sector_factor_len(factor_root_path, sector_name_list)
    create_pnl_file_and_delete_factor(factor_root_path, sector_name_list)
    create_corr_matrix(sector_name_list)
    analyse_corr_matrix_and_delete_factor(factor_root_path, sector_name_list, if_delete_factor=True)

