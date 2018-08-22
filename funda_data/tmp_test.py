import pandas as pd
import numpy as np
import funda_data as fd
from funda_data.funda_data_deal import SectorData
import open_lib.shared_paths.path as pt
import loc_lib.shared_tools.back_test as bt
import os
from datetime import datetime, timedelta
from multiprocessing import Pool


def find_fun(fun_list):
    target_class = fd
    print(fun_list)
    for a in fun_list[:-1]:
        target_class = getattr(target_class, a)
    # print(target_class)
    target_fun = getattr(target_class(), fun_list[-1])
    return target_fun


def load_raw_data(root_path, raw_data_path, xnms, xinx, if_replace):
    raw_data_list = []
    for target_path in raw_data_path:
        tmp_data = bt.AZ_Load_csv(os.path.join('/mnt/mfs/DAT_EQT', target_path)).reindex(columns=xnms, index=xinx)
        if if_replace:
            tmp_data = tmp_data.replace(0, np.nan)
        raw_data_list += [tmp_data]
    return raw_data_list


def create_data_fun(mode, info_path, sector_df, xnms, xinx):
    info = pd.read_pickle(info_path)
    root_path = pt._BinFiles(mode)
    args = info['args']
    fun_list = info['fun'].split('.')
    raw_data_path = info['raw_data_path']
    if_replace = info['if_replace']
    raw_data_list = load_raw_data(root_path, raw_data_path, xnms, xinx, if_replace)

    target_fun = find_fun(fun_list)
    target_df = target_fun(*raw_data_list, sector_df, *args)
    return target_df


if __name__ == '__main__':
    mode = 'bkt'

    begin_date = pd.to_datetime('20100101')
    end_date = datetime.now()

    sector_name = 'market_top_2000'

    save_root_path = '/mnt/mfs/dat_whs/data/new_factor_data/{}'.format(sector_name)
    bt.AZ_Path_create(save_root_path)
    root_path = pt._BinFiles(mode)
    sector_data_class = SectorData(root_path)
    sector_df = sector_data_class.load_sector_data(begin_date, end_date, sector_name)

    xnms = sector_df.columns
    xinx = sector_df.index

    file_name_list = [x[:-4] for x in os.listdir(save_root_path)]
    a = pd.to_datetime('20180601')
    b = pd.to_datetime('20180801')

    config = pd.read_pickle('/mnt/mfs/alpha_whs/config01.pkl')
    factor_info = config['factor_info']
    file_name_list = set(factor_info[['name1', 'name2', 'name3']].values.ravel())
    # pool = Pool(20)

    # file_name_list = ['R_BusinessCycle_First_row_extre_0.3',
    #                   'R_CurrentAssets_TotAssets_First_row_extre_0.3',
    #                   'R_CurrentAssetsTurnover_First_row_extre_0.3',
    #                   'R_DebtAssets_First_row_extre_0.3',
    #                   'R_DebtEqt_First_row_extre_0.3',
    #                   'R_OPCF_IntDebt_QTTM_row_extre_0.3',
    #                   'R_IntDebt_Mcap_First_row_extre_0.3',
    #                   'R_OPEX_sales_TTM_Y3YGR_row_extre_0.3',
    #                   'R_EBITDA_sales_TTM_First_row_extre_0.3',
    #                   'R_EBITDA_sales_TTM_QTTM_row_extre_0.3',
    #                   'R_ROA_TTM_Y3YGR_row_extre_0.3',
    #                   'R_OPCF_TotDebt_QYOY_row_extre_0.3',
    #                   'R_OPCF_TotDebt_First_row_extre_0.3',
    #                   'R_OPCF_NetInc_s_First_row_extre_0.3',
    #                   'R_OPCF_sales_First_row_extre_0.3',
    #                   'R_EBITDA_QTTM_and_R_SUMASSET_First_0.3',
    #                   'R_EBIT2_Y3YGR_and_MCAP_0.3',
    #                   'R_EBITDA_QYOY_and_MCAP_0.3',
    #                   'R_IntDebt_Y3YGR_and_R_SUMASSET_First_0.3',
    #                   'CCI_p120d_limit_12',
    #                   'MACD_20_100',
    #                   'log_price_0.2',
    #                   'bias_turn_p20d',
    #                   'vol_p20d',
    #                   'vol_p20d',
    #                   'evol_p20d'
    #                   ]
    file_name_list = ['R_RevenuePS_s_First_row_extre_0.3']
    for file_name in file_name_list:
        print(file_name)
        factor_to_fun = '/mnt/mfs/dat_whs/data/factor_to_fun'
        info_path = os.path.join(factor_to_fun, file_name)
        base_data = pd.read_pickle('/mnt/mfs/dat_whs/data/new_factor_data/{}/{}.pkl'.format(sector_name, file_name))
        create_data = create_data_fun('bkt', info_path, sector_df, xnms, xinx)
        # print(base_data.loc[a:b].replace(np.nan, 0))
        # print(create_data.loc[a:b].replace(np.nan, 0))
        print((base_data.loc[a:b].replace(np.nan, 0) != create_data.loc[a:b].replace(np.nan, 0)).sum().sum())
