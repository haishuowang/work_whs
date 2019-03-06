import pandas as pd
import numpy as np
import funda_data as fd
from funda_data.funda_data_deal import SectorData
import loc_lib.shared_paths.path as pt
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

    sector_name_list = ['market_top_1000_industry_10_15',
                        'market_top_1000_industry_20_25_30_35',
                        'market_top_1000_industry_40',
                        'market_top_1000_industry_45_50',
                        'market_top_1000_industry_55',
                        'market_top_2000_industry_10_15',
                        'market_top_2000_industry_20_25_30_35',
                        'market_top_2000_industry_40',
                        'market_top_2000_industry_45_50',
                        'market_top_2000_industry_55']

    for sector_name in sector_name_list:
        save_root_path = '/mnt/mfs/dat_whs/data/new_factor_data/{}'.format(sector_name)
        bt.AZ_Path_create(save_root_path)
        # root_path = pt._BinFiles(mode)
        # sector_data_class = SectorData(root_path)
        # sector_df = sector_data_class.load_sector_data(begin_date, end_date, sector_name)

        return_df = pd.read_csv('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv', sep='|', index_col=0,
                                parse_dates=True)
        sector_df = bt.AZ_Load_csv(f'/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_10/{sector_name}.csv')
        sector_df = sector_df[(sector_df.index >= begin_date) & (sector_df.index <= end_date)]
        sector_df = sector_df.reindex(index=return_df.index)

        xnms = sector_df.columns
        xinx = sector_df.index

        file_name_list = [x[:-4] for x in os.listdir(save_root_path)]
        a = pd.to_datetime('20180601')
        b = pd.to_datetime('20180801')

        config = pd.read_pickle('/mnt/mfs/alpha_whs/CRTTINKER01.pkl')
        # config = pd.read_pickle('/mnt/mfs/alpha_whs/CRTMEDUSA01.pkl')
        factor_info = config['factor_info']
        # file_name_list = set(factor_info[['name1', 'name2', 'name3']].values.ravel())
        # pool = Pool(20)

        for file_name in file_name_list:
            print(file_name)
            if file_name.startswith('ADOSC') or file_name.startswith('MFI'):
                continue
            factor_to_fun = '/mnt/mfs/dat_whs/data/factor_to_fun'
            info_path = os.path.join(factor_to_fun, file_name)
            base_data = pd.read_pickle('/mnt/mfs/dat_whs/data/new_factor_data/{}/{}.pkl'.format(sector_name, file_name))
            funda_data = create_data_fun('bkt', info_path, sector_df, xnms, xinx)

            d = base_data.loc[a:b].replace(np.nan, 0).round(10)
            e = funda_data.loc[a:b].replace(np.nan, 0).round(10)
            col = sorted(list(set(d.columns) & set(e.columns)))
            d = d[col]
            e = e[col]
            print((d != e).sum().sum())
