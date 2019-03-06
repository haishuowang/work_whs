import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *

if __name__ == '__main__':

    sector_name_list = [
        'market_top_300plus',
        'market_top_300plus_industry_10_15',
        'market_top_300plus_industry_20_25_30_35',
        'market_top_300plus_industry_40',
        'market_top_300plus_industry_45_50',
        'market_top_300plus_industry_55',

        'market_top_300to800plus',
        'market_top_300to800plus_industry_10_15',
        'market_top_300to800plus_industry_20_25_30_35',
        'market_top_300to800plus_industry_40',
        'market_top_300to800plus_industry_45_50',
        'market_top_300to800plus_industry_55',

        'market_top_800plus',
        'market_top_800plus_industry_10_15',
        'market_top_800plus_industry_20_25_30_35',
        'market_top_800plus_industry_40',
        'market_top_800plus_industry_45_50',
        'market_top_800plus_industry_55',

        # 'market_top_300plus_ind1',
        # 'market_top_300plus_ind2',
        # 'market_top_300plus_ind3',
        # 'market_top_300plus_ind4',
        # 'market_top_300plus_ind5',
        # 'market_top_300plus_ind6',
        # 'market_top_300plus_ind7',
        # 'market_top_300plus_ind8',
        # 'market_top_300plus_ind9',
        # 'market_top_300plus_ind10',
        # 'market_top_300to800plus_ind1',
        # 'market_top_300to800plus_ind2',
        # 'market_top_300to800plus_ind3',
        # 'market_top_300to800plus_ind4',
        # 'market_top_300to800plus_ind5',
        # 'market_top_300to800plus_ind6',
        # 'market_top_300to800plus_ind7',
        # 'market_top_300to800plus_ind8',
        # 'market_top_300to800plus_ind9',
        # 'market_top_300to800plus_ind10',
    ]

    t1 = time.time()
    for sector_name in sector_name_list:
        print('_________________________________________________________________________________________')
        print(sector_name)
        a = time.time()
        main_fun(sector_name)
        b = time.time()
        print(f'{sector_name} Data Updated!, cost time {b-a}\'s!')
    t2 = time.time()
    print(f'totle cost time: {t2-t1}!')