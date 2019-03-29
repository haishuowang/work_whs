import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *

delete_list = [
    'LIQ_all_original',
    'LIQ_all_pure.csv',
    'LIQ_mix.csv',
    'LIQ_p1_original.csv',
    'LIQ_p1_pure.csv',
    'LIQ_p2_original.csv',
    'LIQ_p2_pure.csv',
    'LIQ_p3_original.csv',
    'LIQ_p3_pure.csv',
    'LIQ_p4_original.csv',
    'LIQ_p4_pure.csv',
    'M0',
    'M1',
    'M1_p1',
    'M1_p2',
    'M1_p3',
    'M1_p4, vr_original_20days.csv',
    'dividend_ratio',
    'vr_afternoon_10min_20days',
    'vr_afternoon_last10min_20days.csv',
    'vr_original_45days.csv',
    'vr_original_75days.csv'
]


def key_word_delete(target_path, delete_list):
    all_file_list = os.listdir(target_path)
    delete_file_list = [x for x in all_file_list if x.split('|')[0] in delete_list]
    # print(delete_file_list)
    for delete_file in delete_file_list:
        print(f'{target_path}/{delete_file}')
        os.remove(f'{target_path}/{delete_file}')


if __name__ == '__main__':
    root_path = '/mnt/mfs/dat_whs/data/single_factor_pnl'
    sector_name_list = [
        'index_000300',
        'index_000905',
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
    ]

    for sector_name in sector_name_list:
        key_word_delete(f'{root_path}/{sector_name}', delete_list)
