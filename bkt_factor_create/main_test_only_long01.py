import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *
from work_whs.bkt_factor_create.new_framework_test_2 import FactorTest


def run(factor_test, cut_date, percent, result_save_path, run_num=100):
    pool = Pool(28)
    for i in range(run_num):
        args = (cut_date, percent, result_save_path)
        # factor_test.train_fun(*args)
        pool.apply_async(factor_test.train_fun, args=args)
    pool.close()
    pool.join()


def part_main_fun(if_only_long, hold_time, sector_name, percent):
    root_path = '/mnt/mfs/DAT_EQT'
    if_save = True
    if_new_program = True

    begin_date = pd.to_datetime('20130101')
    end_date = pd.to_datetime('20190411')
    lag = 2
    return_file = ''

    if_hedge = True

    factor_test = FactorTest(root_path, if_save, if_new_program, begin_date, end_date, sector_name, hold_time,
                             lag, return_file, if_hedge, if_only_long)

    result_save_path = f'/mnt/mfs/dat_whs/result_new/only_long01'

    cut_date = '20180101'
    run(factor_test, cut_date, percent, result_save_path, run_num=100)


def main():
    sector_name_list = [
        'index_000300',
        # 'index_000905',

        'market_top_300plus',
        'market_top_300plus_industry_10_15',
        'market_top_300plus_industry_20_25_30_35',
        'market_top_300plus_industry_40',
        'market_top_300plus_industry_45_50',
        'market_top_300plus_industry_55',

        # 'market_top_300to800plus',
        # 'market_top_300to800plus_industry_10_15',
        # 'market_top_300to800plus_industry_20_25_30_35',
        # 'market_top_300to800plus_industry_40',
        # 'market_top_300to800plus_industry_45_50',
        # 'market_top_300to800plus_industry_55'
    ]

    hold_time_list = [5, 10, 20]
    for if_only_long in [True]:
        for sector_name in sector_name_list:
            for percent in [0.1, 0.2]:
                for hold_time in hold_time_list:
                    part_main_fun(if_only_long, hold_time, sector_name, percent)


if __name__ == '__main__':
    main()
    pass
