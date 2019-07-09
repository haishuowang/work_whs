import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load.plt import plot_send_result
from work_whs.bkt_factor_create.base_fun_import import DiscreteClass, ContinueClass, SectorData, \
    DataDeal, TrainFunSet, CorrCheck
from work_whs.bkt_factor_create.new_framework_test_1 import FactorTest


def main_fun(str_1, exe_str, i, test_name):
    print(i)
    print(str_1)
    print(exe_str)
    sector_name, hold_time_str, if_only_long, percent_str = str_1.split('|')

    hold_time = int(hold_time_str)

    percent = float(percent_str)
    if if_only_long == 'False':
        if_only_long = False
    else:
        if_only_long = True

    root_path = '/media/hdd1/DAT_EQT'
    # root_path = '/mnt/mfs/DAT_EQT'
    if_save = True
    if_new_program = True

    begin_date = pd.to_datetime('20130101')
    # end_date = pd.to_datetime('20190411')
    end_date = datetime.now()
    cut_date = pd.to_datetime('20180101')
    lag = 2
    return_file = ''
    if_hedge = True
    # 生成回测脚本
    factor_test = FactorTest(root_path, if_save, if_new_program, begin_date, end_date, sector_name, hold_time,
                             lag, return_file, if_hedge, if_only_long)

    data_deal = DataDeal(begin_date, end_date, root_path, sector_name)
    # 生成回测脚本
    info_df, pnl_df, pos_df = factor_test.get_mix_pnl_df(data_deal, exe_str, cut_date, percent)
    pnl_df.name = f'{sector_name}_{i}'
    print(info_df)
    # 相关性测试
    pnl_save_path = f'{result_root_path}/tmp_pnl/{test_name}'
    bt.AZ_Path_create(pnl_save_path)
    corr_sr = CorrCheck().corr_self_check(pnl_df, pnl_save_path)
    print(corr_sr[corr_sr > 0.5])
    result_df, corr_info_df = bt.commit_check(pd.DataFrame(pnl_df))

    annual_r = pnl_df.sum() / pos_df.abs().sum(1).sum() * 250

    if result_df.prod()[0] == 1 and len(corr_sr[corr_sr > 0.6]) == 0 and info_df['pot'] > 50:
        pnl_df.to_pickle(f'{pnl_save_path}/{sector_name}_{i}')
        plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df),
                         f'[new framewor result]{sector_name}_{i}_{if_only_long} {round(annual_r, 4)}',

                         pd.DataFrame(info_df).to_html() + corr_info_df.to_html() +
                         pd.DataFrame(corr_sr[corr_sr > 0.5]).to_html())
        print('success')
    else:
        plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df),
                         f'[new framewor result]{sector_name}_{i}_{if_only_long} {round(annual_r, 4)} fail',
                         pd.DataFrame(info_df).to_html() + corr_info_df.to_html() +
                         pd.DataFrame(corr_sr[corr_sr > 0.5]).to_html())
        print('fail')
    return info_df, pnl_df, pos_df


if __name__ == '__main__':
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
        'market_top_300to800plus_industry_55'
    ]

    ban_list = [
        'index_000300',
        'index_000905',
        'market_top_300plus',
        'market_top_300plus_industry_10_15',
        'market_top_300plus_industry_20_25_30_35',
        'market_top_300plus_industry_40',
        'market_top_300plus_industry_45_50',
    ]
    filter_i_list = [0, 1]
    result_root_path = '/mnt/mfs/dat_whs/result_new'
    test_name = 'only_long01'

    for sector_name in sector_name_list:
        # if sector_name in ban_list:
        #     continue
        result_path = f'{result_root_path}/{test_name}/{sector_name}.csv'
        if os.path.exists(result_path):
            result_df = pd.read_csv(result_path, header=None)
            for i in result_df.index:
                sector_name = 'index_000300'

                print('*******************************************')
                info_str = result_df.loc[i].values[0]
                str_1, exe_str = info_str.split('#')
                a = time.time()
                info_df, pnl_df, pos_df = main_fun(str_1, exe_str, i, test_name)
                b = time.time()
                print(b - a)
