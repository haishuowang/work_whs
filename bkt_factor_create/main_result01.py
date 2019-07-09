import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load.plt import plot_send_result
from work_whs.bkt_factor_create.base_fun_import import DiscreteClass, ContinueClass, SectorData, \
    DataDeal, SectorFilter, TrainFunSet, CorrCheck
from work_whs.bkt_factor_create.new_framework_test_1 import FactorTest


class FactorTestResult(FactorTest):
    def __init__(self, *args):
        super(FactorTest, self).__init__(*args)

    def load_zscore_factor_b(self, file_name, sector_df_r):
        if 'zscore' in file_name:
            raw_zscore_df = self.load_raw_factor(file_name)
        else:
            raw_zscore_df = self.row_zscore(self.load_raw_factor(file_name), sector_df_r)
        return raw_zscore_df

    @staticmethod
    def check_fun(data_1, data_2):
        a = data_1.loc[pd.to_datetime('20140101'):pd.to_datetime('20190101')]
        b = data_2.loc[pd.to_datetime('20140101'):pd.to_datetime('20190101')]
        c = (a.replace(np.nan, 0) - b.replace(np.nan, 0)).abs().sum().sum()
        d = (a.replace(np.nan, 0) - b.replace(np.nan, 0)).round(8).replace(0, np.nan) \
            .dropna(how='all', axis=1).dropna(how='all', axis=0)
        aa = a.reindex(index=d.index, columns=d.columns)
        bb = b.reindex(index=d.index, columns=d.columns)
        print(c)

    def get_mix_pnl_df(self, data_deal, exe_str, cut_date, percent):
        sector_df_r = SectorData(self.root_path).load_sector_data \
            (self.begin_date, self.end_date, self.sector_name)

        def tmp_fun():
            exe_list = exe_str.split('@')
            way_str_1 = exe_list[0].split('_')[-1]
            name_1 = '_'.join(exe_list[0].split('_')[:-1])
            print(name_1)
            factor_1 = data_deal.count_return_data(name_1) * float(way_str_1)

            factor_1_r = self.load_zscore_factor_b(name_1, sector_df_r) * float(way_str_1)
            # factor_1_r = self.load_raw_factor(name_1) * float(way_str_1)
            self.check_fun(factor_1, factor_1_r)
            for i in range(int((len(exe_list) - 1) / 2)):
                fun_str = exe_list[2 * i + 1]
                way_str_2 = exe_list[2 * i + 2].split('_')[-1]
                name_2 = '_'.join(exe_list[2 * i + 2].split('_')[:-1])
                print(name_2)
                factor_2 = data_deal.count_return_data(name_2) * float(way_str_2)
                factor_2_r = self.load_zscore_factor_b(name_2, sector_df_r) * float(way_str_2)
                self.check_fun(factor_2, factor_2_r)

                factor_1 = getattr(self, fun_str)(factor_1, factor_2)
            return factor_1

        mix_factor = tmp_fun()
        info_df, pnl_df, pos_df = self.back_test(mix_factor, cut_date, percent, return_pos=True)
        pnl_df.name = exe_str
        info_df.name = exe_str
        return info_df, pnl_df, pos_df


def main_fun(str_1, exe_str, i, filter_i):
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
    factor_test = FactorTestResult(root_path, if_save, if_new_program, begin_date, end_date, sector_name, hold_time,
                                   lag, return_file, if_hedge, if_only_long)
    mask_df_list = SectorFilter(root_path).filter_beta(factor_test.if_weight, factor_test.ic_weight)
    mask_df = mask_df_list[filter_i]
    factor_test.sector_df = factor_test.sector_df * mask_df.reindex(index=factor_test.xinx,
                                                                    columns=factor_test.xnms)
    data_deal = DataDeal(begin_date, end_date, root_path, sector_name)
    # 生成回测脚本
    info_df, pnl_df, pos_df = factor_test.get_mix_pnl_df(data_deal, exe_str, cut_date, percent)
    pnl_df.name = f'{sector_name}_{i}'
    print(info_df)
    # 相关性测试
    pnl_save_path = f'{result_root_path}/tmp_pnl/filter_beta/{filter_i}'
    bt.AZ_Path_create(pnl_save_path)
    corr_sr = CorrCheck().corr_self_check(pnl_df, pnl_save_path)
    print(corr_sr[corr_sr > 0.5])
    result_df, corr_info_df = bt.commit_check(pd.DataFrame(pnl_df))
    if result_df.prod()[0] == 1 and len(corr_sr[corr_sr > 0.6]) == 0 and info_df['pot'] > 50:
        pnl_df.to_pickle(f'{pnl_save_path}/{sector_name}_{i}')
        plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df),
                         f'[new framewor result]{filter_i}_{sector_name}_{i}_{if_only_long}',
                         pd.DataFrame(info_df).to_html() + corr_info_df.to_html() +
                         pd.DataFrame(corr_sr[corr_sr > 0.5]).to_html())
        print('success')
    else:
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
    test_name = 'filter_beta'

    pool = Pool(20)
    for sector_name in sector_name_list[:1]:
        # if sector_name in ban_list:
        #     continue
        for filter_i in filter_i_list[:1]:
            filter_i = 1
            result_path = f'{result_root_path}/{test_name}/{filter_i}/{sector_name}.csv'
            if os.path.exists(result_path):
                result_df = pd.read_csv(result_path, header=None)
                for i in result_df.index[:1]:
                    sector_name = 'index_000300'
                    i = 72

                    print('*******************************************')
                    info_str = result_df.loc[i].values[0]
                    str_1, exe_str = info_str.split('#')

                    a = time.time()
                    info_df, pnl_df, pos_df = main_fun(str_1, exe_str, i, filter_i)
                    b = time.time()
                    print(b - a)
