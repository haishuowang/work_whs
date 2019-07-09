import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load.plt import plot_send_result
from work_whs.bkt_factor_create.new_framework_test_2 import FactorTest


class FactorTestOnlyLong02(FactorTest):
    def train_fun_new(self, cut_date, percent, use_fun, result_save_path=None):
        # try:
            def tmp_fun(mix_factor_df, low_corr_factor, low_corr_factor_way):
                tmp_factor_df = self.load_zscore_factor(low_corr_factor) * low_corr_factor_way
                # tmp_factor_df = self.load_raw_factor(low_corr_factor) * low_corr_factor_way
                tmp_mix_df = getattr(self, use_fun)(mix_factor_df, tmp_factor_df)
                if self.if_only_long:
                    info_df, pnl_df = self.back_test(tmp_mix_df, cut_date, percent, ls_para='l')
                else:
                    info_df, pnl_df = self.back_test(tmp_mix_df, cut_date, percent)
                return tmp_mix_df, info_df, pnl_df

            all_file = sorted(os.listdir(f'/mnt/mfs/dat_whs/data/factor_data/{self.sector_name}'))
            all_file = [x[:-4] for x in all_file if 'suspend' not in x and not x.startswith('intra')]
            # 从all_file随机选择select_file
            select_factor = random.sample(all_file, 100)
            # 获取select_file的 info 和 pnl 矩阵
            select_info_df, select_pnl_df = self.get_all_pnl_df(select_factor, cut_date, percent, if_multy=False)
            # 获取select_file的 corr矩阵
            select_corr_df = select_pnl_df.corr()

            select_sp_in = select_info_df.loc['sp_in'].abs()
            select_pot_in = select_info_df.loc['pot_in'].abs()
            # 打分
            select_factor_scores = self.get_scores_fun(select_sp_in, select_pot_in).sort_values(ascending=False)
            # 选择最好的那个 best_factor
            best_factor = select_factor_scores.index[0]
            best_score = select_factor_scores.iloc[0]
            best_info = select_info_df[best_factor]
            best_factor_way = best_info.loc['way_in']
            best_pnl = select_pnl_df[best_factor]
            # 找出跟best_factor相关性低于0.5的factor
            corr_sr = select_corr_df[best_factor].abs().sort_values()
            corr_sr = corr_sr[corr_sr < 0.5]
            low_corr_factor_list = corr_sr.index
            # low_corr_factor_list按照分数排序
            low_corr_factor_scores = select_factor_scores[low_corr_factor_list].sort_values(ascending=False)
            # 初始化变量
            exe_str = f'{best_factor}_{str(best_factor_way)}'
            mix_factor_df = self.load_zscore_factor(best_factor) * best_factor_way
            mix_num = 0
            mix_score = best_score
            mix_info = best_info
            mix_pnl = best_pnl
            # while mix_num<5:
            for low_corr_factor in low_corr_factor_scores.index:
                low_corr_factor_way = select_info_df[low_corr_factor].loc['way_in']
                tmp_mix_df, tmp_info_df, tmp_pnl_df = tmp_fun(mix_factor_df, low_corr_factor, low_corr_factor_way)
                tmp_score = self.get_score_fun(tmp_info_df.loc['sp_in'], tmp_info_df.loc['pot_in'])
                print(exe_str, low_corr_factor, tmp_score)
                if tmp_score > mix_score:
                    print(tmp_info_df)
                    exe_str = f'{exe_str}@{use_fun}@{low_corr_factor}_{str(low_corr_factor_way)}'
                    mix_factor_df = tmp_mix_df
                    mix_score = tmp_score
                    mix_info = tmp_info_df
                    mix_pnl = tmp_pnl_df
                    mix_num += 1
                    # 超过混合上限 跳出
                    if mix_num >= 10:
                        break
            if mix_info.loc['sp'] > 2 and mix_info.loc['pot'] > 50:
                if result_save_path is not None:
                    bt.AZ_Path_create(result_save_path)
                result_df, info_df = bt.commit_check(pd.DataFrame(mix_pnl))
                if result_df.prod().iloc[0] == 1:
                    lock = Lock()
                    result_save_file = f'{result_save_path}/{self.sector_name}.csv'
                    if self.if_save:
                        str_1 = f'{self.sector_name}|{self.hold_time}|{self.if_only_long}|{percent}'
                        write_str = str_1 + '#' + exe_str
                        with lock:
                            f = open(result_save_file, 'a')
                            f.write(write_str + '\n')
                            f.close()

                    plot_send_result(mix_pnl, mix_info.loc['sp'], f'{self.sector_name}|{self.hold_time}'
                                                                  f'|{self.if_only_long}|{percent}|zscore'
                                     , exe_str + '\n' + pd.DataFrame(mix_info).to_html())

        # except Exception as error:
        #     print(error)


def run(factor_test, cut_date, percent, result_save_path, run_num=100):
    pool = Pool(28)
    for i in range(run_num):
        args = (cut_date, percent, 'add_fun_f', result_save_path)
        # factor_test.train_fun_new(*args)
        pool.apply_async(factor_test.train_fun_new, args=args)
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

    factor_test = FactorTestOnlyLong02(root_path, if_save, if_new_program, begin_date, end_date, sector_name,
                                       hold_time, lag, return_file, if_hedge, if_only_long)
    script_name = os.path.basename(__file__).split('.')[0].replace('main_test_', '')
    result_save_path = f'/mnt/mfs/dat_whs/result_new/{script_name}'
    print(result_save_path)
    cut_date = '20180101'
    run(factor_test, cut_date, percent, result_save_path, run_num=100)


def main():
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

    hold_time_list = [5, 10, 20]
    for if_only_long in [True]:
        for sector_name in sector_name_list:
            for percent in [0.1, 0.2]:
                for hold_time in hold_time_list:
                    part_main_fun(if_only_long, hold_time, sector_name, percent)


if __name__ == '__main__':
    main()
    pass
