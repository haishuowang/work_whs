import sys

sys.path.append('/mnt/mfs')
import string
from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load.plt import plot_send_result
from work_whs.loc_lib.pre_load.log import use_time
from work_whs.bkt_factor_create.base_fun_import import CorrCheck
from work_whs.bkt_framework.base_test_import import FactorTest


class FactorTestResult(FactorTest):
    def __init__(self, *args):
        super(FactorTest, self).__init__(*args)

    def load_mix_data(self, file_name):
        if type(file_name) is str:
            target_df = self.load_raw_data(file_name)
        else:
            fun_name, factor_name_list, para_list = file_name
            factor_df_list = [self.load_mix_data(factor_name) for factor_name in factor_name_list]
            target_df = getattr(self, fun_name)(*factor_df_list, *para_list)
            # zcore 根据用到的数据增加权重

            target_df = self.row_zscore(target_df, self.sector_df)  # * np.sqrt(len(factor_name_list))
        return target_df

    def get_pnl_df(self, file_name, cut_date, percent):
        data_df = self.load_mix_data(file_name)
        if self.if_only_long:
            info_df_l, pnl_df_l, pos_df_l = self.back_test(data_df, cut_date, percent, ls_para='l', return_pos=True)
            pnl_df_l.name = str(file_name)
            info_df_l.name = str(file_name)

            info_df_s, pnl_df_s, pos_df_s = self.back_test(data_df, cut_date, percent, ls_para='s', return_pos=True)
            pnl_df_s.name = str(file_name)
            info_df_s.name = str(file_name)
            if info_df_l['sp_in'] > info_df_s['sp_in']:
                info_df = info_df_l
                pnl_df = pnl_df_l
                pos_df = pos_df_l
            else:
                info_df = info_df_s
                pnl_df = pnl_df_s
                pos_df = pos_df_s
        else:
            info_df, pnl_df, pos_df = self.back_test(data_df, cut_date, percent, return_pos=True)
            pnl_df.name = str(file_name)
            info_df.name = str(file_name)
        self.factor_way_dict[str(file_name)] = info_df['way_in']
        return info_df, pnl_df, pos_df


@use_time
def main_fun(base_info_str, exe_str, i):
    # base_info_str = 'index_000300|5|True|0.1'
    sector_name, hold_time_str, if_only_long, percent_str = base_info_str.split('|')

    hold_time = int(hold_time_str)

    percent = float(percent_str)
    if if_only_long == 'False':
        if_only_long = False
    else:
        if_only_long = True

    exe_list = eval(exe_str)

    root_path = '/mnt/mfs/DAT_EQT'
    if_save = True
    if_new_program = True

    begin_date = pd.to_datetime('20130101')
    end_date = datetime.now()
    cut_date = pd.to_datetime('20180101')
    lag = 2
    return_file = ''

    if_hedge = True

    factor_test = FactorTestResult(root_path, if_save, if_new_program, begin_date, end_date, sector_name, hold_time,
                                   lag, return_file, if_hedge, if_only_long)

    info_df, pnl_df, pos_df = factor_test.get_pnl_df(exe_list, cut_date, percent)

    pnl_save_path = f'{result_root_path}/tmp_pnl/{test_name}'
    bt.AZ_Path_create(pnl_save_path)

    corr_sr = CorrCheck().corr_self_check(pnl_df, pnl_save_path)
    print(corr_sr[corr_sr > 0.5])
    result_df, corr_info_df = bt.commit_check(pd.DataFrame(pnl_df))

    annual_r = pnl_df.sum() / pos_df.abs().sum(1).sum() * 250

    if result_df.prod()[0] == 1 and len(corr_sr[corr_sr > 0.6]) == 0 and info_df['pot'] > 50:
        pnl_df.to_pickle(f'{pnl_save_path}/{sector_name}_{i}')
        plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df),
                         f'[new framewor result]{sector_name}|{hold_time}|{if_only_long}|{percent}|{i} '
                         f'{round(annual_r, 4)}',

                         pd.DataFrame(info_df).to_html() + corr_info_df.to_html() +
                         pd.DataFrame(corr_sr[corr_sr > 0.5]).to_html())
        print('success')
    else:
        plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df),
                         f'[new framewor result]{sector_name}|{hold_time}|{if_only_long}|{percent}|{i} '
                         f'{round(annual_r, 4)} fail',
                         pd.DataFrame(info_df).to_html() + corr_info_df.to_html() +
                         pd.DataFrame(corr_sr[corr_sr > 0.5]).to_html())
        print('fail')

    return info_df, pnl_df, pos_df


if __name__ == '__main__':
    result_root_path = '/mnt/mfs/dat_whs/result_new2'
    test_name = 'test02'
    sector_name = 'index_000300'
    with open(f'{result_root_path}/{test_name}/{sector_name}.csv', 'r') as f:
        for i, row in enumerate(f.readlines()):
            base_info_str, exe_str, way = row.split('#')
            info_df, pnl_df, pos_df = main_fun(base_info_str, exe_str, i)