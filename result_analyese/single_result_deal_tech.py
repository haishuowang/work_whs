import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *
import work_whs.AZ_2018_Q2.factor_script.main_file.main_file_single_test as mfst


def mul_fun(a, b):
    a_l = a.where(a > 0, 0)
    a_s = a.where(a < 0, 0)

    b_l = b.where(b > 0, 0)
    b_s = b.where(b < 0, 0)

    pos_l = a_l.mul(b_l)
    pos_s = a_s.mul(b_s)

    pos = pos_l.sub(pos_s)
    return pos


def load_tech_factor(self, file_name):
    load_path = os.path.join('/mnt/mfs/dat_whs/data/new_factor_data/' + self.sector_name)
    target_df = pd.read_pickle(os.path.join(load_path, file_name + '.pkl')) \
        .reindex(index=self.xinx, columns=self.xnms)
    if self.if_only_long:
        target_df = target_df[target_df > 0]
    return target_df


def load_daily_factor(self, file_name):
    load_path = '/mnt/mfs/DAT_EQT/EM_Funda/daily/'
    tmp_df = bt.AZ_Load_csv(os.path.join(load_path, file_name + '.csv')) \
        .reindex(index=self.xinx, columns=self.xnms)

    target_df = self.row_extre(tmp_df, self.sector_df, 0.3)
    if self.if_only_long:
        target_df = target_df[target_df > 0]
    return target_df


def load_whs_factor(self, file_name):
    load_path = '/mnt/mfs/DAT_EQT/EM_Funda/dat_whs/'
    tmp_df = bt.AZ_Load_csv(os.path.join(load_path, file_name + '.csv')) \
        .reindex(index=self.xinx, columns=self.xnms)

    target_df = self.row_extre(tmp_df, self.sector_df, 0.3)
    if self.if_only_long:
        target_df = target_df[target_df > 0]
    return target_df


def get_result_data(root_path, file_name):
    data = pd.read_csv(f'{root_path}/{file_name}', sep='|', index_col=0, header=None)
    print(data)
    data.columns = ['name_1', 'fun_name', 'sector_name', 'in_condition', 'out_condition', 'ic', 'sp_d', 'sp_m', 'sp_u',
                    'pot_in', 'fit_ratio', 'leve_ratio', 'sp_in', 'sp_q_out']
    data_sort = data.sort_values(by='sp_in')
    return data_sort


# root_path = '/mnt/mfs/dat_whs/result/result'
# # def main():
# file_name_list = [x for x in os.listdir(root_path) if 'single_test' in x and
#                   os.path.getsize(os.path.join(root_path, x)) != 0]
# print(file_name_list)
# for file_name in file_name_list:
#     result_df = get_result_data(root_path, file_name)

def get_file_name(sector_name):
    tmp_file_list = os.listdir(f'/media/hdd1/dat_whs/data/single_factor_pnl/{sector_name}')
    target_file_list = [x.split('|') for x in tmp_file_list if '|' in x]
    data = pd.DataFrame(target_file_list, columns=['factor_name', 'sector_name', 'hold_time', 'if_only_long'])
    return data


def get_pnl_table(part_df, sector_name):
    file_name_df = part_df.apply(lambda x: '|'.join(x), axis=1)
    all_pnl_df = pd.DataFrame()
    for file_name in file_name_df:
        # print(file_name)
        pnl_df = pd.read_csv(f'/media/hdd1/dat_whs/data/single_factor_pnl/{sector_name}/{file_name}',
                             index_col=0, parse_dates=True)
        pnl_df.columns = [file_name]
        all_pnl_df = pd.concat([all_pnl_df, pnl_df], axis=1)
    all_pnl_df = all_pnl_df.loc[pd.to_datetime('20130101'):]
    return all_pnl_df


my_factor_dict = dict({
    # 'WILLR_200_40': 'load_tech_factor',
    # 'WILLR_200_30': 'load_tech_factor',
    # 'WILLR_200_20': 'load_tech_factor',
    # 'WILLR_140_40': 'load_tech_factor',
    # 'WILLR_140_30': 'load_tech_factor',
    # 'WILLR_140_20': 'load_tech_factor',
    # 'WILLR_100_40': 'load_tech_factor',
    # 'WILLR_100_30': 'load_tech_factor',
    # 'WILLR_100_20': 'load_tech_factor',
    # 'WILLR_40_40': 'load_tech_factor',
    # 'WILLR_40_30': 'load_tech_factor',
    # 'WILLR_40_20': 'load_tech_factor',
    # 'WILLR_20_40': 'load_tech_factor',
    # 'WILLR_20_30': 'load_tech_factor',
    # 'WILLR_20_20': 'load_tech_factor',
    # 'WILLR_10_40': 'load_tech_factor',
    # 'WILLR_10_30': 'load_tech_factor',
    # 'WILLR_10_20': 'load_tech_factor',
    'BBANDS_10_2': 'load_tech_factor',
    'BBANDS_10_1.5': 'load_tech_factor',
    'BBANDS_10_1': 'load_tech_factor',
    'BBANDS_200_2': 'load_tech_factor',
    'BBANDS_200_1.5': 'load_tech_factor',
    'BBANDS_200_1': 'load_tech_factor',
    'BBANDS_140_2': 'load_tech_factor',
    'BBANDS_140_1.5': 'load_tech_factor',
    'BBANDS_140_1': 'load_tech_factor',
    'BBANDS_100_2': 'load_tech_factor',
    'BBANDS_100_1.5': 'load_tech_factor',
    'BBANDS_100_1': 'load_tech_factor',
    'BBANDS_40_2': 'load_tech_factor',
    'BBANDS_40_1.5': 'load_tech_factor',
    'BBANDS_40_1': 'load_tech_factor',
    'BBANDS_20_2': 'load_tech_factor',
    'BBANDS_20_1.5': 'load_tech_factor',
    'BBANDS_20_1': 'load_tech_factor',
    'MA_LINE_160_60': 'load_tech_factor',
    'MA_LINE_120_60': 'load_tech_factor',
    'MA_LINE_100_40': 'load_tech_factor',
    'MA_LINE_60_20': 'load_tech_factor',
    'MA_LINE_10_5': 'load_tech_factor',
    'MACD_20_60_18': 'load_tech_factor',
    'MACD_12_26_9': 'load_tech_factor',
    'RSI_200_30': 'load_tech_factor',
    'RSI_140_30': 'load_tech_factor',
    'RSI_100_30': 'load_tech_factor',
    'RSI_40_30': 'load_tech_factor',
    'RSI_200_10': 'load_tech_factor',
    'RSI_140_10': 'load_tech_factor',
    'RSI_100_10': 'load_tech_factor',
    'RSI_40_10': 'load_tech_factor',
    'ATR_200_0.2': 'load_tech_factor',
    'ATR_140_0.2': 'load_tech_factor',
    'ATR_100_0.2': 'load_tech_factor',
    'ATR_40_0.2': 'load_tech_factor',
    'ADOSC_60_160_0': 'load_tech_factor',
    'ADOSC_60_120_0': 'load_tech_factor',
    'ADOSC_40_100_0': 'load_tech_factor',
    'ADOSC_20_60_0': 'load_tech_factor',
    'MFI_200_70_30': 'load_tech_factor',
    'MFI_140_70_30': 'load_tech_factor',
    'MFI_100_70_30': 'load_tech_factor',
    'MFI_40_70_30': 'load_tech_factor',
    'CMO_200_0': 'load_tech_factor',
    'CMO_140_0': 'load_tech_factor',
    'CMO_100_0': 'load_tech_factor',
    'CMO_40_0': 'load_tech_factor',
    'AROON_200_80': 'load_tech_factor',
    'AROON_140_80': 'load_tech_factor',
    'AROON_100_80': 'load_tech_factor',
    'AROON_40_80': 'load_tech_factor',
    'ADX_200_20_10': 'load_tech_factor',
    'ADX_140_20_10': 'load_tech_factor',
    'ADX_100_20_10': 'load_tech_factor',
    'ADX_40_20_10': 'load_tech_factor',
    'CCI_p150d_limit_12': 'load_tech_factor',
    'CCI_p120d_limit_12': 'load_tech_factor',
    'CCI_p60d_limit_12': 'load_tech_factor',
    'CCI_p20d_limit_12': 'load_tech_factor',
    'MACD_40_160': 'load_tech_factor',
    'MACD_40_200': 'load_tech_factor',
    'MACD_20_200': 'load_tech_factor',
    'MACD_20_100': 'load_tech_factor',
    'MACD_10_30': 'load_tech_factor',
    'bias_turn_p120d': 'load_tech_factor',
    'bias_turn_p60d': 'load_tech_factor',
    'bias_turn_p20d': 'load_tech_factor',
    'turn_p150d_0.18': 'load_tech_factor',
    'turn_p30d_0.24': 'load_tech_factor',
    'turn_p120d_0.2': 'load_tech_factor',
    'turn_p60d_0.2': 'load_tech_factor',
    'turn_p20d_0.2': 'load_tech_factor',
    'log_price_0.2': 'load_tech_factor',
    'wgt_return_p120d_0.2': 'load_tech_factor',
    'wgt_return_p60d_0.2': 'load_tech_factor',
    'wgt_return_p20d_0.2': 'load_tech_factor',
    'return_p90d_0.2': 'load_tech_factor',
    'return_p30d_0.2': 'load_tech_factor',
    'return_p120d_0.2': 'load_tech_factor',
    'return_p60d_0.2': 'load_tech_factor',
    'return_p20d_0.2': 'load_tech_factor',
    'volume_moment_p20120d': 'load_tech_factor',
    'volume_moment_p1040d': 'load_tech_factor',
    'volume_moment_p530d': 'load_tech_factor',
    'moment_p50300d': 'load_tech_factor',
    'moment_p30200d': 'load_tech_factor',
    'moment_p40200d': 'load_tech_factor',
    'moment_p20200d': 'load_tech_factor',
    'moment_p20100d': 'load_tech_factor',
    'moment_p10100d': 'load_tech_factor',
    'moment_p1060d': 'load_tech_factor',
    'moment_p510d': 'load_tech_factor',
    'continue_ud_p200d': 'load_tech_factor',
    'evol_p200d': 'load_tech_factor',
    'vol_count_down_p200d': 'load_tech_factor',
    'vol_p200d': 'load_tech_factor',
    'continue_ud_p100d': 'load_tech_factor',
    'evol_p100d': 'load_tech_factor',
    'vol_count_down_p100d': 'load_tech_factor',
    'vol_p100d': 'load_tech_factor',
    'continue_ud_p90d': 'load_tech_factor',
    'evol_p90d': 'load_tech_factor',
    'vol_count_down_p90d': 'load_tech_factor',
    'vol_p90d': 'load_tech_factor',
    'continue_ud_p50d': 'load_tech_factor',
    'evol_p50d': 'load_tech_factor',
    'vol_count_down_p50d': 'load_tech_factor',
    'vol_p50d': 'load_tech_factor',
    'continue_ud_p30d': 'load_tech_factor',
    'evol_p30d': 'load_tech_factor',
    'vol_count_down_p30d': 'load_tech_factor',
    'vol_p30d': 'load_tech_factor',
    'continue_ud_p120d': 'load_tech_factor',
    'evol_p120d': 'load_tech_factor',
    'vol_count_down_p120d': 'load_tech_factor',
    'vol_p120d': 'load_tech_factor',
    'continue_ud_p60d': 'load_tech_factor',
    'evol_p60d': 'load_tech_factor',
    'vol_count_down_p60d': 'load_tech_factor',
    'vol_p60d': 'load_tech_factor',
    'continue_ud_p20d': 'load_tech_factor',
    'evol_p20d': 'load_tech_factor',
    'vol_count_down_p20d': 'load_tech_factor',
    'vol_p20d': 'load_tech_factor',
    'continue_ud_p10d': 'load_tech_factor',
    'evol_p10d': 'load_tech_factor',
    'vol_count_down_p10d': 'load_tech_factor',
    'vol_p10d': 'load_tech_factor',
    'volume_count_down_p120d': 'load_tech_factor',
    'volume_count_down_p60d': 'load_tech_factor',
    'volume_count_down_p20d': 'load_tech_factor',
    'volume_count_down_p10d': 'load_tech_factor',
    'price_p120d_hl': 'load_tech_factor',
    'price_p60d_hl': 'load_tech_factor',
    'price_p20d_hl': 'load_tech_factor',
    'price_p10d_hl': 'load_tech_factor',
    'aadj_r_p120d_col_extre_0.2': 'load_tech_factor',
    'aadj_r_p60d_col_extre_0.2': 'load_tech_factor',
    'aadj_r_p20d_col_extre_0.2': 'load_tech_factor',
    'aadj_r_p10d_col_extre_0.2': 'load_tech_factor',
    'aadj_r_p345d_continue_ud': 'load_tech_factor',
    'aadj_r_p345d_continue_ud_pct': 'load_tech_factor',
    'aadj_r_row_extre_0.2': 'load_tech_factor',
    'TVOL_p90d_col_extre_0.2': 'load_tech_factor',
    'TVOL_p30d_col_extre_0.2': 'load_tech_factor',
    'TVOL_p120d_col_extre_0.2': 'load_tech_factor',
    'TVOL_p60d_col_extre_0.2': 'load_tech_factor',
    'TVOL_p20d_col_extre_0.2': 'load_tech_factor',
    'TVOL_p10d_col_extre_0.2': 'load_tech_factor',
    'TVOL_p345d_continue_ud': 'load_tech_factor',
    'TVOL_row_extre_0.2': 'load_tech_factor',

    'BBANDS_alpha_100_1.5_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_100_1.5_0_1': 'load_tech_factor',
    'BBANDS_alpha_100_1.5_1_0': 'load_tech_factor',
    'BBANDS_alpha_100_1_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_100_1_0_1': 'load_tech_factor',
    'BBANDS_alpha_100_1_1_0': 'load_tech_factor',
    'BBANDS_alpha_100_2_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_100_2_0_1': 'load_tech_factor',
    'BBANDS_alpha_100_2_1_0': 'load_tech_factor',
    'BBANDS_alpha_10_1.5_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_10_1.5_0_1': 'load_tech_factor',
    'BBANDS_alpha_10_1.5_1_0': 'load_tech_factor',
    'BBANDS_alpha_10_1_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_10_1_0_1': 'load_tech_factor',
    'BBANDS_alpha_10_1_1_0': 'load_tech_factor',
    'BBANDS_alpha_10_2_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_10_2_0_1': 'load_tech_factor',
    'BBANDS_alpha_10_2_1_0': 'load_tech_factor',
    'BBANDS_alpha_140_1.5_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_140_1.5_0_1': 'load_tech_factor',
    'BBANDS_alpha_140_1.5_1_0': 'load_tech_factor',
    'BBANDS_alpha_140_1_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_140_1_0_1': 'load_tech_factor',
    'BBANDS_alpha_140_1_1_0': 'load_tech_factor',
    'BBANDS_alpha_140_2_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_140_2_0_1': 'load_tech_factor',
    'BBANDS_alpha_140_2_1_0': 'load_tech_factor',
    'BBANDS_alpha_200_1.5_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_200_1.5_0_1': 'load_tech_factor',
    'BBANDS_alpha_200_1.5_1_0': 'load_tech_factor',
    'BBANDS_alpha_200_1_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_200_1_0_1': 'load_tech_factor',
    'BBANDS_alpha_200_1_1_0': 'load_tech_factor',
    'BBANDS_alpha_200_2_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_200_2_0_1': 'load_tech_factor',
    'BBANDS_alpha_200_2_1_0': 'load_tech_factor',
    'BBANDS_alpha_20_1.5_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_20_1.5_0_1': 'load_tech_factor',
    'BBANDS_alpha_20_1.5_1_0': 'load_tech_factor',
    'BBANDS_alpha_20_1_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_20_1_0_1': 'load_tech_factor',
    'BBANDS_alpha_20_1_1_0': 'load_tech_factor',
    'BBANDS_alpha_20_2_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_20_2_0_1': 'load_tech_factor',
    'BBANDS_alpha_20_2_1_0': 'load_tech_factor',
    'BBANDS_alpha_40_1.5_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_40_1.5_0_1': 'load_tech_factor',
    'BBANDS_alpha_40_1.5_1_0': 'load_tech_factor',
    'BBANDS_alpha_40_1_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_40_1_0_1': 'load_tech_factor',
    'BBANDS_alpha_40_1_1_0': 'load_tech_factor',
    'BBANDS_alpha_40_2_0.5_0.5': 'load_tech_factor',
    'BBANDS_alpha_40_2_0_1': 'load_tech_factor',
    'BBANDS_alpha_40_2_1_0': 'load_tech_factor',
    'MACD_alpha_12_26_9_0.5_0.5': 'load_tech_factor',
    'MACD_alpha_12_26_9_0_1': 'load_tech_factor',
    'MACD_alpha_12_26_9_1_0': 'load_tech_factor',
    'MACD_alpha_20_60_18_0.5_0.5': 'load_tech_factor',
    'MACD_alpha_20_60_18_0_1': 'load_tech_factor',
    'MACD_alpha_20_60_18_1_0': 'load_tech_factor',
    'MA_LINE_alpha_100_40_0.5_0.5': 'load_tech_factor',
    'MA_LINE_alpha_100_40_0_1': 'load_tech_factor',
    'MA_LINE_alpha_100_40_1_0': 'load_tech_factor',
    'MA_LINE_alpha_10_5_0.5_0.5': 'load_tech_factor',
    'MA_LINE_alpha_10_5_0_1': 'load_tech_factor',
    'MA_LINE_alpha_10_5_1_0': 'load_tech_factor',
    'MA_LINE_alpha_120_60_0.5_0.5': 'load_tech_factor',
    'MA_LINE_alpha_120_60_0_1': 'load_tech_factor',
    'MA_LINE_alpha_120_60_1_0': 'load_tech_factor',
    'MA_LINE_alpha_160_60_0.5_0.5': 'load_tech_factor',
    'MA_LINE_alpha_160_60_0_1': 'load_tech_factor',
    'MA_LINE_alpha_160_60_1_0': 'load_tech_factor',
    'MA_LINE_alpha_60_20_0.5_0.5': 'load_tech_factor',
    'MA_LINE_alpha_60_20_0_1': 'load_tech_factor',
    'MA_LINE_alpha_60_20_1_0': 'load_tech_factor',
    'RSI_alpha100_1.5_0.5_0.5': 'load_tech_factor',
    'RSI_alpha100_1.5_0_1': 'load_tech_factor',
    'RSI_alpha100_1.5_1_0': 'load_tech_factor',
    'RSI_alpha100_1_0.5_0.5': 'load_tech_factor',
    'RSI_alpha100_1_0_1': 'load_tech_factor',
    'RSI_alpha100_1_1_0': 'load_tech_factor',
    'RSI_alpha100_2_0.5_0.5': 'load_tech_factor',
    'RSI_alpha100_2_0_1': 'load_tech_factor',
    'RSI_alpha100_2_1_0': 'load_tech_factor',
    'RSI_alpha10_1.5_0.5_0.5': 'load_tech_factor',
    'RSI_alpha10_1.5_0_1': 'load_tech_factor',
    'RSI_alpha10_1.5_1_0': 'load_tech_factor',
    'RSI_alpha10_1_0.5_0.5': 'load_tech_factor',
    'RSI_alpha10_1_0_1': 'load_tech_factor',
    'RSI_alpha10_1_1_0': 'load_tech_factor',
    'RSI_alpha10_2_0.5_0.5': 'load_tech_factor',
    'RSI_alpha10_2_0_1': 'load_tech_factor',
    'RSI_alpha10_2_1_0': 'load_tech_factor',
    'RSI_alpha140_1.5_0.5_0.5': 'load_tech_factor',
    'RSI_alpha140_1.5_0_1': 'load_tech_factor',
    'RSI_alpha140_1.5_1_0': 'load_tech_factor',
    'RSI_alpha140_1_0.5_0.5': 'load_tech_factor',
    'RSI_alpha140_1_0_1': 'load_tech_factor',
    'RSI_alpha140_1_1_0': 'load_tech_factor',
    'RSI_alpha140_2_0.5_0.5': 'load_tech_factor',
    'RSI_alpha140_2_0_1': 'load_tech_factor',
    'RSI_alpha140_2_1_0': 'load_tech_factor',
    'RSI_alpha200_1.5_0.5_0.5': 'load_tech_factor',
    'RSI_alpha200_1.5_0_1': 'load_tech_factor',
    'RSI_alpha200_1.5_1_0': 'load_tech_factor',
    'RSI_alpha200_1_0.5_0.5': 'load_tech_factor',
    'RSI_alpha200_1_0_1': 'load_tech_factor',
    'RSI_alpha200_1_1_0': 'load_tech_factor',
    'RSI_alpha200_2_0.5_0.5': 'load_tech_factor',
    'RSI_alpha200_2_0_1': 'load_tech_factor',
    'RSI_alpha200_2_1_0': 'load_tech_factor',
    'RSI_alpha20_1.5_0.5_0.5': 'load_tech_factor',
    'RSI_alpha20_1.5_0_1': 'load_tech_factor',
    'RSI_alpha20_1.5_1_0': 'load_tech_factor',
    'RSI_alpha20_1_0.5_0.5': 'load_tech_factor',
    'RSI_alpha20_1_0_1': 'load_tech_factor',
    'RSI_alpha20_1_1_0': 'load_tech_factor',
    'RSI_alpha20_2_0.5_0.5': 'load_tech_factor',
    'RSI_alpha20_2_0_1': 'load_tech_factor',
    'RSI_alpha20_2_1_0': 'load_tech_factor',
    'RSI_alpha40_1.5_0.5_0.5': 'load_tech_factor',
    'RSI_alpha40_1.5_0_1': 'load_tech_factor',
    'RSI_alpha40_1.5_1_0': 'load_tech_factor',
    'RSI_alpha40_1_0.5_0.5': 'load_tech_factor',
    'RSI_alpha40_1_0_1': 'load_tech_factor',
    'RSI_alpha40_1_1_0': 'load_tech_factor',
    'RSI_alpha40_2_0.5_0.5': 'load_tech_factor',
    'RSI_alpha40_2_0_1': 'load_tech_factor',
    'RSI_alpha40_2_1_0': 'load_tech_factor',
    'pnd_continue_pct_ud_alpha345_0.5_0.5': 'load_tech_factor',
    'pnd_continue_pct_ud_alpha345_0_1': 'load_tech_factor',
    'pnd_continue_pct_ud_alpha345_1_0': 'load_tech_factor',
    'pnd_continue_ud_alpha345_0.5_0.5': 'load_tech_factor',
    'pnd_continue_ud_alpha345_0_1': 'load_tech_factor',
    'pnd_continue_ud_alpha345_1_0': 'load_tech_factor',
})


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


def deal_fun(sector_name, hold_time, if_only_long, name_list, buy_sell_way_list):
    if if_only_long == 'True':
        if_only_long = True
    else:
        if_only_long = False

    root_path = '/mnt/mfs/DAT_EQT'
    if_save = True
    if_new_program = True

    begin_date = pd.to_datetime('20130101')
    cut_date = pd.to_datetime('20160401')
    end_date = pd.to_datetime('20181201')
    lag = 2
    return_file = ''

    if_hedge = True

    if sector_name.startswith('market_top_300plus'):
        if_weight = 1
        ic_weight = 0

    elif sector_name.startswith('market_top_300to800plus'):
        if_weight = 0
        ic_weight = 1

    else:
        if_weight = 0.5
        ic_weight = 0.5

    main = mfst.FactorTestSector(root_path, if_save, if_new_program, begin_date, cut_date, end_date, [], sector_name,
                                 int(hold_time), lag, return_file, if_hedge, bool(if_only_long), if_weight, ic_weight)
    # 因子相加
    mix_factor, in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, \
    pot_in, fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df = main.single_test_real(name_list, buy_sell_way_list)
    # 因子相乘
    # mix_factor, in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, \
    # pot_in, fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df = main.mul_test_c(name_list, buy_sell_way_list)
    # margin_l, margin_s = bt.AZ_ls_margin(mix_factor, main.return_choose)
    # annual_r = bt.AZ_annual_return(mix_factor, main.return_choose)
    return pot_in, fit_ratio, leve_ratio, bt.AZ_Sharpe_y(pnl_df), pnl_df


def plot_send_all(all_pnl_df):
    for file_name in all_pnl_df.index:
        plot_send_result(all_pnl_df[file_name], bt.AZ_Sharpe_y(all_pnl_df), file_name, '')


def select_fun(file_name, pnl_table_c, max_num=10):
    i = 0
    target_pnl = pd.DataFrame(pnl_table_c[file_name])
    target_pnl.columns = ['target_pnl']
    target_sp = bt.AZ_Sharpe_y(target_pnl).values[0]
    if target_sp < 0:
        target_pnl = -target_pnl

    select_list = []
    while i < max_num:
        pnl_corr = pd.concat([pnl_table_c, target_pnl], axis=1).corr()['target_pnl']
        select_name = pnl_corr.abs().sort_values().index[0]
        select_pnl = pnl_table_c[select_name]
        pnl_table_c = pnl_table_c.drop(columns=select_name)
        select_sp = bt.AZ_Sharpe_y(select_pnl)
        if select_sp > 0:
            tmp_pnl = target_pnl.add(select_pnl, axis=0)
        else:
            tmp_pnl = target_pnl.sub(select_pnl, axis=0)
        tmp_sp = bt.AZ_Sharpe_y(tmp_pnl).values[0]
        print(tmp_sp)

        if tmp_sp > target_sp:
            target_pnl = tmp_pnl
            select_list.append(select_name)
            i += 1
        else:
            i += 1
    return select_list


def get_all_pnl_corr(pnl_df, col_name):
    all_pnl_df = pd.read_csv('/mnt/mfs/AATST/corr_tst_pnls', sep='|', index_col=0, parse_dates=True)
    all_pnl_df_c = pd.concat([all_pnl_df, pnl_df], axis=1)
    a = all_pnl_df_c.iloc[-600:].corr()[col_name]
    print(a[a > 0.5])
    return a[a > 0.65]


def part_single_test(file_name, pnl_table, sharpe_mid, sharpe_df, sector_name, hold_time, if_only_long):
    select_list = select_fun(file_name, pnl_table[sharpe_mid.index], max_num=10)
    portfolio_index = [file_name] + list(select_list)

    buy_sell_way_df = sharpe_df[portfolio_index]
    select_pnl_df = pnl_table[portfolio_index]
    buy_sell_way_df[buy_sell_way_df > 0] = 1
    buy_sell_way_df[buy_sell_way_df < 0] = -1
    buy_sell_way_list = buy_sell_way_df.values
    name_list = [x.split('|')[0] for x in portfolio_index]

    target_pnl = (select_pnl_df * buy_sell_way_df).sum(1)
    target_sharpe = bt.AZ_Sharpe_y(target_pnl)
    target_lvr = bt.AZ_Leverage_ratio(target_pnl.cumsum())
    print(target_sharpe, target_lvr)
    # if target_sharpe > 2:
    pot_in, fit_ratio, leve_ratio, sp, pnl_df = \
        deal_fun(sector_name, hold_time, if_only_long, name_list, buy_sell_way_list)
    # plot_send_result(target_pnl, target_sharpe, file_name, '\n'.join(portfolio_index))
    # if sp > 2:
    if pot_in > 30:
        pnl_df = pd.DataFrame(pnl_df, columns=['target_df'])
        result_df, info_df = bt.commit_check(pnl_df)
        check_int = result_df.prod().values[0]
        corr_matric = get_all_pnl_corr(pnl_df, 'target_df')
        if len(corr_matric) < 2 and check_int == 1:
            print(name_list, buy_sell_way_list)
            print('|'.join([x.split('|')[0] for x in portfolio_index]))
            print('|'.join([str(x) for x in [pot_in, fit_ratio, leve_ratio]]))
            plot_send_result(pnl_df, sp, '[MIX]' + file_name + '',
                             '|'.join([x.split('|')[0] for x in portfolio_index]) + '$'
                             + '|'.join([sector_name, hold_time, if_only_long]) +
                             '|'.join([str(x) for x in [pot_in, fit_ratio, leve_ratio]]))
            print(target_sharpe)
            print(sp)


def single_test():
    pool = Pool(20)
    for sector_name in sector_name_list:
        print(sector_name)
        data = get_file_name(sector_name)

        for (hold_time, if_only_long), part_df in data.groupby(['hold_time', 'if_only_long']):
            print(hold_time)
            print(if_only_long)
            # print(part_df)
            pnl_table = get_pnl_table(part_df, sector_name)
            # drop_list = list(my_factor_dict.keys())
            pnl_table = pnl_table[[x for x in pnl_table.columns if 'intra' not in x]]
            # remy pnl
            # pnl_table_remy = pnl_table[[x for x in pnl_table.columns if x.split('|')[0] in drop_list]]
            # remy_list = [x for x in pnl_table.columns if x.split('|')[0] in drop_list]
            sharpe_df = pnl_table.apply(bt.AZ_Sharpe_y)
            a = sharpe_df.sort_values()

            # sharpe_up = a[remy_list][a[remy_list].abs() > 0.8]
            sharpe_up = a[a.abs() > 0.8]
            print(sharpe_up)
            sharpe_mid = a[a.abs() > 0.7]
            for file_name in sharpe_up.index:
                args = (file_name, pnl_table, sharpe_mid, sharpe_df,
                        sector_name, hold_time, if_only_long)
                # part_single_test(*args)
                pool.apply_async(part_single_test, args=args)
    pool.close()
    pool.join()


def mul_test():
    for sector_name in sector_name_list:
        data = get_file_name(sector_name)
        for (hold_time, if_only_long), part_df in data.groupby(['hold_time', 'if_only_long']):
            print(sector_name, hold_time, if_only_long)

            # print(part_df)
            pnl_table = get_pnl_table(part_df, sector_name)
            drop_list = list(my_factor_dict.keys())
            pnl_table = pnl_table[[x for x in pnl_table.columns if 'intra' not in x]]
            # remy pnl
            # pnl_table_remy = pnl_table[[x for x in pnl_table.columns if x.split('|')[0] in drop_list]]
            # remy_list = [x for x in pnl_table.columns if x.split('|')[0] in drop_list]
            sharpe_df = pnl_table.apply(bt.AZ_Sharpe_y)
            a = sharpe_df.sort_values()
            corr_df = pnl_table.corr()

            sharpe_up = a[a.abs() > 0.8]
            print(a)
            print(sharpe_up)
            sharpe_mid = a[a.abs() > 0.7]
            for file_name in sharpe_up.index:
                corr_series = corr_df.reindex(index=sharpe_mid.index, columns=sharpe_mid.index)[file_name]
                modify_index = corr_series.loc[corr_series.abs().sort_values().index].index[:3]
                portfolio_index = [file_name] + list(modify_index)

                buy_sell_way_df = sharpe_df[portfolio_index]

                buy_sell_way_df[buy_sell_way_df > 0] = 1
                buy_sell_way_df[buy_sell_way_df < 0] = -1
                buy_sell_way_list = buy_sell_way_df.values
                name_list = [x.split('|')[0] for x in portfolio_index]
                pot_in, fit_ratio, leve_ratio, sp, pnl_df = \
                    deal_fun(sector_name, hold_time, if_only_long, name_list, buy_sell_way_list)
                print(sp)
                if sp > 2:
                    print(name_list, buy_sell_way_list)
                    print('|'.join([x.split('|')[0] for x in portfolio_index]))
                    print('|'.join([str(x) for x in [pot_in, fit_ratio, leve_ratio]]))
                    plot_send_result(pnl_df, sp, '|'.join([x.split('|')[0] for x in portfolio_index]) + '$'
                                     + '|'.join([sector_name, hold_time, if_only_long]),
                                     '|'.join([str(x) for x in [pot_in, fit_ratio, leve_ratio]]))


if __name__ == '__main__':
    single_test()
