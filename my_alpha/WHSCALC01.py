import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import time
sys.path.append("/mnt/mfs/LIB_ROOT")
# import funda_data as fd
# from funda_data.funda_data_deal import SectorData
import open_lib_c.shared_paths.path as pt
from open_lib_c.shared_tools import send_email
# import warnings
# warnings.filterwarnings('ignore')
# import loc_lib.shared_tools.back_test as bt


class bt:
    @staticmethod
    def AZ_Load_csv(target_path, index_time_type=True):
        target_df = pd.read_table(target_path, sep='|', index_col=0, low_memory=False).round(8)
        if index_time_type:
            target_df.index = pd.to_datetime(target_df.index)
        return target_df

    @staticmethod
    def AZ_Rolling(df, n, min_periods=1):
        return df.rolling(window=n, min_periods=min_periods)

    @staticmethod
    def AZ_Rolling_mean(df, n, min_periods=1):
        target = df.rolling(window=n, min_periods=min_periods).mean()
        target.iloc[:n - 1] = np.nan
        return target

    @staticmethod
    def AZ_Path_create(target_path):
        """
        添加新路径
        :param target_path:
        :return:
        """
        if not os.path.exists(target_path):
            os.makedirs(target_path)


class BaseDeal:
    @staticmethod
    def signal_mean_fun(signal_df):
        return signal_df.abs().sum(axis=1).replace(0, np.nan).dropna() / len(signal_df) > 0.1

    @staticmethod
    def pnd_continue_ud(raw_df, sector_df, n_list):
        def fun(df, n):
            df_pct = df.diff()
            up_df = (df_pct > 0)
            dn_df = (df_pct < 0)
            target_up_df = up_df.copy()
            target_dn_df = dn_df.copy()

            for i in range(n - 1):
                target_up_df = target_up_df * up_df.shift(i + 1)
                target_dn_df = target_dn_df * dn_df.shift(i + 1)
            target_df = target_up_df.fillna(0).astype(int) - target_dn_df.fillna(0).astype(int)
            return target_df

        all_target_df = pd.DataFrame()
        for n in n_list:
            target_df = fun(raw_df, n)
            target_df = target_df * sector_df
            all_target_df = all_target_df.add(target_df, fill_value=0)
        return all_target_df

    @staticmethod
    def pnd_continue_ud_pct(raw_df, sector_df, n_list):
        all_target_df = pd.DataFrame()
        for n in n_list:
            target_df = raw_df.rolling(window=n).apply(lambda x: 1 if (x >= 0).all() and sum(x) > 0
            else (-1 if (x <= 0).all() and sum(x) < 0 else 0), raw=True)
            target_df = target_df * sector_df
            all_target_df = all_target_df.add(target_df, fill_value=0)
        return all_target_df

    @staticmethod
    def row_extre(raw_df, sector_df, percent):
        raw_df = raw_df * sector_df
        target_df = raw_df.rank(axis=1, pct=True)
        target_df[target_df >= 1 - percent] = 1
        target_df[target_df <= percent] = -1
        target_df[(target_df > percent) & (target_df < 1 - percent)] = 0
        return target_df

    @staticmethod
    def pnd_col_extre(raw_df, sector_df, window, percent, min_periods=1):
        dn_df = raw_df.rolling(window=window, min_periods=min_periods).quantile(percent)
        up_df = raw_df.rolling(window=window, min_periods=min_periods).quantile(1 - percent)
        dn_target = -(raw_df < dn_df).astype(int)
        up_target = (raw_df > up_df).astype(int)
        target_df = dn_target + up_target
        return target_df * sector_df

    @staticmethod
    def info_dict_fun(fun, raw_data_path, args, save_path, if_replace):
        info_dict = dict()
        info_dict['fun'] = fun
        info_dict['raw_data_path'] = raw_data_path
        info_dict['args'] = args
        info_dict['if_replace'] = if_replace
        pd.to_pickle(info_dict, save_path)

    @staticmethod
    def pnd_hl(high, low, close, sector_df, n):
        high_n = high.rolling(window=n, min_periods=1).max().shift(1)
        low_n = low.rolling(window=n, min_periods=1).min().shift(1)
        h_diff = (close - high_n)
        l_diff = (close - low_n)

        h_diff[h_diff > 0] = 1
        h_diff[h_diff <= 0] = 0

        l_diff[l_diff >= 0] = 0
        l_diff[l_diff < 0] = -1

        pos = h_diff + l_diff
        return pos * sector_df

    @staticmethod
    def pnd_volume(volume, sector_df, n):
        volume_roll_mean = bt.AZ_Rolling_mean(volume, n) * sector_df
        volume_df_count_down = 1 / (volume_roll_mean.replace(0, np.nan))
        return volume_df_count_down

    @staticmethod
    def pnd_volitality(adj_r, sector_df, n):
        vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5)
        vol_df[vol_df < 0.08] = 0.08
        return vol_df

    @staticmethod
    def pnd_volitality_count_down(adj_r, sector_df, n):
        vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5) * sector_df
        vol_df[vol_df < 0.08] = 0.08
        return 1 / vol_df.replace(0, np.nan)

    @staticmethod
    def pnd_evol(adj_r, sector_df, n):
        vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5)
        vol_df[vol_df < 0.08] = 0.08
        evol_df = bt.AZ_Rolling(vol_df, 30).apply(lambda x: 1 if x[-1] > 2 * x.mean() else 0, raw=True)
        return evol_df * sector_df

    def pnd_vol_continue_ud(self, adj_r, sector_df, n):
        vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5)
        vol_df[vol_df < 0.08] = 0.08
        vol_continue_ud_df = self.pnd_continue_ud(vol_df, sector_df, n_list=[3, 4, 5])
        return vol_continue_ud_df

    @staticmethod
    def pnnd_moment(df, sector_df, n_short=10, n_long=60):
        ma_long = df.rolling(window=n_long, min_periods=1).mean()
        ma_short = df.rolling(window=n_short, min_periods=1).mean()
        ma_dif = ma_short - ma_long
        ma_dif[(ma_dif > -0.00001) & (ma_dif < 0.00001)] = 0
        ma_dif[ma_dif > 0.00001] = 1
        ma_dif[ma_dif < -0.00001] = -1
        return ma_dif * sector_df

    @staticmethod
    def p1d_jump_hl(close, open_, sector_df, split_float_list):
        target_df = pd.DataFrame()
        for split_float in split_float_list:
            jump_df = open_ / close.shift(1) - 1
            tmp_df = pd.DataFrame(index=jump_df.index, columns=jump_df.columns)
            tmp_df[(jump_df > 0.101) | (jump_df < -0.101)] = 0
            tmp_df[(split_float >= jump_df) & (jump_df >= -split_float)] = 0
            tmp_df[jump_df > split_float] = 1
            tmp_df[jump_df < -split_float] = -1
            target_df = target_df.add(tmp_df, fill_value=0)
        return target_df * sector_df

    def judge_save_fun(self, target_df, file_name, save_root_path, fun, raw_data_path, args, if_filter=True,
                       if_replace=False):
        factor_to_fun = '/mnt/mfs/dat_whs/data/factor_to_fun'
        if if_filter:
            target_df.to_pickle(os.path.join(save_root_path, file_name + '.pkl'))
            # 构建factor_to_fun的字典并存储
            self.info_dict_fun(fun, raw_data_path, args, os.path.join(factor_to_fun, file_name), if_replace)
            print(f'{file_name} success!')
        else:
            target_df.to_pickle(os.path.join(save_root_path, file_name + '.pkl'))
            # 构建factor_to_fun的字典并存储
            self.info_dict_fun(fun, raw_data_path, args, os.path.join(factor_to_fun, file_name), if_replace)
            print(f'{file_name} success!')





class FD:
    class funda_data_deal:
        class BaseDeal:
            @staticmethod
            def signal_mean_fun(signal_df):
                return signal_df.abs().sum(axis=1).replace(0, np.nan).dropna() / len(signal_df) > 0.1

            @staticmethod
            def pnd_continue_ud(raw_df, sector_df, n_list):
                def fun(df, n):
                    df_pct = df.diff()
                    up_df = (df_pct > 0)
                    dn_df = (df_pct < 0)
                    target_up_df = up_df.copy()
                    target_dn_df = dn_df.copy()

                    for i in range(n - 1):
                        target_up_df = target_up_df * up_df.shift(i + 1)
                        target_dn_df = target_dn_df * dn_df.shift(i + 1)
                    target_df = target_up_df.fillna(0).astype(int) - target_dn_df.fillna(0).astype(int)
                    return target_df

                all_target_df = pd.DataFrame()
                for n in n_list:
                    target_df = fun(raw_df, n)
                    target_df = target_df * sector_df
                    all_target_df = all_target_df.add(target_df, fill_value=0)
                return all_target_df

            @staticmethod
            def pnd_continue_ud_pct(raw_df, sector_df, n_list):
                all_target_df = pd.DataFrame()
                for n in n_list:
                    target_df = raw_df.rolling(window=n).apply(lambda x: 1 if (x >= 0).all() and sum(x) > 0
                    else (-1 if (x <= 0).all() and sum(x) < 0 else 0),raw=True)
                    target_df = target_df * sector_df
                    all_target_df = all_target_df.add(target_df, fill_value=0)
                return all_target_df

            @staticmethod
            def row_extre(raw_df, sector_df, percent):
                raw_df = raw_df * sector_df
                target_df = raw_df.rank(axis=1, pct=True)
                target_df[target_df >= 1 - percent] = 1
                target_df[target_df <= percent] = -1
                target_df[(target_df > percent) & (target_df < 1 - percent)] = 0
                return target_df

            @staticmethod
            def pnd_col_extre(raw_df, sector_df, window, percent, min_periods=1):
                dn_df = raw_df.rolling(window=window, min_periods=min_periods).quantile(percent)
                up_df = raw_df.rolling(window=window, min_periods=min_periods).quantile(1 - percent)
                dn_target = -(raw_df < dn_df).astype(int)
                up_target = (raw_df > up_df).astype(int)
                target_df = dn_target + up_target
                return target_df * sector_df

            @staticmethod
            def info_dict_fun(fun, raw_data_path, args, save_path, if_replace):
                info_dict = dict()
                info_dict['fun'] = fun
                info_dict['raw_data_path'] = raw_data_path
                info_dict['args'] = args
                info_dict['if_replace'] = if_replace
                pd.to_pickle(info_dict, save_path)

            @staticmethod
            def pnd_hl(high, low, close, sector_df, n):
                high_n = high.rolling(window=n, min_periods=1).max().shift(1)
                low_n = low.rolling(window=n, min_periods=1).min().shift(1)
                h_diff = (close - high_n)
                l_diff = (close - low_n)

                h_diff[h_diff > 0] = 1
                h_diff[h_diff <= 0] = 0

                l_diff[l_diff >= 0] = 0
                l_diff[l_diff < 0] = -1

                pos = h_diff + l_diff
                return pos * sector_df

            @staticmethod
            def pnd_volume(volume, sector_df, n):
                volume_roll_mean = bt.AZ_Rolling_mean(volume, n) * sector_df
                volume_df_count_down = 1 / (volume_roll_mean.replace(0, np.nan))
                return volume_df_count_down

            @staticmethod
            def pnd_volitality(adj_r, sector_df, n):
                vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5)
                vol_df[vol_df < 0.08] = 0.08
                return vol_df

            @staticmethod
            def pnd_volitality_count_down(adj_r, sector_df, n):
                vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5) * sector_df
                vol_df[vol_df < 0.08] = 0.08
                return 1 / vol_df.replace(0, np.nan)

            @staticmethod
            def pnd_evol(adj_r, sector_df, n):
                vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5)
                vol_df[vol_df < 0.08] = 0.08
                evol_df = bt.AZ_Rolling(vol_df, 30).apply(lambda x: 1 if x[-1] > 2 * x.mean() else 0, raw=True)
                return evol_df * sector_df

            def pnd_vol_continue_ud(self, adj_r, sector_df, n):
                vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5)
                vol_df[vol_df < 0.08] = 0.08
                vol_continue_ud_df = self.pnd_continue_ud(vol_df, sector_df, n_list=[3, 4, 5])
                return vol_continue_ud_df

            @staticmethod
            def pnnd_moment(df, sector_df, n_short=10, n_long=60):
                ma_long = df.rolling(window=n_long, min_periods=1).mean()
                ma_short = df.rolling(window=n_short, min_periods=1).mean()
                ma_dif = ma_short - ma_long
                ma_dif[(ma_dif > -0.00001) & (ma_dif < 0.00001)] = 0
                ma_dif[ma_dif > 0.00001] = 1
                ma_dif[ma_dif < -0.00001] = -1
                return ma_dif * sector_df

            @staticmethod
            def p1d_jump_hl(close, open_, sector_df, split_float_list):
                target_df = pd.DataFrame()
                for split_float in split_float_list:
                    jump_df = open_ / close.shift(1) - 1
                    tmp_df = pd.DataFrame(index=jump_df.index, columns=jump_df.columns)
                    tmp_df[(jump_df > 0.101) | (jump_df < -0.101)] = 0
                    tmp_df[(split_float >= jump_df) & (jump_df >= -split_float)] = 0
                    tmp_df[jump_df > split_float] = 1
                    tmp_df[jump_df < -split_float] = -1
                    target_df = target_df.add(tmp_df, fill_value=0)
                return target_df * sector_df

            def judge_save_fun(self, target_df, file_name, save_root_path, fun, raw_data_path, args, if_filter=True,
                               if_replace=False):
                factor_to_fun = '/mnt/mfs/dat_whs/data/factor_to_fun'
                if if_filter:
                    target_df.to_pickle(os.path.join(save_root_path, file_name + '.pkl'))
                    # 构建factor_to_fun的字典并存储
                    self.info_dict_fun(fun, raw_data_path, args, os.path.join(factor_to_fun, file_name), if_replace)
                    print(f'{file_name} success!')
                else:
                    target_df.to_pickle(os.path.join(save_root_path, file_name + '.pkl'))
                    # 构建factor_to_fun的字典并存储
                    self.info_dict_fun(fun, raw_data_path, args, os.path.join(factor_to_fun, file_name), if_replace)
                    print(f'{file_name} success!')

    class EM_Funda:
        class EM_Funda_Deal(BaseDeal):
            def dev_row_extre(self, data1, data2, sector_df, percent):
                target_df = self.row_extre(data1 / data2, sector_df, percent)
                return target_df

    class EM_Tab14:
        class EM_Tab14_Deal(BaseDeal):
            def return_pnd(self, aadj_r, sector_df, n, percent):
                return_pnd_df = bt.AZ_Rolling(aadj_r, n).sum()
                target_df = self.row_extre(return_pnd_df, sector_df, percent)
                return target_df

            def wgt_return_pnd(self, aadj_r, turnratio, sector_df, n, percent):
                aadj_r_c = (aadj_r * turnratio)
                wgt_return_pnd_df = bt.AZ_Rolling(aadj_r_c, n).sum()
                target_df = self.row_extre(wgt_return_pnd_df, sector_df, percent)
                return target_df

            def log_price(self, close, sector_df, percent):
                target_df = self.row_extre(np.log(close), sector_df, percent)
                return target_df

            def turn_pnd(self, turnratio, sector_df, n, percent):
                turnratio_mean = bt.AZ_Rolling(turnratio, n).mean()
                target_df = self.row_extre(turnratio_mean, sector_df, percent)
                return target_df * sector_df

            @staticmethod
            def bias_turn_pnd(turnratio, sector_df, n):
                bias_turnratio = bt.AZ_Rolling(turnratio, n).mean() / bt.AZ_Rolling(turnratio, 480).mean() - 1
                bias_turnratio_up = (bias_turnratio > 0.00001).astype(int)
                bias_turnratio_dn = (bias_turnratio < -0.00001).astype(int)
                target_df = bias_turnratio_up - bias_turnratio_dn
                return target_df * sector_df

            @staticmethod
            def MACD(close, sector_df, n_fast, n_slow):
                EMAfast = close.ewm(span=n_fast, min_periods=n_slow - 1).mean()
                EMAslow = close.ewm(span=n_slow, min_periods=n_slow - 1).mean()
                MACD = EMAfast - EMAslow
                MACDsign = MACD.ewm(span=9, min_periods=8).mean()
                MACDdiff = MACD - MACDsign
                target_df_up = (MACDdiff > 0.00001).astype(int)
                target_df_dn = (MACDdiff < -0.00001).astype(int)
                target_df = target_df_up - target_df_dn
                return target_df * sector_df

            @staticmethod
            def CCI(high, low, close, sector_df, n, limit_list):
                PP = (high + low + close) / 3
                bt.AZ_Rolling(PP, n).std()
                CCI_signal = (PP - bt.AZ_Rolling(PP, n).mean()) / bt.AZ_Rolling(PP, n).std()
                all_target_df = pd.DataFrame()
                for limit in limit_list:
                    CCI_up = (CCI_signal >= limit).astype(int)
                    CCI_dn = -(CCI_signal <= -limit).astype(int)
                    CCI = CCI_up + CCI_dn
                    all_target_df = all_target_df.add(CCI, fill_value=0)
                return all_target_df * sector_df


class SectorData(object):
    def __init__(self, root_path):
        self.root_path = root_path

    # 获取剔除新股的矩阵
    def get_new_stock_info(self, xnms, xinx):
        new_stock_data = bt.AZ_Load_csv(self.root_path.EM_Tab01.CDSY_SECUCODE / 'LISTSTATE.csv')
        new_stock_data.fillna(method='ffill', inplace=True)
        # 获取交易日信息
        return_df = bt.AZ_Load_csv(self.root_path.EM_Funda.DERIVED_14 / 'aadj_r.csv').astype(float)
        trade_time = return_df.index
        new_stock_data = new_stock_data.reindex(index=trade_time).fillna(method='ffill')
        target_df = new_stock_data.shift(40).notnull().astype(int)
        target_df = target_df.reindex(columns=xnms, index=xinx)
        return target_df

    # 获取剔除st股票的矩阵
    def get_st_stock_info(self, xnms, xinx):
        data = bt.AZ_Load_csv(self.root_path.EM_Tab01.CDSY_CHANGEINFO / 'CHANGEA.csv')
        data = data.reindex(columns=xnms, index=xinx)
        data.fillna(method='ffill', inplace=True)

        data = data.astype(str)
        target_df = data.applymap(lambda x: 0 if 'ST' in x or 'PT' in x else 1)
        return target_df

    # 读取 sector(行业 最大市值等)
    def load_sector_data(self, begin_date, end_date, sector_name):
        market_top_n = bt.AZ_Load_csv(self.root_path.EM_Funda.DERIVED_10 / (sector_name + '.csv'))
        market_top_n = market_top_n[(market_top_n.index >= begin_date) & (market_top_n.index < end_date)]
        market_top_n.dropna(how='all', axis='columns', inplace=True)
        xnms = market_top_n.columns
        xinx = market_top_n.index

        new_stock_df = self.get_new_stock_info(xnms, xinx)
        st_stock_df = self.get_st_stock_info(xnms, xinx)
        sector_df = market_top_n * new_stock_df * st_stock_df
        sector_df.replace(0, np.nan, inplace=True)
        return sector_df


def find_fun(fun_list):
    target_class = FD
    # print(fun_list)
    for a in fun_list[:-1]:
        target_class = getattr(target_class, a)
    # print(target_class)
    target_fun = getattr(target_class(), fun_list[-1])
    return target_fun


def load_raw_data(root_path, raw_data_path, xnms, xinx, if_replace, target_date):
    raw_data_list = []
    for target_path in raw_data_path:
        tmp_data = bt.AZ_Load_csv(os.path.join('/media/hdd1/DAT_EQT', target_path)).reindex(columns=xnms, index=xinx)
        if tmp_data.index[-1] != target_date:
            print(send_email)
            send_email(target_path + ' Data Error!',
                       ['whs@yingpei.com'],
                       [],
                       '[{}]'.format(target_date.strftime('%Y%m%d')))
        if if_replace:
            tmp_data = tmp_data.replace(0, np.nan)
        raw_data_list += [tmp_data]
    return raw_data_list


def create_data_fun(mode, info_path, sector_df, xnms, xinx, target_date):
    info = pd.read_pickle(info_path)
    root_path = pt._BinFiles(mode)
    args = info['args']
    fun_list = info['fun'].split('.')
    raw_data_path = list(map(lambda x: str(x)[17:] if str(x).startswith('/mnt/mfs/DAT_EQT') else x,
                             info['raw_data_path']))
    if_replace = info['if_replace']
    raw_data_list = load_raw_data(root_path, raw_data_path, xnms, xinx, if_replace, target_date)

    target_fun = find_fun(fun_list)
    target_df = target_fun(*raw_data_list, sector_df, *args)
    if (target_df.iloc[-1] != 0).sum() == 0:
        send_email.send_email(info_path, ['whs@yingpei.com'], [], 'Data Update Warning')
    return target_df


def main():
    config_path = '/media/hdd1/DAT_PreCalc/PreCalc_whs'
    mode = 'pro'

    begin_date = pd.to_datetime('20120101')
    end_date = datetime.now()

    sector_name = 'market_top_2000'

    save_root_path = f'/media/hdd1/DAT_PreCalc/PreCalc_whs/{sector_name}'

    bt.AZ_Path_create(save_root_path)
    root_path = pt._BinFiles(mode)
    sector_data_class = SectorData(root_path)
    sector_df = sector_data_class.load_sector_data(begin_date, end_date, sector_name)

    target_date = sector_df.index[-1]

    xnms = sector_df.columns
    xinx = sector_df.index

    config1 = pd.read_pickle(f'{config_path}/config01.pkl')
    factor_info1 = config1['factor_info']

    config2 = pd.read_pickle(f'{config_path}/018AUG.pkl')
    factor_info2 = config2['factor_info']

    config3 = pd.read_pickle(f'{config_path}/018JUL.pkl')
    factor_info3 = config3['factor_info']

    config4 = pd.read_pickle(f'{config_path}/018JUN.pkl')
    factor_info4 = config4['factor_info']

    config5 = pd.read_pickle(f'{config_path}/CRTJUN01.pkl')
    factor_info5 = config5['factor_info']

    config6 = pd.read_pickle(f'{config_path}/CRTJUN02.pkl')
    factor_info6 = config6['factor_info']

    file_name_list = list(set(factor_info1[['name1', 'name2', 'name3']].values.ravel()) |
                          set(factor_info2[['name1', 'name2', 'name3']].values.ravel()) |
                          set(factor_info3[['name1', 'name2', 'name3']].values.ravel()) |
                          set(factor_info4[['name1', 'name2', 'name3']].values.ravel()) |
                          set(factor_info5[['name3']].values.ravel()) |
                          set(factor_info6[['name3']].values.ravel())
                          )

    for file_name in file_name_list:
        factor_to_fun = '/mnt/mfs/dat_whs/data/factor_to_fun'
        info_path = os.path.join(factor_to_fun, file_name)
        file_save_path = os.path.join(save_root_path, f'{file_name}.pkl')
        if os.path.exists(file_save_path):
            cut_date = xinx[-5]
            create_data = pd.read_pickle(file_save_path)
            create_data = create_data[(create_data.index <= cut_date)]
            part_create_data = create_data_fun(mode, info_path, sector_df, xnms, xinx[-300:], target_date)
            part_create_data = part_create_data[(part_create_data.index > cut_date)]
            create_data = create_data.append(part_create_data, sort=False)

        else:
            create_data = create_data_fun(mode, info_path, sector_df, xnms, xinx, target_date)

        create_data.to_pickle(file_save_path)


if __name__ == '__main__':
    a = time.time()
    main()
    b = time.time()
    print('pre cal cost time:{} s'.format(b-a))
