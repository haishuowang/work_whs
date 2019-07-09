import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load.sql import conn
from work_whs.loc_lib.pre_load.log import use_time
from work_whs.bkt_factor_create.raw_data_path import base_data_dict


class DiscreteClass:
    """
    生成离散数据的公用函数
    """

    @staticmethod
    def pnd_con_ud(raw_df, sector_df, n_list):
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
    def pnd_con_ud_pct(raw_df, sector_df, n_list):
        all_target_df = pd.DataFrame()
        for n in n_list:
            target_df = raw_df.rolling(window=n).apply(lambda x: 1 if (x >= 0).all() and sum(x) > 0
            else (-1 if (x <= 0).all() and sum(x) < 0 else 0))
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
    def col_extre(raw_df, sector_df, window, percent, min_periods=1):
        dn_df = raw_df.rolling(window=window, min_periods=min_periods).quantile(percent)
        up_df = raw_df.rolling(window=window, min_periods=min_periods).quantile(1 - percent)
        dn_target = -(raw_df < dn_df).astype(int)
        up_target = (raw_df > up_df).astype(int)
        target_df = dn_target + up_target
        return target_df * sector_df

    @staticmethod
    def signal_fun(zscore_df, sector_df, limit):
        zscore_df[(zscore_df < limit) & (zscore_df > -limit)] = 0
        zscore_df[zscore_df >= limit] = 1
        zscore_df[zscore_df <= -limit] = -1
        return zscore_df * sector_df


class ContinueClass:
    """
    生成连续数据的公用函数
    """

    @staticmethod
    def roll_fun_20(raw_df, sector_df):
        return bt.AZ_Rolling_mean(raw_df, 20)

    @staticmethod
    def roll_fun_40(raw_df, sector_df):
        return bt.AZ_Rolling_mean(raw_df, 40)

    @staticmethod
    def roll_fun_100(raw_df, sector_df):
        return bt.AZ_Rolling_mean(raw_df, 100)

    @staticmethod
    def col_zscore(raw_df, sector_df, n, cap=5, min_periods=0):
        return bt.AZ_Col_zscore(raw_df, n, cap, min_periods)

    @staticmethod
    def row_zscore(raw_df, sector_df, cap=5):
        return bt.AZ_Row_zscore(raw_df * sector_df, cap)

    @staticmethod
    def pnd_vol(raw_df, sector_df, n):
        vol_df = bt.AZ_Rolling(raw_df, n).std().round(4) * (250 ** 0.5)
        return vol_df * sector_df

    # @staticmethod
    # def pnd_count_down(raw_df, sector_df, n):
    #     raw_df = raw_df.replace(0, np.nan)
    #     raw_df_mean = bt.AZ_Rolling_mean(raw_df, n) * sector_df
    #     raw_df_count_down = 1 / (raw_df_mean.round(4).replace(0, np.nan))
    #     return raw_df_count_down

    # return fun
    @staticmethod
    def pnd_return_volatility(adj_r, n):
        vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5)
        vol_df[vol_df < 0.08] = 0.08
        return vol_df

    @staticmethod
    def pnd_return_volatility_count_down(adj_r, sector_df, n):
        vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5) * sector_df
        vol_df[vol_df < 0.08] = 0.08
        return 1 / vol_df.replace(0, np.nan)

    @staticmethod
    def pnd_return_evol(adj_r, sector_df, n):
        vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5)
        vol_df[vol_df < 0.08] = 0.08
        evol_df = bt.AZ_Rolling(vol_df, 30).apply(lambda x: 1 if x[-1] > 2 * x.mean() else 0)
        return evol_df * sector_df


class SpecialClass:
    """
    某些数据使用的特殊函数
    """

    @staticmethod
    def pnd_evol(adj_r, sector_df, n):
        vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5)
        vol_df[vol_df < 0.08] = 0.08
        evol_df = bt.AZ_Rolling(vol_df, 30).apply(lambda x: 1 if x[-1] > 2 * x.mean() else 0)
        return evol_df * sector_df

class TrainFunSet:
    @staticmethod
    def mul_fun(a, b):
        a_l = a.where(a > 0, 0)
        a_s = a.where(a < 0, 0)

        b_l = b.where(b > 0, 0)
        b_s = b.where(b < 0, 0)

        pos_l = a_l.mul(b_l)
        pos_s = a_s.mul(b_s)

        pos = pos_l.sub(pos_s)
        return pos

    @staticmethod
    def sub_fun(a, b):
        return a.sub(b)

    @staticmethod
    def add_fun(a, b):
        return a.add(b)


def add_suffix(x):
    if x[0] in ['0', '3']:
        return x + '.SZ'
    elif x[0] in ['6']:
        return x + '.SH'
    else:
        print('error')


def select_astock(x):
    if len(x) == 6:
        if x[:1] in ['0', '3', '6']:
            return True
        else:
            return False
    else:
        return False


def company_to_stock(map_data_c, company_code_df):
    print('company_to_stock')
    # stock_code_df = pd.DataFrame(columns=sorted(self.map_data_c.values))
    company_code_df = company_code_df.reindex(columns=map_data_c.index)
    company_code_df.columns = map_data_c.values
    company_code_df = company_code_df[sorted(company_code_df.columns)]
    company_code_df.columns = [add_suffix(x) for x in company_code_df.columns]
    company_code_df.dropna(how='all', inplace=True, axis='columns')
    return company_code_df


class SectorFilter:
    def __init__(self, root_path):
        map_data = pd.read_sql('SELECT COMPANYCODE, SECURITYCODE, SECURITYTYPE FROM choice_fndb.CDSY_SECUCODE', conn)
        map_data.index = map_data['COMPANYCODE']
        map_data_c = map_data['SECURITYCODE'][map_data['SECURITYTYPE'] == 'A股']
        self.map_data_c = map_data_c[map_data_c.apply(select_astock)]
        self.root_path = root_path

    def load_index_data(self, index_name):
        data = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv')
        target_df = data[index_name]
        return target_df * 0.01

    def filter_market(self):
        market_df = bt.AZ_Load_csv(f'{self.root_path}/')

    def filter_vol(self):
        pass

    def filter_moment(self):
        pass

    def filter_beta(self, if_weight, ic_weight):
        window = 200
        aadj_r = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/DERIVED_14/aadj_r.csv')

        if_df = self.load_index_data('000300').reindex(index=aadj_r.index)
        ic_df = self.load_index_data('000905').reindex(index=aadj_r.index)
        index_df = (if_df * if_weight).add(ic_df * ic_weight, fill_value=0)
        aadj_r_roll = bt.AZ_Rolling_mean(aadj_r, window)
        index_df_roll = bt.AZ_Rolling_mean(index_df, window)
        tmp_df = aadj_r_roll.sub(index_df_roll, axis=0).shift(1)
        beta_mask_1 = tmp_df > 0
        beta_mask_2 = tmp_df <= 0
        # target_df = tmp_df_up - tmp_df_dn
        return beta_mask_1, beta_mask_2

    @use_time
    def filter_inst(self):
        def fun(x):
            xx = x['SHAREHDRATIO'][x['SHAREHDTYPE']
                .apply(lambda x: True if x in ['002', '003', '004', '007', '010', '012', '013',
                                               '014', '015', '016', '017', '018'] else False)].sum()
            # print(xx)
            return xx

        raw_df = pd.read_sql('SELECT COMPANYCODE, REPORTDATE, RANK, SHAREHDTYPE, SHAREHDRATIO, SHAREHDANUM '
                             'FROM choice_fndb.LICO_ES_CIRHOLDERSLIST', conn)

        raw_inst_pct_df = raw_df[['REPORTDATE', 'COMPANYCODE', 'SHAREHDTYPE', 'SHAREHDRATIO']] \
            .groupby(['REPORTDATE', 'COMPANYCODE']).apply(fun).unstack()

        inst_pct_df = raw_inst_pct_df.fillna(method='ffill', limit=250)
        tmp_df = company_to_stock(self.map_data_c, inst_pct_df)
        inst_mask_1 = (tmp_df >= 10)
        inst_mask_2 = (tmp_df >= 5) & (tmp_df < 10)
        inst_mask_3 = (tmp_df >= 0) & (tmp_df < 5)
        return inst_mask_1, inst_mask_2, inst_mask_3


class SectorData:
    """
    load处理数据时的universe
    """
    def __init__(self, root_path):
        self.root_path = root_path

    # 获取剔除新股的矩阵
    def get_new_stock_info(self, xnms, xinx):
        new_stock_data = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/CDSY_SECUCODE/LISTSTATE.csv')
        new_stock_data.fillna(method='ffill', inplace=True)
        # 获取交易日信息
        return_df = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/DERIVED_14/aadj_r.csv').astype(float)
        trade_time = return_df.index
        new_stock_data = new_stock_data.reindex(index=trade_time).fillna(method='ffill')
        target_df = new_stock_data.shift(40).notnull().astype(int)
        target_df = target_df.reindex(columns=xnms, index=xinx)
        return target_df

    # 获取剔除st股票的矩阵
    def get_st_stock_info(self, xnms, xinx):
        data = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/CDSY_CHANGEINFO/CHANGEA.csv')
        data = data.reindex(columns=xnms, index=xinx)
        data.fillna(method='ffill', inplace=True)

        data = data.astype(str)
        target_df = data.applymap(lambda x: 0 if 'ST' in x or 'PT' in x else 1)
        return target_df

    # 读取 sector(行业 最大市值等)
    def load_sector_data(self, begin_date, end_date, sector_name):
        if sector_name.startswith('index'):
            index_name = sector_name.split('_')[-1]
            market_top_n = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/IDEX_YS_WEIGHT_A/SECURITYNAME_{index_name}.csv')
            market_top_n = market_top_n.where(market_top_n != market_top_n, other=1)
        else:
            market_top_n = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/DERIVED_10/{sector_name}.csv')

        market_top_n = market_top_n[(market_top_n.index >= begin_date) & (market_top_n.index < end_date)]
        market_top_n.dropna(how='all', axis='columns', inplace=True)

        xnms = market_top_n.columns
        xinx = market_top_n.index

        new_stock_df = self.get_new_stock_info(xnms, xinx)
        st_stock_df = self.get_st_stock_info(xnms, xinx)
        sector_df = market_top_n * new_stock_df * st_stock_df
        sector_df.replace(0, np.nan, inplace=True)
        return sector_df


class DataDeal(SectorData, DiscreteClass, ContinueClass):
    def __init__(self, begin_date, end_date, root_path, sector_name):
        super(DataDeal, self).__init__(root_path=root_path)
        # self.root_path = root_path
        self.sector_name = sector_name
        self.sector_df = self.load_sector_data(begin_date, end_date, sector_name)

        self.xinx = self.sector_df.index
        self.xnms = self.sector_df.columns

        self.save_root_path = '/mnt/mfs/dat_whs/data/factor_data'
        self.save_sector_path = f'{self.save_root_path}/{self.sector_name}'
        bt.AZ_Path_create(self.save_sector_path)

    def load_raw_data(self, file_name):
        data_path = base_data_dict[file_name]
        if len(data_path) != 0:
            raw_df = bt.AZ_Load_csv(f'{self.root_path}/{data_path}') \
                .reindex(index=self.xinx, columns=self.xnms).round(4)
        else:
            raw_df = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/daily/{file_name}.csv') \
                .reindex(index=self.xinx, columns=self.xnms).round(4)

        return raw_df

    def count_return_data(self, factor_name, z_score=True):
        if len(factor_name.split('|')) == 3:
            str_to_num = lambda x: float(x) if '.' in x else int(x)
            file_name, fun_name, para_str = factor_name.split('|')
            para = [str_to_num(x) for x in para_str.split('_')]
        else:
            file_name, fun_name = factor_name.split('|')
            para = []

        raw_df = self.load_raw_data(file_name)
        fun = getattr(self, fun_name)
        target_df = fun(raw_df, self.sector_df, *para)
        if z_score:
            if 'zscore' in factor_name:
                target_zscore_df = target_df
            else:
                target_zscore_df = self.row_zscore(target_df, self.sector_df)
            return target_zscore_df
        else:
            return target_df


class CorrCheck:
    @staticmethod
    def load_index_data(index_name, xinx):
        data = bt.AZ_Load_csv(os.path.join('/mnt/mfs/DAT_EQT', 'EM_Tab09/INDEX_TD_DAILYSYS/CHG.csv'))
        target_df = data[index_name].reindex(index=xinx)
        return target_df * 0.01

    def get_corr_matrix(self, cut_date=None):
        pos_file_list = [x for x in os.listdir('/mnt/mfs/AAPOS') if x.startswith('WHS')]
        return_df = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv').astype(float)

        index_df_1 = self.load_index_data('000300', return_df.index).fillna(0)
        index_df_2 = self.load_index_data('000905', return_df.index).fillna(0)

        sum_pnl_df = pd.DataFrame()
        for pos_file_name in pos_file_list:
            pos_df = bt.AZ_Load_csv('/mnt/mfs/AAPOS/{}'.format(pos_file_name))

            cond_1 = 'IF01' in pos_df.columns
            cond_2 = 'IC01' in pos_df.columns
            if cond_1 and cond_2:
                hedge_df = 0.5 * index_df_1 + 0.5 * index_df_2
                return_df_c = return_df.sub(hedge_df, axis=0)
            elif cond_1:
                hedge_df = index_df_1
                return_df_c = return_df.sub(hedge_df, axis=0)
            elif cond_2:
                hedge_df = index_df_2
                return_df_c = return_df.sub(hedge_df, axis=0)
            else:
                print('alpha hedge error')
                continue
            pnl_df = (pos_df.shift(2) * return_df_c).sum(axis=1)
            pnl_df.name = pos_file_name
            sum_pnl_df = pd.concat([sum_pnl_df, pnl_df], axis=1)
            # plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df), 'mix_factor')
        if cut_date is not None:
            sum_pnl_df = sum_pnl_df[sum_pnl_df.index > cut_date]
        return sum_pnl_df

    @staticmethod
    def get_all_pnl_corr(pnl_df, col_name):
        all_pnl_df = pd.read_csv('/mnt/mfs/AATST/corr_tst_pnls', sep='|', index_col=0, parse_dates=True)
        all_pnl_df_c = pd.concat([all_pnl_df, pnl_df], axis=1)
        a = all_pnl_df_c.iloc[-600:].corr()[col_name]
        return a[a > 0.6]

    def get_all_pnl_df(self, root_path):
        file_name_list = os.listdir(root_path)
        if len(file_name_list) == 0:
            return pd.DataFrame()
        else:
            result_list = []
            for file_name in file_name_list:
                pnl_df = pd.read_pickle(f'{root_path}/{file_name}')
                result_list.append(pnl_df)
            all_pnl_df = pd.concat(result_list, axis=1)
            return all_pnl_df

    def corr_test_fun(self, pnl_df, alpha_name):
        sum_pnl_df = self.get_corr_matrix(cut_date=None)
        sum_pnl_df_c = pd.concat([sum_pnl_df, pnl_df], axis=1)
        corr_self = sum_pnl_df_c.corr()[[alpha_name]]
        other_corr = self.get_all_pnl_corr(pnl_df, alpha_name)
        print(other_corr)
        self_corr = corr_self[corr_self > 0.6].dropna(axis=0)
        print(corr_self.sort_values(by=alpha_name)[-5:])
        if len(self_corr) >= 2 or len(other_corr) >= 2:
            print('FAIL!')
            send_email.send_email('FAIL!\n' + self_corr.to_html(),
                                  ['whs@yingpei.com'],
                                  [],
                                  '[RESULT DEAL]' + alpha_name)
        else:
            print('SUCCESS!')
            send_email.send_email('SUCCESS!\n' + self_corr.to_html(),
                                  ['whs@yingpei.com'],
                                  [],
                                  '[RESULT DEAL]' + alpha_name)
        print('______________________________________')
        return 0

    def corr_self_check(self, pnl_df, pnl_save_path):
        assert type(pnl_df) == pd.Series
        all_pnl_df = self.get_all_pnl_df(pnl_save_path)
        corr_sr = all_pnl_df.corrwith(pnl_df, axis=0)
        return corr_sr


# if __name__ == '__main__':
    # root_path = '/mnt/mfs/DAT_EQT'
    # a = SectorFilter(root_path).filter_inst()

# index_name = '000905'
# sector_df = bt.AZ_Load_csv(f'{root_path}/EM_Funda/IDEX_YS_WEIGHT_A/SECURITYNAME_{index_name}.csv')
# sector_df[sector_df == sector_df] = 1
#
# table_path = f'{root_path}/EM_Funda/LICO_IM_INCHG'
# target_file_list = [x[:-4] for x in os.listdir(table_path) if 'ZhongXing_Level2' in x]
#
#
# def fun(file_name):
#     xinx = sector_df.index
#     raw_df = bt.AZ_Load_csv(f'{table_path}/{file_name}.csv')
#     xnms = sorted(list(set(sector_df.columns) & set(raw_df.columns)))
#     tmp_df = raw_df.reindex(index=xinx, columns=xnms)
#     target_df = tmp_df * sector_df.reindex(columns=xnms)
#     target_num = target_df.sum(1)
#     target_num.name = file_name
#     return target_num
#
#
# target_num_list = [fun(file_name) for file_name in target_file_list]
# data = pd.concat(target_num_list, axis=1)
# '10002252'
