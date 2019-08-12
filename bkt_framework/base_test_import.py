import sys

sys.path.append('/mnt/mfs')
import string
from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load.plt import plot_send_result
from work_whs.bkt_factor_create.raw_data_path import base_data_dict
import warnings

warnings.filterwarnings("ignore")
lock = Lock()

pool = Pool()


def get_code():
    return ''.join(random.sample(string.ascii_letters + string.digits, 32))


class FunSet:
    @staticmethod
    def div(a, b):
        """
        ratio -～|+~
        :param a:
        :param b:
        :return:
        """
        return a.div(b)

    @staticmethod
    def div_2(a, b):
        """
        ratio -～|+~
        :param a:
        :param b:
        :return:
        """
        return a.div(b.sub(b.min(axis=1) - 1, axis=0))

    @staticmethod
    def diff(a, n):
        """
        diff
        :param a:
        :param n:
        :return:
        """
        return a.diff(n)

    @staticmethod
    def pct_change(a, n):
        """
        ratio
        :param a:
        :param n:
        :return:
        """
        return a.pct_change(n)

    @staticmethod
    def add(a, b):
        """
        no mul +1
        :param a:
        :param b:
        :return:
        """
        return a.add(b)

    @staticmethod
    def mul(a, b):
        """
        no
        :param a:
        :param b:
        :return:
        """
        return a.mul(b)

    @staticmethod
    def sub(a, b):
        """
        diff
        :param a:
        :param b:
        :return:
        """
        return a.sub(b)

    @staticmethod
    def std(a, n):
        """
        std
        :param a:
        :param n:
        :return:
        """
        return bt.AZ_Rolling(a, n).std()

    @staticmethod
    def max_fun(a, n):
        """
        no
        :param a:
        :param n:
        :return:
        """
        return bt.AZ_Rolling(a, n).max()

    @staticmethod
    def min_fun(a, n):
        """
        no
        :param a:
        :param n:
        :return:
        """
        return bt.AZ_Rolling(a, n).min()

    @staticmethod
    def rank_pct(a):
        """
        continue 0|1
        :param a:
        :return:
        """
        return a.rank(axis=1, pct=True)

    @staticmethod
    def corr(a, b, n):
        """
        线性相关性
        continue -1|1
        :param a:
        :param b:
        :param n:
        :return:
        """

        return bt.AZ_Rolling(a, n).corr(b)

    def corr_rank(self, a, b, n):
        """
        线性相关性
        continue -1|1
        :param a:
        :param b:
        :param n:
        :return:
        """
        a = self.rank_pct(a)
        b = self.rank_pct(b)
        return bt.AZ_Rolling(a, n).corr(b)

    @staticmethod
    def ma(a, n):
        """
        no
        :param a:
        :param n:
        :return:
        """
        return bt.AZ_Rolling(a, n).mean()

    @staticmethod
    def shift(a, n):
        """
        no
        :param a:
        :param n:
        :return:
        """
        return a.shift(n)

    @staticmethod
    def abs_fun(a):
        """
        no 0|
        :param a:
        :return:
        """
        return np.abs(a)


class FactorTestBase(FunSet):
    def __init__(self, root_path, if_save, if_new_program, begin_date, end_date, sector_name,
                 hold_time, lag, return_file, if_hedge, if_only_long):
        self.root_path = root_path
        self.if_save = if_save
        self.if_new_program = if_new_program
        self.begin_date = begin_date
        self.end_date = end_date
        self.sector_name = sector_name
        self.hold_time = hold_time
        self.lag = lag
        self.return_file = return_file
        self.if_hedge = if_hedge
        self.if_only_long = if_only_long

        if sector_name.startswith('market_top_300plus') \
                or sector_name.startswith('index_000300'):
            if_weight = 1
            ic_weight = 0

        elif sector_name.startswith('market_top_300to800plus') \
                or sector_name.startswith('index_000905'):
            if_weight = 0
            ic_weight = 1

        else:
            if_weight = 0.5
            ic_weight = 0.5

        self.if_weight = if_weight
        self.ic_weight = ic_weight
        return_df = self.load_return_data()
        self.xinx = return_df.index
        sector_df = self.load_sector_data()
        self.xnms = sector_df.columns

        return_df = return_df.reindex(columns=self.xnms)
        self.sector_df = sector_df.reindex(index=self.xinx)
        print('Loaded sector DataFrame!')
        if if_hedge:
            if ic_weight + if_weight != 1:
                exit(-1)
        else:
            if_weight = 0
            ic_weight = 0

        index_df_1 = self.load_index_data('000300').fillna(0)
        index_df_2 = self.load_index_data('000905').fillna(0)
        hedge_df = if_weight * index_df_1 + ic_weight * index_df_2
        self.return_df = return_df.sub(hedge_df, axis=0)
        print('Loaded return DataFrame!')

        suspendday_df, limit_buy_sell_df = self.load_locked_data()
        limit_buy_sell_df_c = limit_buy_sell_df.shift(-1)
        limit_buy_sell_df_c.iloc[-1] = 1

        suspendday_df_c = suspendday_df.shift(-1)
        suspendday_df_c.iloc[-1] = 1
        self.suspendday_df_c = suspendday_df_c
        self.limit_buy_sell_df_c = limit_buy_sell_df_c
        print('Loaded suspendday_df and limit_buy_sell DataFrame!')
        self.tmp_mix_path = '/media/hdd1/dat_whs/data/tmp'
        # mix_factor_name_dict_path = f'{self.tmp_mix_path}/{self.sector_name}/mix_factor_name_dict.pkl'
        # if os.path.exists(mix_factor_name_dict_path):
        #     self.mix_factor_name_dict = pd.read_pickle(mix_factor_name_dict_path)
        # else:
        self.mix_factor_name_dict = {}
        self.factor_way_dict = {}

    @staticmethod
    def row_zscore(raw_df, sector_df, cap=5):
        return bt.AZ_Row_zscore(raw_df * sector_df, cap)

    def reindex_fun(self, df):
        return df.reindex(index=self.xinx, columns=self.xnms)

    @staticmethod
    def create_log_save_path(target_path):
        top_path = os.path.split(target_path)[0]
        if not os.path.exists(top_path):
            os.mkdir(top_path)
        if not os.path.exists(target_path):
            os.mknod(target_path)

    @staticmethod
    def row_extre(raw_df, sector_df, percent):
        raw_df = raw_df * sector_df
        target_df = raw_df.rank(axis=1, pct=True)
        target_df[target_df >= 1 - percent] = 1
        target_df[target_df <= percent] = -1
        target_df[(target_df > percent) & (target_df < 1 - percent)] = 0
        return target_df

    @staticmethod
    def pos_daily_fun(df, n=5):
        return df.rolling(window=n, min_periods=1).sum()

    def check_factor(self, name_list, file_name, check_path=None):
        if check_path is None:
            load_path = os.path.join('/mnt/mfs/dat_whs/data/new_factor_data/' + self.sector_name)
        else:
            load_path = check_path
        exist_factor = set([x[:-4] for x in os.listdir(load_path)])
        use_factor = set(name_list)
        a = use_factor - exist_factor
        if len(a) != 0:
            print('factor not enough!')
            send_email.send_email(f'{file_name} factor not enough!', ['whs@yingpei.com'], [], 'Factor Test Warning!')

    @staticmethod
    def create_all_para(tech_name_list, funda_name_list):

        target_list_1 = []
        for tech_name in tech_name_list:
            for value in combinations(funda_name_list, 2):
                target_list_1 += [[tech_name] + list(value)]

        target_list_2 = []
        for funda_name in funda_name_list:
            for value in combinations(tech_name_list, 2):
                target_list_2 += [[funda_name] + list(value)]

        target_list = target_list_1 + target_list_2
        return target_list

    # 获取剔除新股的矩阵
    def get_new_stock_info(self, xnms, xinx):
        target_df = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/DERIVED_01/NewStock.csv')
        target_df = target_df.reindex(columns=xnms, index=xinx)
        return target_df

    # 获取剔除st股票的矩阵
    def get_st_stock_info(self, xnms, xinx):
        target_df = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/DERIVED_01/StAndPtStock.csv')
        target_df = target_df.reindex(columns=xnms, index=xinx)
        return target_df

    def load_return_data(self):
        return_df = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Funda/DERIVED_14/aadj_r.csv'))
        return_df = return_df[(return_df.index >= self.begin_date) & (return_df.index < self.end_date)]
        return return_df

    # 获取sector data
    def load_sector_data(self):
        if self.sector_name.startswith('index'):
            index_name = self.sector_name.split('_')[-1]
            market_top_n = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/IDEX_YS_WEIGHT_A/SECURITYNAME_{index_name}.csv')
            market_top_n = market_top_n.where(market_top_n != market_top_n, other=1)
        else:
            market_top_n = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/DERIVED_10/{self.sector_name}.csv')

        market_top_n = market_top_n.reindex(index=self.xinx)
        market_top_n.dropna(how='all', axis='columns', inplace=True)

        xnms = market_top_n.columns
        xinx = market_top_n.index

        new_stock_df = self.get_new_stock_info(xnms, xinx)
        st_stock_df = self.get_st_stock_info(xnms, xinx)
        sector_df = market_top_n * new_stock_df * st_stock_df
        sector_df.replace(0, np.nan, inplace=True)
        return sector_df

    def load_index_weight_data(self, index_name):
        index_info = bt.AZ_Load_csv(self.root_path + f'/EM_Funda/IDEX_YS_WEIGHT_A/SECURITYNAME_{index_name}.csv')
        index_info = self.reindex_fun(index_info)
        index_mask = (index_info.notnull() * 1).replace(0, np.nan)

        mkt_cap = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Funda/LICO_YS_STOCKVALUE/AmarketCapExStri.csv'))
        mkt_roll = mkt_cap.rolling(250, min_periods=0).mean()
        mkt_roll = self.reindex_fun(mkt_roll)

        mkt_roll_qrt = np.sqrt(mkt_roll)
        mkt_roll_qrt_index = mkt_roll_qrt * index_mask
        index_weight = mkt_roll_qrt_index.div(mkt_roll_qrt_index.sum(axis=1), axis=0)
        return index_weight

    # 涨跌停都不可交易
    def load_locked_data(self):
        suspendday_df = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/DERIVED_01/SuspendedStock.csv') \
            .reindex(columns=self.xnms, index=self.xinx)
        limit_buy_sell_df = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/DERIVED_01/LimitedBuySellStock.csv') \
            .reindex(columns=self.xnms, index=self.xinx)
        return suspendday_df, limit_buy_sell_df

    # 获取index data
    def load_index_data(self, index_name):
        data = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/DERIVED_WHS/CHG_{index_name}.csv', header=None)
        target_df = data.iloc[:, 0].reindex(index=self.xinx)
        return target_df

    def signal_to_pos(self, signal_df):
        # 下单日期pos
        order_df = signal_df.replace(np.nan, 0)
        # 排除入场场涨跌停的影响
        order_df = order_df * self.sector_df * self.limit_buy_sell_df_c * self.suspendday_df_c
        order_df = order_df.div(order_df.abs().sum(axis=1).replace(0, np.nan), axis=0)
        order_df[order_df > 0.05] = 0.05
        order_df[order_df < -0.05] = -0.05
        daily_pos = bt.AZ_Rolling_mean(order_df, self.hold_time)
        daily_pos.fillna(0, inplace=True)
        # 排除出场涨跌停的影响
        daily_pos = daily_pos * self.limit_buy_sell_df_c * self.suspendday_df_c
        daily_pos.fillna(method='ffill', inplace=True)
        return daily_pos

    def signal_to_pos_ls(self, signal_df, ls_para):
        if ls_para == 'l':
            signal_df_up = signal_df[signal_df > 0]
            daily_pos = self.signal_to_pos(signal_df_up)
        elif ls_para == 's':
            signal_df_dn = signal_df[signal_df < 0].abs()
            daily_pos = self.signal_to_pos(signal_df_dn)
        elif ls_para == 'ls':
            daily_pos = self.signal_to_pos(signal_df)
        else:
            daily_pos = self.signal_to_pos(signal_df)
        return daily_pos

    @staticmethod
    def judge_way(sharpe):
        if sharpe > 0:
            return 1
        elif sharpe < 0:
            return -1
        else:
            return 0

    def load_raw_data(self, file_name):
        data_path = base_data_dict[file_name]
        if len(data_path) != 0:
            raw_df = bt.AZ_Load_csv(f'{self.root_path}/{data_path}') \
                .reindex(index=self.xinx, columns=self.xnms).round(4)
        else:
            raw_df = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/daily/{file_name}.csv') \
                .reindex(index=self.xinx, columns=self.xnms).round(4)
        return raw_df

    def load_mix_data(self, file_name):
        file_name_str = str(file_name)
        if type(file_name) is str:
            target_df = self.load_raw_data(file_name)
        elif file_name_str in self.mix_factor_name_dict.keys():
            mix_name = self.mix_factor_name_dict[file_name_str]
            target_df = pd.read_pickle(f'{self.tmp_mix_path}/{self.sector_name}/{mix_name}')
            # fun_name, factor_name_list, para_list = file_name
            # factor_df_list = [self.load_mix_data(factor_name) for factor_name in factor_name_list]
            # target_df_count = getattr(self, fun_name)(*factor_df_list, *para_list)
            # # zcore 根据用到的数据增加权重
            # target_df_count = self.row_zscore(target_df_count, self.sector_df) * len(factor_name_list)
            # print((target_df - target_df_count).abs().sum(1))
            # print((target_df - target_df_count).abs().sum(1).sum())
            # print(1)
        else:
            mix_name = get_code()

            fun_name, factor_name_list, para_list = file_name
            factor_df_list = [self.load_mix_data(factor_name) for factor_name in factor_name_list]
            target_df = getattr(self, fun_name)(*factor_df_list, *para_list)
            # zcore 根据用到的数据增加权重
            target_df = self.row_zscore(target_df, self.sector_df)  # * np.sqrt(len(factor_name_list))
            # if 'corr' in fun_name:
            #     print(target_df)

            self.mix_factor_name_dict[file_name_str] = mix_name
            bt.AZ_Path_create(f'{self.tmp_mix_path}/{self.sector_name}')
            pd.to_pickle(target_df, f'{self.tmp_mix_path}/{self.sector_name}/{mix_name}')
            # print('create factor_____________')
            # print(file_name_str, mix_name)
        return target_df

    def back_test(self, data_df, cut_date, percent, return_pos=False, ls_para='ls'):
        cut_time = pd.to_datetime(cut_date)
        signal_df = self.row_extre(data_df, self.sector_df, percent)
        if len(signal_df.abs().sum(1).replace(0, np.nan).dropna()) / len(self.xinx) > 0.7:
            pos_df = self.signal_to_pos_ls(signal_df, ls_para)
            pnl_table = pos_df.shift(self.lag) * self.return_df
            pnl_df = pnl_table.sum(1)
            sample_in_index = (pnl_df.index < cut_time)
            sample_out_index = (pnl_df.index >= cut_time)

            pnl_df_in = pnl_df[sample_in_index]
            pnl_df_out = pnl_df[sample_out_index]

            pos_df_in = pos_df[sample_in_index]
            pos_df_out = pos_df[sample_out_index]

            sp_in = bt.AZ_Sharpe_y(pnl_df_in)
            sp_out = bt.AZ_Sharpe_y(pnl_df_out)

            pot_in = bt.AZ_Pot(pos_df_in, pnl_df_in.sum())
            pot_out = bt.AZ_Pot(pos_df_out, pnl_df_out.sum())

            sp = bt.AZ_Sharpe_y(pnl_df)
            pot = bt.AZ_Pot(pos_df, pnl_df.sum())
            if self.if_only_long:
                if ls_para == 'l':
                    way_in, way_out, way = 1, 1, 1
                elif ls_para == 's':
                    way_in, way_out, way = -1, -1, -1
                else:
                    way_in, way_out, way = self.judge_way(sp_in), self.judge_way(sp_out), self.judge_way(sp)
            else:
                way_in, way_out, way = self.judge_way(sp_in), self.judge_way(sp_out), self.judge_way(sp)
            result_list = [sp_in, sp_out, sp, pot_in, pot_out, pot, way_in, way_out, way]
            info_df = pd.Series(result_list, index=['sp_in', 'sp_out', 'sp',
                                                    'pot_in', 'pot_out', 'pot',
                                                    'way_in', 'way_out', 'way'])
        else:
            info_df = pd.Series([0] * 9, index=['sp_in', 'sp_out', 'sp',
                                                'pot_in', 'pot_out', 'pot',
                                                'way_in', 'way_out', 'way'])
            pnl_df = pd.Series([0] * len(self.xinx), index=self.xinx)
            pos_df = pd.DataFrame(columns=data_df.columns, index=data_df.index)
        if return_pos:
            return info_df, pnl_df, pos_df
        else:
            return info_df, pnl_df


class FactorTest(FactorTestBase):
    def __init__(self, *args):
        super(FactorTest, self).__init__(*args)

        self.window_list = [5, 17, 41, 127]
        self.shift_list = [1, 5]

        self.fun_info_set = OrderedDict({
            'div': ['+ratio', '*1', 0, 2, None, None],
            'diff': ['+diff', '*1', 0, 1, ['window_list'], None],
            'pct_change': ['+ratio', '*1', 0, 1, ['window_list'], None],
            'add': ['no', '+1', 0, 2, None, "p1['type']==p2['type']"],

            'mul': ['no', 'no', 0, 2, None, "p1['type']==p2['type']|p1['type'] in ['single', 'continue']"],
            'sub': ['+diff', '*1', 0, 2, None, "p1['type']==p2['type']"],
            'std': ['+std', '*1', 0, 1, ['window_list'], None],
            'max_fun': ['no', '*1', 0, 1, ['window_list'], None],
            'min_fun': ['no', '*1', 0, 1, ['window_list'], None],

            # 'corr': ['continue', '*1', 0, 2, ['window_list'], "p1['type']!=p2['type']"],
            'corr_rank': ['continue', '*1', 0, 2, ['window_list'], "p1['type']!=p2['type']"],

            'ma': ['no', '*1', 0, 1, ['window_list'], None],
            'shift': ['no', '*1', 0, 1, ['window_list'], None],

            'abs_fun': ['no', '*1', 0, 1, None, None],
        })
        self.fun_info_df = pd.DataFrame.from_dict(self.fun_info_set).T
        self.fun_info_df.columns = ['t_type', 'mul', 'axis', 'data_num', 'para', 'detail']

    def part_mix_factor(self, base_factor_list, fun_use_num):
        """
        给定factor集合 生成下factor
        :param base_factor_list:
        :param fun_use_num:
        :return:
        """
        tmp_fun_2 = lambda x: eval(x) if '[' in x else x

        target_dict = OrderedDict()
        all_fun_list = list(self.fun_info_set.keys())
        use_fun_list = all_fun_list
        use_fun_list.remove('add')
        use_fun_list.remove('sub')
        for fun_name in use_fun_list:
            data_num = self.fun_info_df.loc[fun_name]['data_num']
            para_name_list = self.fun_info_df.loc[fun_name]['para']

            all_factors_list = list(combinations(base_factor_list, data_num))
            if para_name_list is not None:
                all_para_list = list(product(*[getattr(self, para_name) for para_name in para_name_list]))
                tmp_list = list(product(all_factors_list, all_para_list))

            else:
                tmp_list = list(product(all_factors_list, [()]))

            part_target_list = random.sample(tmp_list, fun_use_num)
            target_dict[fun_name] = part_target_list

        if self.factor_way_dict:
            factor_way_sr = pd.Series(dict(self.factor_way_dict)).loc[[str(x) for x in base_factor_list]]
            factor_long_list = [tmp_fun_2(x) for x in factor_way_sr[factor_way_sr == 1].index]
            factor_short_list = [tmp_fun_2(x) for x in factor_way_sr[factor_way_sr == -1].index]
            if len(factor_long_list) > 7 and len(factor_short_list) > 7:
                factors_add_list = list(combinations(factor_long_list[:7], 2)) + \
                                   list(combinations(factor_short_list[:7], 2))
                tmp_add_list = list(product(factors_add_list, [()]))
                if len(tmp_add_list) > 2 * fun_use_num:
                    target_dict['add'] = random.sample(tmp_add_list, 2 * fun_use_num)
                else:
                    target_dict['add'] = tmp_add_list

            elif len(factor_long_list) > 7:
                factors_add_list = list(combinations(factor_long_list[:7], 2))
                tmp_add_list = list(product(factors_add_list, [()]))
                if len(tmp_add_list) > 2 * fun_use_num:
                    target_dict['add'] = random.sample(tmp_add_list, 2 * fun_use_num)
                else:
                    target_dict['add'] = tmp_add_list

            elif len(factor_short_list) > 7:
                factors_add_list = list(combinations(factor_short_list[:7], 2))
                tmp_add_list = list(product(factors_add_list, [()]))
                if len(tmp_add_list) > 2 * fun_use_num:
                    target_dict['add'] = random.sample(tmp_add_list, 2 * fun_use_num)
                else:
                    target_dict['add'] = tmp_add_list
            else:
                pass

            if len(factor_short_list) > 7 and len(factor_long_list) > 7:
                factors_sub_list = list(product(factor_short_list[:7], factor_long_list[:7]))
                tmp_sub_list = list(product(factors_sub_list, [()]))
                if len(tmp_sub_list) > 2 * fun_use_num:
                    target_dict['sub'] = random.sample(tmp_sub_list, 2 * fun_use_num)
                else:
                    target_dict['sub'] = tmp_sub_list

        return target_dict

    def get_mix_factor(self, base_factor_list, fun_use_num):
        mix_factor_list = base_factor_list.copy()
        target_dict = self.part_mix_factor(base_factor_list, fun_use_num)
        for fun_name, target_list in target_dict.items():
            for factor_name_list, para_list in target_list:
                if str([fun_name, factor_name_list, para_list]) not in self.mix_factor_name_dict.keys():
                    mix_factor_list.append([fun_name, factor_name_list, para_list])
        return mix_factor_list

    def get_mix_low_factor(self, best_factor, best_score, low_corr_df, select_info_df, cut_date, percent):
        def tmp_fun(mix_factor, ls_para):
            mix_factor_df = self.load_mix_data(mix_factor)
            if self.if_only_long:
                info_df, pnl_df = self.back_test(mix_factor_df, cut_date, percent, ls_para=ls_para)
            else:
                info_df, pnl_df = self.back_test(mix_factor_df, cut_date, percent)
            return info_df

        print('get_mix_low_factor start:', best_factor)
        if len(low_corr_df) == 0:
            return None, None

        tmp_fun_2 = lambda x: eval(x) if '[' in x else x
        max_add_num = 10
        now_add_num = 0
        best_factor_way = select_info_df[str(best_factor)]['way_in']

        for low_corr_factor in low_corr_df.index:
            print('low_corr_factor info')
            print(select_info_df[str(low_corr_factor)])
            low_corr_factor_way = self.factor_way_dict[str(low_corr_factor)]
            if best_factor_way * low_corr_factor_way > 0:
                tmp_mix_factor = ['add', (best_factor, tmp_fun_2(low_corr_factor)), ()]
            elif best_factor_way * low_corr_factor_way < 0:
                tmp_mix_factor = ['sub', (best_factor, tmp_fun_2(low_corr_factor)), ()]
            else:
                continue

            if best_factor_way > 0:
                ls_para = 'l'
            else:
                ls_para = 's'

            tmp_info_df = tmp_fun(tmp_mix_factor, ls_para)
            tmp_score = self.get_score_fun(tmp_info_df['sp_in'], tmp_info_df['pot_in'], tmp_info_df['sp_out'])
            if tmp_score > best_score:
                best_score = tmp_score
                best_factor = tmp_mix_factor
                best_factor_way = tmp_info_df['way_in']
                print(f'add success {best_factor_way}')
                print(tmp_info_df)
                now_add_num += 1
            else:
                pass
            if now_add_num >= max_add_num:
                break
        if now_add_num == 0:
            return None, None
        else:
            print('get_mix_low_factor end:', best_factor)
            return best_factor, best_score

    def get_pnl_df(self, file_name, cut_date, percent):
        data_df = self.load_mix_data(file_name)
        if self.if_only_long:
            info_df_l, pnl_df_l = self.back_test(data_df, cut_date, percent, ls_para='l')
            pnl_df_l.name = str(file_name)
            info_df_l.name = str(file_name)

            info_df_s, pnl_df_s = self.back_test(data_df, cut_date, percent, ls_para='s')
            pnl_df_s.name = str(file_name)
            info_df_s.name = str(file_name)
            if info_df_l['sp_in'] > info_df_s['sp_in']:
                info_df = info_df_l
                pnl_df = pnl_df_l
            else:
                info_df = info_df_s
                pnl_df = pnl_df_s
        else:
            info_df, pnl_df = self.back_test(data_df, cut_date, percent)
            pnl_df.name = str(file_name)
            info_df.name = str(file_name)
        self.factor_way_dict[str(file_name)] = info_df['way_in']
        return info_df, pnl_df

    @staticmethod
    def get_scores_fun(sp_in, pot_in, sp_out):
        return (sp_in.where(sp_in > 0.4, other=0)) + 5 * (sp_in > 2) + 10 * (sp_in > 2.3) + \
               (pot_in.where(pot_in < 100, other=100)) / 200 + (sp_out.where(sp_out > 0.4, other=0)) / 4
        # + 1 * (pot_in > 50)

    @staticmethod
    def get_score_fun(sp_in, pot_in, sp_out):
        tmp_fun = lambda x: x if x > 0.4 else 0
        return tmp_fun(sp_in) + min(pot_in, 100) / 200 + 5 * (sp_in > 2) + 10 * (sp_in > 2.3) + tmp_fun(sp_out) / 4
        # + 10 * (pot_in > 50)

    def get_all_pnl_df(self, file_list, cut_date, percent, if_multy=True):
        result_list_raw = []
        if if_multy:
            pool = Pool(20)
            for file_name in file_list:
                result_list_raw.append(pool.apply_async(self.get_pnl_df, args=(file_name, cut_date, percent)))
            pool.close()
            pool.join()
            result_list = [res.get() for res in result_list_raw]
        else:
            for file_name in file_list:
                result_list_raw.append(self.get_pnl_df(file_name, cut_date, percent))
            result_list = result_list_raw
        all_info_df = pd.concat([res[0] for res in result_list], axis=1)
        all_pnl_df = pd.concat([res[1] for res in result_list], axis=1)
        return all_info_df, all_pnl_df

    def train_fun(self, fun_use_num, cut_date, percent, evol_num=6, if_multy=False):
        try:
            result_save_path = '/mnt/mfs/dat_whs/result_new2/test03'
            tmp_fun_1 = lambda x: True if '[' in x else False
            tmp_fun_2 = lambda x: eval(x) if '[' in x else x

            all_file = base_data_dict.keys()
            base_factor_list = random.sample([x for x in all_file if not x.startswith('intra')
                                              and 'suspend' not in x], 50)
            best_factor = ''
            best_score = 0
            best_info = pd.DataFrame()
            best_pnl = pd.DataFrame()
            low_corr_df = pd.DataFrame()
            select_info_df = pd.DataFrame()
            if if_multy:
                import multiprocessing
                mgr = multiprocessing.Manager()
                self.mix_factor_name_dict = mgr.dict()
                self.factor_way_dict = mgr.dict()
            # 循环次数
            loop_now = 0
            # 进化成功次数
            evol_now = 0
            # 最大循环次数
            loop_max = 6
            # 下一轮进化
            evol_keep_num = 70
            while loop_now < loop_max:
                loop_now += 1
                print('_______________________________')
                print(f'训练循环第{loop_now}次, 进化成功{evol_now}次')
                all_factor_list = self.get_mix_factor(base_factor_list, fun_use_num)
                add_factor, add_score = self.get_mix_low_factor(best_factor, best_score, low_corr_df,
                                                                select_info_df, cut_date, percent)
                if add_factor is not None and add_factor not in all_factor_list:
                    all_factor_list.append(add_factor)

                print(f'add_factor: {add_factor}, add_score: {add_score}')
                select_info_df, select_pnl_df = self.get_all_pnl_df(all_factor_list, cut_date, percent, if_multy)
                # 相关性

                select_sp_in = select_info_df.loc['sp_in'].abs()
                select_pot_in = select_info_df.loc['pot_in'].abs()
                select_sp_out = select_info_df.loc['sp_out'].abs()
                select_sp = select_info_df.loc['sp'].abs()
                select_sp_sort = select_sp.sort_values(ascending=False).replace(0, np.nan).dropna()
                high_sp_factor = select_sp_sort[select_sp_sort>0.5].index

                select_factor_scores = self.get_scores_fun(select_sp_in, select_pot_in, select_sp_out) \
                    .sort_values(ascending=False)

                if select_factor_scores.iloc[0] > best_score:
                    best_factor = tmp_fun_2(select_factor_scores.index[0])
                    low_corr_df = select_pnl_df.corr()[str(best_factor)][high_sp_factor].fillna(0).abs().sort_values()
                    low_corr_df = low_corr_df[low_corr_df != 0]
                    best_score = select_factor_scores.iloc[0]
                    best_pnl = select_pnl_df[str(best_factor)]
                    best_info = select_info_df[str(best_factor)]
                    evol_now += 1
                    print(f'evolution {evol_now} success!!!:')
                    print(best_score)
                    print(best_factor)
                    print(best_info)

                else:
                    if best_factor is not '':
                        low_corr_df = select_pnl_df.corr()[str(best_factor)][high_sp_factor].fillna(0).sort_values()
                        low_corr_df = low_corr_df[low_corr_df != 0]
                    else:
                        low_corr_df = pd.DataFrame()
                    print('evolution fail...')

                base_factor_list = [tmp_fun_2(x) for x in select_factor_scores.index[:evol_keep_num]]
                # base_delete_list = select_factor_scores.index[evol_keep_num:]

                # delete_factor_list = list(filter(tmp_fun_1, base_delete_list))
                # delete_file_list = []
                # for x in delete_factor_list:
                #     delete_file_list.append(self.mix_factor_name_dict[str(x)])
                #     self.mix_factor_name_dict.pop(str(x))
                #
                # bt.AZ_Delete_file(f'{self.tmp_mix_path}/{self.sector_name}', target_list=delete_file_list)

                if evol_now > evol_num:
                    break

            bt.AZ_Delete_file(f'{self.tmp_mix_path}/{self.sector_name}',
                              target_list=self.mix_factor_name_dict.values())
            print('delete all')
            print(best_info.loc['sp'], best_info.loc['pot'])
            way = self.factor_way_dict[str(best_factor)]
            print(way)
            result_save_file = f'{result_save_path}/{self.sector_name}.csv'
            if self.if_save:
                if best_info.loc['sp_in'] > 2 and best_info.loc['sp'] > 2 and best_info.loc['pot'] > 50:
                    bt.AZ_Path_create(result_save_path)
                    result_df, info_df = bt.commit_check(pd.DataFrame(best_pnl))
                    if result_df.prod().iloc[0] == 1:
                        str_1 = f'{self.sector_name}|{self.hold_time}|{self.if_only_long}|{percent}'

                        write_str = str_1 + '#' + str(best_factor) + '#' + str(way)

                        with lock:
                            f = open(result_save_file, 'a')
                            f.write(write_str + '\n')
                            f.close()

                    plot_send_result(best_pnl, best_info.loc['sp'], f'{self.sector_name}|{self.hold_time}'
                                                                    f'|{self.if_only_long}|{percent}|new framework',
                                     str(best_factor) + '\n' + pd.DataFrame(best_info).to_html())
            return best_factor, best_info

        except Exception as error:
            send_email.send_email(str(error), ['whs@yingpei.com'], [], '[backtest error]')

    def run(self, fun_use_num, cut_date, percent, run_num=300):
        evol_num = 6
        pool = Pool(28)
        for i in range(run_num):
            args = (fun_use_num, cut_date, percent, evol_num, False)
            pool.apply_async(self.train_fun, args=args)
            # self.train_fun(fun_use_num, cut_date, percent, evol_num, True)
        pool.close()
        pool.join()


def part_main_fun(if_only_long, hold_time, sector_name, percent):
    root_path = '/mnt/mfs/DAT_EQT'
    if_save = True
    if_new_program = True

    begin_date = pd.to_datetime('20130101')
    # end_date = pd.to_datetime('20190411')
    end_date = datetime.now()
    lag = 2
    return_file = ''

    if_hedge = True

    factor_test = FactorTest(root_path, if_save, if_new_program, begin_date, end_date, sector_name, hold_time,
                             lag, return_file, if_hedge, if_only_long)

    cut_date = '20180101'
    fun_use_num = 10
    factor_test.run(fun_use_num, cut_date, percent)


def main_fun():
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

    hold_time_list = [5, 10, 20, 30]
    for if_only_long in [True]:
        for sector_name in sector_name_list:
            for percent in [0.1, 0.2]:
                for hold_time in hold_time_list:
                    part_main_fun(if_only_long, hold_time, sector_name, percent)


if __name__ == '__main__':
    main_fun()
    pass
