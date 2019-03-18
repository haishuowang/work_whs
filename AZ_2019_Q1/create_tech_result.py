import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *
import work_whs.AZ_2018_Q2.factor_script.main_file.main_file_return_hedge as mfrh


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
    def add_weight(a, b, wt1, wt2):
        return wt1 * a + wt2 * b

    @staticmethod
    def add_tri(a, b, c):
        return (a + b + c) / 3

    @staticmethod
    def mul(a, b):
        """
        no
        :param a:
        :param b:
        :return:
        """
        return a.add(b)

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
    def compare(a, b):
        """
        signle 0, 1
        :param a:
        :param b:
        :return:
        """
        return (a > b).astype(int)

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
    def order_moment(a, window, n):
        """
        no mul*n
        n阶矩阵
        :param a:
        :param window:
        :param n:
        :return:
        """
        return bt.AZ_Rolling(a, window).apply(lambda x: sum(x ** n))

    @staticmethod
    def order_moment_am(a, window, n):
        """
        std mul*n*2
        n阶矩阵中心
        :param a:
        :param window:
        :param n:
        :return:
        """
        return bt.AZ_Rolling(a, window).apply(lambda x: sum((x - x.mean()) ** n))

    @staticmethod
    def row_extre(a, percent):
        """
        signle -1, 0, 1
        :param a:
        :param sector_df:
        :param percent:
        :return:
        """
        # a = a * sector_df
        target_df = a.rank(axis=1, pct=True)
        target_df[target_df >= 1 - percent] = 1
        target_df[target_df <= percent] = -1
        target_df[(target_df > percent) & (target_df < 1 - percent)] = 0
        return target_df

    @staticmethod
    def pnd_col_extre(a, window, percent, min_periods=1):
        """
        signle -1, 0, 1
        :param a:
        :param window:
        :param percent:
        :param min_periods:
        :return:
        """
        dn_df = a.rolling(window=window, min_periods=min_periods).quantile(percent)
        up_df = a.rolling(window=window, min_periods=min_periods).quantile(1 - percent)
        dn_target = -(a < dn_df).astype(int)
        up_target = (a > up_df).astype(int)
        target_df = dn_target + up_target
        return target_df

    def mean(self, a, b):
        """
        no
        :param a:
        :param b:
        :return:
        """
        return self.add(a, b) / 2

    @staticmethod
    def abs_fun(a):
        """
        no 0|
        :param a:
        :return:
        """
        return np.abs(a)

    @staticmethod
    def pnd_continue_ud(a):
        """
        single 0, +-1, +-2, +-3
        :param a:
        :return:
        """
        n_list = [3, 4, 5]

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
            target_df = fun(a, n)
            target_df = target_df
            all_target_df = all_target_df.add(target_df, fill_value=0)
        return all_target_df

    @staticmethod
    def pnd_hl(high, low, close, n):
        high_n = high.rolling(window=n, min_periods=0).max().shift(1)
        low_n = low.rolling(window=n, min_periods=0).min().shift(1)
        h_diff = (close - high_n)
        l_diff = (close - low_n)

        h_diff[h_diff > 0] = 1
        h_diff[h_diff <= 0] = 0

        l_diff[l_diff >= 0] = 0
        l_diff[l_diff < 0] = -1

        pos = h_diff + l_diff
        return pos

    # @staticmethod
    # def bbangs(a, n):
    #     std_df = bt.AZ_Rolling(a, n).std()
    #     ma_df = bt.AZ_Rolling(a, n).std()
    #     up_df = ma_df + std_df
    #     dn_df = ma_df - std_df
    #     return None


def out_sample_perf_c(pnl_df_out, way=1):
    if way == 1:
        sharpe_out = bt.AZ_Sharpe_y(pnl_df_out)
    else:
        sharpe_out = bt.AZ_Sharpe_y(-pnl_df_out)
    out_condition = sharpe_out > 0.8
    return out_condition, round(sharpe_out * way, 2)


def filter_all(cut_date, pos_df_daily, pct_n, if_return_pnl=False, if_only_long=False):
    pnl_df = (pos_df_daily * pct_n).sum(axis=1)
    pnl_df = pnl_df.replace(np.nan, 0)

    return_in = pct_n[pct_n.index < cut_date]

    pnl_df_in = pnl_df[pnl_df.index < cut_date]
    asset_df_in = pnl_df_in.cumsum()
    last_asset_in = asset_df_in.iloc[-1]
    pos_df_daily_in = pos_df_daily[pos_df_daily.index < cut_date]
    pot_in = bt.AZ_Pot(pos_df_daily_in, last_asset_in)

    leve_ratio = bt.AZ_Leverage_ratio(asset_df_in)
    if leve_ratio < 0:
        leve_ratio = 100
    sharpe_q_in_df = bt.AZ_Rolling_sharpe(pnl_df_in, roll_year=1, year_len=250, min_periods=1,
                                          cut_point_list=[0.3, 0.5, 0.7], output=False)
    sp_in = bt.AZ_Sharpe_y(pnl_df_in)
    fit_ratio = bt.AZ_fit_ratio(pos_df_daily_in, return_in)
    ic = round(bt.AZ_Normal_IC(pos_df_daily_in, pct_n, min_valids=None, lag=0).mean(), 6)
    sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d = sharpe_q_in_df.values
    in_condition_u = sharpe_q_in_df_u > 0.9 and leve_ratio > 1
    in_condition_d = sharpe_q_in_df_d < -0.9 and leve_ratio > 1
    # 分双边和只做多
    if if_only_long:
        in_condition = in_condition_u
    else:
        in_condition = in_condition_u | in_condition_d

    if sharpe_q_in_df_m > 0:
        way = 1
    else:
        way = -1

    # 样本外表现
    pnl_df_out = pnl_df[pnl_df.index >= cut_date]
    out_condition, sharpe_q_out = out_sample_perf_c(pnl_df_out, way=way)
    if if_return_pnl:
        return in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
               fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df
    else:
        return in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
               fit_ratio, leve_ratio, sp_in, sharpe_q_out


class FactorTestSector(mfrh.FactorTest):
    def __init__(self, *args):
        super(FactorTestSector, self).__init__(*args)

    @staticmethod
    def create_all_para_(change_list, ratio_list, tech_list):
        target_list = list(product(change_list, ratio_list, tech_list))
        return target_list

    def save_load_control_single(self, factor_list, suffix_name, file_name):
        # 参数存储与加载的路径控制
        result_save_path = '/mnt/mfs/dat_whs/result'
        if self.if_new_program:
            now_time = datetime.now().strftime('%Y%m%d_%H%M')
            if self.if_only_long:
                file_name = '{}_{}_{}_hold_{}_{}_{}_long.txt' \
                    .format(self.sector_name, self.if_hedge, now_time, self.hold_time, self.return_file, suffix_name)
            else:
                file_name = '{}_{}_{}_hold_{}_{}_{}.txt' \
                    .format(self.sector_name, self.if_hedge, now_time, self.hold_time, self.return_file, suffix_name)

            log_save_file = os.path.join(result_save_path, 'log', file_name)
            result_save_file = os.path.join(result_save_path, 'result', file_name)
            para_save_file = os.path.join(result_save_path, 'para', file_name)
            para_dict = dict()
            para_ready_df = pd.DataFrame(factor_list)
            total_para_num = len(para_ready_df)
            if self.if_save:
                self.create_log_save_path(log_save_file)
                self.create_log_save_path(result_save_file)
                self.create_log_save_path(para_save_file)
                para_dict['para_ready_df'] = para_ready_df
                para_dict['factor_list'] = factor_list
                pd.to_pickle(para_dict, para_save_file)

        else:
            log_save_file = os.path.join(result_save_path, 'log', file_name)
            result_save_file = os.path.join(result_save_path, 'result', file_name)
            para_save_file = os.path.join(result_save_path, 'para', file_name)
            para_tested_df = pd.read_table(log_save_file, sep='|', header=None, index_col=0)
            para_all_df = pd.read_pickle(para_save_file)
            total_para_num = len(para_all_df)
            para_ready_df = para_all_df.loc[sorted(list(set(para_all_df.index) - set(para_tested_df.index)))]
        # print(file_name)
        # print(f'para_num:{len(para_ready_df)}')
        return para_ready_df, log_save_file, result_save_file, total_para_num

    def single_test(self, factor_1, file_name, fun_path):

        daily_pos = self.deal_mix_factor(factor_1).shift(2)
        in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
        fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df = filter_all(self.cut_date, daily_pos, self.return_choose,
                                                                        if_return_pnl=True,
                                                                        if_only_long=self.if_only_long)
        result_list = [in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m,
                       sharpe_q_in_df_d, pot_in, fit_ratio, leve_ratio, sp_in, sharpe_q_out]
        print(fun_path, file_name, bt.AZ_Sharpe_y(pnl_df), pot_in)
        if abs(bt.AZ_Sharpe_y(pnl_df)) > 1 and pot_in > 30:
            plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df),
                             file_name,
                             text=fun_path + '|' + '|'.join([str(x) for x in result_list]))

        return pnl_df, in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
               fit_ratio, leve_ratio, sp_in, sharpe_q_out


class TechFactor(FunSet):
    def __init__(self, root_path, bkt_model):
        self.bkt_model = bkt_model
        self.xinx = bkt_model.xinx
        self.xnms = bkt_model.xnms
        self.close = bt.AZ_Load_csv(f'{root_path}/EM_Funda/DERIVED_14/aadj_p.csv') \
            .reindex(index=self.xinx, columns=self.xnms)
        self.open_price = bt.AZ_Load_csv(f'{root_path}/EM_Funda/DERIVED_14/aadj_p_OPEN.csv') \
            .reindex(index=self.xinx, columns=self.xnms)
        self.low = bt.AZ_Load_csv(f'{root_path}/EM_Funda/DERIVED_14/aadj_p_LOW.csv') \
            .reindex(index=self.xinx, columns=self.xnms)
        self.high = bt.AZ_Load_csv(f'{root_path}/EM_Funda/DERIVED_14/aadj_p_HIGH.csv') \
            .reindex(index=self.xinx, columns=self.xnms)
        self.volume = bt.AZ_Load_csv(f'{root_path}/EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv') \
            .reindex(index=self.xinx, columns=self.xnms)
        # self.avg_price = bt.AZ_Load_csv(f'{root_path}/EM_Funda/DERIVED_14/aadj_.csv')

        self.weight_list = [[0.1 * i, 1 - 0.1 * i] for i in range(1, 10)]
        self.window_list = [5, 17, 41]
        self.shift_list = [1, 5, 11]
        self.order_list = [0.5, 2, 3]
        self.pct_list = [0.2, 0.3]

        self.fun_info_set = dict({
            'div': ['+ratio', '*1', 0, 2, None, None],
            'diff': ['+diff', '*1', 0, 1, ['window_list'], None],
            'pct_change': ['+ratio', '*1', 0, 1, None, None],
            'add': ['no', '+1', 0, 2, None, "p1['type']==p2['type']"],
            'add_weight': ['no', '*1', 0, 2, ['weight_list'], "p1['type']=='price'|p2['type']=='price'"],
            'add_tri': ['no', '*1', 0, 3, None, "p1['type']=='price'|p2['type']=='price'|p3['type']=='price'"],

            'mul': ['no', 'no', 0, 2, None, "p1['type']==p2['type']|p1['type'] in ['single', 'continue']"],
            'sub': ['+diff', '*1', 0, 2, None, "p1['type']==p2['type']"],
            'std': ['+std', '*1', 0, 1, ['window_list'], None],
            'max_fun': ['no', '*1', 0, 1, ['window_list'], None],
            'min_fun': ['no', '*1', 0, 1, ['window_list'], None],
            'compare': ['+signal', '*1', 0, 2, None, "p1['type']==p2['type']"],

            'corr': ['continue', '*1', 0, 2, ['window_list'], "p1['type']!=p2['type']"],
            'corr_rank': ['continue', '*1', 0, 2, ['window_list'], "p1['type']!=p2['type']"],

            'ma': ['no', '*1', 0, 1, ['window_list'], None],
            'shift': ['no', '*1', 0, 1, ['window_list'], None],
            'order_moment': ['no', '**2n', 0, 1, ['window_list', 'order_list'], None],
            'order_moment_am': ['std', '**2n', 0, 1, ['window_list', 'order_list'], None],
            'row_extre': ['signal', '*1', 1, 1, ['pct_list'], None],
            'pnd_col_extre': ['signal', '*1', 0, 1, ['window_list', 'pct_list'], None],
            'mean': ['no', '*1', 0, 2, None, None],
            'abs_fun': ['no', '*1', 0, 1, None, None],
            'pnd_continue_ud': ['signal', '*1', 0, 1, None, None],
            'pnd_hl': ['no', '*1', 0, 3, 'window_list', "set({p1['name'], p2['name'], p3['name']})=="
                                                        "set({'high','low','close'})"]

        })
        self.fun_info_df = pd.DataFrame.from_dict(self.fun_info_set).T
        self.fun_info_df.columns = ['t_type', 'mul', 'axis', 'data_num', 'para', 'detail']

        self.data_info_set = {
            'close': {'data': self.close, 'type': 'price', 'mul': '1', 'axis': 0},
            'open_price': {'data': self.open_price, 'type': 'price', 'mul': 'p', 'axis': 0},
            'low': {'data': self.low, 'type': 'price', 'mul': '1', 'axis': 0},
            'high': {'data': self.high, 'type': 'price', 'mul': '1', 'axis': 0},
            'volume': {'data': self.volume, 'type': 'volume', 'mul': '1', 'axis': 0},
        }

        self.move_tree = dict({
            'div': ['div', 'diff', 'pct_change', 'add', 'mul', 'sub', 'std', 'max_fun', 'min_fun', 'compare',
                    'corr', 'corr_rank', 'ma', 'shift', 'order_moment', 'order_moment_am', 'row_extre',
                    'pnd_col_extre', 'mean', 'abs_fun', 'pnd_continue_ud', 'pnd_hl'],

            'diff': ['div', 'diff', 'pct_change', 'add', 'mul', 'sub', 'std', 'max_fun', 'min_fun', 'compare',
                     'corr', 'corr_rank', 'ma', 'shift', 'order_moment', 'order_moment_am', 'row_extre',
                     'pnd_col_extre', 'mean', 'abs_fun', 'pnd_continue_ud', 'pnd_hl'],

            'pct_change': ['div', 'diff', 'pct_change', 'add', 'mul', 'sub', 'std', 'max_fun', 'min_fun', 'compare',
                           'corr', 'corr_rank', 'ma', 'shift', 'order_moment', 'order_moment_am', 'row_extre',
                           'pnd_col_extre', 'mean', 'abs_fun', 'pnd_continue_ud', 'pnd_hl'],

            'add': ['div', 'diff', 'pct_change', 'add', 'mul', 'sub', 'std', 'max_fun', 'min_fun', 'compare',
                    'corr', 'corr_rank', 'ma', 'shift', 'order_moment', 'order_moment_am', 'row_extre',
                    'pnd_col_extre', 'mean', 'abs_fun', 'pnd_continue_ud', 'pnd_hl'],

            'add_weight': ['div', 'diff', 'pct_change', 'add', 'add_weight', 'add_tri', 'mul', 'sub', 'std',
                           'max_fun', 'min_fun', 'compare', 'corr', 'corr_rank', 'ma', 'shift', 'order_moment',
                           'order_moment_am', 'row_extre', 'pnd_col_extre', 'mean', 'abs_fun',
                           'pnd_continue_ud', 'pnd_hl'],

            'add_tri': ['div', 'diff', 'pct_change', 'add', 'add_weight', 'add_tri', 'mul', 'sub', 'std',
                        'max_fun', 'min_fun', 'compare', 'corr', 'corr_rank', 'ma', 'shift', 'order_moment',
                        'order_moment_am', 'row_extre', 'pnd_col_extre', 'mean', 'abs_fun',
                        'pnd_continue_ud', 'pnd_hl'],

            'mul': ['div', 'diff', 'pct_change', 'add', 'add_weight', 'add_tri', 'mul', 'sub', 'std',
                    'max_fun', 'min_fun', 'compare', 'corr', 'corr_rank', 'ma', 'shift', 'order_moment',
                    'order_moment_am', 'row_extre', 'pnd_col_extre', 'mean', 'abs_fun',
                    'pnd_continue_ud', 'pnd_hl'],

            'sub': ['div', 'diff', 'pct_change', 'add', 'add_weight', 'add_tri', 'mul', 'sub', 'std',
                    'max_fun', 'min_fun', 'compare', 'corr', 'corr_rank', 'ma', 'shift', 'order_moment',
                    'order_moment_am', 'row_extre', 'pnd_col_extre', 'mean', 'abs_fun',
                    'pnd_continue_ud', 'pnd_hl'],

            'max_fun': ['div', 'diff', 'pct_change', 'add', 'add_weight', 'add_tri', 'mul', 'sub', 'std',
                        'min_fun', 'compare', 'corr', 'corr_rank', 'ma', 'shift', 'order_moment',
                        'order_moment_am', 'row_extre', 'pnd_col_extre', 'mean', 'abs_fun',
                        'pnd_continue_ud', 'pnd_hl'],

            'min_fun': ['div', 'diff', 'pct_change', 'add', 'add_weight', 'add_tri', 'mul', 'sub', 'std',
                        'max_fun', 'compare', 'corr', 'corr_rank', 'ma', 'shift', 'order_moment',
                        'order_moment_am', 'row_extre', 'pnd_col_extre', 'mean', 'abs_fun',
                        'pnd_continue_ud', 'pnd_hl'],

            'ma': ['div', 'diff', 'pct_change', 'add', 'add_weight', 'add_tri', 'mul', 'sub', 'std',
                   'max_fun', 'min_fun', 'compare', 'corr', 'corr_rank', 'ma', 'shift', 'order_moment',
                   'order_moment_am', 'row_extre', 'pnd_col_extre', 'mean', 'abs_fun',
                   'pnd_continue_ud', 'pnd_hl'],

            'mean': ['div', 'diff', 'pct_change', 'add', 'mul', 'sub', 'std',
                     'max_fun', 'min_fun', 'compare', 'corr', 'corr_rank', 'ma', 'shift', 'order_moment',
                     'order_moment_am', 'row_extre', 'pnd_col_extre', 'mean', 'pnd_continue_ud', 'pnd_hl'],

            'abs_fun': ['div', 'diff', 'pct_change', 'add', 'mul', 'sub', 'std', 'max_fun', 'min_fun',
                        'compare', 'corr', 'corr_rank', 'ma', 'shift', 'order_moment', 'order_moment_am', 'row_extre',
                        'pnd_col_extre', 'mean', 'pnd_continue_ud', 'pnd_hl'],

            'std': ['div', 'diff', 'pct_change', 'add', 'mul', 'sub', 'std', 'max_fun', 'min_fun', 'compare',
                    'corr', 'corr_rank', 'ma', 'shift', 'row_extre', 'pnd_col_extre', 'mean', 'pnd_continue_ud',
                    'pnd_hl'],

            'shift': ['div', 'diff', 'pct_change', 'add', 'mul', 'sub', 'std', 'max_fun', 'min_fun', 'compare',
                      'corr', 'corr_rank'],

            'corr': ['div', 'diff', 'pct_change', 'add', 'mul', 'sub', 'std', 'max_fun',
                     'min_fun', 'compare', 'ma', 'order_moment', 'order_moment_am', 'row_extre', 'pnd_col_extre',
                     'mean', 'abs_fun', 'pnd_continue_ud', 'pnd_hl'],

            'corr_rank': ['div', 'diff', 'pct_change', 'add', 'mul', 'sub', 'std', 'max_fun',
                          'min_fun', 'compare', 'ma', 'order_moment', 'order_moment_am', 'row_extre', 'pnd_col_extre',
                          'mean', 'abs_fun', 'pnd_continue_ud', 'pnd_hl'],

            'order_moment': ['div', 'diff', 'pct_change', 'add', 'mul', 'sub', 'std',
                             'max_fun', 'min_fun', 'compare', 'corr', 'corr_rank', 'ma', 'shift', 'row_extre',
                             'pnd_col_extre', 'mean', 'abs_fun', 'pnd_continue_ud', 'pnd_hl'],

            'order_moment_am': ['div', 'diff', 'pct_change', 'add', 'mul', 'sub', 'std', 'max_fun', 'min_fun',
                                'compare', 'corr', 'corr_rank', 'ma', 'shift', 'row_extre', 'pnd_col_extre', 'mean',
                                'abs_fun', 'pnd_continue_ud', 'pnd_hl'],
            # signal
            'compare': ['add', 'mul', 'sub', ],
            'pnd_continue_ud': ['add', 'mul', 'sub', 'ma', 'mean', 'abs_fun'],
            'row_extre': ['add', 'mul', 'sub', 'abs_fun'],
            'pnd_hl': ['add', 'mul', 'sub', 'abs_fun'],
            'pnd_col_extre': [],
        })
        self.fun_path_list = self.move_path_fun()
        self.single_fun = ['pnd_continue_ud', 'pnd_col_extre', 'row_extre']

    @staticmethod
    def if_use_fun_too_much(x, mark_list, n=2):
        """
        限制函数使用次数
        :param x:
        :param mark_list:
        :param n:
        :return:
        """
        count_dict = Counter(mark_list)
        if x in count_dict.keys():
            if count_dict[x] >= n:
                return True
            else:
                return False
        else:
            return False

    def tmp_move_fun(self, fun_set, mark_list, fun_info_df_1, n, target_list):
        """
        循环获取函数执行路径
        :param fun_set:
        :param mark_list:
        :param fun_info_df_1:
        :param n:
        :param target_list:
        :return:
        """
        for x in fun_set:
            t_type = self.fun_info_set[x][0]

            if self.if_use_fun_too_much(x, mark_list, n=3):
                continue
            else:
                pass

            if t_type != 'signal' and n <= 4:
                target_list.append(mark_list + [x])
                print(mark_list + [x])
                self.tmp_move_fun(list(set(fun_info_df_1.index) & set(self.move_tree[x])),
                                  mark_list + [x], fun_info_df_1, n + 1, target_list)
        if n == 0:
            return target_list

    def move_path_fun(self):
        """
        函数执行路径 集合
        :return:
        """
        fun_info_df_1 = self.fun_info_df[self.fun_info_df['data_num'] == 1]
        mark_list = []
        fun_move_list = self.tmp_move_fun(fun_info_df_1.index, mark_list, fun_info_df_1, 0, [])
        return fun_move_list

    def fun_exe(self, fun_name, data_list, para):
        target_df = getattr(self, fun_name)(*data_list, *para)
        return target_df

    def fun_para_exe(self, fun_name, data_list):
        para_name_list = self.fun_info_df.loc[fun_name]['para']
        if para_name_list is None:
            target_df = self.fun_exe(fun_name, data_list, [])
            yield target_df, []
        else:
            para_list = [getattr(self, x) for x in para_name_list]
            for para in product(*para_list):
                target_df = self.fun_exe(fun_name, data_list, para)
                yield target_df, para

    def fun_path_exe(self, fun_path, data_name):
        data_df = getattr(self, data_name)
        target_df = data_df.copy()
        para_name_list = [self.fun_info_df.loc[fun_name]['para'] for fun_name in fun_path]
        all_para_set = list(map(lambda x: list(product(*[getattr(self, a) for a in x])) if x is not None else [[]],
                                para_name_list))
        for all_para_list in list(product(*all_para_set)):
            for i in range(len(fun_path)):
                fun_name = fun_path[i]
                para = all_para_list[i]
                detail = self.fun_info_df.loc[fun_name]['detail']
                if detail is not None:
                    exec(f'p1=self.data_info_set[data_name]')
                    exec(f'fun_judge = {detail}')
                    if not fun_judge:
                        yield None
                target_df = self.fun_exe(fun_name, [target_df], para)
            yield target_df, all_para_list

    def tmp_fpe_1(self, fun_path, data_price_list, data_other_list):
        print(fun_path)
        for tmp_df, all_para_list in self.fun_path_exe(fun_path, data_price_list[0]):
            if tmp_df is not None:
                for signal_fun in self.single_fun:
                    for single_df, single_para in self.fun_para_exe(signal_fun, [tmp_df]):
                        yield single_df, all_para_list, signal_fun, single_para, None, []
                if len(data_other_list) != 0:
                    print('Volume')
                    for data_other_name in data_other_list:
                        data_other_df = getattr(self, data_other_name)
                        for combine_fun in ['corr', 'corr_rank', 'div']:
                            for target_df, combine_para in self.fun_para_exe(combine_fun, [tmp_df, data_other_df]):
                                for signal_fun in self.single_fun:
                                    for single_df, single_para in self.fun_para_exe(signal_fun, [target_df]):
                                        yield single_df, all_para_list, signal_fun, single_para, \
                                              combine_fun, combine_para
                else:
                    pass
            else:
                pass

    def tmp_fpe_2(self, fun_path, data_price_list, data_other_list):
        print(fun_path)
        for tmp_df_1, all_para_list_1 in self.fun_path_exe(fun_path, data_price_list[0]):
            for tmp_df_2, all_para_list_2 in self.fun_path_exe(fun_path, data_price_list[1]):
                for concat_fun in ['div', 'sub', 'compare', 'mean']:
                    for tmp_df, concat_para in self.fun_para_exe(concat_fun, [tmp_df_1, tmp_df_2]):
                        if tmp_df is not None:
                            for signal_fun in self.single_fun:
                                for single_df, single_para in self.fun_para_exe(signal_fun, [tmp_df]):
                                    yield single_df, all_para_list_1, all_para_list_2, concat_fun, concat_para, \
                                          signal_fun, single_para, None, []
                            if len(data_other_list) != 0:
                                for data_other_name in data_other_list:
                                    data_other_df = getattr(self, data_other_name)
                                    for combine_fun in ['corr', 'corr_rank', 'div']:
                                        for target_df, combine_para in self.fun_para_exe(combine_fun,
                                                                                         [tmp_df, data_other_df]):
                                            for signal_fun in self.single_fun:
                                                for single_df, single_para in self.fun_para_exe(signal_fun,
                                                                                                [target_df]):
                                                    yield single_df, all_para_list_1, all_para_list_2, concat_fun, \
                                                          concat_para, signal_fun, single_para, \
                                                          combine_fun, combine_para
                            else:
                                pass
                        else:
                            pass

    @staticmethod
    def get_file_name(i, info_data, data_price_num):
        if data_price_num == 1:
            all_para_list, signal_fun, single_para, combine_fun, combine_para = info_data
            all_para_str = '_'.join([str(x) for x in all_para_list])
            signal_fun_str = str(signal_fun)
            single_para_str = '_'.join([str(x) for x in single_para])
            combine_fun_str = str(combine_fun)
            combine_para_str = '_'.join([str(x) for x in combine_para])
            tmp_list = [all_para_str, combine_fun_str, combine_para_str, signal_fun_str, single_para_str]
            file_name = ''.join([f'fun_path{i}', '|'.join(tmp_list), str(data_price_num)])

        elif data_price_num == 2:
            all_para_list_1, all_para_list_2, concat_fun, concat_para, signal_fun, \
            single_para, combine_fun, combine_para = info_data

            all_para_str_1 = '_'.join([str(x) for x in all_para_list_1])
            all_para_str_2 = '_'.join([str(x) for x in all_para_list_2])
            concat_fun_str = str(concat_fun)
            concat_para_str = '_'.join([str(x) for x in concat_para])
            signal_fun_str = str(signal_fun)
            single_para_str = '_'.join([str(x) for x in single_para])
            combine_fun_str = str(combine_fun)
            combine_para_str = '_'.join([str(x) for x in combine_para])
            tmp_list = [all_para_str_1, all_para_str_2, concat_fun_str, concat_para_str,
                        combine_fun_str, combine_para_str, signal_fun_str, single_para_str]
            file_name = '|'.join([f'fun_path{i}', '|'.join(tmp_list), str(data_price_num)])

        else:
            file_name = 'error'
        return file_name

    def part_train_model(self, i, data_price_list, data_other_list):
        data_price_num = len(data_price_list)
        fun_path = self.fun_path_list[i]
        if data_price_num == 1:
            for single_df, *info_data in self.tmp_fpe_1(fun_path, data_price_list, data_other_list):
                file_name = self.get_file_name(i, info_data, data_price_num)
                self.bkt_model.single_test(single_df, file_name, fun_path)

        elif data_price_num == 2:
            for single_df, *info_data in self.tmp_fpe_2(fun_path, data_price_list, data_other_list):
                file_name = self.get_file_name(i, info_data, data_price_num)
                self.bkt_model.single_test(single_df, file_name, fun_path)

        elif data_price_num == 3:
            pass
        elif data_price_num == 4:
            pass
        else:
            print('error')

    def train_model(self, data_price_list, data_other_list):
        # data_price_num = len(data_price_list)
        pool = Pool(10)
        for i in range(len(self.fun_path_list)):
            # self.part_train_model(i, data_price_list, data_other_list)
            pool.apply_async(self.part_train_model, args=(i, data_price_list, data_other_list))
        pool.close()
        pool.join()

    def main_fun(self):
        data_list = list(self.data_info_set.keys())
        data_list.remove('volume')
        all_para_list = list(combinations(data_list, 1))
        # all_para_list = list(combinations(data_list, 2))
        # all_para_list = list(combinations(data_list, 1)) + list(combinations(data_list, 2)) + \
        #                 list(combinations(data_list, 3)) + list(combinations(data_list, 4))

        for args in all_para_list:
            self.train_model(list(args), ['volume'])


def get_bkt_model(root_path):
    sector_name = 'index_000905'
    hold_time = 5
    if_only_long = False
    time_para_dict = []
    # root_path = '/mnt/mfs/DAT_EQT'
    if_save = False
    if_new_program = True

    begin_date = pd.to_datetime('20130101')
    cut_date = pd.to_datetime('20160401')
    # end_date = pd.to_datetime('2019030５')
    end_date = datetime.today()
    lag = 2
    return_file = ''

    if_hedge = True
    # if_only_long = False

    if sector_name.startswith('market_top_300plus'):
        if_weight = 1
        ic_weight = 0

    elif sector_name.startswith('market_top_300to800plus'):
        if_weight = 0
        ic_weight = 1

    else:
        if_weight = 0.5
        ic_weight = 0.5

    bkt_model = FactorTestSector(root_path, if_save, if_new_program, begin_date, cut_date, end_date, time_para_dict,
                                 sector_name, hold_time, lag, return_file, if_hedge, if_only_long,
                                 if_weight, ic_weight)

    return bkt_model


if __name__ == '__main__':
    root_path = '/mnt/mfs/DAT_EQT'
    bkt_model = get_bkt_model(root_path)
    tech_factor = TechFactor(root_path, bkt_model)
    # tech_factor.train_modle(data_name_list)
    tech_factor.main_fun()
