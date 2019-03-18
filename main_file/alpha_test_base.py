import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *


def mul_fun(a, b):
    a_l = a[a > 0]
    a_s = a[a < 0]

    b_l = b[b > 0]
    b_s = b[b < 0]

    pos_l = a_l.mul(b_l)
    pos_s = a_s.mul(b_s)

    pos = pos_l.sub(pos_s)
    return pos


def sub_fun(a, b):
    return a.sub(b)


def add_fun(a, b):
    return a.add(b)


def filter_all(cut_date, pos_df_daily, pct_n, if_return_pnl=False, if_only_long=False):
    pnl_df = (pos_df_daily * pct_n).sum(axis=1)
    pnl_df = pnl_df.replace(np.nan, 0)
    # pnl_df = pd.Series(pnl_df)
    # 样本内表现
    return_in = pct_n[pct_n.index < cut_date]

    pnl_df_in = pnl_df[pnl_df.index < cut_date]
    asset_df_in = pnl_df_in.cumsum()
    last_asset_in = asset_df_in.iloc[-1]
    pos_df_daily_in = pos_df_daily[pos_df_daily.index < cut_date]
    pot_in = AZ_Pot(pos_df_daily_in, last_asset_in)

    leve_ratio = AZ_Leverage_ratio(asset_df_in)
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


def out_sample_perf_c(pnl_df_out, way=1):
    # 根据sharpe大小,统计样本外的表现
    # if cut_point_list is None:
    #     cut_point_list = [0.30]
    # if way == 1:
    #     rolling_sharpe, cut_sharpe = \
    #         bt.AZ_Rolling_sharpe(pnl_df_out, roll_year=0.5, year_len=250, cut_point_list=cut_point_list, output=True)
    # else:
    #     rolling_sharpe, cut_sharpe = \
    #         bt.AZ_Rolling_sharpe(-pnl_df_out, roll_year=0.5, year_len=250, cut_point_list=cut_point_list, output=True)
    if way == 1:
        sharpe_out = bt.AZ_Sharpe_y(pnl_df_out)
    else:
        sharpe_out = bt.AZ_Sharpe_y(-pnl_df_out)
    out_condition = sharpe_out > 0.8
    return out_condition, round(sharpe_out * way, 2)


def AZ_Leverage_ratio(asset_df):
    """
    返回250天的return/(负的 一个月的return)
    :param asset_df:
    :return:
    """
    asset_20 = asset_df - asset_df.shift(20)
    asset_250 = asset_df - asset_df.shift(250)
    if asset_250.mean() > 0:
        return round(asset_250.mean() / (-asset_20.min()), 2)
    else:
        return round(asset_250.mean() / (-asset_20.max()), 2)


def pos_daily_fun(df, n=5):
    return df.rolling(window=n, min_periods=1).sum()


def AZ_Pot(pos_df_daily, last_asset):
    trade_times = pos_df_daily.diff().abs().sum().sum()
    if trade_times == 0:
        return 0
    else:
        pot = last_asset / trade_times * 10000
        return round(pot, 2)


class FactorTest:
    def __init__(self, root_path, if_save, if_new_program, begin_date, cut_date, end_date, time_para_dict, sector_name,
                 hold_time, lag, return_file, if_hedge, if_only_long, if_weight=0.5, ic_weight=0.5,
                 para_adj_set_list=None):
        self.root_path = root_path
        self.if_save = if_save
        self.if_new_program = if_new_program
        self.begin_date = begin_date
        self.cut_date = cut_date
        self.end_date = end_date
        self.time_para_dict = time_para_dict
        self.sector_name = sector_name
        self.hold_time = hold_time
        self.lag = lag
        self.return_file = return_file
        self.if_hedge = if_hedge
        self.if_only_long = if_only_long
        self.if_weight = if_weight
        self.ic_weight = ic_weight
        if para_adj_set_list is None:
            self.para_adj_set_list = [
                {'pot_in_num': 50, 'leve_ratio_num': 2, 'sp_in': 1.5, 'ic_num': 0.0, 'fit_ratio': 2},
                {'pot_in_num': 40, 'leve_ratio_num': 2, 'sp_in': 1.5, 'ic_num': 0.0, 'fit_ratio': 2},
                {'pot_in_num': 50, 'leve_ratio_num': 2, 'sp_in': 1, 'ic_num': 0.0, 'fit_ratio': 1},
                {'pot_in_num': 50, 'leve_ratio_num': 1, 'sp_in': 1, 'ic_num': 0.0, 'fit_ratio': 2},
                {'pot_in_num': 50, 'leve_ratio_num': 1, 'sp_in': 1, 'ic_num': 0.0, 'fit_ratio': 1},
                {'pot_in_num': 40, 'leve_ratio_num': 1, 'sp_in': 1, 'ic_num': 0.0, 'fit_ratio': 1}]

        return_choose = self.load_return_data()
        self.xinx = return_choose.index
        sector_df = self.load_sector_data()
        self.xnms = sector_df.columns

        return_choose = return_choose.reindex(columns=self.xnms)
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

        self.return_choose = return_choose.sub(hedge_df, axis=0)
        print('Loaded return DataFrame!')

        suspendday_df, limit_buy_sell_df = self.load_locked_data()
        limit_buy_sell_df_c = limit_buy_sell_df.shift(-1)
        limit_buy_sell_df_c.iloc[-1] = 1

        suspendday_df_c = suspendday_df.shift(-1)
        suspendday_df_c.iloc[-1] = 1
        self.suspendday_df_c = suspendday_df_c
        self.limit_buy_sell_df_c = limit_buy_sell_df_c
        print('Loaded suspendday_df and limit_buy_sell DataFrame!')

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

    def check_factor(self, name_list, file_name):
        load_path = os.path.join('/mnt/mfs/dat_whs/data/new_factor_data/' + self.sector_name)
        exist_factor = set([x[:-4] for x in os.listdir(load_path)])
        print()
        use_factor = set(name_list)
        a = use_factor - exist_factor
        if len(a) != 0:
            print('factor not enough!')
            print(a)
            print(len(a))
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
        new_stock_data = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Tab01/CDSY_SECUCODE/LISTSTATE.csv'))
        new_stock_data.fillna(method='ffill', inplace=True)
        # 获取交易日信息
        return_df = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Funda/DERIVED_14/aadj_r.csv')).astype(float)
        trade_time = return_df.index
        new_stock_data = new_stock_data.reindex(index=trade_time).fillna(method='ffill')
        target_df = new_stock_data.shift(40).notnull().astype(int)
        target_df = target_df.reindex(columns=xnms, index=xinx)
        return target_df

    # 获取剔除st股票的矩阵
    def get_st_stock_info(self, xnms, xinx):
        data = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Tab01/CDSY_CHANGEINFO/CHANGEA.csv'))
        data = data.reindex(columns=xnms, index=xinx)
        data.fillna(method='ffill', inplace=True)

        data = data.astype(str)
        target_df = data.applymap(lambda x: 0 if 'ST' in x or 'PT' in x else 1)
        return target_df

    def load_return_data(self):
        return_choose = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Funda/DERIVED_14/aadj_r.csv'))
        return_choose = return_choose[(return_choose.index >= self.begin_date) & (return_choose.index < self.end_date)]
        return return_choose

    # 获取sector data
    def load_sector_data(self):
        market_top_n = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Funda/DERIVED_10/' + self.sector_name + '.csv'))
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
        raw_suspendday_df = bt.AZ_Load_csv(
            os.path.join(self.root_path, 'EM_Funda/TRAD_TD_SUSPENDDAY/SUSPENDREASON.csv'))
        suspendday_df = raw_suspendday_df.isnull().astype(int)
        suspendday_df = suspendday_df.reindex(columns=self.xnms, index=self.xinx, fill_value=True)
        suspendday_df.replace(0, np.nan, inplace=True)

        return_df = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Funda/DERIVED_14/aadj_r.csv')).astype(float)
        limit_buy_sell_df = (return_df.abs() < 0.095).astype(int)
        limit_buy_sell_df = limit_buy_sell_df.reindex(columns=self.xnms, index=self.xinx, fill_value=1)
        limit_buy_sell_df.replace(0, np.nan, inplace=True)
        return suspendday_df, limit_buy_sell_df

    # 获取index data
    def load_index_data(self, index_name):
        data = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Funda/INDEX_TD_DAILYSYS/CHG.csv'))
        target_df = data[index_name].reindex(index=self.xinx)
        return target_df * 0.01

    # 读取factor
    def load_factor(self, file_name):
        factor_set = OrderedDict()
        load_path = os.path.join('/mnt/mfs/dat_whs/data/new_factor_data/' + self.sector_name)
        target_df = pd.read_pickle(os.path.join(load_path, file_name + '.pkl'))
        factor_set[file_name] = target_df.reindex(columns=self.xnms, index=self.xinx).fillna(0)
        return factor_set

    def deal_mix_factor(self, mix_factor):
        if self.if_only_long:
            mix_factor = mix_factor[mix_factor > 0]
        # 下单日期pos
        order_df = mix_factor.replace(np.nan, 0)
        # 排除入场场涨跌停的影响
        order_df = order_df * self.sector_df * self.limit_buy_sell_df_c * self.suspendday_df_c
        order_df = order_df.div(order_df.abs().sum(axis=1).replace(0, np.nan), axis=0)
        order_df[order_df > 0.05] = 0.05
        order_df[order_df < -0.05] = -0.05
        daily_pos = pos_daily_fun(order_df, n=self.hold_time)
        daily_pos.fillna(0, inplace=True)
        # 排除出场涨跌停的影响
        daily_pos = daily_pos * self.limit_buy_sell_df_c * self.suspendday_df_c
        daily_pos.fillna(method='ffill', inplace=True)
        return daily_pos

    @staticmethod
    def create_all_para_(change_list, ratio_list, tech_list):
        target_list = list(product(change_list, ratio_list, tech_list))
        return target_list

    def save_load_control_(self, index_list, suffix_name):
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

            para_dict = dict()
            total_para_num = len(index_list)
            if self.if_save:
                self.create_log_save_path(log_save_file)
                self.create_log_save_path(result_save_file)
                para_dict['index_list'] = index_list

            print(file_name)
            return log_save_file, result_save_file, total_para_num
        else:
            print('save load control error!')
            exit(-1)

    def load_tech_factor(self, file_name):
        load_path = os.path.join('/mnt/mfs/dat_whs/data/new_factor_data/' + self.sector_name)
        target_df = pd.read_pickle(os.path.join(load_path, file_name + '.pkl')) \
            .reindex(index=self.xinx, columns=self.xnms)
        return target_df

    def load_notice_factor(self, file_name):
        load_path = '/mnt/mfs/dat_whs/EM_Funda/my_data_test'
        tmp_df = bt.AZ_Load_csv(os.path.join(load_path, file_name + '.csv')) \
            .reindex(index=self.xinx, columns=self.xnms)
        target_df = self.row_extre(tmp_df, self.sector_df, 0.3)
        return target_df

    def part_test_index_3_(self, name_1, log_save_file, result_save_file):
        lock = Lock()
        start_time = time.time()
        load_time_1 = time.time()
        # load因子,同时根据stock_universe筛选数据.
        index_factor = self.load_notice_factor(name_1)
        # '/mnt/mfs/dat_whs/EM_Funda/my_data_test'
        # mix_factor = mul_fun(index_factor, tech_factor)
        mix_factor = index_factor
        print(index_factor.sum().sum())
        # print(mix_factor.sum().sum())

        daily_pos = self.deal_mix_factor(mix_factor).shift(2)

        # 返回样本内筛选结果
        *result_dict, pnl_df = filter_all(self.cut_date, daily_pos, self.return_choose,
                                          if_return_pnl=True, if_only_long=self.if_only_long)
        if bt.AZ_Sharpe_y(pnl_df) > 0.8:
            plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df),
                             f'{name_1}_{self.sector_name}_{self.hold_time}',
                             str(bt.AZ_Sharpe_y(pnl_df)))
        in_condition, *filter_result = result_dict
        # result 存储
        if in_condition:
            if self.if_save:
                with lock:
                    f = open(result_save_file, 'a')
                    write_list = [name_1, self.sector_name, in_condition] + filter_result
                    f.write('|'.join([str(x) for x in write_list]) + '\n')
            print([in_condition, name_1] + filter_result)
        end_time = time.time()
        # 参数存储
        if self.if_save:
            with lock:
                f = open(log_save_file, 'a')
                write_list = [name_1, self.sector_name, round(end_time - start_time, 4)]
                f.write('|'.join([str(x) for x in write_list]) + '\n')

        print('{}, cost {} seconds'.format(name_1, round(end_time - start_time, 2)))

    def test_index_3_(self, index_list, pool_num=20, suffix_name=''):
        log_save_file, result_save_file, total_para_num = self.save_load_control_(index_list, suffix_name)
        self.check_factor(index_list, result_save_file)
        a_time = time.time()
        # pool = Pool(pool_num)
        for name_1 in index_list:
            print(name_1)
            args_list = (name_1, log_save_file, result_save_file)
            # pool.apply_async(self.part_test_index_3_, args=args_list)
            self.part_test_index_3_(*args_list)

        # pool.close()
        # pool.join()

        b_time = time.time()
        print('Success!Processing end, Cost {} seconds'.format(round(b_time - a_time, 2)))


def main_fun(sector_name, hold_time):
    root_path = '/mnt/mfs/DAT_EQT'
    if_save = False
    if_new_program = True

    begin_date = pd.to_datetime('20100101')
    cut_date = pd.to_datetime('20160401')
    end_date = pd.to_datetime('20180901')
    lag = 2
    return_file = ''

    if_hedge = True
    if_only_long = True
    time_para_dict = OrderedDict()

    time_para_dict['time_para_1'] = [pd.to_datetime('20100101'), pd.to_datetime('20150101'),
                                     pd.to_datetime('20150401'), pd.to_datetime('20150701'),
                                     pd.to_datetime('20151001'), pd.to_datetime('20160101')]

    time_para_dict['time_para_2'] = [pd.to_datetime('20110101'), pd.to_datetime('20160101'),
                                     pd.to_datetime('20160401'), pd.to_datetime('20160701'),
                                     pd.to_datetime('20161001'), pd.to_datetime('20170101')]

    time_para_dict['time_para_3'] = [pd.to_datetime('20120601'), pd.to_datetime('20170601'),
                                     pd.to_datetime('20170901'), pd.to_datetime('20171201'),
                                     pd.to_datetime('20180301'), pd.to_datetime('20180601')]

    time_para_dict['time_para_4'] = [pd.to_datetime('20130601'), pd.to_datetime('20180601'),
                                     pd.to_datetime('20180901'), pd.to_datetime('20180901'),
                                     pd.to_datetime('20180901'), pd.to_datetime('20180901')]

    time_para_dict['time_para_5'] = [pd.to_datetime('20130701'), pd.to_datetime('20180701'),
                                     pd.to_datetime('20180901'), pd.to_datetime('20180901'),
                                     pd.to_datetime('20180901'), pd.to_datetime('20180901')]

    time_para_dict['time_para_6'] = [pd.to_datetime('20130801'), pd.to_datetime('20180801'),
                                     pd.to_datetime('20180901'), pd.to_datetime('20180901'),
                                     pd.to_datetime('20180901'), pd.to_datetime('20180901')]

    if sector_name.startswith('market_top_300plus'):
        if_weight = 1
        ic_weight = 0

    elif sector_name.startswith('market_top_300to800plus'):
        if_weight = 0
        ic_weight = 1

    else:
        if_weight = 0.5
        ic_weight = 0.5

    main_model = FactorTest(root_path, if_save, if_new_program, begin_date, cut_date, end_date, time_para_dict,
                            sector_name, hold_time, lag, return_file, if_hedge, if_only_long,
                            if_weight, ic_weight)

    index_list = ['staff_changes',
                  'funds',
                  'meeting_decide',
                  'restricted_shares',
                  'son_company',
                  'suspend',
                  'shares',
                  ]

    # tech_list = ['BBANDS_20_1.5',
    #              'BBANDS_40_1.5',
    #              'BBANDS_100_1.5',
    #              'BBANDS_140_1.5',
    #              'BBANDS_200_1.5',
    #              'MACD_12_26_9',
    #              'MACD_20_60_18',
    #              'MA_LINE_10_5',
    #              'MA_LINE_60_20',
    #              'MA_LINE_120_60',
    #              ]

    pool_num = 20
    main_model.test_index_3_(index_list, pool_num, suffix_name='single')


if __name__ == '__main__':
    sector_name_list = ['market_top_300plus',
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

                        'market_top_800plus',
                        'market_top_800plus_industry_10_15',
                        'market_top_800plus_industry_20_25_30_35',
                        'market_top_800plus_industry_40',
                        'market_top_800plus_industry_45_50',
                        'market_top_800plus_industry_55'
                        ]

    hold_time_list = [5, 20]
    for hold_time in hold_time_list:
        for sector_name in sector_name_list:
            main_fun(sector_name, hold_time)
