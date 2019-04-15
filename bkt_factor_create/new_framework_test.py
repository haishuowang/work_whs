import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *
from work_whs.bkt_factor_create.base_fun_import import DiscreteClass, ContinueClass


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


class FactorTestBase:
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
            market_top_n[market_top_n == market_top_n] = 1
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
        data = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Funda/INDEX_TD_DAILYSYS/CHG.csv'))
        target_df = data[index_name].reindex(index=self.xinx)
        return target_df * 0.01

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
            signal_df_up = (signal_df > 0).astype(int)
            daily_pos = self.signal_to_pos(signal_df_up)
        elif ls_para == 's':
            signal_df_dn = (signal_df < 0).astype(int)
            daily_pos = self.signal_to_pos(signal_df_dn)
        elif ls_para == 'ls':
            daily_pos = self.signal_to_pos(signal_df)
        else:
            daily_pos = self.signal_to_pos(signal_df)
        return daily_pos

    def save_load_control(self, tech_name_list, funda_name_list, suffix_name, file_name):
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
            para_ready_df = pd.DataFrame(list(self.create_all_para(tech_name_list, funda_name_list)))
            total_para_num = len(para_ready_df)
            if self.if_save:
                self.create_log_save_path(log_save_file)
                self.create_log_save_path(result_save_file)
                self.create_log_save_path(para_save_file)
                para_dict['para_ready_df'] = para_ready_df
                para_dict['tech_name_list'] = tech_name_list
                para_dict['funda_name_list'] = funda_name_list
                pd.to_pickle(para_dict, para_save_file)

        else:
            log_save_file = os.path.join(result_save_path, 'log', file_name)
            result_save_file = os.path.join(result_save_path, 'result', file_name)
            para_save_file = os.path.join(result_save_path, 'para', file_name)

            para_tested_df = pd.read_table(log_save_file, sep='|', header=None, index_col=0)
            para_all_df = pd.read_pickle(para_save_file)
            total_para_num = len(para_all_df)
            para_ready_df = para_all_df.loc[sorted(list(set(para_all_df.index) - set(para_tested_df.index)))]
        print(file_name)
        print(f'para_num:{len(para_ready_df)}')
        return para_ready_df, log_save_file, result_save_file, total_para_num


class FactorTest(FactorTestBase, DiscreteClass, ContinueClass, TrainFunSet):
    def __init__(self, *args):
        super(FactorTest, self).__init__(*args)

    def load_raw_data(self, file_name):
        raw_df = pd.read_pickle(f'/mnt/mfs/dat_whs/data/factor_data/{self.sector_name}/{file_name}.pkl')
        raw_df = raw_df.reindex(index=self.xinx)
        return raw_df

    def load_zscore_data(self, file_name):
        raw_zscore_df = self.row_zscore(self.load_raw_data(file_name), self.sector_df)
        return raw_zscore_df

    # def load_all_raw_data(self, file_list):
    #     all_raw_zscore_df_dict = OrderedDict()
    #     for file_name in file_list:
    #         all_raw_zscore_df_dict.update(self.load_zscore_data(file_name))
    #     return all_raw_zscore_df_dict

    @staticmethod
    def judge_way(sharpe):
        if sharpe > 0:
            return 1
        elif sharpe < 0:
            return -1
        else:
            return 0

    def back_test(self, data_df, cut_date, percent, return_pos=False):
        cut_time = pd.to_datetime(cut_date)
        signal_df = self.row_extre(data_df, self.sector_df, percent)
        if len(signal_df.abs().sum(1).replace(0, np.nan).dropna()) / len(self.xinx) > 0.7:
            pos_df = self.signal_to_pos(signal_df)
            pnl_table = pos_df.shift(self.lag) * self.return_df
            pnl_df = pnl_table.sum(1)
            sample_in_index = (self.xinx < cut_time)
            sample_out_index = (self.xinx >= cut_time)

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

    def get_pnl_df(self, file_name, cut_date, percent):
        data_df = self.load_zscore_data(file_name)
        info_df, pnl_df = self.back_test(data_df, cut_date, percent)
        pnl_df.name = file_name
        info_df.name = file_name
        # print(info_df, pnl_df)
        return info_df, pnl_df

    def get_all_pnl_df(self, file_list, cut_date, percent, if_multy=True):
        result_list = []
        if if_multy:
            pool = Pool(20)
            for file_name in file_list:
                # result_list.append(self.get_pnl_df(file_name, cut_date, percent))
                result_list.append(pool.apply_async(self.get_pnl_df, args=(file_name, cut_date, percent)))
            result_list_c = [res.get() for res in result_list]
            all_info_list = pd.concat([res[0] for res in result_list_c], axis=1)
            all_pnl_df = pd.concat([res[1] for res in result_list_c], axis=1)
        else:
            for file_name in file_list:
                result_list.append(self.get_pnl_df(file_name, cut_date, percent))

            all_info_list = pd.concat([res[0] for res in result_list], axis=1)
            all_pnl_df = pd.concat([res[1] for res in result_list], axis=1)
        return all_info_list, all_pnl_df

    def get_mix_pnl_df(self, exe_str, cut_date, percent):
        def tmp_fun():
            exe_list = exe_str.split('@')
            name_1, way_str = exe_list[0].split('_')
            factor_1 = self.load_raw_data(name_1) * int(way_str)
            for i in range(int((len(exe_list) - 1) / 2)):
                fun_str = exe_list[i + 1]
                name_2, way_str = exe_list[i + 2].split('_')
                factor_2 = self.load_raw_data(name_2) * way_str
                factor_1 = getattr(self, fun_str)(factor_1, factor_2)
            return factor_1

        mix_factor = tmp_fun()
        info_df, pnl_df = self.back_test(mix_factor, cut_date, percent)
        pnl_df.name = exe_str
        info_df.name = exe_str
        return info_df, pnl_df

    @staticmethod
    def get_scores_fun(sp_in, pot_in):
        return (sp_in.where(sp_in > 0.4, other=0)) / 1 + (pot_in.where(pot_in < 100, other=100)) / 200

    @staticmethod
    def get_score_fun(sp_in, pot_in):
        tmp_fun = lambda x: x if x > 0.4 else 0
        return tmp_fun(sp_in) / 1 + min(pot_in, 100) / 200

    def train_fun(self, cut_date, percent):
        try:
            def tmp_fun(mix_factor_df, low_corr_factor, low_corr_factor_way):
                tmp_factor_df = self.load_raw_data(low_corr_factor) * low_corr_factor_way
                tmp_mix_df = mix_factor_df + tmp_factor_df
                info_df, pnl_df = self.back_test(tmp_mix_df, cut_date, percent)
                return tmp_mix_df, info_df, pnl_df

            all_file = sorted(os.listdir('/mnt/mfs/dat_whs/data/factor_data/index_000300'))
            all_file = [x[:-4] for x in all_file if 'suspend' not in x]
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
            mix_factor_df = self.load_raw_data(best_factor) * best_factor_way
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
                    exe_str = f'{exe_str}@add_fun@{low_corr_factor}_{str(low_corr_factor_way)}'
                    mix_factor_df = tmp_mix_df
                    mix_score = tmp_score
                    mix_info = tmp_info_df
                    mix_pnl = tmp_pnl_df
                    mix_num += 1
                    # 超过混合上限 跳出
                    if mix_num >= 10:
                        break
            if mix_info.loc['sp'] > 2 and mix_info.loc['pot'] > 50:
                plot_send_result(mix_pnl, mix_info.loc['sp'], f'{self.sector_name}|{self.hold_time}'
                                                              f'|{self.if_only_long}|{percent}'
                                 , exe_str + '\n' + pd.DataFrame(mix_info).to_html())

        except Exception as error:
            print(error)

    def run(self, cut_date, percent, run_num=100):
        pool = Pool(28)
        for i in range(run_num):
            pool.apply_async(self.train_fun, args=(cut_date, percent))
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

    factor_test = FactorTest(root_path, if_save, if_new_program, begin_date, end_date, sector_name, hold_time,
                             lag, return_file, if_hedge, if_only_long)

    cut_date = '20180101'
    factor_test.run(cut_date, percent, run_num=10)


def main_fun():
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
    for if_only_long in [False, True]:
        for sector_name in sector_name_list:
            for percent in [0.1, 0.2]:
                for hold_time in hold_time_list:
                    part_main_fun(if_only_long, hold_time, sector_name, percent)


if __name__ == '__main__':
    main_fun()
