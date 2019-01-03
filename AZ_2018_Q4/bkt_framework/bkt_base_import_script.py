import sys
from multiprocessing import Lock

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *


def mul_fun(a, b):
    a_l = a.where(a > 0, 0)
    a_s = a.where(a < 0, 0)

    b_l = b.where(b > 0, 0)
    b_s = b.where(b < 0, 0)

    pos_l = a_l.mul(b_l)
    pos_s = a_s.mul(b_s)

    pos = pos_l.sub(pos_s)
    return pos


def sub_fun(a, b):
    return a.sub(b)


def add_fun(a, b):
    return a.add(b)


def out_sample_perf_c(pnl_df_out, way=1):
    if way == 1:
        sharpe_out = bt.AZ_Sharpe_y(pnl_df_out)
    else:
        sharpe_out = bt.AZ_Sharpe_y(-pnl_df_out)
    out_condition = sharpe_out > 0.8
    return out_condition, round(sharpe_out * way, 2)


def simu_fun(cut_date, pos_df_daily, pct_n, if_return_pnl=False, if_only_long=False):
    pnl_df = (pos_df_daily * pct_n).sum(axis=1)
    pnl_df = pnl_df.replace(np.nan, 0)
    # 样本内表现
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
    # ic = round(bt.AZ_Normal_IC(pos_df_daily_in, pct_n, min_valids=None, lag=0).mean(), 6)
    ic = 0
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


def simu_time_para_fun(time_para_dict, pos_df_daily, adj_return, if_return_pnl=False, if_only_long=False):
    pnl_df = (pos_df_daily * adj_return).sum(axis=1)

    pnl_df = pnl_df.replace(np.nan, 0)
    result_dict = OrderedDict()
    xinx = adj_return.index
    for time_key in time_para_dict.keys():
        begin_para, cut_para, end_para_1, end_para_2, end_para_3, end_para_4 = time_para_dict[time_key]

        # 样本内索引
        sample_in_index = (xinx >= begin_para) & (xinx < cut_para)
        # 样本外索引
        sample_out_index_1 = (xinx >= cut_para) & (xinx < end_para_1)
        sample_out_index_2 = (xinx >= cut_para) & (xinx < end_para_2)
        sample_out_index_3 = (xinx >= cut_para) & (xinx < end_para_3)
        sample_out_index_4 = (xinx >= cut_para) & (xinx < end_para_4)
        # 样本内表现
        pos_df_daily_in = pos_df_daily[sample_in_index]
        adj_return_in = adj_return[sample_in_index]
        pnl_df_in = pnl_df[sample_in_index]

        asset_df_in = pnl_df_in.cumsum()
        last_asset_in = asset_df_in.iloc[-1]

        pot_in = bt.AZ_Pot(pos_df_daily_in, last_asset_in)

        leve_ratio = bt.AZ_Leverage_ratio(asset_df_in)

        if leve_ratio < 0:
            leve_ratio = 100
        sharpe_q_in_df = bt.AZ_Rolling_sharpe(pnl_df_in, roll_year=1, year_len=250, min_periods=1,
                                              cut_point_list=[0.3, 0.5, 0.7], output=False)
        sharpe_q_in_df = round(sharpe_q_in_df, 4)
        sp_in = bt.AZ_Sharpe_y(pnl_df_in)
        fit_ratio = bt.AZ_fit_ratio(pos_df_daily_in, adj_return_in)

        ic = 0
        sp_in_u, sp_in_m, sp_in_d = sharpe_q_in_df.values

        in_condition_u = sp_in_u > 0.9 and leve_ratio > 1
        in_condition_d = sp_in_d < -0.9 and leve_ratio > 1
        # 分双边和只做多
        if if_only_long:
            in_condition = in_condition_u
        else:
            in_condition = in_condition_u | in_condition_d

        if sp_in_m > 0:
            way = 1
        else:
            way = -1

        # 样本外表现
        pnl_df_out_1 = pnl_df[sample_out_index_1]
        pnl_df_out_2 = pnl_df[sample_out_index_2]
        pnl_df_out_3 = pnl_df[sample_out_index_3]
        pnl_df_out_4 = pnl_df[sample_out_index_4]

        out_condition_1, sp_out_1 = out_sample_perf_c(pnl_df_out_1, way=way)
        out_condition_2, sp_out_2 = out_sample_perf_c(pnl_df_out_2, way=way)
        out_condition_3, sp_out_3 = out_sample_perf_c(pnl_df_out_3, way=way)
        out_condition_4, sp_out_4 = out_sample_perf_c(pnl_df_out_4, way=way)
        if if_return_pnl:
            result_dict[time_key] = [in_condition, out_condition_1, out_condition_2, out_condition_3, out_condition_4,
                                     ic, sp_in_u, sp_in_m, sp_in_d, pot_in, fit_ratio, leve_ratio,
                                     sp_in, sp_out_1, sp_out_2, sp_out_3, sp_out_4, pnl_df]
        else:
            result_dict[time_key] = [in_condition, out_condition_1, out_condition_2, out_condition_3, out_condition_4,
                                     ic, sp_in_u, sp_in_m, sp_in_d, pot_in, fit_ratio, leve_ratio,
                                     sp_in, sp_out_1, sp_out_2, sp_out_3, sp_out_4]
    return result_dict


class FactorTestBase:
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

    def load_index_data(self, index_name):
        data = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Funda/INDEX_TD_DAILYSYS/CHG.csv'))
        target_df = data[index_name].reindex(index=self.xinx)
        return target_df * 0.01

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

    @staticmethod
    def pos_daily_fun(df, n=5):
        return df.rolling(window=n, min_periods=1).mean() * 100

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
        daily_pos = self.pos_daily_fun(order_df, n=self.hold_time)
        daily_pos.fillna(0, inplace=True)
        # 排除出场涨跌停的影响
        daily_pos = daily_pos * self.limit_buy_sell_df_c * self.suspendday_df_c
        daily_pos.fillna(method='ffill', inplace=True)
        return daily_pos

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
    def create_all_para_(*args):
        target_list = list(product(*args))
        return target_list

    @staticmethod
    def create_log_save_path(target_path):
        top_path = os.path.split(target_path)[0]
        if not os.path.exists(top_path):
            os.mkdir(top_path)
        if not os.path.exists(target_path):
            os.mknod(target_path)

    def save_load_control(self, suffix_name, file_name, *args):
        # 参数存储与加载的路径控制
        result_save_path = '/mnt/mfs/dat_whs/result_new'
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
            para_ready_df = pd.DataFrame(list(self.create_all_para_(*args)))
            total_para_num = len(para_ready_df)
            if self.if_save:
                self.create_log_save_path(log_save_file)
                self.create_log_save_path(result_save_file)
                self.create_log_save_path(para_save_file)
                para_dict['para_ready_df'] = para_ready_df
                para_dict['args'] = args
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


class FactorTest(FactorTestBase):
    def __init__(self, key_fun, *args):
        super(FactorTest, self).__init__(*args)
        self.key_fun = key_fun

    @staticmethod
    def save_file_fun(target_file, write_list, if_save, lock):
        if if_save:
            with lock:
                f = open(target_file, 'a')
                f.write('|'.join([str(x) for x in write_list]) + '\n')
                f.close()

    def part_test_fun(self, key, log_save_file, result_save_file, total_para_num, filter_name, *factor_name_list):
        lock = Lock()
        start_time = time.time()
        load_time_1 = time.time()
        percent = 0.3
        mix_factor = self.key_fun.create_mix_factor(*factor_name_list, xinx=self.xinx, xnms=self.xnms,
                                                    sector_df=self.sector_df, if_only_long=self.if_only_long,
                                                    percent=percent)

        filter_factor = self.key_fun.load_filter_data(filter_name, self.xinx, self.xnms,
                                                      self.sector_df, self.if_only_long)
        load_time_2 = time.time()
        load_delta = round(load_time_2 - load_time_1, 2)
        signal_df = mix_factor * filter_factor

        pos_df = self.deal_mix_factor(signal_df).shift(2)
        result_dict = simu_time_para_fun(self.time_para_dict, pos_df, self.return_choose,
                                         if_return_pnl=False, if_only_long=self.if_only_long)

        for time_key in result_dict.keys():
            in_condition, *filter_result = result_dict[time_key]
            # result 存储
            if in_condition:
                self.save_file_fun(result_save_file,
                                   [time_key, key, filter_name, *factor_name_list,
                                    self.sector_name, in_condition] + filter_result,
                                   self.if_save, lock)
                print([time_key, in_condition, filter_name, *factor_name_list] + filter_result)

        end_time = time.time()
        run_delta = round(end_time - start_time, 2)
        # 参数存储
        self.save_file_fun(log_save_file,
                           [key, filter_name, *factor_name_list, self.sector_name, run_delta, load_delta],
                           self.if_save, lock)
        print('{}%, {}, cost {} seconds, load_cost {} seconds'
              .format(round(key / total_para_num * 100, 4), key, run_delta, load_delta), *factor_name_list)

    def main_test_fun(self, *args, pool_num=20, suffix_name='', old_file_name=''):
        para_ready_df, log_save_file, result_save_file, total_para_num = \
            self.save_load_control(suffix_name, old_file_name, *args)

        pool = Pool(pool_num)
        for key in list(para_ready_df.index):
            # print(para_ready_df)
            args = para_ready_df.loc[key]
            args_list = (key, log_save_file, result_save_file, total_para_num, *args)
            # self.part_test_fun(*args_list)
            pool.apply_async(self.part_test_fun, args=args_list)
        pool.close()
        pool.join()
        pass

    def single_test(self, filter_name, *factor_name_list):
        percent = 0.3
        mix_factor = self.key_fun.create_mix_factor(*factor_name_list, xinx=self.xinx, xnms=self.xnms,
                                                    sector_df=self.sector_df, if_only_long=self.if_only_long,
                                                    percent=percent)

        filter_factor = self.key_fun.load_filter_data(filter_name, self.xinx, self.xnms,
                                                      self.sector_df, self.if_only_long)
        signal_df = mix_factor * filter_factor
        pos_df = self.deal_mix_factor(signal_df).shift(2)
        in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
        fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df = simu_fun(self.cut_date, pos_df, self.return_choose,
                                                                      if_return_pnl=True,
                                                                      if_only_long=self.if_only_long)
        return mix_factor, in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, \
               pot_in, fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df

# time_para_dict = OrderedDict()
#
# time_para_dict['time_para_1'] = [pd.to_datetime('20100101'), pd.to_datetime('20150101'),
#                                  pd.to_datetime('20150401'), pd.to_datetime('20150701'),
#                                  pd.to_datetime('20151001'), pd.to_datetime('20160101')]
#
# time_para_dict['time_para_2'] = [pd.to_datetime('20110101'), pd.to_datetime('20160101'),
#                                  pd.to_datetime('20160401'), pd.to_datetime('20160701'),
#                                  pd.to_datetime('20161001'), pd.to_datetime('20170101')]
#
# time_para_dict['time_para_3'] = [pd.to_datetime('20130101'), pd.to_datetime('20180101'),
#                                  pd.to_datetime('20180401'), pd.to_datetime('20180701'),
#                                  pd.to_datetime('20181001'), pd.to_datetime('20181001')]
#
# time_para_dict['time_para_4'] = [pd.to_datetime('20130601'), pd.to_datetime('20180601'),
#                                  pd.to_datetime('20181001'), pd.to_datetime('20181001'),
#                                  pd.to_datetime('20181001'), pd.to_datetime('20181001')]
#
# time_para_dict['time_para_5'] = [pd.to_datetime('20130701'), pd.to_datetime('20180701'),
#                                  pd.to_datetime('20181001'), pd.to_datetime('20181001'),
#                                  pd.to_datetime('20181001'), pd.to_datetime('20181001')]
#
# time_para_dict['time_para_6'] = [pd.to_datetime('20130801'), pd.to_datetime('20180801'),
#                                  pd.to_datetime('20181001'), pd.to_datetime('20181001'),
#                                  pd.to_datetime('20181001'), pd.to_datetime('20181001')]

# if __name__ == '__main__':
#     main_fun('market_top_300plus', 20, time_para_dict)
