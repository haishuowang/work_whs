import sys

sys.path.append('/mnt/mfs')
import string
from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load.plt import plot_send_result, savfig_send
from work_whs.loc_lib.pre_load.sql import conn
from work_whs.bkt_factor_create.raw_data_path import base_data_dict


# (subject='tmp', text='', to=None, filepath=None)

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

        # if sector_name.startswith('market_top_300plus') \
        #         or sector_name.startswith('index_000300'):
        #     if_weight = 1
        #     ic_weight = 0
        #
        # elif sector_name.startswith('market_top_300to800plus') \
        #         or sector_name.startswith('index_000905'):
        if_weight = 0
        ic_weight = 1

        # else:
        #     if_weight = 0.5
        #     ic_weight = 0.5

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
            index_df_1 = self.load_index_data('000300').fillna(0)
            index_df_2 = self.load_index_data('000905').fillna(0)
            hedge_df = if_weight * index_df_1 + ic_weight * index_df_2
            self.return_df = return_df.sub(hedge_df, axis=0)

        else:
            self.return_df = return_df

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
        data = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/DERIVED_WHS/CHG_{index_name}.csv', header=None)
        target_df = data[1].reindex(index=self.xinx)
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


def main(factor_test, industry_df, index_name='000905', q=True, sector_name='market_top_2000',
         limit=0.02, begin_d=7, end_d=7):
    def sector_filter(x):
        stock_id = x.name
        result_list = []
        for i in x:
            if pd.isna(i):
                result_list.append(np.nan)
            else:
                if sector_df[stock_id][i] == 1:
                    result_list.append(1)
                else:
                    result_list.append(np.nan)
        return result_list

    def fun_index(x):
        return index_price_sr.loc[x]

    def fun_stock(x):
        stock_id = x.name
        if stock_id not in adj_price.columns:
            return [None] * len(x)
        else:
            result_list = []

            for i in x:
                result_list.append(adj_price[stock_id][i])
            return result_list

    # def fun_1(x):
    #     if x in factor_test.xinx or pd.isna(x):
    #         return x
    #     elif x < factor_test.xinx[0] or x > factor_test.xinx[-1]:
    #         return pd.NaT
    #     else:
    #         return factor_test.xinx[factor_test.xinx > x][0]

    def fun_2(x):
        stock_id = x.name
        if stock_id not in adj_price.columns:
            return [None] * len(x)
        else:
            result_list = []
            for i in x:
                if i < sector_df.index[0] or i > sector_df.index[-1] or pd.isna(i):
                    result_list.append(pd.NaT)
                elif i in sector_df.index:
                    result_list.append(next_date_df[stock_id][i])
                else:
                    result_list.append(next_date_df[stock_id][sector_df.index[sector_df.index > i][0]])
            return result_list

    def fun(df, n):
        up_df = (df > 0)
        dn_df = (df < 0)
        target_up_df = up_df.copy()
        target_dn_df = dn_df.copy()

        for i in range(n - 1):
            target_up_df = target_up_df * up_df.shift(i + 1)
            target_dn_df = target_dn_df * dn_df.shift(i + 1)
        target_df = target_up_df.fillna(0).astype(int) - target_dn_df.fillna(0).astype(int)
        return target_df

    target_path = '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_EVA/REM5_scores.csv'
    data = bt.AZ_Load_csv(target_path)
    # 涨跌停的股票mask
    limit_buy_sell_df = factor_test.limit_buy_sell_df_c.shift(1) * factor_test.suspendday_df_c.shift(1)
    sector_df = (data >= 7).astype(int).reindex(factor_test.xinx) * limit_buy_sell_df
    # 加上行业测试
    sector_df = sector_df.replace(0, np.nan)  # * industry_df

    tmp_next_date_df = pd.DataFrame([limit_buy_sell_df.index] * len(limit_buy_sell_df.columns)).T
    tmp_next_date_df.index = limit_buy_sell_df.index
    tmp_next_date_df.columns = limit_buy_sell_df.columns
    next_date_df = tmp_next_date_df[limit_buy_sell_df == limit_buy_sell_df].fillna(method='bfill')
    next_date_df.loc[pd.NaT] = np.nan
    # notice_date_df = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/LICO_FN_RGINCOME/UpSampleDate_NETPROFIT_First.csv')

    if q:
        a = notice_date_df
    else:
        a = notice_date_df[notice_date_df.index.str.endswith('Q2')]

    # 把columns reindex到对应sector
    b = a.apply(lambda x: pd.to_datetime(x)).reindex(columns=factor_test.xnms)

    adj_price = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv')
    adj_price.loc[pd.NaT] = np.nan

    index_price_df = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/NEW.csv')
    index_price_sr = index_price_df[index_name]
    index_price_sr.loc[pd.NaT] = np.nan

    b_begin_df = (b - timedelta(begin_d)).apply(fun_2)
    # 把交易不到的点过滤
    filter_df = b_begin_df < b

    b_end_df = (b + timedelta(end_d)).apply(fun_2)

    # 获取 公告日期前7天 和 公告日期后7天index的价格
    index_begin_price = b_begin_df.applymap(fun_index)
    index_end_price = b_end_df.applymap(fun_index)
    # index_price = b_df.applymap(fun_index)
    # 获取 公告日期前7天 和 公告日期后7天index的return
    index_pct_df = (index_end_price - index_begin_price) / index_begin_price

    # 获取 公告日期前7天 和 公告日期后7天股票的价格
    price_begin_price = b_begin_df.apply(fun_stock)
    price_end_price = b_end_df.apply(fun_stock)
    # 获取 公告日期前7天 和 公告日期后7天股票的return
    price_pct_df = (price_end_price - price_begin_price) / price_begin_price
    # 对应时间 超额收益 df
    alpha_pct_df = price_pct_df - index_pct_df

    con_df = (alpha_pct_df > limit).astype(int)
    # 过去3年超过limit的 接下来一年交易
    pos_df = fun(con_df, 3).shift(1) * filter_df
    # 计算超额收益,
    pnl_table = pos_df * alpha_pct_df
    pnl_df = pnl_table.sum(1)
    pot = pnl_df.sum() / (2 * pos_df.sum().sum())
    print(pnl_df)
    return pnl_df.replace(0, np.nan).dropna(), pot


def part_main(factor_test, industry_df, index_name, q, sector_name, limit, begin_d_list, end_d_list):
    print(1)
    plt.figure(figsize=[16, 10])
    for begin_d in begin_d_list:
        for end_d in end_d_list:
            pnl_df, pot = main(factor_test, industry_df, index_name, q,
                               sector_name, limit, begin_d, end_d)
            plt.plot(pnl_df.cumsum().values, label=f'begin_d={begin_d}, end_d={end_d}, pot={pot}')
    plt.legend()
    savfig_send(subject=f'sector_name=REM5_scores, index_name={index_name}, '
                        f'Q={q}, limit={limit}')


notice_date_df = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/LICO_FN_SIGQUAFINA/UpSampleDate_NOTICEDATE_First.csv',
                                parse_dates=False)

if __name__ == '__main__':
    index_name_list = ['000300', '000905']
    q_list = [True, False]
    sector_name_list = ['market_top_1000', 'market_top_2000']
    limit_list = [0.01, 0.02, 0.03]
    begin_d_list = [2, 7, 14]
    end_d_list = [2, 7, 14]

    root_path = '/mnt/mfs/DAT_EQT'
    if_save = True
    if_new_program = True

    begin_date = pd.to_datetime('20050101')
    end_date = datetime.now()
    lag = 2
    return_file = ''

    if_hedge = False
    hold_time = 10
    if_only_long = True
    pool = Pool(20)
    for sector_name in sector_name_list:
        factor_test = FactorTestBase(root_path, if_save, if_new_program, begin_date, end_date, sector_name,
                                     hold_time, lag, return_file, if_hedge, if_only_long)
        # file_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
        # for file_num in file_list:
        #     industry_df = bt.AZ_Load_csv(f'/mnt/mfs/DAT_EQT/EM_Funda/LICO_IM_INCHG/Global_Level1_{file_num}.csv')
        industry_df = 1
        for q in q_list:
            for index_name in index_name_list:
                for limit in limit_list:
                    args = (factor_test, industry_df, index_name, q, sector_name, limit, begin_d_list, end_d_list)
                    pool.apply_async(part_main, args)
                    # part_main(*args)
    pool.close()
    pool.join()
