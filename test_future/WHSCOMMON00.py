import sys

sys.path.append('/mnt/mfs')
from work_dmgr_fut.loc_lib.pre_load import *
from work_dmgr_fut.loc_lib.pre_load.plt import savfig_send
from work_dmgr_fut.fut_script.FutDataLoad import FutData, ReshapeData
from work_dmgr_fut.fut_script.signal_fut_fun import FutIndex, Signal, Position
from work_whs.loc_lib.pre_load.senior_tools import SignalAnalysis
import talib as ta
import warnings
warnings.filterwarnings("ignore")


def ana_fun(data_df, fut_name, col_name):
    # raw_return_df = data_df['price_return']
    raw_return_df = data_df['pnl']
    signal_df = data_df[col_name]
    SignalAnalysis.CDF(signal_df, raw_return_df, hold_time=1, title=f'{fut_name} {col_name} CDF Figure', lag=1)


class FutMinPosBase(FutData):
    def __init__(self, fut_name_list, usecols_list, cut_num=None,
                 date_now=datetime.now(), lag=1,
                 live_path='/mnt/mfs/temp/LIVE_MIN_BAR', if_save=False, *args):
        super(FutMinPosBase, self).__init__(*args)
        self.live_path = live_path
        self.trade_time = pd.read_csv(f'{self.root_path}/DailyPX/TradeTime', sep='|', index_col=0)['TradeTime']
        self.trade_intra_date = self.trade_date + timedelta(hours=16)

        self.start_time_dict = self.get_start_time_dict(fut_name_list)
        self.end_time_dict = self.get_end_time_dict(fut_name_list)

        self.today_str, self.last_trd_str, self.next_trd_str = self.judge_trade_date(date_now)
        self.last_day_str = (pd.to_datetime(self.today_str) + timedelta(1)).strftime('%Y%m%d')
        self.active_df, self.aadj_factor_df = self.get_active_df('Volume')
        # self.active_df = self.active_df.truncate(before='2016-01-01')
        self.save_exe_path = '/mnt/mfs/dat_whs/DAT_FUT'
        self.save_bkt_path = '/mnt/mfs/dat_whs/DAT_FUT'

        self.active_df = self.active_df.truncate(before='2016-01-01')
        self.live_time_dict = self.load_live_time_dict(fut_name_list)
        self.usecols_list = usecols_list
        self.cut_num = cut_num
        self.alpha_name = ''
        self.lag = lag
        self.if_save = if_save

    @staticmethod
    def get_active_df(data_name):
        root_path = '/mnt/mfs/dat_whs/DAT_FUT'
        active_df = bt.AZ_Load_csv(f'{root_path}/Active_{data_name}')
        adj_factor_df = bt.AZ_Load_csv(f'{root_path}/adj_factor_{data_name}')
        aadj_factor_df = adj_factor_df.cumprod()
        return active_df, aadj_factor_df

    def judge_trade_date(self, date_now=datetime.now()):
        today_str = self.trade_date[date_now < self.trade_intra_date][0].strftime('%Y%m%d')
        next_trd_str = self.trade_date[date_now < self.trade_intra_date][1].strftime('%Y%m%d')
        last_trd_str = self.trade_date[date_now > self.trade_intra_date][-1].strftime('%Y%m%d')
        return today_str, last_trd_str, next_trd_str

    def get_start_time(self, fut_name):
        trade_time = pd.read_csv(f'{self.root_path}/DailyPX/TradeTime', sep='|', index_col=0)['TradeTime']
        start_time_list = []
        for trade_range_str in trade_time.loc[fut_name].split(','):
            begin_str, end_str = trade_range_str.split('-')
            start_time_list.append(begin_str)
        return start_time_list

    def get_start_time_dict(self, fut_name_list):
        target_dict = {}
        for fut_name in fut_name_list:
            target_dict[fut_name] = self.get_start_time(fut_name)
        return target_dict

    def get_end_time(self, fut_name):
        trade_time = pd.read_csv(f'{self.root_path}/DailyPX/TradeTime', sep='|', index_col=0)['TradeTime']
        end_time_list = []
        for trade_range_str in trade_time.loc[fut_name].split(','):
            begin_str, end_str = trade_range_str.split('-')
            end_time_list.append(end_str)
        return end_time_list

    def get_end_time_dict(self, fut_name_list):
        target_dict = {}
        for fut_name in fut_name_list:
            target_dict[fut_name] = self.get_end_time(fut_name)
        return target_dict

    def get_next_time_range(self, fut_name):
        def str_to_time(target_str):
            a = '16:00'
            b = '03:00'
            if target_str < b:
                return pd.to_datetime(self.last_day_str + ' ' + target_str)
            if b < target_str < a:
                return pd.to_datetime(self.today_str + ' ' + target_str)
            else:
                return pd.to_datetime(self.last_trd_str + ' ' + target_str)

        trade_range = []
        for trade_range_str in self.trade_time.loc[fut_name].split(','):
            begin_str, end_str = trade_range_str.split('-')
            begin_time = str_to_time(begin_str) + timedelta(minutes=1)
            end_time = str_to_time(end_str)
            time_range = list(pd.date_range(begin_time, end_time, freq='T').values)
            trade_range += time_range
        return sorted(trade_range)

    def load_live_time(self, fut_name):
        live_time_raw = pd.read_csv(f'{self.root_path}/T_Calendar/Active_Trading', sep='|',
                                    index_col=0, parse_dates=True)[fut_name].dropna().truncate(
            after=datetime.now() + timedelta(2))
        delete_time_list = self.start_time_dict[fut_name]
        # delete_time_list = self.get_start_time(fut_name)
        live_time = np.array([x for x in live_time_raw.index if x.strftime('%H:%M') not in delete_time_list])

        return live_time[live_time > pd.to_datetime(self.last_trd_str) + timedelta(hours=16)]

    def load_live_time_dict(self, fut_name_list):
        target_dict = {}

        for fut_name in fut_name_list:
            target_dict[fut_name] = self.load_live_time(fut_name)
        return target_dict

    def load_live_data(self, contract_id, usecols_list, cut_time=None):
        print(f'Deal {contract_id}')
        fut_name = re.sub('\d', '', contract_id.split('.')[0])
        live_time = self.live_time_dict[fut_name]

        if contract_id.split('.')[-1] != 'CZC':
            live_file = contract_id.split('.')[0] + '_data.csv'
            live_file_path = f'{self.live_path}/{live_file.lower()}'
        else:
            num = re.sub('\D', '', contract_id.split('.')[0])
            live_file = re.sub('\d', '', contract_id.split('.')[0]) + num[1:] + '_data.csv'
            live_file_path = f'{self.live_path}/{live_file}'

        if os.path.exists(live_file_path):
            # 去掉第一行 最后一行
            live_data = pd.read_csv(live_file_path, sep='|').iloc[:-1]
            new_data = pd.DataFrame(index=pd.to_datetime(live_data['Time'].apply(lambda x: self.last_trd_str
            if x > '16:00' else self.today_str) + ' ' + live_data['Time']) + timedelta(minutes=1))
            new_data.index.name = 'TradeDate'
            new_data['Date'] = pd.to_datetime(self.today_str).strftime('%Y-%m-%d')

            new_data['Time'] = new_data.index.strftime('%H:%M')
            new_data['Open'] = live_data['Open'].values
            new_data['High'] = live_data['High'].values
            new_data['Low'] = live_data['Low'].values
            new_data['Close'] = live_data['Close'].values
            new_data['Volume'] = (live_data['Volume'] - live_data['Volume'].shift(1).fillna(0)).values
            new_data['Turnover'] = (live_data['Turnover'] - live_data['Turnover'].shift(1).fillna(0)).values
            new_data['OpenInterest'] = live_data['IO'].values
            new_data = new_data.loc[sorted(list(set(new_data.index) & set(live_time)))]
            new_data.dropna(how='all', subset=['Volume'], inplace=True)

            return new_data[['Date', 'Time'] + usecols_list]
        else:
            return pd.DataFrame()

    def load_live_reshape_data(self, contract_id, usecols_list, cut_num=5):
        new_data = self.load_live_data(contract_id, usecols_list)
        end_time_list = self.end_time_dict[re.sub('\d', '', contract_id.split('.')[0])]
        # end_time_list = self.get_end_time(re.sub('\d', '', contract_id.split('.')[0]))
        today_end = [x for x in end_time_list if '12:00' < x < '16:00']
        if not new_data.empty:
            new_data_reshape = ReshapeData(self.today_str, today_end[0]).run(new_data, cut_num)
            return new_data_reshape
        else:
            return pd.DataFrame()

    def get_old_new_data(self, contract_id, update_live_data=True):
        fut_name = re.sub('\d', '', contract_id.split('.')[0])
        if self.cut_num:
            old_data = self.load_intra_reshape_data(contract_id, self.usecols_list, self.cut_num)
        else:
            old_data = self.load_intra_data(contract_id, self.usecols_list)
        if update_live_data:
            if self.cut_num:
                new_data = self.load_live_reshape_data(contract_id, self.usecols_list, self.cut_num)
            else:
                new_data = self.load_live_data(contract_id, self.usecols_list)
            all_data = old_data.combine_first(new_data)[['Date', 'Time'] + self.usecols_list]
            # all_data = new_data.combine_first(old_data)[['Date', 'Time'] + self.usecols_list]
        else:
            all_data = old_data

        part_info_df = self.active_df[self.active_df[fut_name + '01'] == contract_id]
        active_begin = self.last_trade_date(part_info_df.index[0]) + timedelta(hours=17)
        active_end = part_info_df.index[-1] + timedelta(hours=17)
        if update_live_data:
            part_data_df = all_data.truncate(before=active_begin - timedelta(20))
        else:
            part_data_df = all_data.truncate(before=active_begin - timedelta(20), after=active_end)
        return part_data_df, active_begin, active_end

    def pos_to_bkt(self, data_df, concat=True):
        pos_diff = data_df['position'] - data_df['position'].shift(1).fillna(0)
        pos_change = data_df.loc[pos_diff.replace(0, np.nan).dropna().index]
        pos_change = pos_change[~pos_change['exe_time'].duplicated('last')]
        pos_df = pos_change.pivot('exe_time', 'con_id', 'position')
        con_id_list = list(pos_df.columns)
        pos_df['e_time'] = pos_df.index.strftime('%Y-%m-%d %H:%M')
        pos_df['w_time'] = datetime.now().strftime('%Y-%m-%d %H:%M')
        pos_df = pos_df[['w_time', 'e_time'] + con_id_list]
        pos_path = f'{self.save_bkt_path}/{self.alpha_name}.pos'
        if os.path.exists(pos_path) and concat:
            old_pos_df = pd.read_csv(pos_path, sep='|')
            old_pos_df.index = old_pos_df['e_time']
            pos_df = old_pos_df.combine_first(pos_df)
            pos_df = pos_df[['w_time', 'e_time'] + sorted(list(set(pos_df.columns) - {'w_time', 'e_time'}))]
        if self.if_save:
            pos_df.to_csv(pos_path, sep='|', index=None)

    def pos_to_exe(self, data_df):
        if not data_df.empty:
            pos_diff = (data_df['position'] - data_df['position'].shift(1).fillna(0))
            pos_diff.iloc[0] = 1

            diff_index = pos_diff.replace(0, np.nan).dropna().index
            pos_change = data_df.loc[diff_index]
            pos_change = pos_change[~pos_change['exe_time'].duplicated('last')]
            pos_df = pos_change.pivot('exe_time', 'con_id', 'position')
            con_id_list = list(pos_df.columns)
            pos_df['e_time'] = pos_df.index.strftime('%Y-%m-%d %H:%M')
            pos_df['w_time'] = datetime.now().strftime('%Y-%m-%d %H:%M')
            pos_df = pos_df[['w_time', 'e_time'] + con_id_list]
            exe_path = f'{self.save_exe_path}/{self.today_str}/{self.alpha_name}.pos'
            if os.path.exists(exe_path):
                old_pos_df = pd.read_csv(exe_path, sep='|')
                old_pos_df.index = old_pos_df['e_time']
                pos_df = old_pos_df.combine_first(pos_df)
                pos_df = pos_df[['w_time', 'e_time'] + sorted(list(set(pos_df.columns) - {'w_time', 'e_time'}))]
            if self.if_save:
                pos_df.to_csv(exe_path, sep='|', index=None)

    def generation(self, fut_name, window, limit, concat=True):
        result_list = []
        pool = Pool(20)
        for con_id, part_info_df in self.active_df[[f'{fut_name}01']].groupby(f'{fut_name}01'):
            # active_begin = self.last_trade_date(part_info_df.index[0]) + timedelta(hours=17)
            # active_end = part_info_df.index[-1] + timedelta(hours=17)
            args = [fut_name, con_id, window, limit]
            result_list.append(pool.apply_async(self.part_generation, args=args))
            # self.part_generation(*args)
        pool.close()
        pool.join()

        pnl_df = pd.concat([res.get()[0] for res in result_list], axis=0)
        turnover_df = pd.concat([res.get()[1] for res in result_list], axis=0)
        data_df = pd.concat([res.get()[2] for res in result_list], axis=0)

        pot = pnl_df.sum() / turnover_df.sum() * 10000
        sp = bt.AZ_Sharpe_y(pnl_df)
        print(fut_name, sp, pot, window, limit)
        if abs(sp) > 1.2 and abs(pot) > 8:
            plt.figure(figsize=[16, 8])
            pnl_df.index = pd.to_datetime(pnl_df.index)
            plt.plot(pnl_df.cumsum())
            plt.grid()
            savfig_send(f'{fut_name} sp:{sp} pot={pot} CCI_window:{window}, CCI_limit:{limit} cut_num={self.cut_num}')

            part_data_df = data_df
            col_name = 'Time'
            raw_return_df = part_data_df['pnl']
            signal_df = part_data_df[col_name].shift(2).replace(np.nan, '00:00')
            SignalAnalysis.CDF_c(signal_df, raw_return_df, hold_time=1,
                                 title=f'{fut_name} {col_name} CDF Figure', lag=0)

            ana_fun(data_df, fut_name, 'test_score')
            ana_fun(data_df, fut_name, 'boll_score')
            ana_fun(data_df, fut_name, 'CCI_score')

            ana_fun(data_df, fut_name, 'trend_indicator')

            ana_fun(data_df, fut_name, 'macd')
            ana_fun(data_df, fut_name, 'RSI')
            ana_fun(data_df, fut_name, 'obv')
            ana_fun(data_df, fut_name, 'atr')
            ana_fun(data_df, fut_name, 'adx')

            ana_fun(data_df, fut_name, 'trix')
            ana_fun(data_df, fut_name, 'willr')

            ana_fun(data_df, fut_name, 'Vol_OI')

        self.pos_to_bkt(data_df, concat)
        return data_df, sp, pot

    def update(self, fut_name, window, limit, update_live_data=True):
        con_id = self.active_df[fut_name + '01'].iloc[-1]
        part_pnl_df, part_turnover, part_data_df = self.part_generation(fut_name, con_id, window, limit,
                                                                        update_live_data)
        self.pos_to_bkt(part_data_df)
        self.pos_to_exe(part_data_df[part_data_df.index > pd.to_datetime(self.last_trd_str) + timedelta(hours=16)])
        return part_pnl_df, part_turnover, part_data_df

    def loop(self, fut_name, window, limit, slp_time=30):
        while datetime.now() < pd.to_datetime(self.today_str + ' ' + '15:30'):
            t1 = time.time()
            self.update(fut_name, window, limit)
            t2 = time.time()
            print(t2 - t1)
            time.sleep(slp_time)

    @staticmethod
    def run(*args, **kwargs):
        return None, None, None

    def part_generation(self, *args):
        return pd.Series(), pd.Series(), pd.Series()


class FutMinPosTest(FutMinPosBase):
    def __init__(self, *args):
        super(FutMinPosTest, self).__init__(*args)
        self.alpha_name = os.path.basename(__file__).split('.')[0]

    def part_generation(self, fut_name, con_id, window, limit, update_live_data=False):
        try:
            old_min_trade_time = self.load_intra_time(con_id)
            part_data_df, active_begin, active_end = self.get_old_new_data(con_id, update_live_data)
            all_min_trade_time = list(old_min_trade_time) + \
                                 list(self.live_time_dict[fut_name][self.live_time_dict[fut_name]
                                                                    > old_min_trade_time[-1]])
            all_min_trade_time_c = pd.Series(all_min_trade_time, index=all_min_trade_time).apply(lambda x: x if
            x.strftime('%H:%M') not in self.end_time_dict[fut_name] else None).fillna(method='bfill').shift(-self.lag)

            part_data_df['exe_time'] = all_min_trade_time_c.reindex(part_data_df.index)
            all_min_df = self.load_intra_data(con_id, ['Close'])
            if update_live_data:
                part_data_df['next_trade_close'] = 0
            else:
                next_trade_close = all_min_df['Close'].shift(-1).loc[part_data_df['exe_time'].values]\
                    .fillna(method='ffill')
                part_data_df['next_trade_close'] = next_trade_close.values

            part_pnl_df, part_turnover, part_data_df = self.run(part_data_df, con_id, window, limit, active_begin)
            return part_pnl_df, part_turnover, part_data_df
        except Exception as error:
            print(error)
            return pd.Series(), pd.Series(), pd.Series()

    @staticmethod
    def run(part_data_df, con_id, window, limit, active_begin=None):
        part_data_df['weekday'] = pd.to_datetime(part_data_df['Date']).dt.weekday
        part_data_df['weekday_pos'] = (
            (part_data_df['weekday'] != 3)
            # & (part_data_df['weekday'] != 3)
            # & (part_data_df['weekday'] != 3)
            # (part_data_df['weekday'] != 4)
        ).astype(int)
        part_data_df['Vol_OI'] = part_data_df['Volume'] / part_data_df['OpenInterest']
        part_data_df['Vol_OI_pos'] = (part_data_df['Vol_OI'] < 0.4).astype(int)

        macd, macdsignal, macdhist = ta.MACD(part_data_df['Close'], 12, 26, 9)
        part_data_df['macd'] = macd
        part_data_df['macd_pos'] = (macd < 25).astype(int)

        macd, macdsignal, macdhist = ta.MACD(part_data_df['Close'], 12, 26, 9)
        part_data_df['macd'] = macd
        part_data_df['macd_pos'] = (macd < 25).astype(int)
        RSI = ta.RSI(part_data_df['Close'], window)
        RSI = RSI - 50
        RSI[RSI > 20] = 20
        RSI[RSI < -20] = -20
        part_data_df['RSI'] = RSI
        part_data_df['RSI_signal'] = Signal.fun_1(part_data_df['RSI'], 1)
        part_data_df['RSI_pos'] = Position.fun_1(part_data_df['RSI_signal'], limit=60)

        # aroondown, aroonup = ta.AROON(part_data_df['High'], part_data_df['Low'], test_window)

        obv = ta.OBV(part_data_df['Close'], part_data_df['Volume'])
        part_data_df['obv'] = obv
        part_data_df['obv_pos'] = (obv < 4).astype(int)

        atr = ta.ATR(part_data_df['High'], part_data_df['Low'], part_data_df['Close'], window)
        part_data_df['atr'] = atr
        part_data_df['atr_pos'] = (atr < 13).astype(int)

        adx = ta.ADX(part_data_df['High'], part_data_df['Low'], part_data_df['Close'], window)
        part_data_df['adx'] = adx

        trix = ta.TRIX(part_data_df['Close'], 60)
        part_data_df['trix'] = trix

        willr = ta.WILLR(part_data_df['High'], part_data_df['Low'], part_data_df['Close'], window)
        part_data_df['willr'] = willr
        part_data_df['willr_pos'] = ((willr < -75) | (willr > -25)).astype(int)

        part_data_df['test_score'] = FutIndex.test_fun(part_data_df['Close'], window,
                                                       cap=5, num=3, return_line=False)
        part_data_df['test_signal'] = Signal.fun_1(part_data_df['test_score'], limit)

        part_data_df['test_pos'] = Position.fun_1(part_data_df['test_signal'])

        part_data_df['boll_score'], ma_n, md_n = FutIndex.boll_fun(part_data_df['Close'], window, return_line=True)
        part_data_df['boll_signal'] = Signal.fun_1(part_data_df['boll_score'], limit)
        part_data_df['boll_pos'] = Position.fun_1(part_data_df['boll_signal'])

        part_data_df['CCI_score'] = FutIndex.CCI_fun(part_data_df['High'],
                                                     part_data_df['Low'],
                                                     part_data_df['Close'], window)

        part_data_df['CCI_signal'] = Signal.fun_1(part_data_df['CCI_score'], limit)

        part_data_df['CCI_pos'] = Position.fun_1(part_data_df['CCI_signal'])

        part_data_df['trend_indicator'] = bt.AZ_Rolling(part_data_df['Close'], 100). \
            apply(lambda x: (x[-1] / x[0] - 1) / (max(x) / min(x) - 1), raw=False)

        part_data_df['trend_pos'] = (part_data_df['trend_indicator'].abs() > 0.25).astype(int)

        part_data_df['signal'] = part_data_df['boll_signal']
        part_data_df['position'] = Position.fun_1(part_data_df['signal']) * part_data_df['weekday_pos']

        part_data_df['position_exe'] = part_data_df['position'].shift(1)
        part_data_df['position_sft'] = part_data_df['position'].shift(2)

        part_data_df['price_return'] = part_data_df['next_trade_close'] / part_data_df['next_trade_close'].shift(1) - 1

        part_data_df['pnl'] = part_data_df['position_exe'] * part_data_df['price_return']

        part_data_df = part_data_df.truncate(before=active_begin)

        part_data_df['turnover'] = part_data_df['position_sft'] - part_data_df['position_sft'].shift(1)

        part_data_df['asset'] = part_data_df['pnl'].cumsum()

        part_data_df['con_id'] = con_id
        part_pnl_df = part_data_df.groupby('Date')['pnl'].sum()
        part_turnover = part_data_df.groupby('Date')['turnover'].apply(lambda x: sum(abs(x)))

        # plt.figure(figsize=[64, 64])
        # ax1 = plt.subplot(4, 1, 1)
        # ax2 = plt.subplot(4, 1, 2)
        # ax3 = plt.subplot(4, 1, 3)
        # ax1.plot(part_data_df['asset'].values)
        # # ax2.plot(part_data_df['Close'].values, '--', color='#75bbfd')
        # ax2.scatter(np.array(range(len(part_data_df.index)))[(part_data_df['position_sft'] > 0)],
        #             part_data_df['Close'][part_data_df['position_sft'] > 0], s=0.5, color='red')
        #
        # ax2.scatter(np.array(range(len(part_data_df.index)))[(part_data_df['position_sft'] == 0)],
        #             part_data_df['Close'][part_data_df['position_sft'] == 0], s=0.5, color='black')
        #
        # ax2.scatter(np.array(range(len(part_data_df.index)))[(part_data_df['position_sft'] < 0)],
        #             part_data_df['Close'][part_data_df['position_sft'] < 0], s=0.5, color='blue')

        # ax2.scatter(np.array(range(len(part_data_df.index)))[(part_data_df['past_min_pct_signal'] > 0)],
        #             part_data_df['Close'][part_data_df['past_min_pct_signal'] > 0], s=20, color='y')
        # ax2.scatter(np.array(range(len(part_data_df.index)))[(part_data_df['past_min_pct_signal'] < 0)],
        #             part_data_df['Close'][part_data_df['past_min_pct_signal'] < 0], s=20, color='#f504c9')

        # ax3.bar(range(len(part_data_df.index)), part_data_df['Volume'].values)
        #
        # ax1.grid()
        # ax2.grid()
        # ax3.grid()
        # plt.title(con_id)
        # savfig_send(con_id)

        return part_pnl_df, part_turnover, part_data_df


if __name__ == '__main__':
    usecols_list = ['High', 'Low', 'Close', 'Volume', 'OpenInterest']
    # fut_name, window, limit, cut_num = 'J', 40, 1, 20
    fut_name_list = ['RB', 'RU', 'M', 'MA', 'BU', 'SC', 'JD', 'CU']
    window_list = [10, 20, 30, 40, 60, 120]
    limit_list = [1, 1.5, 2]
    cut_num_list = [3, 5, 10, 20, 30]
    # for cut_num in cut_num_list:
    #     for window in window_list:
    #         for fut_name in fut_name_list:
    #             for limit in limit_list:
    #                 fut_min_pos = FutMinPosTest([fut_name], usecols_list, cut_num)
    #                 data_df, sp, pot = fut_min_pos.generation(fut_name, window, limit, concat=False)

    # RB 1.7056 10.368174633647946 30 1
    # MA sp:1.2876 pot=24.899168410996474 CCI_window:120, CCI_limit:1
    # RB sp:1.3339 pot=16.364210221856695 CCI_window:60, CCI_limit:1
    # SC sp:2.5136 pot=58.424689192789835 CCI_window:120, CCI_limit:1
    # SC 2.8288 156.99334314823741 120 1.5
    fut_name, window, limit, cut_num = 'RB', 30, 1, 5
    fut_min_pos = FutMinPosTest([fut_name], usecols_list, cut_num)
    data_df, sp, pot = fut_min_pos.generation(fut_name, window, limit, concat=False)