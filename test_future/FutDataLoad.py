import sys

sys.path.append('/mnf/mfs')

from work_dmgr_fut.loc_lib.pre_load import *
from work_dmgr_fut.loc_lib.pre_load.plt import savfig_send

root_path = '/mnt/mfs/DAT_FUT'

FutClass = dict({
    '黑色': ['RB', 'HC', 'J', 'JM', 'ZC', 'I', 'WR', 'SF', 'SM'],
    '化工': [
        # 'RU', 'FU', 'L', 'V', 'J', 'TA', 'BU', 'SC', 'MA',
        'FG', 'PP', 'FB', 'BB', 'EG'],
    '有色': ['AU', 'AG', 'CU', 'PB', 'ZN', 'SN', 'NI', 'AL'],
    '农产品': ['OI', 'RS', 'RM', 'WH', 'JR', 'SR', 'CF', 'RI', 'LR', 'CY', 'AP', 'P', 'B', 'M', 'JD',
            'Y', 'C', 'A', 'CS'],
    '金融': ['IF', 'IH', 'IC', 'T', 'TF', 'TS']
})


def get_active_df(data_name):
    root_path = '/mnt/mfs/dat_whs/DAT_FUT'
    active_df = bt.AZ_Load_csv(f'{root_path}/Active_{data_name}')
    adj_factor_df = bt.AZ_Load_csv(f'{root_path}/adj_factor_{data_name}')
    aadj_factor_df = adj_factor_df.cumprod()
    return active_df, aadj_factor_df


class FutData:
    def __init__(self, root_path='/mnt/mfs/DAT_FUT'):
        self.root_path = root_path
        self.reshape_path = '/mnt/mfs/dat_whs/DAT_FUT'
        self.act_info_df = bt.AZ_Load_csv(f'{root_path}/DailyPX/Contract').shift(1)
        self.trade_date = bt.AZ_Load_csv(f'{root_path}/DailyPX/TradeDates').index
        self.active_trading = bt.AZ_Load_csv(f'{root_path}/T_Calendar/Active_Trading')
        #
        # active_trading = active_trading[(active_trading.index.strftime('%H:%M') != '21:00') &
        #                                 (active_trading.index.strftime('%H:%M') != '09:00')
        #                                 ]
        # self.active_trading = active_trading

    def last_trade_date(self, raw_date):
        return self.trade_date[self.trade_date < raw_date][-1]

    def next_trade_date(self, raw_date):
        return self.trade_date[self.trade_date > raw_date][0]

    def load_fut_data(self, fut_name, file_name):
        raw_df = bt.AZ_Load_csv(f'{self.root_path}/day/{fut_name}/{file_name}')
        return raw_df

    def load_act_fut_data(self, fut_name, file_name):
        raw_df = self.load_fut_data(fut_name, file_name)
        act_info_sr = self.act_info_df[f'{fut_name}01']
        target_df = raw_df.apply(lambda x, y: x * (y == x.name), args=(act_info_sr,))
        target_df = target_df.replace(0, np.nan).dropna(how='all', axis='columns')
        target_sr = target_df.sum(1)
        target_sr.name = f'{fut_name}01'
        return target_sr

    def load_act_fut_data_r(self, fut_name, file_name, act_info_df, active_num='01'):
        raw_df = self.load_fut_data(fut_name, file_name)
        act_info_sr = act_info_df[f'{fut_name}{active_num}'].reindex(index=raw_df.index)
        target_df = raw_df.apply(lambda x, y: x * (y == x.name), args=(act_info_sr,))
        target_df = target_df.replace(0, np.nan).dropna(how='all', axis='columns')
        target_sr = target_df.sum(1).replace(0, np.nan)
        target_sr.name = f'{fut_name}01'
        return target_sr

    def load_act_intra_data(self, fut_name, usecols_list, active_df):
        result_list = []
        for con_id, part_info_df in active_df[[f'{fut_name}01']].groupby(f'{fut_name}01'):

            active_begin = self.last_trade_date(part_info_df.index[0]) + timedelta(hours=17)
            active_end = part_info_df.index[-1] + timedelta(hours=17)
            print(con_id, active_begin, active_end)
            target_intra = self.load_intra_data(con_id, usecols_list)
            target_intra['con_id'] = con_id
            result_list.append(target_intra.truncate(active_begin, active_end))
        return pd.concat(result_list, axis=0)

    def load_act_adj_fut_data(self, fut_name, file_name, act_info_df, aadj_factor_df):
        target_sr = self.load_act_fut_data_r(fut_name, file_name, act_info_df)
        adj_target_sr = target_sr * aadj_factor_df[fut_name]

        return adj_target_sr

    def load_contract_data(self, contract_id, file_name):
        fut_name = re.sub('\d', '', contract_id.split('.')[0])
        raw_df = self.load_fut_data(fut_name, file_name)
        target_df = raw_df[[contract_id]]
        return target_df

    def load_spot_data(self, fut_name, file_name):
        """
        仓单库存, 方坯, 现货价格
        :param fut_name:
        :param file_name:
        :return:
        """
        print(f'{self.root_path}/spot/{fut_name}/{file_name}.csv')
        raw_df = bt.AZ_Load_csv(f'{self.root_path}/spot/{fut_name}/{file_name}.csv').fillna(method='ffill')
        return raw_df

    def load_intra_time(self, contract_id):
        fut_name = re.sub('\d', '', contract_id.split('.')[0])
        load_path = f'{self.root_path}/intraday/fut_1mbar/{fut_name}/{contract_id}'

        if os.path.exists(load_path):
            data = bt.AZ_Load_csv(load_path, usecols=['TradeDate', 'Date', 'Time'])
            # active_trading_cut = active_trading_fut[active_trading_fut > data.index[-1]]
            return np.array(list(data.index))
        else:
            return None

    def load_intra_data(self, contract_id, usecols_list):
        fut_name = re.sub('\d', '', contract_id.split('.')[0])
        load_path = f'{self.root_path}/intraday/fut_1mbar/{fut_name}/{contract_id}'
        if os.path.exists(load_path):
            data = bt.AZ_Load_csv(f'{self.root_path}/intraday/fut_1mbar/{fut_name}/{contract_id}',
                                  usecols=['TradeDate', 'Date', 'Time'] + usecols_list)
            return data
        else:
            return pd.DataFrame()

    def load_intra_reshape_data(self, contract_id, usecols_list, cut_num=5):
        fut_name = re.sub('\d', '', contract_id.split('.')[0])
        load_path = f'{self.reshape_path}/intraday/fut_{cut_num}mvolbar/{fut_name}/{contract_id}'
        if os.path.exists(load_path):
            data = bt.AZ_Load_csv(load_path, usecols=['TradeDate', 'Date', 'Time'] + usecols_list)
            return data
        else:
            return pd.DataFrame()

    @staticmethod
    def load_spot_data_wind(path_name, file_name):
        # raw_df = pd.read_csv(f'/mnt/mfs/dat_whs/{path_name}/{file_name}.csv')
        raw_df = bt.AZ_Load_csv(f'/mnt/mfs/DAT_FUT/spot/{path_name}/{file_name}.csv', sep='|').fillna(method='ffill')
        return raw_df


class ReshapeData:
    def __init__(self, trade_date, daily_end_time):
        self.daily_end_time = pd.to_datetime(trade_date + ' ' + daily_end_time)

    # @staticmethod
    def run(self, part_data, cut_num):
        part_data_copy = part_data.copy('deep')

        cut_vol_num = sum(part_data['Volume'].iloc[:cut_num]) - 0.01
        part_data_copy['Volume'] = part_data_copy['Volume'].cumsum()

        if cut_vol_num != -0.01:
            # print(cut_vol_num)
            print(part_data['Date'].iloc[0], cut_vol_num)
            vol_cum = part_data['Volume'].cumsum()

            cut_vol_sr = vol_cum.apply(
                lambda x: int(x / cut_vol_num) if x % cut_vol_num == 0 else int(x / cut_vol_num) + 1)
            if datetime.now() > self.daily_end_time:
                cut_vol_sr = cut_vol_sr.shift(1).fillna(1)
                cut_vol_df = pd.DataFrame(cut_vol_sr)

                target_index = cut_vol_df.groupby(by=['Volume']).apply(lambda x: x.index[-1]).values
            else:
                target_index = (cut_vol_sr - cut_vol_sr.shift(1).fillna(1)).replace(0, np.nan).dropna().index

            target_index = sorted(target_index)
            target_df = part_data_copy.loc[target_index]
            # print(target_df['Volume'])

            target_df['Volume'] = target_df['Volume'] - target_df['Volume'].shift(1).fillna(0)
            return target_df
        else:
            return pd.Series()

    def deal_contract(self, con_df, cut_num):
        reshape_mul_df = con_df.groupby(by=['Date']).apply(self.run, cut_num)
        target_index = reshape_mul_df.index.droplevel('Date')
        reshape_df = reshape_mul_df.set_index(target_index)
        return reshape_df


def save_data_fun(fut_name):
    # j_intra_data = fut_data.load_act_intra_data('J', ['Open', 'High', 'Low', 'Close', 'Volume', 'OpenInterest'],
    #                                             active_df)
    # jm_intra_data = fut_data.load_act_intra_data('JM', ['Open', 'High', 'Low', 'Close', 'Volume', 'OpenInterest'],
    #                                              active_df)
    # j_intra_data.to_csv('/mnt/mfs/temp/j_intra_data.csv', sep='|')
    # jm_intra_data.to_csv('/mnt/mfs/temp/jm_intra_data.csv', sep='|')

    intra_data = fut_data.load_act_intra_data(fut_name, ['Open', 'High', 'Low', 'Close', 'Volume', 'OpenInterest'],
                                                active_df)
    intra_data.to_csv(f'/mnt/mfs/temp/{fut_name.lower()}_intra_data.csv', sep='|')


if __name__ == '__main__':
    active_df, aadj_factor_df = get_active_df('Volume')
    fut_data = FutData()
    # a = fut_data.load_intra_time('J1909.DCE')
    save_data_fun('HC')
    save_data_fun('RB')
    save_data_fun('I')
    pass
