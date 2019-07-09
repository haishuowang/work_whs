import sys

sys.path.append('/mnf/mfs')
from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load import log
from work_whs.loc_lib.pre_load.plt import savfig_send
from work_whs.loc_lib.pre_load.senior_tools import SignalAnalysis
from work_whs.test_future.FutDataLoad import FutData
from work_whs.test_future.signal_fut_fun import FutIndex, Signal, Position


def fun(x, begin_p, end_p):
    # 开盘的价格
    begin_price = x.at_time(begin_p)
    # 在14:00的价格
    end_price = x.at_time(end_p)
    if len(end_price) != 0:
        return end_price.iloc[0] / begin_price.iloc[0] - 1
    else:
        return None


fut_data = FutData()


class DataCreate:
    def __init__(self, begin_p, end_p, trade_p):
        self.tmp_path = '/mnt/mfs/dat_whs/tmp'
        self.script_name = os.path.basename(__file__).split('.')[0]
        self.save_root_path = f'{self.tmp_path}/{self.script_name}'
        bt.AZ_Path_create(self.save_root_path)
        self.begin_p = begin_p
        self.end_p = end_p
        self.trade_p = trade_p

    def deal_fut(self, fut_name):
        result_part_signal_list = []
        result_part_end_end_list = []
        for con_id, part_info_df in fut_data.act_info_df[[f'{fut_name}01']].groupby(f'{fut_name}01'):
            begin_time = part_info_df.index[0] - timedelta(1)
            end_time = part_info_df.index[-1] + timedelta(1)
            print(con_id, begin_time, end_time)
            data_df = fut_data.load_intra_data(con_id, ['Close'])

            if data_df is None:
                continue

            part_data_df = data_df.truncate(before=begin_time, after=end_time)

            # part_return_signal = part_data_df.groupby(by=['Date'])['Close'] \
            #     .apply(fun, args=(self.begin_p, self.end_p,)).dropna()

            begin_time_df = part_data_df['Close'].at_time(self.begin_p).dropna()
            begin_time_df.index = begin_time_df.index.strftime('%Y-%m-%d')
            end_time_df = part_data_df['Close'].at_time(self.end_p).dropna()
            end_time_df.index = end_time_df.index.strftime('%Y-%m-%d')

            part_return_signal = end_time_df / begin_time_df - 1

            trade_time_df = part_data_df['Close'].at_time(self.trade_p).dropna()

            part_return_end_end = (trade_time_df / trade_time_df.shift(1) - 1)
            part_return_end_end.index = part_return_end_end.index.strftime('%Y-%m-%d')

            result_part_signal_list.append(part_return_signal)
            result_part_end_end_list.append(part_return_end_end)

        return_signal = pd.concat(result_part_signal_list, axis=0)
        return_signal = return_signal[~return_signal.index.duplicated(keep='last')]
        return_signal.name = fut_name

        return_end_end = pd.concat(result_part_end_end_list, axis=0)
        return_end_end = return_end_end[~return_end_end.index.duplicated(keep='last')]
        return_end_end.name = fut_name

        print(f'__________________{fut_name}__________________')

        len_signal = len(return_signal.index)
        len_end_end = len(return_end_end.index)
        print(len_signal, len_end_end)
        if len_signal != len_end_end:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        return return_signal, return_end_end
        # except Exception as error:
        #     print(error)

    def run(self, fut_class):
        result_signal_list = []
        result_end_end_list = []

        for class_name, [fut_list, weight_list] in fut_class.items():
            weight_list = np.array(weight_list) / sum(np.array(weight_list))
            fut_result_signal_list = []
            fut_result_end_end_list = []
            for fut_name, weight in zip(fut_list, weight_list):
                fut_return_signal, fut_return_end_end = self.deal_fut(fut_name)

                fut_return_signal = fut_return_signal * weight
                fut_return_end_end = fut_return_end_end * weight

                fut_return_signal.to_csv(f'{self.save_root_path}/{fut_name}_signal.csv', sep="|")
                fut_return_end_end.to_csv(f'{self.save_root_path}/{fut_name}_end_end.csv', sep="|")

                fut_result_signal_list.append(fut_return_signal)
                fut_result_end_end_list.append(fut_return_end_end)

            class_return_signal = pd.concat(fut_result_signal_list, axis=1, sort=True).mean(1)
            class_return_signal.name = class_name

            class_return_end_end = pd.concat(fut_result_end_end_list, axis=1, sort=True).mean(1)
            class_return_end_end.name = class_name

            class_return_signal.to_csv(f'{self.save_root_path}/{class_name}_signal.csv', sep="|")
            class_return_end_end.to_csv(f'{self.save_root_path}/{class_name}_end_end.csv', sep="|")

            result_signal_list.append(class_return_signal)
            result_end_end_list.append(class_return_end_end)

        return_signal = pd.concat(result_signal_list, axis=1, sort=True)
        return_end_end = pd.concat(result_end_end_list, axis=1, sort=True)
        return return_signal, return_end_end

        # '黑色': ['RB', 'HC', 'J', 'JM', 'ZC', 'I', 'WR', 'SF', 'SM'],
        # '化工': [
        #     'RU', 'FU', 'L', 'V', 'J', 'TA', 'BU', 'SC', 'MA',
        #     'FG', 'PP', 'FB', 'BB', 'EG'],
        # '有色': ['AU', 'AG', 'CU', 'PB', 'ZN', 'SN', 'NI', 'AL'],
        # '农产品': ['OI', 'RS', 'RM', 'WH', 'JR', 'SR', 'CF', 'RI', 'LR', 'CY', 'AP', 'P', 'B', 'M', 'JD',
        #           'Y', 'C', 'A', 'CS'],
        # '金融': ['IF', 'IH', 'IC', 'T', 'TF', 'TS']


if __name__ == '__main__':
    a = time.time()

    fut_class_list = [
        OrderedDict({
            'black': [['RB', 'HC', 'J', 'I'], [1, 0, 0, 0]],
            'chem': [['RU', 'BU', 'MA'], [1, 0, 0]],
            'metal': [['CU', 'ZN', 'NI', 'AL'], [1, 0, 0, 0]],
            'precious': [['AU', 'AG'], [1, 0]],
            'agro1': [['C', 'AP', 'JD', 'OI', 'P'], [1, 0, 0, 0, 0]],
            'agro2': [['M', 'RM', 'SR', 'CF'], [1, 0, 0, 0]],
            'eq': [['IF'], [1]],
            'bond': [['T'], [1]],
        }),
        OrderedDict({
            'black': [['RB', 'HC', 'J', 'I'], [1, np.nan, np.nan, np.nan]],
            'chem': [['RU', 'BU', 'MA'], [1, np.nan, np.nan]],
            'metal': [['CU', 'ZN', 'NI', 'AL'], [1, np.nan, np.nan, np.nan]],
            'precious': [['AU', 'AG'], [1, np.nan]],
            'agro1': [['C', 'AP', 'JD', 'OI', 'P'], [1, np.nan, np.nan, np.nan, np.nan]],
            'agro2': [['M', 'RM', 'SR', 'CF'], [1, np.nan, np.nan, np.nan]],
            'eq': [['IF'], [1]],
            'bond': [['T'], [1]],
        }),

    ]

    para_list = [
        ['09:35', '11:00', '11:05'],
        ['09:35', '13:00', '13:05'],
        ['09:35', '13:30', '13:35'],
        ['09:35', '14:00', '14:05'],
        ['09:35', '14:30', '14:35'],

        ['10:00', '11:00', '11:05'],
        ['10:00', '13:00', '13:05'],
        ['10:00', '13:30', '13:35'],
        ['10:00', '14:00', '14:05'],
        ['10:00', '14:30', '14:35'],

        ['11:00', '13:00', '13:05'],
        ['11:00', '13:30', '13:35'],
        ['11:00', '14:00', '14:05'],
        ['11:00', '14:30', '14:35'],

        ['11:30', '13:30', '13:35'],
        ['11:30', '14:00', '14:05'],
        ['11:30', '14:30', '14:35'],
    ]
    for fut_class in fut_class_list:
        for begin_p, end_p, trade_p in para_list:
            return_signal, return_end_end = DataCreate(begin_p, end_p, trade_p).run(fut_class)
            b = time.time()
            print(b - a)
            # target_dict

            vol_signal = bt.AZ_Rolling(return_signal, n=20).std()
            vol_end_end = bt.AZ_Rolling(return_end_end, n=20).std()

            factor_1 = return_signal / vol_signal
            factor_1_m = factor_1.sub(factor_1.mean(1), axis=0)

            factor_2 = return_end_end / vol_end_end
            factor_2_m = factor_2.sub(factor_2.mean(1), axis=0)

            weight_pos_df_1 = factor_1_m
            weight_pos_df_2 = factor_2_m

            pnl_df_1 = (weight_pos_df_1.shift(1) * return_end_end).sum(1)
            pnl_df_2 = (weight_pos_df_2.shift(1) * return_end_end).sum(1)

            pot_1 = pnl_df_1.sum() / weight_pos_df_1.diff().abs().sum().sum() * 10000
            pot_2 = pnl_df_2.sum() / weight_pos_df_2.diff().abs().sum().sum() * 10000

            plt.plot(pnl_df_1.cumsum().values, label=round(pot_1, 4))
            plt.plot(pnl_df_2.cumsum().values, label=round(pot_2, 4))
            plt.legend()
            savfig_send(subject=f'test_1 , {begin_p}, {end_p}, {trade_p} pot_1:{pot_1}, pot_2:{pot_2}')
