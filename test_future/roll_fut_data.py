import sys

sys.path.append('/mnf/mfs')
from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load import log
from work_whs.loc_lib.pre_load.plt import savfig_send
from work_whs.loc_lib.pre_load.senior_tools import SignalAnalysis
from work_whs.test_future.FutDataLoad import FutData, FutClass


class RollData(FutData):
    def __init__(self, *args):
        super(RollData, self).__init__(*args)
        instrument_list = [
            'CU', 'ZN', 'AL', 'PB', 'AU', 'RB', 'RU', 'WR', 'FU', 'AG', 'BU', 'HC', 'NI', 'SN',
            'CF', 'SR', 'TA', 'WH', 'RI', 'JR', 'FG', 'OI', 'RM', 'RS', 'LR', 'SF', 'SM', 'MA',
            'ZC', 'CY', 'AP', 'A', 'B', 'C', 'J', 'L', 'M', 'P', 'V', 'Y', 'JD', 'JM', 'I',
            'FB', 'BB', 'PP', 'CS', 'SC', 'EG'
        ]
        error_list = [
            'JR', 'WR', 'FU', 'RI', 'LR', 'PB', 'CY', 'RS',
            'OI', 'ZC', 'SM', 'BB', 'FB', 'B',
        ]
        self.instrument_list = sorted(list(set(instrument_list) - set(error_list)))

    def get_adj_factor(self, instrument, raw_act_df):
        close_df = self.load_fut_data(instrument, 'Close')

        a = (close_df * raw_act_df.shift(1)).sum(1).replace(0, np.nan)
        b = (close_df * raw_act_df).sum(1).replace(0, np.nan)

        adj_factor = (a / b).fillna(1)
        adj_factor.name = instrument
        return adj_factor

    def part_run(self, instrument, data_name):
        # openinterest_df = self.load_fut_data(instrument, 'OpenInterest')
        # volume_df = self.load_fut_data(instrument, 'Volume')
        try:
            raw_df = self.load_fut_data(instrument, data_name)

            def fun(x):
                x_s = x.rank(ascending=False).sort_values()
                return x_s.iloc[:1]

            # 找出raw_df里最活跃和次活跃合约
            raw_act_df = raw_df.apply(fun, axis=1)
            # 计算adj factor
            adj_factor = self.get_adj_factor(instrument, raw_act_df)
            # 将raw_active_df处理成 columns=[Date, Contact, active]的格式
            tmp_df = raw_act_df.unstack().reset_index().dropna(how='all', subset=[0])
            tmp_df[0] = instrument + tmp_df[0].astype(int).astype(str).str.zfill(2)
            # index=Date columns=[active01, active02, ...]的格式
            tmp_act_df = tmp_df.pivot(index='Date', columns=0, values='level_0')
            # act_df
            x = tmp_act_df[(tmp_act_df[instrument + '01'].shift(1) < tmp_act_df[instrument + '01'])]
            act_df = x.reindex(tmp_act_df.index).fillna(method='ffill').shift(1)
            return act_df, adj_factor
        except Exception as error:
            print(instrument, error)
            return pd.DataFrame()

    def run(self, data_name):
        result_list = []
        t1 = time.time()
        pool = Pool(20)
        for instrument in self.instrument_list:
            # self.part_run(instrument, data_name)
            result_list.append(pool.apply_async(self.part_run, args=(instrument, data_name)))
        pool.close()
        pool.join()

        result_list = [x.get() for x in result_list]
        target_df = pd.concat([x[0] for x in result_list], axis=1)
        adj_factor_df = pd.concat([x[1] for x in result_list], axis=1)

        target_df.to_csv(f'/mnt/mfs/dat_whs/DAT_FUT/Active_{data_name}', sep='|')
        adj_factor_df.to_csv(f'/mnt/mfs/dat_whs/DAT_FUT/adj_factor_{data_name}', sep='|')
        t2 = time.time()
        print(t2 - t1)
        return target_df, adj_factor_df


if __name__ == '__main__':
    fut_root_path = '/mnt/mfs/DAT_FUT'
    data_name_list = ['OpenInterest', 'Volume']
    roll_data = RollData(fut_root_path)
    for data_name in data_name_list:
        target_df, adj_factor_df = roll_data.run(data_name)
