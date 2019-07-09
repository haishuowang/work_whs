import sys

sys.path.append('/mnf/mfs')

from work_whs.loc_lib.pre_load import *

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


class FutData:
    def __init__(self, root_path='/mnt/mfs/DAT_FUT'):
        self.root_path = root_path
        self.act_info_df = bt.AZ_Load_csv(f'{root_path}/DailyPX/Contract').shift(1)

    def load_fut_data(self, fut_name, file_name):
        raw_df = bt.AZ_Load_csv(f'{self.root_path}/day/{fut_name}/{file_name}')
        return raw_df

    # def load_act_fut_data(self, fut_name, file_name):
    #     raw_df = self.load_fut_data(fut_name, file_name)
    #
    #     target_df = raw_df.apply(lambda x, y: x * (y == x.name), args=(act_info_sr,))
    #     target_df = target_df.replace(0, np.nan).dropna(how='all', axis='columns')
    #     target_sr = target_df.sum(1)
    #     target_sr.name = f'{fut_name}01'
    #     return target_sr

    def load_act_fut_data_r(self, fut_name, file_name, act_info_df):
        raw_df = self.load_fut_data(fut_name, file_name)
        act_info_sr = act_info_df[f'{fut_name}01'].reindex(index=raw_df.index)
        target_df = raw_df.apply(lambda x, y: x * (y == x.name), args=(act_info_sr,))
        target_df = target_df.replace(0, np.nan).dropna(how='all', axis='columns')
        target_sr = target_df.sum(1).replace(0, np.nan)
        target_sr.name = f'{fut_name}01'
        return target_sr

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

    def load_intra_data(self, contract_id, usecols_list):
        fut_name = re.sub('\d', '', contract_id.split('.')[0])
        load_path = f'{self.root_path}/intraday/fut_1mbar/{fut_name}/{contract_id}'
        if os.path.exists(load_path):
            data = bt.AZ_Load_csv(f'{self.root_path}/intraday/fut_1mbar/{fut_name}/{contract_id}',
                                  usecols=['TradeDate', 'Date', 'Time'] + usecols_list)
            return data
        else:
            return None

    def load_act_intra_data(self, fut_name, file_name):
        pass

    @staticmethod
    def load_spot_data_wind(path_name, file_name):
        # raw_df = pd.read_csv(f'/mnt/mfs/dat_whs/{path_name}/{file_name}.csv')
        raw_df = bt.AZ_Load_csv(f'/mnt/mfs/dat_whs/{path_name}/{file_name}.csv', sep=',').fillna(method='ffill')
        return raw_df


if __name__ == '__main__':
    fut_data = FutData('/mnt/mfs/DAT_FUT')
    data_df = fut_data.load_intra_data('IC1906.CFE', ['Close', 'Volume'])
    path_name = '能源_动力煤'
    file_name = '国内外煤炭库存'
    data = fut_data.load_spot_data_wind(path_name, file_name)
    print(data_df)
