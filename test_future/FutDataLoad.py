import sys

sys.path.append('/mnf/mfs')

from work_whs.loc_lib.pre_load import *

root_path = '/mnt/mfs/DAT_FUT'

FutClass = dict({
    '黑色': ['RB', 'HC', 'J', 'JM', 'ZC', 'I', 'WR', 'SF', 'SM'],
    '化工': ['RU', 'FU', 'L', 'V', 'J', 'TA', 'BU', 'SC', 'MA', 'FG', 'PP', 'FB', 'BB', 'EG'],
    '有色': ['AU', 'AG', 'CU', 'PB', 'ZN', 'SN', 'NI', 'AL'],
    '农产品': ['OI', 'RS', 'RM', 'WH', 'JR', 'SR', 'CF', 'RI', 'LR', 'CY', 'AP', 'P', 'B', 'M', 'JD',
            'Y', 'C', 'A', 'CS'],
    '金融': ['IF', 'IH', 'IC', 'T', 'TF', 'TS']
})


class FutData:
    def __init__(self, root_path):
        self.root_path = root_path
        self.act_info_df = bt.AZ_Load_csv(f'{root_path}/DailyPX/Contract')

    def load_fut_data(self, fut_name, file_name):
        raw_df = bt.AZ_Load_csv(f'{self.root_path}/day/union/{fut_name}/{file_name}')
        return raw_df

    def load_act_fut_data(self, fut_name, file_name):
        raw_df = self.load_fut_data(fut_name, file_name)
        act_info_sr = self.act_info_df[f'{fut_name}01'].reindex(index=raw_df.index)
        target_df = raw_df.apply(lambda x, y: x * (y == x.name), args=(act_info_sr,))
        target_df = target_df.replace(0, np.nan).dropna(how='all', axis='columns')
        target_sr = target_df.sum(1)
        target_sr.name = f'{fut_name}01'
        return target_sr

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
        raw_df = bt.AZ_Load_csv(f'{self.root_path}/Inventory/{fut_name}/{file_name}.csv')
        return raw_df

    def load_intra_data(self, contract_id, usecols_list):
        fut_name = re.sub('\d', '', contract_id.split('.')[0])
        data = bt.AZ_Load_csv(f'{self.root_path}/intraday/fut_1mbar/{fut_name}/{contract_id}',
                              usecols=['TradeDate', 'Date', 'Time'] + usecols_list)
        return data

    def load_act_intra_data(self, fut_name, file_name):
        pass


if __name__ == '__main__':
    fut_data = FutData('/mnt/mfs/DAT_FUT')
    data_df = fut_data.load_intra_data('IC1906.CFE', ['Close', 'Volume'])
    print(data_df)
