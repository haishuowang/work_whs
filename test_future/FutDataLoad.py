import sys

sys.path.append('/mnf/mfs')

from work_whs.loc_lib.pre_load import *

root_path = '/mnt/mfs/DAT_FUT'


class FutData:
    def __init__(self, root_path):
        self.root_path = root_path
        self.act_info_df = bt.AZ_Load_csv(f'{root_path}/day/DailyPX/Contract')

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
        fut_name = re.sub('\d', '', contract_id)
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

    # def load_act_intra_data(self):

    # def load_intra_data(self):
