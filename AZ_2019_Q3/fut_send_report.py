import sys

sys.path.append('/mnt/mfs')
import string
from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load.plt import plot_send_result
from work_whs.test_future.FutDataLoad import FutData

fut_data = FutData('/mnt/mfs/DAT_FUT')
# data_df = fut_data.load_intra_data('IC1906.CFE', ['Close', 'Volume'])
data = fut_data.load_intra_data('IC1906.CFE', ['Close'])


class FutReport:
    def __init__(self):
        self.fut_pos_bkt_path = '/mnt/mfs/AAFUTPOS/BKT'
        self.fut_pos_exe_path = '/mnt/mfs/AAFUTPOS/EXE'
        # all_pos_file = os.listdir(fut_pos_root_path)
        # all_pos_file = ['RZJ19Q10101.pos']

    def deal_contract(self, part_pos_sr, contract_id):
        close_df = fut_data.load_intra_data(contract_id, ['Close'])
        if close_df is not None:
            diff_sr = close_df.reindex(index=part_pos_sr.index)['Close'].diff()
            part_pnl_sr = part_pos_sr * diff_sr
            part_pnl_sr.name = contract_id
            return part_pnl_sr
        else:
            return pd.Series(index=part_pos_sr.index, name=contract_id)

    def generation(self, begin_time=None, end_time=None):
        all_pos_file = ['RZJ19Q10101.pos']
        pnl_sr_list = []
        for pos_file in all_pos_file[:1]:
            pos_df = bt.AZ_Load_csv(f'{self.fut_pos_bkt_path}/{pos_file}', index_col='e_time').shift(1)
            pos_df.drop(columns='w_time', inplace=True)
            pos_df.dropna(how='all', axis='columns', inplace=True)
            pos_df.truncate(before=begin_time, after=end_time)
            part_pnl_sr_list = []
            pool = Pool(5)
            for contract_id, part_pos_sr in pos_df.items():
                part_pnl_sr_list.append(pool.apply_async(self.deal_contract, (part_pos_sr.dropna(), contract_id)))
                # pnl_df_list.append(self.deal_contract(part_pos_df.dropna(), contract_id))
            pool.close()
            pool.join()
            pnl_table = pd.concat([x.get() for x in part_pnl_sr_list], axis=1)
            pnl_sr = pnl_table.sum(1)
            pnl_sr.name = pos_file
            # pnl_df = pd.concat(pnl_df_list)
            pnl_sr_list.append(pnl_sr)
        pnl_df = pd.concat(pnl_sr_list, axis=1)
        return pnl_df


a = time.time()
pnl_df = FutReport().generation()
b = time.time()
print(b-a)
