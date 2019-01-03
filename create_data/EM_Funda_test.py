import sys
sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *
import work_whs.funda_data.funda_data_deal as fdd

BaseDeal = fdd.BaseDeal
FundaBaseDeal = fdd.FundaBaseDeal
SectorData = fdd.SectorData
TechBaseDeal = fdd.TechBaseDeal


class EM_Funda_test_Deal(BaseDeal):
    def __init__(self, sector_df, root_path, para_name, save_root_path, if_replace=False):
        xnms = sector_df.columns
        xinx = sector_df.index
        data = bt.AZ_Load_csv(os.path.join(root_path, 'EM_Funda', 'dat_whs', f'stock_code_df_{para_name}'))

        if if_replace:
            self.raw_df = data.replace(0, np.nan).reindex(columns=xnms, index=xinx)
        else:
            self.raw_df = data.reindex(columns=xnms, index=xinx)
        self.sector_df = sector_df
        self.save_root_path = save_root_path
        self.factor_to_fun = '/mnt/mfs/dat_whs/data/factor_to_fun'
        self.if_replace = if_replace
        self.para_name = para_name
        self.root_path = root_path

    def row_extre_(self, percent):
        target_df = self.row_extre(self.raw_df, self.sector_df, percent)
        # print(target_df)
        file_name = self.para_name + '_row_extre_{}'.format(percent)
        fun = 'EM_Funda_test.EM_Funda_test_Deal.row_extre'
        raw_data_path = (f'EM_Funda/dat_whs/stock_code_df_{self.para_name}',)
        args = (percent,)
        self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args,
                            if_replace=self.if_replace)


def common_deal(sector_df, save_root_path):
    para_name_list = ['tab1_1', 'tab1_2', 'tab1_5', 'tab1_7',
                      'tab1_9', 'tab2_1', 'tab2_4', 'tab2_7', 'tab2_8',
                      'tab2_9', 'tab4_1', 'tab4_2',
                      'tab4_5', 'tab5_13', 'tab5_14', 'tab5_15']

    # para_name_list = ['tab5_14']
    root_path = "/mnt/mfs/DAT_EQT"
    percent = 0.3
    for para_name in para_name_list:
        test_deal = EM_Funda_test_Deal(sector_df, root_path, para_name, save_root_path)
        test_deal.row_extre_(percent)
