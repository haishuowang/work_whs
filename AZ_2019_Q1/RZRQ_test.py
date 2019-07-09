import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load.plt import plot_send_result, savfig_send
from work_whs.bkt_factor_create.new_framework_test_1 import FactorTestBase
from work_whs.bkt_factor_create.base_fun_import import DiscreteClass, ContinueClass, SectorData
from multiprocessing import Lock


class FactorTest(FactorTestBase):
    def load_rzrq_data(self, file_name):
        data_path = f'{self.root_path}/EM_Funda/TRAD_MT_MARGIN/{file_name}'
        data = bt.AZ_Load_csv(data_path)
        return data

    def back_test_c(self, data, file_name, add_info):
        data_c = data * self.sector_df
        a = data_c.rank(axis=1, ascending=False)
        signal_df = (a <= 20).astype(int)
        print(signal_df.iloc[-10].replace(0, np.nan).dropna())
        self.pos_daily_fun(signal_df)
        pos_df = self.signal_to_pos_ls(signal_df, 'l')
        pnl_df = (pos_df.shift(2) * self.return_df).sum(1)
        plt.figure(figsize=[16, 10])
        plt.plot(pnl_df.cumsum())
        savfig_send(subject=f'{add_info}|{file_name[:-4]}|{self.sector_name}|{self.if_only_long}'
                            f'|{self.hold_time}|{bt.AZ_Sharpe_y(pnl_df)}')


turnover = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TVALCNY.csv')
turnover_20 = bt.AZ_Rolling_mean(turnover, 20)
turnover_10 = bt.AZ_Rolling_mean(turnover, 10)

market = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/LICO_YS_STOCKVALUE/AmarketCapExStri.csv')
market_20 = bt.AZ_Rolling_mean(market, 20)
market_10 = bt.AZ_Rolling_mean(market, 10)


def part_main_fun(if_only_long, hold_time, sector_name):
    root_path = '/mnt/mfs/DAT_EQT'
    if_save = True
    if_new_program = True

    begin_date = pd.to_datetime('20130101')
    end_date = pd.to_datetime('20190411')
    end_date = datetime.now()
    lag = 2
    return_file = ''

    if_hedge = True

    factor_test = FactorTest(root_path, if_save, if_new_program, begin_date, end_date, sector_name, hold_time,
                             lag, return_file, if_hedge, if_only_long)

    file_name_list = ['RQCHL.csv', 'RQMCL.csv', 'RQYE.csv', 'RQYL.csv', 'RZCHE.csv', 'RZMRE.csv',
                      'RZRQYE.csv', 'RZYE.csv']
    for file_name in file_name_list:
        print(file_name)
        data = factor_test.load_rzrq_data(file_name).astype(float)
        data_t10 = data / turnover_10
        data_t20 = data / turnover_20
        data_m10 = data / market_10
        data_m20 = data / market_20

        factor_test.back_test_c(data, file_name, 'data')
        factor_test.back_test_c(data_t10, file_name, 'data_t10')
        factor_test.back_test_c(data_t20, file_name, 'data_t20')
        factor_test.back_test_c(data_m10, file_name, 'data_m10')
        factor_test.back_test_c(data_m20, file_name, 'data_m20')


def main_fun():
    sector_name_list = [
        'index_000300',
        'index_000905',
        # 'market_top_300plus',
        # 'market_top_300plus_industry_10_15',
        # 'market_top_300plus_industry_20_25_30_35',
        # 'market_top_300plus_industry_40',
        # 'market_top_300plus_industry_45_50',
        # 'market_top_300plus_industry_55',
        #
        # 'market_top_300to800plus',
        # 'market_top_300to800plus_industry_10_15',
        # 'market_top_300to800plus_industry_20_25_30_35',
        # 'market_top_300to800plus_industry_40',
        # 'market_top_300to800plus_industry_45_50',
        # 'market_top_300to800plus_industry_55'
    ]
    pool = Pool(20)
    hold_time_list = [5, 10, 20, 30]
    for if_only_long in [True]:
        for sector_name in sector_name_list:
            # for percent in [0.1, 0.2]:
            for hold_time in hold_time_list:
                pool.apply_async(part_main_fun, (if_only_long, hold_time, sector_name,))
    pool.close()
    pool.join()


if __name__ == '__main__':
    main_fun()
