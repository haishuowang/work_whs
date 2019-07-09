import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *
from work_whs.bkt_factor_create.base_fun_import import DiscreteClass, ContinueClass
from work_whs.bkt_factor_create.raw_data_path import base_data_dict


class SectorData:
    def __init__(self, root_path):
        self.root_path = root_path

    # 获取剔除新股的矩阵
    def get_new_stock_info(self, xnms, xinx):
        new_stock_data = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/CDSY_SECUCODE/LISTSTATE.csv')
        new_stock_data.fillna(method='ffill', inplace=True)
        # 获取交易日信息
        return_df = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/DERIVED_14/aadj_r.csv').astype(float)
        trade_time = return_df.index
        new_stock_data = new_stock_data.reindex(index=trade_time).fillna(method='ffill')
        target_df = new_stock_data.shift(40).notnull().astype(int)
        target_df = target_df.reindex(columns=xnms, index=xinx)
        return target_df

    # 获取剔除st股票的矩阵
    def get_st_stock_info(self, xnms, xinx):
        data = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/CDSY_CHANGEINFO/CHANGEA.csv')
        data = data.reindex(columns=xnms, index=xinx)
        data.fillna(method='ffill', inplace=True)

        data = data.astype(str)
        target_df = data.applymap(lambda x: 0 if 'ST' in x or 'PT' in x else 1)
        return target_df

    # 读取 sector(行业 最大市值等)
    def load_sector_data(self, begin_date, end_date, sector_name):
        if sector_name.startswith('index'):
            index_name = sector_name.split('_')[-1]
            market_top_n = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/IDEX_YS_WEIGHT_A/SECURITYNAME_{index_name}.csv')
            market_top_n = market_top_n.where(market_top_n != market_top_n, other=1)
        else:
            market_top_n = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/DERIVED_10/{sector_name}.csv')

        market_top_n = market_top_n[(market_top_n.index >= begin_date) & (market_top_n.index < end_date)]
        market_top_n.dropna(how='all', axis='columns', inplace=True)

        xnms = market_top_n.columns
        xinx = market_top_n.index

        new_stock_df = self.get_new_stock_info(xnms, xinx)
        st_stock_df = self.get_st_stock_info(xnms, xinx)
        sector_df = market_top_n * new_stock_df * st_stock_df
        sector_df.replace(0, np.nan, inplace=True)
        return sector_df


class DataDeal(SectorData, DiscreteClass, ContinueClass):
    def __init__(self, begin_date, end_date, root_path, sector_name):
        super(DataDeal, self).__init__(root_path=root_path)
        # self.root_path = root_path
        self.sector_name = sector_name
        self.sector_df = self.load_sector_data(begin_date, end_date, sector_name)

        self.xinx = self.sector_df.index
        self.xnms = self.sector_df.columns

        self.save_root_path = '/mnt/mfs/dat_whs/data/factor_data'
        self.save_sector_path = f'{self.save_root_path}/{self.sector_name}'
        bt.AZ_Path_create(self.save_sector_path)

    def count_save_data(self, file_name, fun_name, raw_df, *para):
        """
        计算并存储数据
        :param file_name:
        :param fun_name:
        :param raw_df:
        :param para:
        :return:
        """
        fun = getattr(self, fun_name)
        target_df = fun(raw_df, self.sector_df, *para)
        if len(para) != 0:
            para_str = '_'.join([str(x) for x in para])
            save_file = f'{file_name}|{fun_name}|{para_str}'
        else:
            save_file = f'{file_name}|{fun_name}'
        if save_file == 'R_COMMPAY_QTTM|pnd_vol|20':
            print(1)
        save_path = f'{self.save_sector_path}/{save_file}.pkl'
        target_df.to_pickle(save_path)
        return target_df

    def count_return_data(self, factor_name):
        if len(factor_name.split('|')) == 3:
            str_to_num = lambda x: float(x) if '.' in x else int(x)
            file_name, fun_name, para_str = factor_name.split('|')
            para = [str_to_num(x) for x in para_str.split('_')]
        else:
            file_name, fun_name = factor_name.split('|')
            para = []

        raw_df = self.load_raw_data(file_name)
        fun = getattr(self, fun_name)
        target_df = fun(raw_df, self.sector_df, *para)
        # target_zscore_df = self.row_zscore(target_df, self.sector_df)
        return target_df

    def load_raw_data(self, file_name):
        data_path = base_data_dict[file_name]
        if len(data_path) != 0:
            raw_df = bt.AZ_Load_csv(f'{self.root_path}/{data_path}') \
                .reindex(index=self.xinx, columns=self.xnms).round(4)
        else:
            raw_df = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/daily/{file_name}.csv') \
                .reindex(index=self.xinx, columns=self.xnms).round(4)
        return raw_df

    def fun_info_dict_deal(self, data_name, raw_df, fun_info_dict):
        """
        传入 参数 和 fun_info_dict
        :param data_name: 数据名称
        :param raw_df: 数据
        :param fun_info_dict:函数信息
        :return:
        """
        for fun_name in fun_info_dict.keys():
            para_list = fun_info_dict[fun_name]
            if para_list is None:
                self.count_save_data(data_name, fun_name, raw_df)
            else:
                for para in para_list:
                    self.count_save_data(data_name, fun_name, raw_df, *para)

    def part_common_fun(self, data_name):
        """
        输入数据名 用通用方式处理数据
        :param data_name:
        :return:
        """
        try:
            print(data_name)
            raw_df = self.load_raw_data(data_name)
            fun_info_dict = OrderedDict({
                'row_zscore': None,
                'col_zscore': [[5], [20], [60], [120]],
                'pnd_vol': [[5], [20], [60], [120]],
            })
            self.fun_info_dict_deal(data_name, raw_df, fun_info_dict)
        except Exception as error:
            print(error)

    def common_fun(self, data_name_list, cpu_num=28):
        pool = Pool(cpu_num)
        for data_name in data_name_list:
            # self.part_common_fun(data_name)
            pool.apply_async(self.part_common_fun, (data_name,))
        pool.close()
        pool.join()

    def run(self, info_dict):
        """
        info_dict = {data_name:{fun_name: para_list}}
        :param info_dict:
        :return:
        """
        for data_name in info_dict.keys():
            raw_df = self.load_raw_data(data_name)
            fun_info_dict = info_dict[data_name]
            self.fun_info_dict_deal(data_name, raw_df, fun_info_dict)


def check_fun(data_1, data_2):
    a = data_1.loc[pd.to_datetime('20140101'):pd.to_datetime('20190101')]
    b = data_2.loc[pd.to_datetime('20140101'):pd.to_datetime('20190101')]
    c = (a.replace(np.nan, 0) - b.replace(np.nan, 0)).abs().sum().sum()
    d = (a.replace(np.nan, 0) - b.replace(np.nan, 0)).round(8).replace(0, np.nan) \
        .dropna(how='all', axis=1).dropna(how='all', axis=0)
    aa = a.reindex(index=d.index, columns=d.columns)
    bb = b.reindex(index=d.index, columns=d.columns)
    print(c)
    return aa, bb, d, c


if __name__ == '__main__':
    begin_date = '20130101'
    end_date = '20190411'
    root_path = '/mnt/mfs/DAT_EQT'
    sector_name_list = [
        'index_000300',
        'index_000905',
        'market_top_300plus',
        'market_top_300plus_industry_10_15',
        'market_top_300plus_industry_20_25_30_35',
        'market_top_300plus_industry_40',
        'market_top_300plus_industry_45_50',
        'market_top_300plus_industry_55',

        'market_top_300to800plus',
        'market_top_300to800plus_industry_10_15',
        'market_top_300to800plus_industry_20_25_30_35',
        'market_top_300to800plus_industry_40',
        'market_top_300to800plus_industry_45_50',
        'market_top_300to800plus_industry_55',
    ]

    data_name_list = [
        'intra_oc_15min_volume_div',
        'intra_last_15min_volume_pct',
        'intra_hl_pct',
        'intra_today_sharpe',
        'intra_last_15_min_ud',
        'intra_r_vol_last_15min',
        'intra_r_vol',
        'intra_p_vol_last_15min',
        'intra_p_vol',
        'intra_money_flow2',
        'intra_money_flow1',
        'intra_dn_15_bar_div_daily',
        'intra_up_15_bar_div_daily',
        'intra_up_15_bar_div_dn_15_bar',
        'intra_dn_div_daily',
        'intra_up_div_daily',
        'intra_up_div_dn',
        'intra_dn_vwap',
        'intra_up_vwap',
        'intra_dn_15_bar_vwap',
        'intra_up_15_bar_vwap',
        'intra_daily_vwap',
        'intra_dn_15_bar_vol',
        'intra_up_15_bar_vol',
        'intra_dn_vol',
        'intra_up_vol',
    ]

    a = time.time()
    for sector_name in sector_name_list:
        print('**********************************************************')
        print(f'{sector_name}')
        data_deal = DataDeal(begin_date, end_date, root_path, sector_name)
        # data_deal.common_fun(base_data_dict.keys())
        data_deal.common_fun(data_name_list)
    b = time.time()
    print(f'花费时间：{b-a}')
