import os
import pandas as pd
import numpy as np
from multiprocessing import Pool
from collections import OrderedDict


# import work_whs.loc_lib.shared_tools.back_test as bt


def AZ_filter_stock(stock_list):  # 筛选相应股票池
    target_list = [x for x in stock_list if x[:2] == 'SH' and x[2] == '6' or
                   x[:2] == 'SZ' and x[2] in ['0', '3']]
    return target_list


class DailyDealFunSet:
    @staticmethod
    def intra_open_min_return(day, close, cut_num):
        part_open_min_return = close.iloc[cut_num - 1] / close.iloc[0].replace(0, np.nan) - 1
        part_open_min_return.name = day
        return part_open_min_return

    @staticmethod
    def intra_open_min_vol(day, volume, cut_num):
        volume = volume[AZ_filter_stock(volume.columns)]
        part_open_min_vol = volume.iloc[:cut_num].sum()
        part_open_min_vol.name = day
        return part_open_min_vol

    @staticmethod
    def intra_stock_return(day, close, begin_num, end_num):
        part_stock_return = close.iloc[end_num - 1] / close.iloc[begin_num - 1].replace(0, np.nan) - 1
        part_stock_return.name = day
        return part_stock_return

    @staticmethod
    def intra_index_return(day, close, index_name, begin_num, end_num):
        if f'SH{index_name}' in close.columns:
            part_index_return = close[[f'SH{index_name}']].iloc[end_num - 1] / \
                                close[[f'SH{index_name}']].iloc[begin_num - 1] - 1
        else:
            part_index_return = pd.Series([0], index=[f'SH{index_name}'])
        part_index_return.name = day
        return part_index_return

    # @staticmethod
    # def intra_target_close(day, close, cut_num):
    #     close.iloc


class GetIntraData(DailyDealFunSet):
    def __init__(self, begin_str, end_str, fun_dict, cup_num=20):
        assert isinstance(fun_dict, OrderedDict)
        self.begin_str = begin_str
        self.end_str = end_str

        self.fun_dict = fun_dict
        self.save_path = '/mnt/mfs/dat_whs/intra_data'
        self.cup_num = cup_num

    @staticmethod
    def save_fun(data, target_path):
        data.columns = [x[2:] + '.' + x[:2] for x in data.columns]
        data = data.reindex(columns=sorted(data.columns))
        data.to_csv(target_path, sep='|')

    @staticmethod
    def str_to_list(para_str):
        tmp_fun = lambda x: x if len(x) == 6 else int(x)
        para = [tmp_fun(x) for x in para_str.split('_')]
        return para

    def daily_deal_fun(self, day, day_path):
        print(day)
        result_list = []
        volume = pd.read_csv(os.path.join(day_path, 'Volume.csv'), index_col=0).astype(float)
        close = pd.read_csv(os.path.join(day_path, 'Close.csv'), index_col=0).astype(float)

        for file_name in self.fun_dict.keys():
            fun_name, para_str = file_name.split('|')
            para = self.str_to_list(para_str)

            target_fun = getattr(self, fun_name)
            # str to data
            data_list = []
            for x in self.fun_dict[file_name]['data']:
                data_list.append(locals()[x])

            result_list.append(target_fun(day, *data_list, *para))
        return result_list

    def run(self):
        begin_year, begin_month, begin_day = self.begin_str[:4], self.begin_str[:6], self.begin_str
        end_year, end_month, end_day = self.end_str[:4], self.end_str[:6], self.end_str
        intraday_path = '/mnt/mfs/DAT_EQT/intraday/eqt_1mbar'

        result_list = []

        pool = Pool(20)
        year_list = [x for x in os.listdir(intraday_path) if (x >= begin_year) & (x <= end_year)]
        for year in sorted(year_list):
            year_path = os.path.join(intraday_path, year)
            month_list = [x for x in os.listdir(year_path) if (x >= begin_month) & (x <= end_month)]
            for month in sorted(month_list):
                month_path = os.path.join(year_path, month)
                day_list = [x for x in os.listdir(month_path) if (x >= begin_day) & (x <= end_day)]
                for day in sorted(day_list):
                    day_path = os.path.join(month_path, day)

                    args = (day, day_path)
                    # self.daily_deal_fun(*args)
                    result_list.append(pool.apply_async(self.daily_deal_fun, args=args))
        pool.close()
        pool.join()

        for i, file_name in enumerate(self.fun_dict.keys()):
            target_df = pd.concat([x.get()[i] for x in result_list], axis=1, sort=True)
            self.save_fun(target_df.T, f'{self.save_path}/{file_name}.csv')


def main_fun(fun_dict, begin_str, end_str):
    get_intra_data = GetIntraData(begin_str, end_str, fun_dict)
    get_intra_data.run()


if __name__ == '__main__':
    fun_dict = OrderedDict({'intra_open_min_return|15': dict({'data': ['close']}),
                            'intra_open_min_vol|15': dict({'data': ['volume']})
                            })
    begin_str = '20190109'
    end_str = '20190408'
    main_fun(fun_dict, begin_str, end_str)
