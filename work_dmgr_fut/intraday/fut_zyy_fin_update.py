import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import OrderedDict
from multiprocessing import Pool
import re


def AZ_Path_create(target_path):
    """
    添加新路径
    :param target_path:
    :return:
    """
    if not os.path.exists(target_path):
        os.makedirs(target_path)


def save_fun(new_data, now_path):
    path_split = os.path.split(now_path)
    new_path = path_split[0] + '/.' + path_split[1] + '.temp'
    new_data.to_csv(new_path, sep='|', encoding='utf-8')
    os.rename(new_path, now_path)


class TaobaoFutDeal:
    def __init__(self):
        self.root_path = '/mnt/mfs/DAT_PUBLIC'
        self.p_num = re.compile(r'\d+')
        self.p_str = re.compile(r'\D+')
        self.columns_list = ['TradeDate', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover', 'OpenInterest']
        self.usecols = ['时间', '开', '高', '低', '收', '成交量', '成交额', '持仓量']

        self.deal_info_dict = self.get_deal_info_dict()

    def get_deal_info_dict(self):
        year_name_list = [
            'FutSF_Min1_Std_2010',
            'FutSF_Min1_Std_2011',
            'FutSF_Min1_Std_2012',
            'FutSF_Min1_Std_2013',
            'FutSF_Min1_Std_2014',
            'FutSF_Min1_Std_2015',
            'FutSF_Min1_Std_2016',
            'FutSF_Min1_Std_2017',
            'FutSF_Min1_Std_2018',
            'FutSF_Min1_Std_2019',
            'FutSF_Min1_Std_201903'
        ]

        deal_info_dict = OrderedDict()
        for i, year_name in enumerate(year_name_list):
            contract_name_list = [x[:-4] for x in sorted(os.listdir(f'{self.root_path}/{year_name}'))]
            for contract_name in contract_name_list:
                contract_num = re.sub('\D', '', contract_name)
                if len(contract_num) != 0:
                    if contract_name in deal_info_dict.keys():
                        deal_info_dict[contract_name].append(year_name)
                    else:
                        deal_info_dict.update({contract_name: [year_name]})
        return deal_info_dict

    def part_deal_fun(self, contract_name):
        try:
            print(contract_name)
            future_name = re.sub('\d', '', contract_name)
            result_list = []
            if len(self.deal_info_dict[contract_name]) > 1:
                print(contract_name, '!!!!!!')
            for year_name in self.deal_info_dict[contract_name]:
                print(year_name, contract_name)
                raw_df = pd.read_csv(f'{self.root_path}/{year_name}/{contract_name}.csv',
                                     encoding='gbk', usecols=self.usecols)
                raw_df.columns = self.columns_list
                result_list.append(raw_df)
                raw_df['Date'] = [x[:10] for x in raw_df['TradeDate']]
                raw_df['Time'] = [x[11:16] for x in raw_df['TradeDate']]
                result_list.append(raw_df)
            target_df = pd.concat(result_list, axis=0)
            target_df = target_df[['TradeDate', 'Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover',
                                   'OpenInterest']]
            save_path = f'/mnt/mfs/DAT_FUT/intraday/fut_1mbar/{future_name}'
            AZ_Path_create(save_path)
            target_df.to_csv(f'{save_path}/{contract_name}.CFE', sep='|', index=False)

        except Exception as error:
            print(error)

    def run(self):
        deal_info_dict = self.get_deal_info_dict()
        pool = Pool(20)
        for contract_name in list(deal_info_dict.keys()):
            args = (contract_name,)
            # self.part_deal_fun(*args)
            pool.apply_async(self.part_deal_fun, args=args)
        pool.close()
        pool.join()


class ZYYFutDeal:
    def __init__(self):
        self.usecols = ['更新时间', '最新价', '成交量', '持仓量', '换手率']
        self.columns_list = ['TradeDate', 'New', 'Volume', 'OpenInterest', 'Turnover']
        self.root_path = '/mnt/mfs/DAT_PUBLIC'
        self.deal_info_dict = self.get_deal_info_dict()
        self.save_root_path = '/mnt/mfs/DAT_FUT/intraday/fut_1mbar'
        self.columns_sorted = ['Date', 'Time', 'Close', 'High', 'Low', 'Open',
                               'Volume', 'Turnover', 'OpenInterest']

    def get_deal_info_dict(self):
        deal_info_dict = OrderedDict()
        file_name_list = sorted(os.listdir(f'{self.root_path}/FIN_FUTURE_DATA'))
        for file_name in file_name_list:
            year, month, day, contract_id, _, _ = file_name.split('_')
            if contract_id in deal_info_dict.keys():
                deal_info_dict[contract_id].append(file_name[:-4])
            else:
                deal_info_dict[contract_id] = [file_name[:-4]]
        return deal_info_dict

    def get_deal_date_dict(self, target_date):
        target_date = target_date.strftime('%Y%m%d')
        deal_info_dict = OrderedDict()
        file_name_list = sorted(os.listdir(f'{self.root_path}/FIN_FUTURE_DATA'))
        for file_name in file_name_list:
            year, month, day, contract_id, _, _ = file_name.split('_')
            if target_date == year + month + day:
                if contract_id in deal_info_dict.keys():
                    deal_info_dict[contract_id].append(file_name[:-4])
                else:
                    deal_info_dict[contract_id] = [file_name[:-4]]
            else:
                pass
        return deal_info_dict

    def bad_time_deal(self, raw_df, fut_name):
        if fut_name in ['IF', 'IC', 'IH']:
            raw_df['Time'].replace('15:00', '14:59', inplace=True)
            raw_df['Time'].replace('11:30', '11:29', inplace=True)
        elif fut_name in ['T', 'TF', 'TS']:
            raw_df['Time'].replace('11:30', '11:29', inplace=True)
            raw_df['Time'].replace('15:15', '15:14', inplace=True)
        return raw_df

    def deal_daily_fun(self, file_name):
        raw_df = pd.read_csv(f'{self.root_path}/FIN_FUTURE_DATA/{file_name}.csv', encoding='GBK',
                             usecols=self.usecols)
        print(len(raw_df.index))
        if len(raw_df.index) != 0:
            raw_df.columns = self.columns_list
            if isinstance(raw_df['New'].iloc[0], str):
                raw_df = raw_df[raw_df['New'] != '最新价']
                raw_df[['New', 'Volume', 'OpenInterest', 'Turnover']] = \
                    raw_df[['New', 'Volume', 'OpenInterest', 'Turnover']].astype(float)
            year, month, day, contract_id, _, _ = file_name.split('_')
            raw_df['Time'] = np.array([x[:5] for x in raw_df['TradeDate'].values])
            raw_df = self.bad_time_deal(raw_df, re.sub('\d', '', contract_id))
            raw_df['TradeDate'] = pd.to_datetime(f'{year}-{month}-{day} ' + raw_df['Time'])
            price_df = raw_df.groupby(['TradeDate'])['New'].apply(lambda x: {
                'High': max(x),
                'Open': x.iloc[0],
                'Low': min(x),
                'Close': x.iloc[-1]}).unstack()

            other_df = raw_df.groupby(['TradeDate'])[['Volume', 'Turnover', 'OpenInterest']] \
                .apply(lambda x: x.iloc[-1])
            other_df['Volume'] = other_df['Volume'].sub(other_df['Volume'].shift(1), fill_value=0)
            other_df['Turnover'] = other_df['Turnover'].sub(other_df['Turnover'].shift(1), fill_value=0)
            other_df['Date'] = f'{year}-{month}-{day}'

            target_df = pd.concat([price_df, other_df], axis=1)
            target_df.index = target_df.index + timedelta(minutes=1)
            target_df['Time'] = [x.strftime('%H:%M') for x in target_df.index]
            # 过滤时间段
            target_df = target_df[(target_df['Time'] > '09:00') & (target_df['Time'] <= '15:15')]

            return target_df
        else:
            return pd.DataFrame()

    def deal_contract_fun(self, contract_id):
        result_list = []
        for file_name in self.deal_info_dict[contract_id]:
            result_list.append(self.deal_daily_fun(file_name))
        target_df = pd.concat(result_list, axis=0)
        return target_df

    def part_run(self, contract_id):

        try:
            print(contract_id)
            fut_name = re.sub('\d', '', contract_id)
            save_path = f'{self.save_root_path}/{fut_name}/{contract_id}.CFE'
            tmp_df = self.deal_contract_fun(contract_id)
            if os.path.exists(save_path):
                raw_df = pd.read_csv(save_path, sep='|', index_col=0)
                target_df = raw_df.combine_first(tmp_df)[self.columns_sorted]
                print(0)
            else:
                target_df = tmp_df[self.columns_sorted]
                print(1)
            print("hahah")
            # target_df[self.columns_sorted].to_csv(save_path, sep='|')
        except Exception as error:
            print(error)

    def run(self):
        pool = Pool(20)
        for contract_id in self.deal_info_dict.keys():
            # self.part_run(contract_id)
            pool.apply_async(self.part_run, (contract_id,))
        pool.close()
        pool.join()

    def part_update(self, contract_id, file_name, target_date):
        print(contract_id)
        fut_name = re.sub('\d', '', contract_id)
        save_path = f'{self.save_root_path}/{fut_name}/{contract_id}.CFE'
        tmp_df = self.deal_daily_fun(file_name)
        print(save_path)
        if os.path.exists(save_path):
            raw_df = pd.read_csv(save_path, sep='|', index_col=0)
            raw_df = raw_df[raw_df['Date'] != target_date.strftime('%Y-%m-%d')]
            target_df = raw_df.combine_first(tmp_df)[self.columns_sorted]
        else:
            target_df = tmp_df[self.columns_sorted]
        save_fun(target_df[self.columns_sorted], save_path)
        # print('haha')

    def update(self, target_date):
        deal_info_dict = self.get_deal_date_dict(target_date)
        for contract_id in deal_info_dict.keys():
            print(deal_info_dict[contract_id])
            self.part_update(contract_id, deal_info_dict[contract_id][0], target_date)


if __name__ == '__main__':
    # taobao_fut_deal = TaobaoFutDeal()
    # taobao_fut_deal.run()
    #
    # target_date = datetime.now()
    # zyy_fut_deal = ZYYFutDeal()
    # target_date = pd.to_datetime('20190401')
    # while target_date < pd.to_datetime('20190420'):
    #     print(target_date)
    #     zyy_fut_deal.update(target_date)
    #     target_date += timedelta(days=1)
    pass