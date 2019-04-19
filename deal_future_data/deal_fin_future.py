import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *
import csv


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
                contract_num = self.p_num.findall(contract_name)
                if len(contract_num) != 0:
                    if contract_name in deal_info_dict.keys():
                        deal_info_dict[contract_name].append(year_name)
                    else:
                        deal_info_dict.update({contract_name: [year_name]})
        return deal_info_dict

    def part_deal_fun(self, contract_name):
        try:
            print(contract_name)
            if contract_name == 'IC1904':
                print(1)
            future_name = self.p_str.findall(contract_name)[0]
            result_list = []
            if len(self.deal_info_dict[contract_name]) > 1:
                print(contract_name, '!!!!!!')
            for year_name in self.deal_info_dict[contract_name]:
                raw_df = pd.read_csv(f'{self.root_path}/{year_name}/{contract_name}.csv',
                                     encoding='gbk', usecols=self.usecols)
                raw_df.columns = self.columns_list
                raw_df['Date'] = [x[:10] for x in raw_df['TradeDate']]
                raw_df['Time'] = [x[11:16] for x in raw_df['TradeDate']]
                result_list.append(raw_df)
            target_df = pd.concat(result_list, axis=0)
            target_df = target_df[['TradeDate', 'Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover',
                                   'OpenInterest']]
            save_path = f'/mnt/mfs/DAT_EQT/intraday/fut_1mbar/{future_name}'
            bt.AZ_Path_create(save_path)
            target_df.to_csv(f'{save_path}/{contract_name}.csv', sep='|', index=False)
        except Exception as error:
            print(error)

    def run(self):
        deal_info_dict = self.get_deal_info_dict()
        pool = Pool(20)
        for contract_name in list(deal_info_dict.keys()):
            args = (contract_name,)
            pool.apply_async(self.part_deal_fun, args=args)
        pool.close()
        pool.join()


class ZYYFutDeal:
    def __init__(self):
        self.usecols = ['更新时间', '最新价', '成交量', '持仓量', '换手率']
        self.columns_list = ['TradeDate', 'New', 'Volume', 'OpenInterest', 'Turnover']
        self.root_path = '/mnt/mfs/DAT_PUBLIC'
        self.deal_info_dict = self.get_deal_info_dict()
        self.save_root_path = '/mnt/mfs/DAT_EQT/intraday/fut_1mbar'
        self.p_str = re.compile(r'\D+')
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

    def deal_daily_fun(self, file_name):
        raw_df = pd.read_csv(f'{self.root_path}/FIN_FUTURE_DATA/{file_name}.csv', encoding='GBK', usecols=self.usecols)
        raw_df.columns = self.columns_list
        if isinstance(raw_df['New'].loc[0], str):
            raw_df = raw_df[raw_df['New'] != '最新价']
            raw_df[['New', 'Volume', 'OpenInterest', 'Turnover']] = \
                raw_df[['New', 'Volume', 'OpenInterest', 'Turnover']].astype(float)
        year, month, day, contract_id, _, _ = file_name.split('_')

        raw_df['Time'] = np.array([x[:5] for x in raw_df['TradeDate'].values])

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
        target_df = target_df[(target_df['Time'] > '09:00') & (target_df['Time'] <= '15:00')]
        return target_df

    def deal_contract_fun(self, contract_id):
        result_list = []
        for file_name in self.deal_info_dict[contract_id]:
            result_list.append(self.deal_daily_fun(file_name))
        target_df = pd.concat(result_list, axis=0)
        target_df = target_df[target_df.index > pd.to_datetime('20190401')]
        return target_df

    def part_run(self, contract_id):
        # try:
        print(contract_id)
        if contract_id == 'IC1904':
            print(1)
        fut_name = self.p_str.findall(contract_id)[0]
        save_path = f'{self.save_root_path}/{fut_name}/{contract_id}.csv'
        tmp_df = self.deal_contract_fun(contract_id)
        if os.path.exists(save_path):
            raw_df = pd.read_csv(save_path, sep='|', index_col='TradeDate')
            target_df = raw_df.combine_first(tmp_df)[self.columns_sorted]
            print(0)
        else:
            target_df = tmp_df[self.columns_sorted]
            print(1)
        target_df[self.columns_sorted].to_csv(save_path, sep='|')
        # except Exception as error:
        #     print(error)

    def run(self):
        # pool = Pool(20)
        for contract_id in self.deal_info_dict.keys():
            self.part_run(contract_id)
        #     pool.apply_async(self.part_run, (contract_id,))
        # pool.close()
        # pool.join()

    def update(self, target_date):
        self.get_deal_date_dict(target_date)
        for contract_id in self.deal_info_dict.keys():
            self.part_run(contract_id)


if __name__ == '__main__':
    taobao_fut_deal = TaobaoFutDeal()
    taobao_fut_deal.run()
    taobao_fut_deal.part_deal_fun('IC1904')

    zyy_fut_deal = ZYYFutDeal()
    # a = zyy_fut_deal.deal_contract_fun('IF1906')
    zyy_fut_deal.run()
