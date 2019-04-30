import sys

sys.path.append('/mnf/mfs')

from work_whs.loc_lib.pre_load import *

root_path = '/mnt/mfs/DAT_FUT'


def plot_send_data(raw_df, subject, text=''):
    figure_save_path = os.path.join('/mnt/mfs/dat_whs', 'tmp_figure')
    raw_df.plot(figsize=[16, 10], legend=True, grid=True)
    plt.savefig(f'{figure_save_path}/{subject}.png')
    plt.close()
    to = ['whs@yingpei.com']
    filepath = [f'{figure_save_path}/{subject}.png']
    send_email.send_email(text, to, filepath, subject)


class FutData:
    def __init__(self, root_path):
        self.root_path = root_path
        self.act_info_df = bt.AZ_Load_csv(f'{root_path}/day/DailyPX/Contract')

    def load_fut_data(self, fut_name, file_name):
        raw_df = bt.AZ_Load_csv(f'{self.root_path}/day/union/{fut_name}/{file_name}')
        return raw_df

    def load_contract_data(self, contract_id, file_name):
        fut_name = re.sub('\d', '', contract_id)
        raw_df = self.load_fut_data(fut_name, file_name)
        target_df = raw_df[[contract_id]]
        return target_df

    def load_act_fut_data(self, fut_name, file_name):
        raw_df = self.load_fut_data(fut_name, file_name)
        act_info_sr = self.act_info_df[f'{fut_name}01'].reindex(index=raw_df.index)
        target_df = raw_df.apply(lambda x, y: x * (y == x.name), args=(act_info_sr,))
        target_df = target_df.replace(0, np.nan).dropna(how='all', axis='columns')
        target_sr = target_df.sum(1)
        target_sr.name = f'{fut_name}01'
        return target_sr

    def load_spot_data(self, fut_name, file_name):
        """
        仓单库存, 方坯, 现货价格
        :param fut_name:
        :param file_name:
        :return:
        """
        raw_df = bt.AZ_Load_csv(f'{self.root_path}/Inventory/{fut_name}/{file_name}.csv')
        return raw_df


class SignalSet:
    @staticmethod
    def fun_1(raw_df):
        tmp_df = raw_df - raw_df.shift(1)
        signal_df_up = (tmp_df > 0).astype(int)
        signal_df_dn = (tmp_df < 0).astype(int)
        signal_df = (signal_df_up - signal_df_dn).replace(0, np.nan).fillna(method='ffill')
        return signal_df

    @staticmethod
    def fun_2(raw_df):
        signal_df_up = (raw_df > 0).astype(int)
        signal_df_dn = (raw_df < 0).astype(int)
        signal_df = (signal_df_up - signal_df_dn).replace(0, np.nan).fillna(method='ffill')
        return signal_df


class SignalAnalysis:
    @staticmethod
    def CDF(signal_df, raw_return_df, hold_time, title='CDF Figure', lag=2):
        signal_df = signal_df.shift(lag).values
        return_df = bt.AZ_Rolling_sum(raw_return_df, hold_time).shift(-hold_time + 1).values

        save_path = '/mnt/mfs/dat_whs/tmp_figure/CDF_Figure.png'
        f_return_m = return_df - return_df.mean()
        a = np.argsort(signal_df)
        plt.figure(figsize=(10, 6))
        p1 = plt.subplot(221)
        p2 = plt.subplot(222)
        p3 = plt.subplot(223)
        p4 = plt.subplot(224)
        p1.plot(np.cumsum(return_df[a]))
        p1.set_title('cumsum return')
        p1.grid(1)

        p2.plot(signal_df[a], np.cumsum(return_df[a]))
        p2.set_title('signal and cumsum return')
        p2.grid(1)

        p3.plot(np.cumsum(f_return_m[a]))
        p3.set_title('cumsum mean return')
        p3.grid(1)

        p4.plot(signal_df[a], np.cumsum(f_return_m[a]))
        p4.set_title('signal and cumsum mean return')
        p4.grid(1)

        plt.suptitle(title)
        plt.savefig(save_path)
        plt.close()
        to = ['whs@yingpei.com']
        filepath = [save_path]
        send_email.send_email('', to, filepath, title)


class SpotDataTest(FutData, SignalSet):
    def __init__(self, root_path, fut_name, hold_time, lag=2, use_num=3):
        super(SpotDataTest, self).__init__(root_path=root_path)
        self.fut_name = fut_name
        self.act_fut_sr = self.load_act_fut_data(fut_name, 'adj_r')
        self.return_df = bt.AZ_Rolling_sum(self.act_fut_sr, hold_time).shift(-hold_time + 1)
        self.lag = lag
        self.use_num = use_num

    def spot_test(self, spot_name, fun_name, lag=2, way=1):
        print(spot_name)
        all_spot_data = self.load_spot_data(self.fut_name, spot_name)
        all_spot_data = all_spot_data.reindex(index=self.act_fut_sr.index).fillna(method='ffill')
        # 数据回测
        signal_df = getattr(self, fun_name)(all_spot_data).reindex(index=self.act_fut_sr.index)
        pnl_df = signal_df.shift(lag).mul(self.return_df, axis=0)
        # 指标计算
        asset_df = pnl_df.sum()
        trade_times = signal_df.diff().abs().sum()
        pot_df = (asset_df / trade_times) * 10000
        sp_sorted = bt.AZ_Sharpe_y(pnl_df).sort_values(na_position='first')
        if way == 1:
            select_data = sp_sorted.iloc[-self.use_num:].index
        else:
            select_data = sp_sorted.iloc[:self.use_num].index

        print(pot_df[select_data])
        print(sp_sorted[select_data])

        # a = spot_name.encode('utf-8')
        # plot_send_data(pnl_df.cumsum(), f'{a}:spot_pnl_result')
        # plot_send_data(all_spot_data, f'{a}:spot_data', text='')
        print(bt.AZ_Sharpe_y(pnl_df[select_data].sum(1)))
        return signal_df * way, pnl_df[select_data].sum(1) * way

    def all_spot_test(self, select_spot_name_dict, lag=2):
        target_pnl = pd.Series()
        for spot_name in select_spot_name_dict.keys():
            fun_name, way = select_spot_name_dict[spot_name]
            signal_df, pnl_df = self.spot_test(spot_name, fun_name, lag=lag, way=way)
            target_pnl = target_pnl.add(pnl_df, fill_value=0)
        print(bt.AZ_Sharpe_y(target_pnl))
        plot_send_data(target_pnl.cumsum(), f'total:sum_pnl_result')


if __name__ == '__main__':
    fut_name = 'RB'
    hold_time = 1
    lag = 2
    select_spot_name_dict = OrderedDict({
        '涂镀': ['fun_1', 1],
        '废钢': ['fun_1', 1],
        # '铁矿石2': ['fun_1', 1],
        '铁矿石1': ['fun_1', 1],
        '方坯': ['fun_1', 1],
        '现货价格': ['fun_1', 1],
        '库存': ['fun_2', -1],
    })
    spot_data_test = SpotDataTest(root_path, fut_name, hold_time, lag=lag)
    # spot_data_test.spot_test('库存', lag=2, way=-1)
    spot_data_test.all_spot_test(select_spot_name_dict, lag=2)
    # main_deal(fut_name, ['库存'], use_num=3, hold_time=1, lag=1)
    # fut_name = 'ZC'
    # hold_time = 1
    # lag = 1
    # select_spot_name_list = ['动力煤现货价格']
    # main_deal(fut_name, select_spot_name_list, hold_time=hold_time, lag=lag)
