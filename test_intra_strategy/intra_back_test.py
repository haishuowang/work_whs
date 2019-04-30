import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *
from work_whs.test_intra_strategy import intra_data_create
import work_whs.loc_lib.shared_tools.senior_tools as st
import string


class DataCreate:
    @staticmethod
    def run():
        begin_str = '20100101'
        end_str = '20190408'
        cut_num_list = [15, 30, 45, 60]
        end_num_list = [120, 240]
        index_code_list = ['000905', '000300']

        fun_dict = OrderedDict()
        for cut_num in cut_num_list:
            begin_num = cut_num + 2
            for end_num in end_num_list:
                fun_dict.update({
                    f'intra_open_min_return|{cut_num}': dict({'data': ['close']}),
                    f'intra_open_min_vol|{cut_num}': dict({'data': ['volume']}),
                    f'intra_stock_return|{begin_num}_{end_num}': dict({'data': ['close']}),
                })
                for index_code in index_code_list:
                    fun_dict.update({
                        f'intra_index_return|{index_code}_{begin_num}_{end_num}': dict({'data': ['close']}),
                    })

        # 生成daily数据
        a1 = time.time()
        intra_data_create.main_fun(fun_dict, begin_str, end_str)
        a2 = time.time()
        print(f'初始数据生成花费时间:{round((a2-a1),4)}')


class FilterFun(st.Signal):
    @staticmethod
    def filter_plus_minus(raw_df):
        return (raw_df > 0).astype(int).replace(0, -1)

    def all_filter_signal(self, raw_df, sector_df, window, pct, limit, cap=3):
        col_zscore_df = bt.AZ_Col_zscore(raw_df, window, cap)
        raw_zscore_df = bt.AZ_Row_zscore(raw_df * sector_df, cap)

        plus_minus_df = self.filter_plus_minus(raw_df)
        col_zscore_row_extre_df = self.row_extre(col_zscore_df, pct)
        col_zscore_signal_fun_df = self.signal_fun(col_zscore_df, limit)

        raw_zscore_row_extre_df = self.row_extre(raw_zscore_df, pct)
        raw_zscore_signal_fun_df = self.signal_fun(raw_zscore_df, limit)
        part_signal_df_dict = OrderedDict({'plus_minus_df': plus_minus_df,
                                           'col_zscore_row_extre_df': col_zscore_row_extre_df,
                                           'col_zscore_signal_fun_df': col_zscore_signal_fun_df,
                                           'raw_zscore_row_extre_df': raw_zscore_row_extre_df,
                                           'raw_zscore_signal_fun_df': raw_zscore_signal_fun_df}
                                          )
        return part_signal_df_dict


class IntraBackTest(FilterFun):
    def __init__(self, root_path, begin_str, end_str, cut_num, end_num, index_code):
        self.root_path = root_path

        self.begin_str = begin_str
        self.end_str = end_str

        self.cut_num, self.end_num, self.index_code = cut_num, end_num, index_code
        # sector_df
        self.sector_df = bt.AZ_Load_csv(f'{root_path}/EM_Funda/IDEX_YS_WEIGHT_A/SECURITYNAME_{index_code}.csv')
        self.sector_df[self.sector_df == self.sector_df] = 1
        self.sector_df = self.sector_df[(self.sector_df.index >= pd.to_datetime(self.begin_str)) &
                                        (self.sector_df.index <= pd.to_datetime(self.end_str))]
        self.xinx = self.sector_df.index
        self.xnms = self.sector_df.columns
        #
        # bar_num_df = bt.AZ_Load_csv(f'/media/hdd1/DAT_EQT/EM_Funda/dat_whs/bar_num_df.csv')
        # self.bar_roll_num_df = bt.AZ_Rolling_mean(bar_num_df, 10)
        # 数据文件名
        self.open_min_return_df = self.load_data(f'intra_open_min_return|{cut_num}')
        self.open_min_vol_df = self.load_data(f'intra_open_min_vol|{cut_num}')

        # 回测数据列表
        self.raw_data_list = [self.open_min_return_df, self.open_min_vol_df]

        self.stock_return_df = self.load_data(f'intra_stock_return|{cut_num+2}_{end_num}')
        self.index_return_df = self.load_data(f'intra_index_return|{index_code}_{cut_num+2}_{end_num}')

        self.return_df = None

    def load_data(self, file_name):
        data = bt.AZ_Load_csv(f'/mnt/mfs/dat_whs/intra_data/{file_name}.csv').reindex(index=self.xinx)
        return data

    # def save_data():

    def get_return_table(self, if_hegde):
        if if_hegde:
            return_df = self.stock_return_df.sub(self.index_return_df[f'{self.index_code}.SH'], axis=0)
        else:
            return_df = self.stock_return_df
        return return_df

    @staticmethod
    def deal_signal(signal_df_raw, name):
        if name == 'up':
            signal_df = (signal_df_raw > 0).astype(int)
        elif name == 'dn':
            signal_df = (signal_df_raw < 0).astype(int)
        else:
            signal_df = signal_df_raw
        return signal_df

    def get_pnl_fun(self, signal_df_raw, name):
        signal_df = self.deal_signal(signal_df_raw, name)
        if signal_df.abs().sum().sum() != 0:
            pnl_df = (signal_df * self.return_df).sum(1)
            pnl_df.name = name
            pot = round(pnl_df.sum() / signal_df.abs().sum().sum() / 2, 4)
            sharpe = bt.AZ_Sharpe_y(pnl_df)
        else:
            pnl_df = pd.Series([0] * len(signal_df.index), name=name, index=signal_df.index)
            pot = 0
            sharpe = 0
        return pnl_df, sharpe, pot

    def get_all_pnl_fun(self, signal_df, signal_name):
        # signal 做多做空双边情况
        side_list = ['both', 'up', 'dn']
        sharpe_list = []
        pot_list = []
        pnl_list = []
        for side in side_list:
            pnl, sharpe, pot = self.get_pnl_fun(signal_df, side)
            sharpe_list.append(abs(sharpe))
            pot_list.append(abs(pot))
            pnl_list.append(pnl)
        if (np.array(pot_list)[np.array(sharpe_list) > 2] > 0.001).any():
            salt = ''.join(random.sample(string.ascii_letters + string.digits, 10))
            figure_path = f'/mnt/mfs/dat_whs/tmp_figure/{salt}.png'
            plt.figure(figsize=[16, 8])
            for pnl_df, sharpe, pot in zip(*[pnl_list, sharpe_list, pot_list]):
                plt.plot(pnl_df.cumsum(), label=f'name={pnl_df.name}, sharpe={sharpe}, pot={pot}')
            plt.title(signal_name)
            plt.legend()
            plt.grid()
            plt.savefig(figure_path)
            plt.close()
            return figure_path
        else:
            return None

    def plot_send_fun(self, signal_df_dict, subject):
        figure_path_list = []
        step = 20
        for signal_name in signal_df_dict.keys():
            signal_df = signal_df_dict[signal_name]
            figure_path = self.get_all_pnl_fun(signal_df, signal_name)
            if figure_path is not None:
                figure_path_list.append(figure_path)
                if len(figure_path_list) >= step:
                    send_email.send_email('', ['whs@yingpei.com'], figure_path_list, '[intratest]' + subject)
                    figure_path_list = []
                else:
                    pass
            else:
                pass

        if len(figure_path_list) != 0:
            send_email.send_email('', ['whs@yingpei.com'], figure_path_list, '[intratest]' + subject)

    def plot_send_fun_mix(self, mix_signal_iter, subject):
        figure_path_list = []
        step = 20
        for signal_name, signal_df in mix_signal_iter:
            figure_path = self.get_all_pnl_fun(signal_df, signal_name)
            if figure_path is not None:
                figure_path_list.append(figure_path)
                if len(figure_path_list) >= step:
                    send_email.send_email('', ['whs@yingpei.com'], figure_path_list, '[intratest]' + subject)
                    figure_path_list = []
                else:
                    pass
            else:
                pass
        if len(figure_path_list) != 0:
            send_email.send_email('', ['whs@yingpei.com'], figure_path_list, '[intratest]' + subject)

    def mul_signal_fun(self, signal_dict_list):
        sides_list = ['both', 'up', 'dn']
        single_keys_list = list(product(*[x.keys() for x in signal_dict_list]))
        for key_list in single_keys_list:
            raw_signal_list = [signal_dict_list[i][key] for i, key in enumerate(key_list)]
            for signal_sides_list in product(sides_list, repeat=len(signal_dict_list)):
                signal_list = [self.deal_signal(raw_signal_list[i], side) for i, side in enumerate(signal_sides_list)]
                signal_df = None
                for tmp_signal in signal_list:
                    if signal_df is None:
                        signal_df = tmp_signal
                    else:
                        signal_df = signal_df.mul(tmp_signal)
                yield '|'.join(['@'.join(list(key_list)), '@'.join(list(signal_sides_list))]), signal_df

    def run(self, limit, window, pct, if_hegde):
        print(self.cut_num, self.end_num, self.index_code, limit, window, pct, if_hegde)
        # try:
        self.return_df = self.get_return_table(if_hegde)
        # 根据数据生成 dict
        open_min_return_signal_dict = \
            self.all_filter_signal(self.open_min_return_df, self.sector_df, window, pct, limit)
        open_min_vol_signal_dict = \
            self.all_filter_signal(self.open_min_vol_df, self.sector_df, window, pct, limit)
        vol_df = pd.read_pickle(f'/mnt/mfs/dat_whs/data/new_factor_data/'
                                f'index_{self.index_code}/vol_p100d.pkl').shift(1)

        vol_df_signal_dict = \
            self.all_filter_signal(vol_df, self.sector_df, window, pct, limit)

        # bar_roll_num_signal_dict = \
        #     self.all_filter_signal(self.bar_roll_num_df, self.sector_df, window, pct, limit)

        suject_suffix = f'{self.index_code}|{limit}|{window}|{pct}|{if_hegde}'
        signal_dict_list = [open_min_vol_signal_dict, open_min_return_signal_dict, vol_df_signal_dict]
        mix_signal_dict = self.mul_signal_fun(signal_dict_list)

        # self.plot_send_fun(open_min_return_signal_dict, 'open_min_return_signal_list' + suject_suffix)
        # self.plot_send_fun(open_min_vol_signal_dict, 'open_min_vol_signal_list' + suject_suffix)
        self.plot_send_fun(vol_df_signal_dict, 'vol_df_signal_list' + suject_suffix)

        # 多重filter
        self.plot_send_fun_mix(mix_signal_dict, 'mix_signal_list' + suject_suffix)

        # except Exception as error:
        #     send_email.send_email(str(error), ['whs@yingpei.com'], [],
        #                           f'[intratest]{self.cut_num}, {self.end_num}, {self.index_code}, '
        #                           f'{limit}, {window}, {pct}, {if_hegde} error')


def main_fun(cut_num_list, end_num_list, index_code_list, limit_list, window_list, pct_list, if_hegde_list):
    root_path = '/mnt/mfs/DAT_EQT'
    begin_str, end_str = '20130101', '20190404'
    pool = Pool(15)
    for cut_num, end_num, index_code in product(cut_num_list, end_num_list, index_code_list):

        intra_back_test = IntraBackTest(root_path, begin_str, end_str, cut_num, end_num, index_code)
        for limit, window, pct, if_hegde in product(limit_list, window_list, pct_list, if_hegde_list):
            args = (limit, window, pct, if_hegde)
            # intra_back_test.run(*args)
            pool.apply_async(intra_back_test.run, args=args)
    pool.close()
    pool.join()


if __name__ == '__main__':
    # 生成数据
    # DataCreate.run()

    # root_path = '/mnt/mfs/DAT_EQT'
    # begin_str, end_str = '20130101', '20190404'
    # cut_num, end_num, index_code, limit, window, pct, if_hegde = 30, 240, '000905', 2, 10, 0.2, True
    # intra_back_test = IntraBackTest(root_path, begin_str, end_str, cut_num, end_num, index_code)
    # intra_back_test.run(limit, window, pct, if_hegde)

    cut_num_list = [15, 30, 45, 60]
    end_num_list = [120, 240]
    index_code_list = ['000905', '000300']

    limit_list = [1.8, 2, 2.4]
    window_list = [5, 10, 20]
    pct_list = [0.1, 0.2]
    if_hegde_list = [True, False]

    limit_list = [1.8, 2, 2.4]
    window_list = [5, 10, 20]
    main_fun(cut_num_list, end_num_list, index_code_list, limit_list, window_list, pct_list, if_hegde_list)
