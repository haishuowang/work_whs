import sys

sys.path.append('/mnf/mfs')
from work_whs.loc_lib.pre_load import *
from work_whs.test_future.FutDataLoad import FutData, FutClass
from work_whs.loc_lib.pre_load.senior_tools import SignalAnalysis

root_path = '/mnt/mfs/DAT_FUT'


def plot_send_data(raw_df, subject, text=''):
    figure_save_path = os.path.join('/mnt/mfs/dat_whs', 'tmp_figure')
    raw_df.plot(figsize=[16, 10], legend=True, grid=True)
    plt.savefig(f'{figure_save_path}/{subject}.png')
    plt.close()
    to = ['whs@yingpei.com']
    filepath = [f'{figure_save_path}/{subject}.png']
    send_email.send_email(text, to, filepath, subject)


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

    @staticmethod
    def fun_3(raw_df):
        tmp_df = raw_df - raw_df.shift(1)
        # tmp_df[tmp_df.abs() > 50] = 0
        signal_df_up = (tmp_df > 0).astype(int)
        signal_df_dn = (tmp_df < 0).astype(int)
        signal_df = signal_df_up - signal_df_dn
        return signal_df


class SpotDataTest(FutData, SignalSet):
    def __init__(self, begin_time, end_time, root_path, fut_name, hold_time, lag=2, use_num=3):
        super(SpotDataTest, self).__init__(root_path=root_path)
        self.fut_name = fut_name
        self.act_fut_sr = self.load_act_fut_data(fut_name, 'adj_r')

        self.return_df = bt.AZ_Rolling_sum(self.act_fut_sr, hold_time).shift(-hold_time + 1) \
            .truncate(begin_time, end_time)

        self.xinx = self.return_df.index
        self.lag = lag
        self.use_num = use_num

    def spot_test(self, fut_name, spot_name, fun_name, lag=2, way=1):
        print(spot_name)
        # load spot data
        # all_spot_data = self.load_spot_data(fut_name, spot_name)
        all_spot_data = self.load_spot_data_wind(fut_name, spot_name)

        all_spot_data = all_spot_data.reindex(index=self.act_fut_sr.index).fillna()
        # all_spot_data = all_spot_data[['秦皇岛港:平仓价:动力末煤(Q4500):山西产']]
        # all_spot_data = all_spot_data[['含税价:废钢:上海']]
        # 数据回测
        signal_df = getattr(self, fun_name)(all_spot_data).reindex(index=self.act_fut_sr.index)
        pnl_df = signal_df.shift(lag).mul(self.return_df, axis=0)
        # 指标计算
        asset_df = pnl_df.sum()
        trade_times = signal_df.diff().abs().sum()
        pot_df = (asset_df / trade_times) * 10000

        if way == 1:
            sp_sorted = bt.AZ_Sharpe_y(pnl_df).sort_values(na_position='first')
            select_data = sp_sorted.iloc[-self.use_num:].index
        else:
            sp_sorted = bt.AZ_Sharpe_y(pnl_df).sort_values(na_position='last')
            select_data = sp_sorted.iloc[:self.use_num].index

        print(pot_df[select_data])
        print(sp_sorted[select_data])

        a = spot_name.encode('utf-8')
        plot_send_data(pnl_df.cumsum(), f'{a}:spot_pnl_result')
        plot_send_data(all_spot_data, f'{a}:spot_data', text='')
        print(bt.AZ_Sharpe_y(pnl_df[select_data].sum(1)))

        # tmp_df = all_spot_data - all_spot_data.shift(1)
        # tmp_df[tmp_df.abs() > 50] = 0
        # for col in signal_df.columns:
        #     part_tmp_df = tmp_df[col]
        #     SignalAnalysis.CDF(part_tmp_df, self.return_df, 1, title=f'{col} CDF Figure', lag=2, zero_drop=True)
        return signal_df * way, pnl_df[select_data].sum(1) * way

    def all_spot_test(self, select_spot_name_dict, lag=2):
        target_pnl = pd.Series()
        for fut_name in select_spot_name_dict.keys():
            info_dict = select_spot_name_dict[fut_name]
            for spot_name in info_dict:
                fun_name, way = info_dict[spot_name]
                signal_df, pnl_df = self.spot_test(fut_name, spot_name, fun_name, lag=lag, way=way)
                target_pnl = target_pnl.add(pnl_df, fill_value=0)
        print(bt.AZ_Sharpe_y(target_pnl))
        plot_send_data(target_pnl.cumsum(), f'total:sum_pnl_result')


# fut_name = 'RB'
# hold_time = 1
# lag = 2
# select_spot_name_dict = OrderedDict({
#     '涂镀': ['fun_1', 1],
#     '废钢': ['fun_1', 1],
#     # '铁矿石2': ['fun_1', 1],
#     '铁矿石1': ['fun_1', 1],
#     '方坯': ['fun_1', 1],
#     '现货价格': ['fun_1', 1],
#     '库存': ['fun_2', -1],
# })
# spot_data_test = SpotDataTest(root_path, fut_name, hold_time, lag=lag)
# # spot_data_test.spot_test('库存', lag=2, way=-1)
# spot_data_test.all_spot_test(select_spot_name_dict, lag=2)

# fut_name = 'ZC'
# hold_time = 1
# lag = 2
# select_spot_name_dict = OrderedDict({
#     '动力煤现货价格': ['fun_1', 1],
# })

# fut_name = 'J'
# hold_time = 1
# lag = 2
# select_spot_name_dict = OrderedDict({
#     '工业萘': ['fun_1', 1],
#     # '洗油': ['fun_1', 1],
#     '炼钢生铁': ['fun_1', 1],
#     # '焦炭': ['fun_1', 1],
#     '煤焦油': ['fun_1', 1],
#     # '粗苯': ['fun_1', 1],
#     # '蒽油': ['fun_1', 1],
#     '顺酐': ['fun_1', 1],
# })
#
# spot_data_test = SpotDataTest(root_path, fut_name, hold_time, lag=lag)
# # spot_data_test.spot_test('库存', lag=2, way=-1)
# spot_data_test.all_spot_test(select_spot_name_dict, lag=2)

# fut_name = 'J'
# hold_time = 1
# lag = 2
# select_spot_name_dict = OrderedDict({
#     '工业萘': ['fun_1', 1],
#     # '洗油': ['fun_1', 1],
#     '炼钢生铁': ['fun_1', 1],
#     # '焦炭': ['fun_1', 1],
#     '煤焦油': ['fun_1', 1],
#     # '粗苯': ['fun_1', 1],
#     # '蒽油': ['fun_1', 1],
#     '顺酐': ['fun_1', 1],
# })

# fut_name = 'RB'
# hold_time = 1
# lag = 2
# select_spot_name_dict = OrderedDict({
#     '涂镀': ['fun_1', 1],
#     '废钢': ['fun_1', 1],
#     # '铁矿石2': ['fun_1', 1],
#     '铁矿石1': ['fun_1', 1],
#     '方坯': ['fun_1', 1],
#     '现货价格': ['fun_1', 1],
#     '库存': ['fun_2', -1],
# })

# fut_name = 'ZC'
# hold_time = 1
# lag = 2
# select_spot_name_dict = OrderedDict({
#     'ZC': OrderedDict({
#         '国内外动力煤价格1': ['fun_1', 1]
#     }),
# })

# fut_name = 'CF'
# hold_time = 1
# lag = 1
# select_spot_name_dict = OrderedDict({
#     'SR': OrderedDict({
#         '天气': ['fun_1', 1]
#     }),
# })


# 'ZC', '国内外煤炭库存', '煤炭库存:广州港集团'

info_dict = {'能源_动力煤': {
    '国内外动力煤价格1':
        [
            '京唐港:平仓价:动力末煤(Q5800):山西产',
            '宁波港:库提价(含税):动力煤(Q6000):中国产',
            '天津港:平仓价(含税):动力煤(Q5000):中国产',
            '秦皇岛港:平仓价:动力末煤(Q4500):山西产',
            '船板价:烟煤(A21%,V31%,0.7%S,Q5000):江苏:徐州',
            '秦皇岛港:平仓价:动力末煤(Q5000):山西产',
            '宁波港:库提价(含税):动力煤(Q5500):中国产',
            '秦皇岛港:平仓价:动力末煤(Q5800):山西产',
            '京唐港:平仓价:动力末煤(Q5500):山西产',
            '秦皇岛港:平仓价:动力末煤(Q5500):山西产',
        ],
    '国内外动力煤价格2':
        [
            '市场价:烟煤(优混):天津',
            '价格:动力煤(Q5000):陕西:关中',
            '天津港:现货价:块煤(Q6000)',
            '上海港:到岸价:动力末煤(Q4500)',
            '黄骅港:平仓价:动力煤(Q4500)',
            '平均价:动力煤:西北地区',
            '上海港:到岸价:动力末煤(Q5000)',
            '上海港:到岸价:动力末煤(Q5800)',
            '黄骅港:平仓价:动力煤(Q5800)',
            '黄骅港:平仓价:动力煤(Q5000)',
            '上海港:到岸价:动力末煤(Q5500)',
            '平均价:动力煤:陕西',
            '黄骅港:平仓价:动力煤(Q5500)',
            '市场价:动力煤(Q:5500,榆林产):陕西',
            '市场价:动力煤(Q:5000,S:1%,韩城产):陕西',
        ],
    '国内外煤炭库存':
        [
            '国有重点煤矿库存:北京',
            '煤炭库存:广州港集团',
            '国有重点煤矿库存:华北地区',
            '国有重点煤矿库存:宁夏区',
            '煤炭库存:秦皇岛港:内贸',
            '煤炭库存:秦皇岛港',
            '煤炭库存可用天数:大唐',
            '煤炭库存:上电',
            '煤炭库存可用天数:6大发电集团:直供总计',
            '国有重点煤矿库存:吉林',
            '煤炭库存:大唐',
            '国有重点煤矿库存:山西',
            '国有重点煤矿库存:江西',
            '国有重点煤矿库存:陕西',
            '煤炭库存:浙电',
        ],
    '国内外煤炭运价':
        [
            'CBCFI:运价:煤炭:秦皇岛-宁波(1.5-2万DWT)',
            'CBCFI:运价:煤炭:秦皇岛-南京(3-4万DWT)',
            '煤炭运价:好望角型船:120000吨级:汉普敦-鹿特丹',
            'CBCFI:运价:煤炭:秦皇岛-张家港(2-3万DWT)',
            'CBCFI:运价:煤炭:黄骅-上海(3-4万DWT)',
            'CBCFI:运价:煤炭:秦皇岛-福州(3-4万DWT)',
            '煤炭运价:巴拿马型船:70000吨级:美湾-ARA',
            'CBCFI:运价:煤炭:秦皇岛-上海(4-5万DWT)',
            'CBCFI:运价:煤炭:秦皇岛-张家港(4-5万DWT)',
            '煤炭运价:好望角型船:130000/140000吨级:昆士兰-日本',
            'CBCFI:运价:煤炭:秦皇岛-广州(5-6万DWT)',
            '煤炭运价:巴拿马型船:70000吨级:玻利瓦尔-ARA',
            '煤炭运价:好望角型船:140000吨级:理查德湾-鹿特丹',
            'CBCFI:运价:煤炭:秦皇岛-厦门(5-6万DWT)',
            'CBCFI:运价:煤炭:天津-镇江(1-1.5万DWT)',
        ],
    '电力':
        [
            '日均耗煤量:浙电',
            '产量:发电量:当月值',
            '用电量占比:第三产业',
            '城乡居民生活用电量:当月值',
            '城乡居民生活用电量:当月同比',
            '乡村居民生活用电量:环比',
            '产量:火电:当月值',
            '产量:火电:当月同比',
        ]
},
}

if __name__ == '__main__':
    # fut_name = 'SR'
    # hold_time = 1
    # lag = 2
    # select_spot_name_dict = OrderedDict({
    #     'SR': OrderedDict({
    #         # '国内外动力煤价格2': ['fun_1', 1],
    #         # '国内外煤炭库存': ['fun_1', 1],
    #         # '国内外煤炭运价': ['fun_1', 1],
    #         '天气': ['fun_1', 1]
    #     }),
    # })

    fut_name = 'RB'
    hold_time = 1
    lag = 1
    select_spot_name_dict = OrderedDict({
        '煤焦钢矿_螺纹线材': OrderedDict({
            '废钢': ['fun_1', 1]
        }),
    })

    # all_spot_data = fut_data.load_spot_data(fut_name, '现货价格')
    begin_time = pd.to_datetime('20100101')
    end_time = datetime.now()
    # end_time = pd.to_datetime('20180201')
    spot_data_test = SpotDataTest(begin_time, end_time, root_path, fut_name, hold_time, lag=lag, use_num=3)
    spot_data_test.all_spot_test(select_spot_name_dict, lag)

    # data = pd.read_excel('/mnt/mfs/dat_whs/国内外动力煤价格.xls', index_col=0)
    # data = data.iloc[:-2]
    # data.index = pd.to_datetime(data.index)
    # data = data.truncate(begin_time, end_time)
    # diff_df = data - data.shift(1)
    # signal_df_up = (diff_df > 0).astype(int)
    # signal_df_dn = (diff_df < 0).astype(int)
    # pos_df = (signal_df_up - signal_df_dn).replace(0, np.nan).fillna(method='ffill')
    # pos_df = pos_df.reindex(index=spot_data_test.return_df.index)
    # pnl_df = pos_df.iloc[:, 0].shift(lag) * spot_data_test.return_df
    # print(bt.AZ_Sharpe_y(pnl_df))
    # plot_send_data(pnl_df.cumsum(), subject='test')

