import sys

# sys.path = ['/mnt/mfs'] + sys.path
sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load.plt import savfig_send
from work_whs.test_future.FutDataLoad import FutData, FutClass
from work_dmgr_fut.fut_script.signal_fut_fun import FutIndex, Signal, Position
import talib as ta


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


def get_return_df(fut_name):
    act_fut_sr = fut_data.load_act_fut_data(fut_name, 'adj_r')
    return_df = act_fut_sr.truncate(begin_time, end_time)
    # return_df = bt.AZ_Rolling_sum(act_fut_sr, ).shift(-hold_time + 1) \
    #     .truncate(begin_time, end_time)
    return return_df


fut_data = FutData()


def test_fun(fut_name, path_name, file_name):
    root_path = '/mnt/mfs/DAT_PUBLIC/spot_data'
    ####################
    data_df = pd.read_excel(f'{root_path}/{path_name}/{file_name}', index_col=0,
                            skiprows=[0, 1, 3, 4, 5, 6, 7, 8, 9], parse_dates=True)

    ####################
    # data_df = pd.read_excel(f'{root_path}/{path_name}/{file_name}', index_col=0,
    #                         skiprows=[1, 2], parse_dates=True)

    data_df = data_df.dropna(how='all', axis=0)
    data_df.index = pd.to_datetime(data_df.index)

    data_df = data_df.truncate(begin_time, end_time)

    return_df = get_return_df(fut_name)
    data_df = data_df.reindex(return_df.index).fillna(method='ffill')
    # pos_df = SignalSet().fun_1(data_df)

    score_df, _, _ = FutIndex.boll_fun(data_df, 10, return_line=True)
    signal_df = Signal.fun_1(score_df, 0)
    pos_df = Position.fun_1(signal_df)

    pnl_df = pos_df.shift(2).mul(return_df, axis=0)
    pnl_df.name = fut_name

    # plt.figure(figsize=[16, 10])
    # plt.plot(pnl_df.index, pnl_df.cumsum().values)
    # plt.legend()
    # plt.grid()
    # savfig_send(subject=f"{fut_name}|{path_name.encode()}|{file_name.encode()}")

    sharpe_sr = bt.AZ_Sharpe_y(pnl_df)
    sharpe_sr.name = 'sharpe'
    asset_df = pnl_df.sum()
    trade_times = pos_df.diff().abs().sum()
    pot_sr = (asset_df / trade_times) * 10000
    pot_sr.name = 'pot'

    result_df = pd.concat([sharpe_sr, pot_sr], axis=1) \
        .sort_values(by='sharpe', na_position='last')
    result_df = result_df[result_df['sharpe'].abs() > 1]
    if len(result_df) != 0:
        print(fut_name, path_name, file_name)
        print(result_df)
        plt.figure(figsize=[16, 10])
        plt.plot(pnl_df.index, pnl_df[result_df.index].cumsum().values)
        plt.legend()
        plt.grid()
        savfig_send(text=f"{fut_name}|{path_name.encode()}|{file_name.encode()}",
                    subject=f"{fut_name}|{path_name.encode()}|{file_name.encode()}")
    return pos_df, pnl_df, data_df, result_df


if __name__ == '__main__':
    begin_time = pd.to_datetime('2014-01-01')
    # end_time = pd.to_datetime('2019-01-01')
    end_time = None
    # #
    fut_name_list = ['AP', 'A', 'B', 'BU', 'C', 'CF', 'CS', 'CU', 'EG', 'FG', 'HC', 'I', 'J', 'JD', 'JM', 'L',
                     'M', 'MA', 'NI', 'OI', 'P', 'PP', 'RB', 'RM', 'RU', 'SC', 'SM', 'SR', 'TA', 'V', 'Y', 'ZN']

    # for fut_name in fut_name_list:
    #     print(fut_name)
    #     path_name_list = os.listdir('/mnt/mfs/DAT_PUBLIC/spot_data')
    #     path_name_list = ['能源_动力煤', '谷物_玉米', '化工_甲醇', '油脂油料_菜油', '化工_橡胶',
    #                       '油脂油料_豆粕', '油脂油料_菜油']
    #     # path_name_list = ['化工_尿素', '化工_库存', '化工_沥青', '化工_涤纶', '化工_聚乙烯', '化工_苯乙烯']
    #     for path_name in path_name_list:
    #         file_name_list = os.listdir(f'/mnt/mfs/DAT_PUBLIC/spot_data/{path_name}')
    #         for file_name in file_name_list:
    #             try:
    #                 test_fun(fut_name, path_name, file_name)
    #             except Exception as error:
    #                 print(error)

    # 金灿荣、戴旭、罗援、王洪光
    # B 能源_动力煤 国内外动力煤价格2.xls
    # 市场价:动力煤(Q:5000,淮北产):安徽
    # 市场价:动力煤(Q:4800-5000,宿州产):安徽
    # 坑口价:动力煤(Q4200):鄂尔多斯
    # 车板价:褐煤(Q3200):锡林郭勒
    # 价格:块煤:陕西:咸阳

    # B 能源_动力煤 国内外动力煤价格1.xls
    # 秦皇岛港: 市场价:动力煤(Q: 5800, 山西产)
    # 秦皇岛港: 市场价:动力煤(Q: 5000, 山西产)

    # BU 能源_动力煤 国内外动力煤价格2.xls
    # -车板价: 直达煤(Q5500, S≤1 %, 晋城产):山西
    # -车板价:直达煤(Q5700,S≤0.5%,长治产):山西

    # BU 化工_甲醇 甲醇价格2.xls
    # 市场估价(高端价):甲醇:太仓
    # 市场估价(低端价):甲醇:太仓
    # 市场估价(中间价):甲醇:太仓

    # BU 化工_甲醇 期货市场.xls
    # 仓单数量:甲醇:江阴澄利

    # CF 能源_动力煤 国内外煤炭运价.xls
    # CBCFI:运价:煤炭:秦皇岛-广州(6-7万DWT)
    # CBCFI:运价:煤炭:秦皇岛-张家港(4-5万DWT)

    # CF 能源_动力煤 国内外动力煤价格2.xls
    # 市场价:动力煤(Q:4800-5200,徐州产):江苏
    # 平均价:动力煤:宁夏
    # 市场价:动力煤(Q:6000,石嘴山产):宁夏
    # 平均价:动力煤:江苏

    # NI 能源_动力煤 国内外动力煤价格1.xls
    # 广州港:平均价:动力煤
    # 秦皇岛港:市场价:动力煤(Q:4500,山西产)
    # 曹妃甸港:市场价:动力煤(Q:4500,山西产)

    # M 能源_动力煤 国内外动力煤价格2.xls
    # 车板价:动力煤(Q5500):鄂尔多斯
    # 市场价:动力煤(Q:4800-5000,V:28%,S<1.0%,兖州产):山东
    # 价格:动力煤(Q5000):陕西:关中
    # 坑口价:动力煤(Q4200):鄂尔多斯

    # 仓单数量:甲醇:江阴澄利

    # BU 能源_动力煤 国内外动力煤价格2.xls 车板价:直达煤(Q5500,S≤1%,晋城产):山西
    # BU 能源_动力煤 国内外动力煤价格2.xls 车板价:直达煤(Q5700,S≤0.5%,长治产):山西

    #
    # 市场价(现货基准价):醋酸:华南  1.6952  72.617087

    # result_list = [
    #     ['C', '谷物_玉米', '玉米衍生品价格.xls', '出厂价:玉米淀粉糖:山东:诸城'],
    #     ['C', '谷物_玉米', '玉米衍生品价格.xls', '出厂价:玉米淀粉糖:河南:鹤壁'],
    #     ['C', '谷物_玉米', '玉米衍生品价格.xls', '出厂价:玉米淀粉糖:吉林:长春'],
    #     ['C', '谷物_玉米', '玉米衍生品价格.xls', '出厂价:玉米胚芽油:山东:邹平'],
    #     ['C', '谷物_玉米', '玉米衍生品价格.xls', '出厂价:玉米淀粉糖:河北:石家庄'],
    #     ['C', '谷物_玉米', '玉米衍生品价格.xls', '出厂价:玉米胚芽油:吉林:长春'],
    # ]
    #
    # result_list = [
    #     ['AP', '谷物_玉米', '玉米衍生品价格.xls', '出厂价:玉米淀粉糖:吉林:长春'],
    #     ['AP', '谷物_玉米', '玉米衍生品价格.xls', '出厂价:玉米淀粉糖:河南:鹤壁'],
    #     ['AP', '谷物_玉米', '玉米衍生品价格.xls', '出厂价:玉米淀粉糖:山东:诸城'],
    # ]

    # result_list = [
    #     ['SR', '软产品_白糖', '天气.xls', '旬平均气温:预报值:下限:广西:玉林'],
    #     ['SR', '软产品_白糖', '天气.xls', '旬平均气温:预报值:上限:广西:玉林'],
    # ]

    # result_list = [
    #     ['RU', '软产品_白糖', '现货价.xls', '现货价:白砂糖:长春'],
    #     ['RU', '软产品_白糖', '现货价.xls', '现货价:白砂糖:徐州'],
    #     ['RU', '软产品_白糖', '现货价.xls', '现货价:白砂糖:广州'],
    #     ['RU', '软产品_白糖', '现货价.xls', '现货价:白砂糖:哈尔滨'],
    # ]

    # result_list = [
    #     ['RB', '能源_动力煤', '国内煤炭产量（月）.xls', '']
    # ]
    #
    # # 出矿价: 动力煤(Q:4500, 鄂尔多斯产):内蒙古
    # # 市场价: 动力煤(Q:5500, 榆林产):陕西
    # # 市场价: 动力煤(Q:5000, S: 1 %, 韩城产):陕西
    # # 曹妃甸港: 平仓价:动力煤(Q5000K)
    #
    # pnl_list = []
    # pos_list = []
    # for res in result_list:
    #     # fut_name, path_name, file_name = 'BU', '化工_聚丙烯', 'PP价格.xls'
    #     # pos_df, pnl_df, data_df, result_df = test_fun(fut_name, path_name, file_name)
    #     # fut_name, path_name, file_name, sp_name = 'NI', '化工_聚丙烯', 'PP价格.xls', '出厂价:聚丙烯PP(T30S):齐鲁石化'
    #     fut_name, path_name, file_name, sp_name = res
    #     pos_df, pnl_df, data_df, result_df = test_fun(fut_name, path_name, file_name)
    #     pnl_sr = pnl_df[sp_name]
    #     pos_sr = pos_df[sp_name]
    #     pnl_sr.name = fut_name
    #     pos_sr.name = fut_name
    #     pnl_list.append(pnl_sr)
    #     pos_list.append(pos_sr)
    #     # 出厂价: 聚丙烯PP(T30S):齐鲁石化
    #     plt.figure(figsize=[16, 10])
    #     plt.plot(pnl_df.index, pnl_df[sp_name].cumsum())
    #     plt.legend()
    #     plt.grid()
    #     savfig_send(subject=f"result {result_df.loc[sp_name]['sharpe']} {result_df.loc[sp_name]['pot']}")
    # pnl_sum = pd.concat(pnl_list, axis=1)
    # pos_concat = pd.concat(pos_list, axis=1)
    # pos_concat.index + timedelta(hours=14, minutes=50)
    # plt.figure(figsize=[16, 10])
    # pnl_cumsum = pnl_sum.mean(1).cumsum()
    # pnl_cumsum = pnl_cumsum.replace(0, np.nan).dropna()
    # plt.plot(pnl_cumsum.index, pnl_cumsum.values)
    # plt.legend()
    # plt.grid()
    # savfig_send(subject='result')

    #     score_df, _, _ = FutIndex.boll_fun(data_df, 5, return_line=True)
    # signal_df = Signal.fun_1(score_df, 1)

    # Y 谷物_玉米 玉米衍生品价格.xls
    #                  sharpe         pot
    # 出厂价:玉米胚芽油:吉林:四平  1.1638  251.423154
    # 出厂价:玉米胚芽油:吉林:长春  1.2810  229.507836

    # V 化工_甲醇 甲醇价格2.xls
    #                  sharpe        pot
    # 出厂估价(高端价):甲醇:甘肃 -1.3741 -72.998811
    # 出厂估价(中间价):甲醇:甘肃 -1.3354 -69.544145
    # 出厂估价(低端价):甲醇:甘肃 -1.1827 -65.551678
    # 出厂估价(低端价):甲醇:宁夏 -1.1023 -67.538280
    # 出厂估价(中间价):甲醇:宁夏 -1.0159 -60.185240

    # V 能源_动力煤 国内外动力煤价格2.xls
    #                                     sharpe         pot
    # 出矿价:动力煤(Q:4500,S:1.2-1.6%,乌海产):内蒙古  1.0418  420.498636
    # 出矿价:动力煤(Q:4500,鄂尔多斯产):内蒙古           1.0418  420.498636
    # 平均价:动力煤:内蒙古                         1.0985  443.288123
    # 纽卡斯尔NEWC动力煤现货价                      1.1256   34.194712
    # 出矿价:动力煤(Q:5500,鄂尔多斯产):内蒙古           1.1749  473.959258

    # TA 谷物_玉米 玉米衍生品价格.xls
    #                  sharpe         pot
    # 出厂价:玉米酒精:黑龙江:肇东  1.0449  212.544380
    # 出厂价:DDGS:吉林:四平   1.0546  173.335969
    # 出厂价:玉米酒精:吉林:松原   1.2760  268.379898
    # 出厂价:玉米酒精:吉林:四平   1.5078  306.000844

    result_list = [
        # ['ZN', '能源_动力煤', '国内外动力煤价格2.xls', '综合平均价格指数:环渤海动力煤(Q5500K)'],
        # ['ZN', '能源_动力煤', '国内外动力煤价格2.xls', '市场价:烟煤(优混):上海'],
        ['ZN', '能源_动力煤', '国内外动力煤价格1.xls', '京唐港:平均价:动力煤(山西产)'],
        ['ZN', '能源_动力煤', '国内外动力煤价格1.xls', '含税价:烟煤(优混):上海'],
        ['RU', '谷物_玉米', '玉米农户出售价.xls', '农户出售价:玉米:河南:漯河', -1],
        ['RU', '谷物_玉米', '玉米农户出售价.xls', '农户出售价:玉米:河南:郑州', -1],
        # ['J', '谷物_玉米', '玉米站台价.xls', '站台价:玉米(国标一等):湖南:岳阳', -1],
        # ['J', '谷物_玉米', '玉米站台价.xls', '站台价:玉米(国标一等):浙江:杭州', -1],
    ]

    pnl_list = []
    pos_list = []
    for res in result_list:
        # fut_name, path_name, file_name = 'BU', '化工_聚丙烯', 'PP价格.xls'
        # pos_df, pnl_df, data_df, result_df = test_fun(fut_name, path_name, file_name)
        # fut_name, path_name, file_name, sp_name = 'NI', '化工_聚丙烯', 'PP价格.xls', '出厂价:聚丙烯PP(T30S):齐鲁石化'
        if len(res) == 4:
            fut_name, path_name, file_name, sp_name = res
            way = 1
        else:
            fut_name, path_name, file_name, sp_name, way = res
        pos_df, pnl_df, data_df, result_df = test_fun(fut_name, path_name, file_name)
        pnl_sr = pnl_df[sp_name] * way
        pos_sr = pos_df[sp_name] * way
        pnl_sr.name = fut_name
        pos_sr.name = fut_name
        pnl_list.append(pnl_sr)
        pos_list.append(pos_sr)
        # 出厂价: 聚丙烯PP(T30S):齐鲁石化
        plt.figure(figsize=[16, 10])
        plt.plot(pnl_df.index, pnl_df[sp_name].cumsum())
        plt.legend()
        plt.grid()
        savfig_send(subject=f"result {result_df.loc[sp_name]['sharpe']} {result_df.loc[sp_name]['pot']}")
    pnl_sum = pd.concat(pnl_list, axis=1)

    pos_concat = pd.concat(pos_list, axis=1)
    pos_concat.index + timedelta(hours=14, minutes=50)
    plt.figure(figsize=[16, 10])
    pnl_cumsum = pnl_sum.mean(1).cumsum()
    pnl_cumsum = pnl_cumsum.replace(0, np.nan).dropna()
    print(bt.AZ_Sharpe_y(pnl_sum.mean(1)))
    plt.plot(pnl_cumsum.index, pnl_cumsum.values)
    plt.legend()
    plt.grid()
    savfig_send(subject='result')
