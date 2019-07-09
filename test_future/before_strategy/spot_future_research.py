# -*- coding:utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import re
import sys
import datetime
import os
from pylab import mpl
import collections
import get_daily_data as gdd

mpl.rcParams['font.sans-serif'] = ['FangSong']
matplotlib.rcParams['axes.unicode_minus'] = False


def sharpe_ratio(assets, turnover):
    pnl = np.diff(assets)
    sharpe = 16 * pnl.mean() / pnl.std()
    pot = 10000 * assets[-1] / sum(turnover)
    return sharpe, pot


class SpotResearch(object):
    def __init__(self, instrument, begin_date='20100101', end_date='20160101', price_method='ClosePrice'):
        """
        初始状态设定
        :param instrument: 品种
        :param begin_date: 回测开始时间
        :param end_date: 回测结束时间
        :param price_method: 价格类型
        """
        assert instrument in ['RB', 'SF', 'SM', 'J', 'JM', 'I', 'OI', 'A', 'Y', 'C', 'ZC', 'CF', 'RU', 'SR', 'MA']
        future, contract = gdd.get_daily_data(instrument, begin_date, end_date)
        self.instru_dict = {'RB': '煤焦钢矿_螺纹线材',
                            'SF': '煤焦钢矿_硅铁锰硅',
                            'SM': '煤焦钢矿_硅铁锰硅',
                            'J': '煤焦钢矿_焦煤焦炭',
                            'JM': '煤焦钢矿_焦煤焦炭',
                            'I': '煤焦钢矿_铁矿石',
                            'OI': '油脂油料_菜油',
                            'A': '油脂油料_大豆',
                            'Y': '油脂油料_豆油',
                            'C': '谷物_玉米',
                            'ZC': '能源_动力煤',
                            'CF': '软商品_棉花',
                            'RU': '化工_橡胶',
                            'SR': '软产品_白糖',
                            'MA': '化工_甲醇'}
        self.begin_date = datetime.datetime.strptime(begin_date, '%Y%m%d')
        self.end_date = datetime.datetime.strptime(end_date, '%Y%m%d')
        self.instrument = instrument
        self.all_future_data = future
        self.price = future[price_method].values
        self.contract = contract.values
        self.use_index = future.index
        self.file_path = f'/mnt/mfs/dat_whs/spot_data/{self.instru_dict[instrument]}'

        self.save_path = f'/mnt/mfs/dat_whs/spot research/20160101/pnl/{instrument}'

        self.all_spot_data = None
        self.spot_data = None
        self.result = {}
        self.file_name = None
        self.spot_name = None
        self.signal = None
        self.pos = None

    def get_all_spot_data(self, file_name):
        """
        根据文件名获取spot data
        :param file_name: 
        :return: 
        """
        self.file_name = file_name
        all_spot_path = os.path.join(self.file_path, file_name)
        all_spot_data = pd.read_excel(all_spot_path, index_col=0, header=2).ix[7:]
        all_spot_data.index = pd.to_datetime(all_spot_data.index)
        self.all_spot_data = all_spot_data
        return all_spot_data

    def get_spot_data(self, spot_name):
        """
        根据数据名获取spot data
        :param spot_name: 
        :return: 
        """
        self.spot_name = spot_name
        self.spot_data = pd.DataFrame(self.all_spot_data[spot_name])
        return self.spot_data

    def back_test(self, pos, if_plt=False, if_trade=False):
        """
        回测与图像储存
        :param pos:
        :param if_plt:
        :param if_trade:
        :return:
        """
        length = len(pos)
        self.pos = pos
        cash = [0.] * length
        dcash = [0.] * length
        portfolio = np.array([0.] * length)
        assets = [0.] * length
        turnover_ask = np.array([0.] * length)
        turnover_bid = np.array([0.] * length)
        times = 0
        win_times = 0
        win_sum = 0.
        loss_times = 0
        loss_sum = 0.
        last_trade = [0, 0]
        pos[0] = 0
        # portfolio[0] = pos[0] * price[0]
        for t in range(1, length):
            cash[t] = cash[t - 1]
            portfolio[t] = pos[t] * self.price[t]
            if pos[t] != pos[t - 1]:
                dcash[t] = self.price[t] * (pos[t] - pos[t - 1])
                times += 1
                profit = last_trade[0] * (self.price[t] - self.price[last_trade[1]])
                if profit > 0:
                    win_times += 1
                    win_sum += profit
                elif profit < 0:
                    loss_times += 1
                    loss_sum += profit
                last_trade[0] = pos[t]
                last_trade[1] = t
            cash[t] = cash[t] - dcash[t]
            assets[t] = cash[t] + portfolio[t]
        dif_pos = pos[1:] - pos[:-1]
        dif_pos = np.array([0] + list(dif_pos))

        turnover_ask[dif_pos > 0] = np.abs(dif_pos[dif_pos > 0] * self.price[dif_pos > 0])
        turnover_bid[dif_pos < 0] = np.abs(dif_pos[dif_pos < 0] * self.price[dif_pos < 0])
        turnovers = turnover_ask + turnover_bid
        average_position_times = 2 * sum(np.abs(portfolio)) / sum(turnovers)
        self.result.update({'assets': assets,
                            'times': times,
                            'turnovers': turnovers,
                            'win_times': win_times,
                            'win_sum': win_sum,
                            'loss_times': loss_times,
                            'loss_sum': loss_sum,
                            'hold': round(average_position_times, 2)})
        sharpe, pot = self.sharpe_and_pot()
        self.result.update({'sharpe': round(sharpe, 3),
                            'pot': round(pot, 3)})
        print('sharpe={},pot={},pnl={},times={},hold={}'.format(sharpe, pot, assets[-1], times, average_position_times))
        file_way = os.path.join(self.save_path, self.file_name[:-4])
        if not os.path.exists(file_way):
            os.makedirs(file_way)
        rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
        title = re.sub(rstr, "_", self.spot_name)
        if if_plt:
            self.plot_pnl()
            print('fig saved')
            plt.savefig(os.path.join(file_way, title + '.jpg'), dpi=300)
            plt.close()
        if if_trade:
            trade_df = pd.DataFrame()
            trade_df['contract'] = self.contract
            trade_df['price'] = self.price
            trade_df['spot'] = self.spot_data.ix[self.use_index].values.ravel()
            trade_df['signal'] = self.signal
            trade_df['pos'] = self.pos
            # trade_df['price'] = self.price
            trade_df['assets'] = assets
            trade_df['turnovers'] = turnovers
            trade_df.index = self.use_index
            print('data_saved')
            trade_df.to_csv(os.path.join(file_way, title + '.csv'))
        return assets, times, turnovers, win_times, win_sum, loss_times, loss_sum, average_position_times

    def plot_pnl(self):
        plt.figure(figsize=[12, 6])
        plt.plot(self.use_index, self.result['assets'], label='sharpe={},pot={},pnl={},times={},hold={}\n'
                                                              'win_times={},win_sum={},loss_times={},loss_sum={}'
                 .format(self.result['sharpe'], self.result['pot'], self.result['assets'][-1],
                         self.result['times'], self.result['hold'], self.result['win_times'],
                         self.result['win_sum'], self.result['loss_times'], self.result['loss_sum']))
        loc1 = np.where(self.pos > 0)
        if len(loc1[0]) != 0:
            plt.scatter(self.use_index[loc1], self.price[loc1], s=5, c='r')
        loc0 = np.where(self.pos == 0)
        if len(loc0[0]) != 0:
            plt.scatter(self.use_index[loc0], self.price[loc0], s=5, c='black')
        loc2 = np.where(self.pos < 0)
        if len(loc2[0]) != 0:
            plt.scatter(self.use_index[loc2], self.price[loc2], s=5, c='lime')
        plt.plot(self.use_index, self.price, c='black')
        plt.ylabel('PNL')
        plt.xlabel('DATE')
        plt.title(self.spot_data.columns[0])
        plt.legend()
        plt.grid()

    def sharpe_and_pot(self):
        dates = pd.to_datetime(self.use_index)
        assets = pd.DataFrame(self.result['assets'], index=dates, columns=['pnl'])
        daily_pnl = assets.resample('D').last().dropna()
        daily_pnl_diff = (daily_pnl - daily_pnl.shift(1)).dropna().values.ravel()
        sharpe_radio = (16 * daily_pnl_diff.mean()) / daily_pnl_diff.std()
        pot = 10000 * assets.values.ravel()[-1] / sum(self.result['turnovers'])
        return sharpe_radio, pot

    def spot_signal(self, method='ffill'):
        """
        简单的diff
        :param method: 现货数据空缺时填充方式
        :return:信号
        """
        signal = self.spot_data.dropna() - self.spot_data.dropna().shift(1)
        signal = signal.ix[self.use_index].fillna(method=method).fillna(0)
        self.signal = signal.values.ravel()
        return self.signal

    def spot_average_signal(self, n, method='ffill'):
        """
        均值信号
        :param n: 周期
        :param method: 现货数据空缺时填充方式
        :return: 信号
        """
        spot = self.spot_data.ix[self.use_index].fillna(method=method)
        ma = self.spot_data.shift(1).ix[self.use_index].fillna(method=method)\
            .rolling(window=n).mean().fillna(method=method)
        # print spot
        # print ma
        signal = (spot - ma).values.ravel()
        self.signal = signal
        return signal

    def spot_rsi_signal(self, n=20, method='ffill'):
        """
        spot data 生成RSI信号
        :param n: 周期
        :param method: 现货数据空缺时填充方式
        :return:信号
        """
        spot = self.spot_data.ix[self.use_index].fillna(method=method)
        signal = spot.diff(1).rolling(window=n).apply(lambda y:
                                                      0 if float(sum(abs(y)))*100 == 0
                                                      else sum([x for x in y if x > 0])/float(sum(abs(y)))*100-50)

        self.signal = signal.values.ravel()
        return self.signal

    def position(self, signal, limit=0):
        """
        根据signal生成position
        :param signal: 信号
        :param limit: 参数
        :return: 
        """
        pos = np.array([0] * len(signal))
        for i in range(1, len(signal)):
            if np.isnan(signal[i]):
                pos[i] = 0
            else:
                if signal[i] > limit:
                    pos[i] = 1
                elif signal[i] < -limit:
                    pos[i] = -1
                else:
                    pos[i] = pos[i - 1]
        for i in range(1, len(signal)):
            if i < len(signal) - 1:
                if self.contract[i] != self.contract[i + 1]:
                    pos[i] = 0
        self.pos = pos
        return pos

    def log_return(self, step=1):
        """
        计算期货数据的log return
        :param step: 周期
        :return:
        """
        a = np.array([np.log(x) for x in self.price])
        assert type(step) == int and step != 0
        if step < 0:
            df_log = pd.DataFrame(a[-step:] - a[:step], index=self.use_index[:step], columns=['ClosePrice'])
            return df_log
        else:
            df_log = pd.DataFrame(a[step:] - a[:-step], index=self.use_index[step:], columns=['ClosePrice'])
            return df_log

    def cdf_markout(self, signal, step=1, step_list=list(range(1, 40)), if_save=False):
        """
        给定signal，绘制cdf和markout图像
        :param signal:信号
        :param step: 信号周期
        :param step_list: retrun周期，用来绘制markout
        :param if_save: 图像存储
        :return:
        """
        signal = signal[:-step]
        f_return = self.log_return(step).values.ravel()
        # print signal, f_return
        f_return_m = f_return - f_return.mean()
        a = np.argsort(signal)
        plt.figure(figsize=(12, 10))
        p1 = plt.subplot(321)
        p2 = plt.subplot(322)
        p3 = plt.subplot(323)
        p4 = plt.subplot(324)
        p5 = plt.subplot(313)
        plt.tight_layout(pad=5, h_pad=2, w_pad=2, rect=None)
        # print len(f_return), len(a), a
        p1.plot(np.cumsum(f_return[a]))
        p1.set_title('cumsum return')
        p1.grid(axis='both', linestyle='--')

        p2.plot(signal[a], np.cumsum(f_return[a]))
        p2.set_title('signal and cumsum return')
        p2.grid(axis='both', linestyle='--')

        p3.plot(np.cumsum(f_return_m[a]))
        p3.set_title('cumsum mean return')
        p3.grid(axis='both', linestyle='--')

        p4.plot(signal[a], np.cumsum(f_return_m[a]))
        p4.set_title('signal and cumsum mean return')
        p4.grid(axis='both', linestyle='--')

        mark_out = []
        for x in step_list:
            f_return = self.log_return(x).values.ravel()
            mark_out.append(sum(f_return * self.pos[:-x]) / (x * len(f_return)))
        p5.plot(step_list, mark_out)
        p5.grid(axis='both', linestyle='--')
        p5.set_title('mark out')
        plt.suptitle(self.spot_data.columns[0] + ' CDF figure')
        if if_save:
            file_way = os.path.join(self.save_path, self.file_name[:-4])
            rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
            title = re.sub(rstr, "_", self.spot_name)
            plt.savefig(os.path.join(file_way, title + '_CDF&markout.jpg'), dpi=300)

    def yearly_analysis(self, if_save=False):
        """
        策略每年的sharpe，pot等图像绘制
        :param if_save: 图片存储
        :return: None
        """
        year_list = ['20100104', '20110103', '20120103', '20130103', '20140103', '20150105', '20160104',
                     '20170103', '20180103', '20190101']
        assets_dict = collections.OrderedDict()
        plt.figure(figsize=[12, 10])
        p1 = plt.subplot(3, 2, 1)
        p2 = plt.subplot(3, 2, 2)
        p3 = plt.subplot(3, 2, 3)
        p4 = plt.subplot(3, 2, 4)
        p5 = plt.subplot(3, 2, 5)
        p6 = plt.subplot(3, 2, 6)
        plt.suptitle(self.spot_name)
        plt.tight_layout(pad=3, h_pad=2, w_pad=1, rect=None)
        assets_df = pd.DataFrame()
        assets_df['assets'] = self.result['assets']
        assets_df['turnovers'] = self.result['turnovers']
        assets_df.index = self.use_index
        static_list = [[], [], []]

        p1.plot(self.use_index, self.spot_data.ix[self.use_index].fillna(method='ffill').values, color='gray')
        for i in range(len(year_list) - 1):
            year_begin = pd.to_datetime(year_list[i])
            year_end = pd.to_datetime(year_list[i + 1])
            year_cut = assets_df.loc[year_begin:year_end]
            if len(year_cut):
                assets_dict[year_list[i][:4]] = year_cut
                # = sharpe_ratio(year_cut['assets'].values, year_cut['turnovers'].values)
                sharpe, pot = sharpe_ratio(year_cut['assets'].values - year_cut['assets'].values[0],
                                           year_cut['turnovers'].values)
                static_list[0].append(year_list[i][:4])
                static_list[1].append(sharpe)
                static_list[2].append(pot)
                p1.plot(year_cut['assets'].index, year_cut['assets'].values, )
                if year_list[i] != '20180103':
                    p1.axvline(year_end, linestyle='--')
                p2.plot(year_cut['assets'].values - year_cut['assets'].values[0], label='{}'.format(year_list[i][:4]))
                # label='{},sharpe={},pot={}'.format(year_list[i][:4], sharpe, pot))
        p1.plot(self.use_index, self.price, c='black',
                label='{}, sharpe={},pot={},hold={}'
                .format(self.instrument, self.result['sharpe'], self.result['pot'], self.result['hold']))
        p1.set_ylabel('PNL')
        p1.grid(axis='y', linestyle='--')
        p1.legend(loc='best', fontsize=10)

        p2.grid(axis='y', linestyle='--')
        p2.set_ylabel('PNL')
        p2.legend(loc='best', fontsize=10)

        # print self.use_index, self.signal, len(self.use_index), len(self.signal)
        p3.plot(self.use_index, self.signal)
        p3.set_xlabel('signal')
        p3.grid(axis='y', linestyle='--')
        p3.legend(loc='best', fontsize=10)

        cut_num = int(0.02 * len(self.signal))
        print(cut_num)
        print(sorted(self.signal))
        print(sorted(self.signal)[cut_num:-cut_num])
        p4.hist(sorted(self.signal)[cut_num:-cut_num], bins=30, color='green')
        p4.grid(axis='both', linestyle='--')
        p4.set_xlabel('signal')

        ind = np.linspace(0.5, len(static_list[0]) - 0.5, len(static_list[0]))
        p5.set_xticks(ind)
        p5.set_xticklabels(static_list[0])
        p5.grid(axis='y', linestyle='--')
        p5.set_xlabel('sharpe')
        p5.bar(ind, static_list[1], 0.2, color='green')

        p6.set_xticks(ind)
        p6.set_xticklabels(static_list[0])
        p6.grid(axis='y', linestyle='--')
        p6.set_xlabel('POT')
        p6.bar(ind, static_list[2], 0.2, color='green')

        if if_save:
            file_way = os.path.join(self.save_path, self.file_name[:-4])
            rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
            title = re.sub(rstr, "_", self.spot_name)
            plt.savefig(os.path.join(file_way, title + '_analysis.jpg'), dpi=300)

    def multiply_plot(self, pos_set, title=[]):
        """
        把多个仓位信号绘制在一张图上
        :param pos_set:  position set
        :return: figure
        """
        if not title:
            title = range(len(title))
        plt.figure(figsize=[12, 6])
        for i in range(len(pos_set)):
            pos = pos_set[i]
            length = len(pos)
            self.pos = pos
            cash = [0.] * length
            dcash = [0.] * length
            portfolio = np.array([0.] * length)
            assets = [0.] * length
            turnover_ask = np.array([0.] * length)
            turnover_bid = np.array([0.] * length)
            times = 0
            win_times = 0
            win_sum = 0.
            loss_times = 0
            loss_sum = 0.
            last_trade = [0, 0]
            pos[0] = 0
            for t in range(1, length):
                cash[t] = cash[t - 1]
                portfolio[t] = pos[t] * self.price[t]
                if pos[t] != pos[t - 1]:
                    dcash[t] = self.price[t] * (pos[t] - pos[t - 1])
                    times += 1
                    profit = last_trade[0] * (self.price[t] - self.price[last_trade[1]])
                    if profit > 0:
                        win_times += 1
                        win_sum += profit
                    elif profit < 0:
                        loss_times += 1
                        loss_sum += profit
                    last_trade[0] = pos[t]
                    last_trade[1] = t
                cash[t] = cash[t] - dcash[t]
                assets[t] = cash[t] + portfolio[t]
            dif_pos = pos[1:] - pos[:-1]
            dif_pos = np.array([0] + list(dif_pos))

            turnover_ask[dif_pos > 0] = np.abs(dif_pos[dif_pos > 0] * self.price[dif_pos > 0])
            turnover_bid[dif_pos < 0] = np.abs(dif_pos[dif_pos < 0] * self.price[dif_pos < 0])
            turnovers = turnover_ask + turnover_bid
            average_position_times = 2 * sum(np.abs(portfolio)) / sum(turnovers)
            self.result = {}
            self.result.update({'assets': assets,
                                'times': times,
                                'turnovers': turnovers,
                                'win_times': win_times,
                                'win_sum': win_sum,
                                'loss_times': loss_times,
                                'loss_sum': loss_sum,
                                'hold': round(average_position_times, 2)})
            sharpe, pot = self.sharpe_and_pot()
            self.result.update({'sharpe': round(sharpe, 3),
                                'pot': round(pot, 3)})
            print('sharpe={},pot={},pnl={},times={},hold={}'.format(sharpe, pot, assets[-1], times,
                                                                    average_position_times))
            if i == 0:
                plt.plot(self.use_index, self.result['assets'],
                         label='{},sharpe={},pot={},pnl={},times={},hold={}\n'
                               'win_times={},win_sum={},loss_times={},loss_sum={}'
                         .format(title[i], self.result['sharpe'], self.result['pot'], self.result['assets'][-1],
                                 self.result['times'], self.result['hold'], self.result['win_times'],
                                 self.result['win_sum'], self.result['loss_times'], self.result['loss_sum']))
            else:
                plt.plot(self.use_index, self.result['assets'], linestyle='--',
                         label='{},sharpe={},pot={},pnl={},times={},hold={}\n'
                               'win_times={},win_sum={},loss_times={},loss_sum={}'
                         .format(title[i], self.result['sharpe'], self.result['pot'], self.result['assets'][-1],
                                 self.result['times'], self.result['hold'], self.result['win_times'],
                                 self.result['win_sum'], self.result['loss_times'], self.result['loss_sum']))
        plt.ylabel('PNL')
        plt.xlabel('DATE')
        plt.title('multiply signal compare')
        plt.legend()
        plt.grid()


if __name__ == '__main__':
    instrument_list = ['ZC']
    for instrument1 in instrument_list:
        print('___________________________________________________')
        print(instrument1)
        begin_time = '20100101'
        end_time = '20180220'
        instru_info = SpotResearch(instrument1, begin_time, end_time, 'ClosePrice')
        # plt.ion()
        for file_name1 in os.listdir(instru_info.file_path):
            file_name1 = u'国内外动力煤价格1.xls'
            # file_name1 = u'天气.xls'
            print('file name is' + file_name1)
            instru_info.get_all_spot_data(file_name1)
            for spot_name1 in instru_info.all_spot_data.columns:
                spot_name1 = u'秦皇岛港:平仓价:动力末煤(Q4500):山西产'
                # spot_name1 = u'旬平均气温:预报值:下限:广西:来宾'
                # u'旬平均气温:预报值:上限:广西:百色'
                print('spot name is' + spot_name1)
                instru_info.get_spot_data(spot_name1)
                signal1 = instru_info.spot_signal()
                signal2 = instru_info.spot_rsi_signal(3, method='ffill')
                signal3 = instru_info.spot_rsi_signal(5, method='ffill')
                signal4 = instru_info.spot_rsi_signal(10, method='ffill')
                signal5 = instru_info.spot_rsi_signal(20, method='ffill')

                pos1 = instru_info.position(signal1)
                # pos2 = instru_info.position(signal2, limit=30)
                # pos3 = instru_info.position(signal3, limit=30)
                # pos4 = instru_info.position(signal4, limit=30)
                # pos5 = instru_info.position(signal5, limit=30)
                # instru_info.back_test(pos1, if_plt=True, if_trade=True)
                # instru_info.yearly_analysis(if_save=True)
                # instru_info.cdf_markout(signal1, step=2, if_save=True)
                # instru_info.multiply_plot([pos1, pos2, pos3, pos4, pos5], ['raw', 'RSI3', 'RSI5', 'RSI10', 'RSI20'])
                fig_save = r'/mnt/mfs/dat_whs/spot_research/result_save/'
                rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
                title = re.sub(rstr, "_", spot_name1)
                if not os.path.exists(os.path.join(fig_save, instrument1, file_name1)):
                    os.makedirs(os.path.join(fig_save, instrument1, file_name1))

                plt.savefig(os.path.join(fig_save, instrument1, file_name1, title + '.jpg'))
                plt.show()
