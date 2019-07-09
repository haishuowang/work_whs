# -*- coding:utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import datetime

sys.path.append(r'..\assetManagement')
sys.path.append(r'..\get_data')
sys.path.append(r'..\backtest')
import my_email
from email_program import df2html
import backtest as bt
import get_daily_data as gdd
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong']
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import os

xmajorLocator = MultipleLocator(200)  # 将x主刻度标签设置为20的倍数
xmajorFormatter = FormatStrFormatter('%1.1f')  # 设置x轴标签文本的格式
xminorLocator = MultipleLocator(200)  # 将x轴次刻度标签设置为5的倍数


def MDD(arr):
    return arr - np.maximum.accumulate(arr)


def display_format(raw):
    if np.fabs(raw) > 1e12:
        dis = raw / float(1e12)
        dis = '{:.2f}t'.format(dis)
    elif np.fabs(raw) > 1e9:
        dis = raw / float(1e9)
        dis = '{:.2f}b'.format(dis)
    elif np.fabs(raw) > 1e6:
        dis = raw / float(1e6)
        dis = '{:.2f}m'.format(dis)
    elif np.fabs(raw) > 1e3:
        dis = raw / float(1e3)
        dis = '{:.2f}k'.format(dis)
    else:
        dis = '{:.2f}'.format(raw)

    return dis


def portfolio_info(df):
    add_df = pd.DataFrame(df.sum()).T
    add_df['Future'] = 'portfolio'
    df = df.append(add_df)
    return df


def plot_report(df):
    pnl_plot = df['assets']
    turnover_plot = df['Turnover']
    GMV_plot = df['GMV']
    Notional_plot = df['Notional']
    # date_list = df.index
    date_list = pd.Series([x.strftime('%Y%m%d') for x in df.index], index=df.index)
    print(date_list)
    fig = plt.figure()
    fig.suptitle('report figure', fontsize=40)

    ax0 = fig.add_subplot(3, 2, 1)
    if np.diff(pnl_plot).std() > 0:
        sharpe_ratio = (np.diff(pnl_plot).mean()) / (np.diff(pnl_plot).std()) * 16
    else:
        sharpe_ratio = 0
    ar = np.diff(pnl_plot).mean()
    vol = np.diff(pnl_plot).std()
    pot = pnl_plot.iloc[-1] / turnover_plot.sum() * 10000
    hold = 2 * sum(df['GMV'].values) / sum(df['Turnover'].values)
    ax0.plot(pnl_plot.values, label='PnL: sharpe_ratio:{:.2f} PoT:{:.2f} \n        Vol:{} AP:{} Hold:{:.2f}'.
             format(sharpe_ratio, pot, display_format(vol), display_format(ar), hold))
    ax0.legend(loc='best', fontsize=10)
    label_num = min(8, len(date_list) - 1)
    xmin, xmax = ax0.set_xlim()
    locs = np.linspace(0, len(date_list) - 1, label_num)
    date_locs = np.linspace(0, len(date_list) - 1, label_num)
    locs = pd.Series(locs).apply(int)
    date_locs = pd.Series(date_locs).apply(int)
    ax0.set_xticks(locs)
    ax0.set_xticklabels(date_list[date_locs], rotation=30)
    ax0.set_ylim(-400000, 1400000)
    ylocs = np.linspace(-200000, 1000000, 6)
    # ymin, ymax = ax0.set_ylim()
    # locs = np.linspace(ymin, ymax, label_num)
    ax0.set_yticks(ylocs)
    yticks = (pd.Series(ax0.get_yticks()).apply(display_format)).values
    ax0.set_yticklabels(yticks)
    ax0.grid()

    ax1 = fig.add_subplot(3, 2, 2)
    ax1.plot(turnover_plot.values, label='Turnover')
    ax1.legend(loc='best', fontsize=10)
    label_num = min(8, len(date_list) - 1)
    xmin, xmax = ax1.set_xlim()
    locs = np.linspace(0, len(date_list) - 1, label_num)
    date_locs = np.linspace(0, len(date_list) - 1, label_num)
    locs = pd.Series(locs).apply(int)
    date_locs = pd.Series(date_locs).apply(int)
    ax1.set_xticks(locs)
    ax1.set_xticklabels(date_list[date_locs], rotation=30)
    ax1.set_ylim(-600000, 6000000)
    ylocs = np.linspace(-600000, 6000000, 10)
    # ymin, ymax = ax0.set_ylim()
    # locs = np.linspace(ymin, ymax, label_num)
    ax1.set_yticks(ylocs)
    yticks = (pd.Series(ax1.get_yticks()).apply(display_format)).values
    ax1.set_yticklabels(yticks)
    ax1.grid()

    ax2 = fig.add_subplot(3, 2, 3)
    ax2.plot(GMV_plot.values, label='GMV')
    ax2.legend(loc='best', fontsize=10)
    label_num = min(8, len(date_list) - 1)
    xmin, xmax = ax2.set_xlim()
    locs = np.linspace(0, len(date_list) - 1, label_num)
    date_locs = np.linspace(0, len(date_list) - 1, label_num)
    locs = pd.Series(locs).apply(int)
    date_locs = pd.Series(date_locs).apply(int)
    ax2.set_xticks(locs)
    ax2.set_xticklabels(date_list[date_locs], rotation=30)
    ax2.set_ylim(0, 6000000)
    ylocs = np.linspace(0, 6000000, 11)
    ax2.set_yticks(ylocs)
    yticks = (pd.Series(ax2.get_yticks()).apply(display_format)).values
    ax2.set_yticklabels(yticks)
    ax2.grid()

    ax3 = fig.add_subplot(3, 2, 4)
    ax3.plot(Notional_plot.values, label='Notional')
    ax3.legend(loc='best', fontsize=10)
    label_num = min(8, len(date_list) - 1)
    xmin, xmax = ax3.set_xlim()
    locs = np.linspace(0, len(date_list) - 1, label_num)
    date_locs = np.linspace(0, len(date_list) - 1, label_num)
    locs = pd.Series(locs).apply(int)
    date_locs = pd.Series(date_locs).apply(int)
    ax3.set_xticks(locs)
    ax3.set_xticklabels(date_list[date_locs], rotation=30)
    ax3.set_ylim(-6000000, 6000000)
    ylocs = np.linspace(-6000000, 6000000, 10)
    ax3.set_yticks(ylocs)
    yticks = (pd.Series(ax3.get_yticks()).apply(display_format)).values
    ax3.set_yticklabels(yticks)
    ax3.grid()

    MDS = MDD(pnl_plot)
    ax4 = fig.add_subplot(3, 1, 3)
    ax4.bar(left=range(0, len(date_list)), height=MDS, label='DrawDown')
    ax4.legend(loc='best', fontsize=10)
    label_num = min(10, len(date_list) - 1)
    xmin, xmax = ax4.set_xlim()
    locs = np.linspace(0, len(date_list) - 1, label_num)
    date_locs = np.linspace(0, len(date_list) - 1, label_num)
    locs = pd.Series(locs).apply(int)
    date_locs = pd.Series(date_locs).apply(int)
    ax4.set_xticks(locs)
    ax4.set_xticklabels(date_list[date_locs], rotation=30)
    ax4.set_ylim(-400000, 0)
    ylocs = np.linspace(-400000, 0, 7)
    ax4.set_yticks(ylocs)
    yticks = (pd.Series(ax4.get_yticks()).apply(display_format)).values
    ax4.set_yticklabels(yticks)
    ax4.grid()

    fig.set_size_inches(15, 15)
    plt.savefig(r'\\192.168.126.38\ResearchWorks\Fulltime\haishuo.wang\Trade_file\plot_figure.png')


def creat_paper_report(instrument_list):
    month = datetime.datetime.today().year * 10000 + datetime.datetime.today().month * 100 + 1
    # month = 20171101
    report_data = pd.DataFrame()
    plot_data = pd.DataFrame()
    position_data = pd.DataFrame()

    report_columns = ['Future', 'pnl', 'tpnl', 'ypnl', 'Fee', 'NetPnl', 'Notional', 'GMV', 'turnover', 'POT', 'YTD',
                      'MTD', 'Vol5', 'Vol60']
    position_columns = ['Future', 'TodayPosition', 'NewContract', 'NewPosiChange', 'OldContract', 'OldPosiChange']
    for instrument in instrument_list:
        save_data = pd.read_csv(
            r'\\192.168.126.38\ResearchWorks\Fulltime\haishuo.wang\Trade_file\paper_trading_1\{}_lag=0.csv'
            .format(instrument))
        pnl = save_data['PNL'].values[-1]
        tpnl = 0
        ypnl = pnl
        fee = 0
        netpnl = pnl
        Notional = save_data['Notional'].values[-1]
        GMV = save_data['GMV'].values[-1]
        Turnover = save_data['Turnover'].values[-1]
        if Turnover == 0:
            POT = 0
        else:
            POT = float(pnl) / Turnover
        YTD = save_data['Assets'].values[-1] - save_data[save_data['date'] < 20180101]['Assets'].values[-1]
        MTD = save_data['Assets'].values[-1] - save_data[save_data['date'] < month]['Assets'].values[-1]
        Vol5 = save_data['PNL'].values[-5:].std()
        Vol60 = save_data['PNL'].values[-60:].std()
        report_data = report_data.append(pd.DataFrame([[instrument, pnl, tpnl, ypnl, fee, netpnl, Notional, GMV,
                                                        Turnover, POT, YTD, MTD, Vol5, Vol60]], columns=report_columns))

        TodayPosition = save_data['Position'].values[-1]
        NewContract = save_data['Contract'].values[-1]
        NewPosiChange = save_data['Position'].values[-1] - save_data['Position'].values[-2]
        OldContract = save_data['Contract'].values[-2]
        OldPosiChange = 0
        position_data = position_data.append(pd.DataFrame([[instrument, TodayPosition, NewContract, NewPosiChange,
                                                            OldContract, OldPosiChange]], columns=position_columns))
        if len(plot_data) == 0:
            plot_data['assets'] = save_data['Assets'].values
            plot_data['Turnover'] = save_data['Turnover'].values
            plot_data['GMV'] = save_data['GMV'].values
            plot_data['Notional'] = save_data['Notional'].values
            plot_data.index = pd.to_datetime(save_data['date'].astype(str))
        else:
            plot_data['assets'] += save_data['Assets'].values
            plot_data['Turnover'] += save_data['Turnover'].values
            plot_data['GMV'] += save_data['GMV'].values
            plot_data['Notional'] += save_data['Notional'].values
    plot_report(plot_data)
    print(report_data)
    add_df = pd.DataFrame(report_data.sum()).T
    add_df['Future'] = 'portfolio'
    report_data_sum = report_data.append(add_df)
    # report_data_sum = portfolio_info(report_data)
    return report_data_sum, position_data


def sell_or_buy(x):
    if x > 0:
        return 'BUY'
    elif x < 0:
        return 'SELL'
    else:
        return 0


def Q1(signal, limit, hold_time=1):
    pos_row = np.array([0] * len(signal))
    pos = np.array([0.] * len(signal))
    for i in range(len(signal)):
        if signal[i] > limit:
            pos_row[i] = 1
        elif signal[i] < -limit:
            pos_row[i] = -1
        else:
            pos_row[i] = pos_row[i - 1]
    # print(hold_time)
    for i in range(len(signal)):
        if i < hold_time - 1:
            pos[i] = sum(pos_row[0:i + 1]) / float(len(pos_row[:i + 1]))
        else:
            pos[i] = sum(pos_row[i - hold_time + 1:i + 1]) / float(hold_time)
            # print(sum(pos_row[i-hold_time+1:i+1]), hold_time, pos[i])
    return pos


def position(signal, top, bottom):
    pos = np.array([0] * len(signal))
    for i in range(len(signal)):
        if np.isnan(signal[i]):
            pos[i] = 0
        else:
            if signal[i] > top:
                pos[i] = 1
            elif signal[i] < bottom:
                pos[i] = -1
            else:
                pos[i] = pos[i - 1]
    return pos


def position_reverse(signal, top, bottom):
    pos = np.array([0] * len(signal))
    for i in range(len(signal)):
        if np.isnan(signal[i]):
            pos[i] = 0
        else:
            if signal[i] > top:
                pos[i] = -1
            elif signal[i] < bottom:
                pos[i] = 1
            else:
                pos[i] = pos[i - 1]
    return pos


def trade_info(future_price, signal, lag=0):
    info = pd.concat([future_price, signal], axis=1)
    for i in range(1, len(info)):
        if np.isnan(info.ix[:, 0].values[i]):
            info.ix[:, 0].values[i] = info.ix[:, 0].values[i - 1]
        if np.isnan(info.ix[:, 1].values[i]):
            info.ix[:, 1].values[i] = 0
    info = info.dropna()
    price = info.ix[:, 0].values
    signal = info.ix[:, 1].values
    pos = position(signal[1:], 0, 0)
    df = pd.DataFrame()
    df['price'] = price[1:]
    df['position'] = pos
    df['signal'] = signal[1:]
    df.index = info.index[1:]
    df.ix[:, 1:] = df.ix[:, 1:].shift(lag)
    df.ix[:lag, 1:] = 0
    return df


def future_pos(spot_data, future_price):
    spot_data = spot_data.dropna()
    signal = pd.DataFrame(np.diff(spot_data.values.astype(float)), index=pd.to_datetime(spot_data.index[1:]),
                          columns=[spot_data.name])
    df = trade_info(future_price, signal)
    return signal, df


def future_pos_mix(spot_data, future_price):
    spot_data = spot_data.dropna()
    pos = ([0.] * len(spot_data.index))
    for i in range(5):
        lag = i + 1
        signal = spot_data.values.astype(float) - spot_data.shift(lag).values.astype(float)
        a = Q1(signal, limit=0, hold_time=1)
        pos += a
    pos = pos / 5
    print(pos)
    df = pd.DataFrame()
    df['price'] = future_price[spot_data.index]
    df['position'] = pos
    df.index = pd.to_datetime(np.array(spot_data.index).astype(str))
    return pos, df


def future_info(instrument, future_price, lag=0):
    use_data = pd.read_csv(
        r'\\192.168.126.38\ResearchWorks\Fulltime\haishuo.wang\Spot\spot research\20160101\pnl\{}\{}_data.csv'
            .format(instrument, instrument), index_col=0, encoding='gbk')

    use_data = use_data.ix[7:]
    use_data.ix[:, :] = use_data.ix[:, :].shift(lag)
    use_data.ix[:lag, :] = 0

    df_list = np.array([None] * len(use_data.columns))
    for i in range(len(use_data.columns)):
        spot_data = use_data.ix[:, i]
        signal, df = future_pos(spot_data, future_price)
        df_list[i] = df
    return df_list


def date_pos(df_list, date):
    """
    用时间切片的方式获取每个数据的position，不会出错。
    :param df_list:
    :param date:
    :return:
    """
    date = pd.to_datetime(date)
    pos_list = [None] * len(df_list)
    for i in range(len(df_list)):
        df = df_list[i]
        if len(df.ix[df.index <= date, 1].values) == 0:
            pos_list[i] = 0
        else:
            pos_list[i] = df.ix[df.index <= date, 1].values[-1]
    print(date)
    pos = sum(pos_list)
    return pos, pos_list


def spot_bt(future_data, contract, df_list, contract_size, weight):
    """
    paper trading 1的backtest
    :param future_data: 期货数据
    :param contract: 主力合约数据
    :param df_list: 由现货数据生成的仓位信息
    :param contract_size: 合约规模
    :param weight: 调整Vol的权重
    :return:
    """
    df = pd.DataFrame()
    length = len(future_data.index)
    cash = np.array([0.] * length)
    dcash = np.array([0.] * length)
    portfolio = np.array([0.] * length)
    assets = np.array([0.] * length)
    pos = np.array([0.] * length)
    pos_all = [None] * length
    turnover_ask = np.array([0.] * length)
    turnover_bid = np.array([0.] * length)
    price = np.array(future_data['ClosePrice'].values)
    initial_account = 500000
    can_buy_or_sell = np.array([0.] * length)
    account_assets = np.array([0.] * length)
    account_assets[0] = initial_account
    account_pos = np.array([0.] * length)
    account_cash = np.array([0.] * length)
    account_cash[0] = initial_account
    account_dcash = np.array([0.] * length)
    account_portfolio = np.array([0.] * length)
    for i in range(1, length):
        can_buy_or_sell[i] = int(initial_account / (price[i] * contract_size))
        date = future_data.index[i]
        pos[i], pos_all[i] = date_pos(df_list, date)
        # 根据weight控制Vol
        pos[i] = round(pos[i] * weight, 0)
        portfolio[i] = pos[i] * price[i]

        if pos[i] != pos[i - 1]:
            dcash[i] = price[i] * (pos[i] - pos[i - 1])
            if int(can_buy_or_sell[i] * pos[i]) != int(can_buy_or_sell[i] * pos[i - 1]):
                account_pos[i] = int(can_buy_or_sell[i] * pos[i])
                account_dcash[i] = contract_size * price[i] * (account_pos[i] - account_pos[i - 1])
            else:
                account_pos[i] = account_pos[i - 1]
        else:
            account_pos[i] = account_pos[i - 1]

        account_portfolio[i] = account_pos[i] * contract_size * price[i]
        cash[i] = cash[i - 1] - dcash[i]
        account_cash[i] = account_cash[i - 1] - account_dcash[i]
        assets[i] = cash[i] + portfolio[i]
        account_assets[i] = account_portfolio[i] + account_cash[i]
    GMV = abs(portfolio)
    pos_all[0] = [0] * len(pos_all[1])
    dif_pos = pos[1:] - pos[:-1]
    dif_pos = np.array([0] + list(dif_pos))
    turnover_ask[dif_pos > 0] = np.abs(dif_pos[dif_pos > 0] * price[dif_pos > 0])
    turnover_bid[dif_pos < 0] = np.abs(dif_pos[dif_pos < 0] * price[dif_pos < 0])
    turnovers = turnover_ask + turnover_bid
    Hold = 2 * sum(GMV) / sum(turnovers)

    account_dif_pos = account_pos[1:] - account_pos[:-1]
    account_dif_pos = np.array([0] + list(account_dif_pos))
    account_turnovers = np.abs(account_dif_pos * price * contract_size)
    sharpe, PoT = bt.sharpe_and_pot(future_data.index, assets, turnovers)
    max_draw_down = bt.maxdrawdown(assets)
    loc1 = np.where(pos > 0)
    if len(loc1[0]) != 0:
        plt.scatter(future_data.index[loc1], price[loc1], s=5, c='r')
    loc0 = np.where(pos == 0)
    if len(loc0[0]) != 0:
        plt.scatter(future_data.index[loc0], price[loc0], s=5, c='black')
    loc2 = np.where(pos < 0)
    if len(loc2[0]) != 0:
        plt.scatter(future_data.index[loc2], price[loc2], s=5, c='lime')
    plt.fill_between(future_data.index, max_draw_down, alpha='0.5')
    plt.plot(future_data.index, price, color='black', linestyle='--')
    plt.plot(future_data.index, assets * contract_size, color='darkorchid')
    plt.legend(['close',
                'sharpe=' + str(sharpe[0]) + ', ' + 'PoT=' + str(PoT) + ', '
                + 'assets=' + str(assets[-1]) + ', ' + 'Hold=' + str(Hold)])
    plt.grid()
    df['Position'] = pos
    df['Contract'] = contract.values
    df['Price'] = price
    df['Turnover'] = turnovers * contract_size
    df['Notional'] = portfolio * contract_size
    df['GMV'] = np.abs(portfolio) * contract_size
    df['PNL'] = np.array([0] + list(np.diff(assets))) * contract_size
    df['Assets'] = assets * contract_size
    df.index = future_data.index
    plt.xticks(rotation=30)
    plt.axvline(x=pd.to_datetime('20160101'))
    return df, pos_all


if __name__ == '__main__':
    contract_size = {'j': 100, 'jm': 60, 'rb': 10}
    # 控制权重（仓位）
    weights = {'j': 0.6, 'jm': 1.2, 'rb': 10}
    exchange = {'j': [90, 'DCE'], 'jm': [90, 'DCE'], 'rb': [89, 'SHFE']}
    trade_df = pd.DataFrame()
    instrument_list = ['j', 'jm', 'rb']
    begin_date = '20170101'
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    # end_date = '20180226'
    # to = ['haishuo.wang@dfc.sh']
    to = ['tamago.hu@dfc.sh', 'michael.xie@dfc.sh', 'midas@dfc.sh']
    for instrument in instrument_list:
        num = contract_size[instrument]
        weight = weights[instrument]
        # 期货数据
        future_data, contract = gdd.get_daily_data(instrument, begin_date, end_date)

        if future_data.index[-1].strftime('%Y%m%d') != end_date:
            print('{} error'.format(instrument))
            continue
        future_ClosePrice = future_data['SettlementPrice']
        df_list = future_info(instrument, future_ClosePrice, lag=0)
        df, pos_all = spot_bt(future_data, contract, df_list, num, weight)
        df.index = [x.strftime('%Y%m%d') for x in df.index]
        df.index.name = 'date'
        df.to_csv(
            r'\\192.168.126.38\ResearchWorks\Fulltime\haishuo.wang\Trade_file\paper_trading_1\{}_lag=0.csv'.format(
                instrument),
            encoding='utf-8')
        portfolio_info = exchange[instrument]
        part_trade_df = pd.DataFrame()
        pos = df['Position'].values
        pos_change = df['Position'].values[-1] - df['Position'].values[-2]
        price_path = r'\\192.168.126.38\DFCData\RawData\ExtradayData\Wind\Future\WindData_Futures' \
                     r'\Futures_DailyData_Wind\{}\{}\{}'.format(end_date[:4], end_date[:6], end_date)

        if contract[-1] != contract[-2]:
            contract_1 = pos[-1]
            contract_2 = -pos[-2]
            way_1 = sell_or_buy(contract_1)
            way_2 = sell_or_buy(contract_2)

            part_trade_df['FromIQ'] = ['DFC_LMH_F_Berserker'] * 2
            part_trade_df['FromAccount'] = [portfolio_info[0]] * 2
            part_trade_df['FromPortfolio'] = ['MD_CashInfo_{}_MP1'.format(instrument)] * 2
            FeedCode_1 = instrument + str(int(contract[-1])) + '.' + portfolio_info[1]
            FeedCode_2 = instrument + str(int(contract[-2])) + '.' + portfolio_info[1]
            part_trade_df['FeedCode'] = [FeedCode_1, FeedCode_2]
            part_trade_df['Qty'] = [int(abs(contract_1)), int(abs(contract_2))]
            price_1 = pd.read_csv(os.path.join(price_path, FeedCode_1 + '.csv'))['ClosePrice'][0]
            price_2 = pd.read_csv(os.path.join(price_path, FeedCode_1 + '.csv'))['ClosePrice'][0]
            part_trade_df['Price'] = [price_1, price_2]
            part_trade_df['Way'] = [way_1, way_2]
            part_trade_df['ShType'] = ['SPECULATING'] * 2
            part_trade_df['ToPortfolio'] = ['MD_WDTWAP_{}1'.format(portfolio_info[1])] * 2
            print(part_trade_df)
            trade_df = trade_df.append(part_trade_df)

        else:
            way = sell_or_buy(pos_change)
            part_trade_df['FromIQ'] = ['DFC_LMH_F_Berserker']
            part_trade_df['FromAccount'] = [portfolio_info[0]]
            part_trade_df['FromPortfolio'] = ['MS_CashInfo_{}_MP1'.format(instrument)]
            FeedCode = instrument + str(int(contract[-1])) + '.' + portfolio_info[1]
            part_trade_df['FeedCode'] = [FeedCode]
            part_trade_df['Qty'] = [int(abs(pos_change))]
            contract_price = pd.read_csv(os.path.join(price_path, FeedCode + '.csv'))['ClosePrice'][0]
            part_trade_df['Price'] = [df['Price'].values[-1]]
            # part_trade_df['Price'] = [df['Price'].values[-1]]
            part_trade_df['Way'] = [way]
            part_trade_df['ShType'] = ['SPECULATING']
            part_trade_df['ToPortfolio'] = ['MD_WDDTWAP_{}1'.format(portfolio_info[1])]
            print(part_trade_df)
            trade_df = trade_df.append(part_trade_df)

    df, position_df = creat_paper_report(instrument_list)

    text = df2html(position_df, 1)
    subject = r'[Paper trading position] Cash info version1.0 ' + end_date
    my_email.send_email(text, to, [], subject)
    trade_df = trade_df.ix[trade_df['Qty'] != 0]
    # 247mxrf7a4
