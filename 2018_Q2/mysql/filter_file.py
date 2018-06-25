import pandas as pd
import os
from collections import OrderedDict
from multiprocessing import Pool
import time


def path_create(target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)


# down_dict = OrderedDict({'大宗交易': 'TRAD_BT_DAILY', '股票增仓排名表': 'TRAD_SK_RANK', '个股日行情筹码分布': 'TRAD_SK_DAILYCH',
#                          '交易公开信息_股票': 'TIT_S_PUB_STOCK', '交易公开信息_营业部': 'TIT_S_PUB_SALES',
#                          '融资融券标的表': 'TRAD_MT_TARGET', '融资融券交易表': 'TRAD_MT_MARGIN',
#                          '证券交易停复牌': 'TRAD_TD_SUSPEND', '证券市场交易结算资金表': 'TRAD_ST_TRADESETFUND',
#                          '证券停复牌每日情况表': 'TRAD_TD_SUSPENDDAY', '转融券交易汇总表': 'TRAD_MT_ZRQJYHZ',
#                          '转融券交易明细': 'TRAD_MT_ZRQJYMX', '转融通标的证券信息': 'TRAD_MT_ZRTTARGET',
#                          '转融通可充抵保证金证券及折算率': 'TRAD_MT_ZRTBZJJZSL', '转融通期限费率': 'TRAD_MT_ZRTQXFL',
#                          '转融资每日交易汇总表': 'TRAD_MT_ZRZMRJYHZ'})
# fail_list = OrderedDict({'转融资每日交易汇总表': 'TRAD_MT_ZRZMRJYHZ', '证券市场交易结算资金表': 'TRAD_ST_TRADESETFUND',
#                          '交易公开信息_营业部': 'TIT_S_PUB_SALES', '转融通期限费率': 'TRAD_MT_ZRTQXFL'})


raw_root_path = '/media/hdd1/whs_data/raw_data/table14'
adj_save_path = '/media/hdd1/whs_data/adj_data/table14/EQA'


def filter_stock_type(raw_root_path, adj_save_path):
    path_create(adj_save_path)
    file_list = os.listdir(raw_root_path)
    print('^^^^^^^^^^^^^^^^^^^^^^^^')
    print(len(file_list))
    info = pd.read_pickle('/media/hdd1/whs_data/raw_data/base_info/CDSY_SECUCODE.pkl')
    universe_EQA2 = info[info['SECURITYTYPECODE'] == '058001001']['SECURITYCODE'].values

    for file_name in file_list:
        print('**********************************')
        drop_columns = ['EID', 'ESEQID', 'EITIME', 'EUTIME', 'EGETTIME', 'EISDEL']

        load_path = os.path.join(raw_root_path, file_name)
        try:
            data = pd.read_pickle(load_path)
        except IOError:
            print('PATH ERROR')
            continue
        # 删除错误数据
        data = data[data['EISDEL'].astype(int) == 0]
        data.drop(columns=list(set(drop_columns) & set(data.columns)), inplace=True)
        if 'SECURITYCODE' in data.columns:
            # 筛选A股数据
            universe_EQA = sorted(list(set(universe_EQA2) & set(data['SECURITYCODE'])))

            if 'TRADEDATE' in data.index:
                data_EQA = data.loc[universe_EQA].sort_values(by='TRADEDATE')
            else:
                data_EQA = data.loc[universe_EQA]
            data_EQA.index = range(len(data_EQA))
            data_EQA.to_pickle(os.path.join(adj_save_path, file_name))
            print('{} success filter'.format(file_name))
        else:
            data.to_pickle(os.path.join(adj_save_path, file_name))
            print('{} fail filter, SECURITYCODE not in columns'.format(file_name))


def TRAD_BT_DAILY_deal():
    """
    大宗交易
    :return:
    """
    root_path = '/media/hdd1/whs_data/adj_data/table14'
    file_name = 'TRAD_BT_DAILY.pkl'
    load_path = os.path.join(root_path, 'EQA', file_name)
    data = pd.read_pickle(load_path)
    path_create(os.path.join(root_path, 'TRAD_BT_DAILY'))
    data.index = data['SECURITYCODE'].values
    data.drop_duplicates()

    target_tval = data.groupby(['TRADEDATE', 'SECURITYCODE'])['TVAL'].mean().unstack()
    target_tval.fillna(0, inplace=True)
    target_tval.to_pickle(os.path.join(root_path, 'TRAD_BT_DAILY', 'TRAD_BT_DAILY_VAL.pkl'))
    target_time = data.groupby(['TRADEDATE', 'SECURITYCODE'])['TVAL'].apply(len).unstack()
    target_time.fillna(0, inplace=True)
    target_time.to_pickle(os.path.join(root_path, 'TRAD_BT_DAILY', 'TRAD_BT_DAILY_TIME.pkl'))


def TRAD_SK_DAILYCH_deal():
    """
    个股日行情筹码分布
    :param file_name:
    :return:
    """
    file_name = 'TRAD_SK_DAILYCH'
    root_path = '/media/hdd1/whs_data/adj_data/table14'
    load_path = os.path.join(root_path, 'EQA', file_name + '.pkl')
    data = pd.read_pickle(load_path)

    columns_list = list(set(data.columns) - {'TRADEDATE', 'SECURITYCODE'})
    path_create(os.path.join(root_path, file_name, 'triangle'))
    path_create(os.path.join(root_path, file_name, 'average'))
    data.index = data['SECURITYCODE'].values
    data.drop_duplicates()
    for dbtype, part_data in data.groupby('DBTYPE'):
        for col_name in columns_list:
            target_df = part_data.groupby(['TRADEDATE', 'SECURITYCODE'])[col_name].sum().unstack()
            if dbtype == '三角形分布':
                target_df.to_pickle(os.path.join(root_path, file_name, 'triangle', col_name + '.pkl'))
            else:
                target_df.to_pickle(os.path.join(root_path, file_name, 'average', col_name + '.pkl'))


def TRAD_SK_RANK_deal():
    """
    股票增仓排名表
    :param filename:
    :return:
    """
    file_name = 'TRAD_SK_RANK'
    root_path = '/media/hdd1/whs_data/adj_data/table14'
    load_path = os.path.join(root_path, 'EQA', file_name + '.pkl')
    data = pd.read_pickle(load_path)

    columns_list = list(set(data.columns) - {'TRADEDATE', 'SECURITYCODE'})
    path_create(os.path.join(root_path, file_name))
    data.index = data['SECURITYCODE'].values
    data.drop_duplicates()
    for col_name in columns_list:
        target_df = data.groupby(['TRADEDATE', 'SECURITYCODE'])[col_name].sum().unstack()
        target_df.to_pickle(os.path.join(root_path, file_name, col_name + '.pkl'))


def TIT_S_PUB_STOCK():
    """
    交易公开信息_股票
    记录了异常股票信息, 用到时处理

    137001 有涨跌幅限制的异动证券
    137002 无价格涨跌幅限制的证券
    137003 单只标的证券的当日融资买卖数量
    137004 实施特别停牌的证券
    137005 其它异常波动的证券
    137006 权证信息
    137007 当日有涨跌幅限制的A股，连续2个交易日触及涨幅限制，
    在这2个交易日中同一营业部净买入股数占当日总成交股数的比重30％以上，且上市公司未有重大事项公告的
    137008 当日无价格涨跌幅限制的B股，出现异常波动停牌的
    137009 风险警示股票盘中换手率达到或超过30%
    137010 退市整理的证券
    137011 风险警示期交易

    :param file_name:
    :return:
    """
    file_name = 'TIT_S_PUB_STOCK'
    root_path = '/media/hdd1/whs_data/adj_data/table14'
    load_path = os.path.join(root_path, 'EQA', file_name + '.pkl')
    data = pd.read_pickle(load_path)

    columns_list = list(set(data.columns) - {'TRADEDATE', 'SECURITYCODE'})
    path_create(os.path.join(root_path, file_name))
    data.index = data['SECURITYCODE'].values
    data.drop_duplicates()
    data_ctype = data[['CTYPE',  'TRADEDATE', 'SECURITYCODE']]
    odd_code_set = ['137001', '137002', '137003', '137004', '137005', '137006', '137007', '137008',
                     '137009', '137010', '137011']
    # global_para = {'data_ctype': data_ctype, 'a': odd_code_set}
    for odd_code in odd_code_set:
        save_path = os.path.join(root_path, file_name, 'set_' + odd_code + '.pkl')
        exec('set_{0} = data_ctype[list(map(lambda x: True if x.startswith(\'{0}\') else False, '
             'data_ctype[\'CTYPE\']))]'.format(odd_code))
        exec('condition = len(set_{0})!=0'.format(odd_code))
        if condition:
            exec('df_{0}=set_{0}.groupby([\'TRADEDATE\', \'SECURITYCODE\'])[\'CTYPE\'].apply(lambda x: x.iloc[0])'
                 '.unstack()'.format(odd_code))
            exec('df_{0}.to_pickle(save_path)'.format(odd_code))


def TIT_S_PUB_SALES_deal():
    """
    交易公开信息_营业部
    :param file_name:
    :return:
    """
    file_name = 'TIT_S_PUB_SALES'
    root_path = '/media/hdd1/whs_data/adj_data/table14'
    load_path = os.path.join(root_path, 'EQA', file_name + '.pkl')
    data = pd.read_pickle(load_path)

    columns_list = list(set(data.columns) - {'TRADEDATE', 'SECURITYCODE'})
    path_create(os.path.join(root_path, file_name))
    data.index = data['SECURITYCODE'].values
    data.drop_duplicates()
    for col_name in columns_list:
        target_df = data.groupby(['TRADEDATE', 'SECURITYCODE'])[col_name].sum().unstack()
        target_df.to_pickle(os.path.join(root_path, file_name, col_name + '.pkl'))


def TRAD_MT_TARGET_deal():
    """
    融资融券标的表
    :param file_name:
    :return:
    """
    file_name = 'TRAD_MT_TARGET'
    root_path = '/media/hdd1/whs_data/adj_data/table14'
    load_path = os.path.join(root_path, 'EQA', file_name + '.pkl')
    data = pd.read_pickle(load_path)

    columns_list = list(set(data.columns) - {'TRADEDATE', 'SECURITYCODE'})
    path_create(os.path.join(root_path, file_name))
    data.index = data['SECURITYCODE'].values
    data.drop_duplicates()
    for tatype, part_data in data.groupby(['TATYPE']):
        for col_name in columns_list:
            target = part_data.groupby(['TRADEDATE', 'SECURITYCODE'])[col_name].apply(lambda x: x.iloc[0]).unstack()
            if tatype == '0':
                target.to_pickle(os.path.join(root_path, file_name, 'financing_target', col_name + '.pkl'))
            else:
                target.to_pickle(os.path.join(root_path, file_name, 'security_target', col_name + '.pkl'))


def TRAD_MT_MARGIN_deal():
    """
    融资融券交易表
    :param file_name:
    :return:
    """
    file_name = 'TRAD_MT_MARGIN'
    root_path = '/media/hdd1/whs_data/adj_data/table14'
    load_path = os.path.join(root_path, 'EQA', file_name + '.pkl')
    data = pd.read_pickle(load_path)

    columns_list = list(set(data.columns) - {'TRADEDATE', 'SECURITYCODE'})
    path_create(os.path.join(root_path, file_name))
    data.index = data['SECURITYCODE'].values
    data.drop_duplicates()
    for col_name in columns_list:
        target = data.groupby(['TRADEDATE', 'SECURITYCODE'])[col_name].sum().unstack()
        target.to_pickle(os.path.join(root_path, file_name, col_name + '.pkl'))


def TRAD_TD_SUSPEND_deal():
    """
    证券交易停复牌 TRAD_TD_SUSPEND
    证券市场交易结算资金表 TRAD_ST_TRADESETFUND
    证券停复牌每日情况表 TRAD_TD_SUSPENDDAY
    :return:
    """
    file_name = 'TRAD_TD_SUSPEND'
    root_path = '/media/hdd1/whs_data/adj_data/table14'
    load_path = os.path.join(root_path, 'EQA', file_name + '.pkl')
    data = pd.read_pickle(load_path)

    columns_list = list(set(data.columns) - {'TRADEDATE', 'SECURITYCODE'})
    path_create(os.path.join(root_path, file_name))
    data.index = data['SECURITYCODE'].values
    data.drop_duplicates()
    for col_name in columns_list:
        target = data.groupby(['TRADEDATE', 'SECURITYCODE'])[col_name].sum().unstack()
        target.to_pickle(os.path.join(root_path, file_name, col_name + '.pkl'))


def TRAD_MT_ZRQJYHZ_deal():
    """
    转融券交易汇总表
    :return:
    """
    file_name = 'TRAD_MT_ZRQJYHZ'
    root_path = '/media/hdd1/whs_data/adj_data/table14'
    load_path = os.path.join(root_path, 'EQA', file_name + '.pkl')
    data = pd.read_pickle(load_path)

    columns_list = list(set(data.columns) - {'TRADEDATE', 'SECURITYCODE'})
    path_create(os.path.join(root_path, file_name))
    data.index = data['SECURITYCODE'].values
    data.drop_duplicates()
    for col_name in columns_list:
        target = data.groupby(['TRADEDATE', 'SECURITYCODE'])[col_name].sum().unstack()
        target.to_pickle(os.path.join(root_path, file_name, col_name + '.pkl'))


def TRAD_MT_ZRTTARGET_deal():
    """
    转融通标的证券信息
    :return:
    """
    file_name = 'TRAD_MT_ZRTTARGET'
    root_path = '/media/hdd1/whs_data/adj_data/table14'
    load_path = os.path.join(root_path, 'EQA', file_name + '.pkl')
    data = pd.read_pickle(load_path)

    columns_list = list(set(data.columns) - {'TRADEDATE', 'SECURITYCODE'})
    path_create(os.path.join(root_path, file_name))
    data.index = data['SECURITYCODE'].values
    data.drop_duplicates()
    for col_name in columns_list:
        target = data.groupby(['TRADEDATE', 'SECURITYCODE'])[col_name].sum().unstack()
        target.to_pickle(os.path.join(root_path, file_name, col_name + '.pkl'))


def TRAD_MT_ZRTBZJJZSL_deal():
    """
    转融通标的证券信息
    :return:
    """
    file_name = 'TRAD_MT_ZRTBZJJZSL'
    root_path = '/media/hdd1/whs_data/adj_data/table14'
    load_path = os.path.join(root_path, 'EQA', file_name + '.pkl')
    data = pd.read_pickle(load_path)

    columns_list = list(set(data.columns) - {'TRADEDATE', 'SECURITYCODE'})
    path_create(os.path.join(root_path, file_name))
    data.index = data['SECURITYCODE'].values
    data.drop_duplicates()

    data['CFTSM'] = data['CFTSM'].astype(float)
    target = data.groupby(['TRADEDATE', 'SECURITYCODE'])['CFTSM'].sum().unstack()
    target.to_pickle(os.path.join(root_path, file_name, 'CFTSM' + '.pkl'))


def TRAD_MT_ZRTQXFL_deal():
    """
    转融通期限费率
    :return:
    """
    file_name = 'TRAD_MT_ZRTQXFL'
    root_path = '/media/hdd1/whs_data/adj_data/table14'
    load_path = os.path.join(root_path, 'EQA', file_name + '.pkl')
    data = pd.read_pickle(load_path)

    path_create(os.path.join(root_path, file_name))
    data.index = data['TRADEDATE'].values
    data.drop_duplicates()
    ZRC_target = data.groupby(['TRADEDATE', 'DLINE'])['ZRCNLV'].sum().unstack()
    ZRC_target.to_pickle(os.path.join(root_path, file_name, 'ZRCNLV' + '.pkl'))

    ZRR_target = data.groupby(['TRADEDATE', 'DLINE'])['ZRRNLV'].sum().unstack()
    ZRR_target.to_pickle(os.path.join(root_path, file_name, 'ZRRNLV' + '.pkl'))


def TRAD_MT_ZRZMRJYHZ_deal():
    """
    转融资每日交易汇总表
    :return:
    """
    file_name = 'TRAD_MT_ZRZMRJYHZ'
    root_path = '/media/hdd1/whs_data/adj_data/table14'
    load_path = os.path.join(root_path, 'EQA', file_name + '.pkl')
    data = pd.read_pickle(load_path)

    path_create(os.path.join(root_path, file_name))
    data.index = data['TRADEDATE'].values
    data.drop_duplicates()
    data.index = range(len(data))
    data.to_pickle(os.path.join(root_path, file_name, file_name + '.pkl'))


def common_deal(file_name):
    """
    通用处理 分 TRADEDATE, SECURITYCODE 储存文件
    :param file_name:
    :return:
    """
    root_path = '/media/hdd1/whs_data/adj_data/table14'
    load_path = os.path.join(root_path, 'EQA', file_name + '.pkl')
    data = pd.read_pickle(load_path)

    columns_list = list(set(data.columns) - {'TRADEDATE', 'SECURITYCODE'})
    path_create(os.path.join(root_path, file_name))
    data.index = data['SECURITYCODE'].values
    data.drop_duplicates()
    for col_name in columns_list:
        target = data.groupby(['TRADEDATE', 'SECURITYCODE'])[col_name].sum().unstack()
        target.to_pickle(os.path.join(root_path, file_name, col_name + '.pkl'))


down_dict = OrderedDict({'相对估值指标': 'TRAD_SK_REVALUATION',
                         '新股首日交易营业部统计': 'TRAD_SK_FISTDPLACE', '新股首日投资者类型统计': 'TRAD_SK_FISTDCOUNT',
                         '行业市盈率表（申万发布）': 'LICO_IR_PESW', '行业市盈率表（中证发布）': 'LICO_IR_PECSI'})

if __name__ == '__main__':

    # filter_stock_type(raw_root_path, adj_save_path)

    # TRAD_BT_DAILY_deal()
    # TRAD_SK_DAILYCH_deal('TRAD_SK_DAILYCH')
    # TRAD_SK_RANK_deal('TRAD_SK_RANK')
    # TRAD_SK_RANK_deal('TRAD_SK_RANK')
    # TRAD_MT_MARGIN_deal()
    # TRAD_MT_ZRQJYHZ_deal()
    # TRAD_MT_ZRTTARGET_deal()
    # filter_stock_type(raw_root_path, adj_save_path)
    # for file_name in down_dict.values():
    #     common_deal(file_name)
    TIT_S_PUB_STOCK()
