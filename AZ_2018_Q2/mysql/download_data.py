import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
import time
from collections import OrderedDict


def path_create(target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)


# # table 01
# down_dict = OrderedDict({'公司概况变动': 'LICO_BS_COPMCHG', '机构别名表': 'ORGA_BI_INSTOTHERNAME',
#                          '机构当事人': 'ORGA_BI_ORGPARTY', '机构基本资料表': 'ORGA_BI_ORGBASEINFO',
#                          '机构类型表': 'ORGA_BI_COMPTYPETEMP', '机构资格对应情况': 'ORGA_BI_QUALIFYINFO',
#                          '人物档案表': 'CDSY_PERSONINFO', '上市公司子公司基本信息表': 'INCH_CM_MINESUBINFO',
#                          '市场类型表': 'SPTM_MARKETRELATION', '系统参数表': 'CFP_PVALUE',
#                          '证券代码表': 'CDSY_SECUCODE', '证券代码对应表': 'CDSY_CORRES',
#                          '证券名称变更表': 'CDSY_CHANGEINFO', '证券上市状态变动': 'CDSY_CHANGESTATE',
#                          })


# table 09
down_dict = OrderedDict({'系列指数分类表': 'INDEX_BA_SMTYPE', '指数概况': 'INDEX_BA_INFO',
                         '指数年行情指标表': 'TRAD_ID_YEARSYS', '指数日行情': 'INDEX_TD_DAILY',
                         '指数日行情指标表': 'INDEX_TD_DAILYSYS', '指数入选样本券表': 'INDEX_BA_SAMPLE',
                         '指数资金流向': 'INDEX_TD_FUNDFLOW'})


# # table 14
# down_dict = OrderedDict({'大宗交易': 'TRAD_BT_DAILY', '个股日行情筹码分布': 'TRAD_SK_DAILYCH',
#                          '股票复权因子(行情计算）':'TRAD_SK_FACTOR1', '股票增仓排名表': 'TRAD_SK_RANK',
#                          '交易公开信息_股票': 'TIT_S_PUB_STOCK', '交易公开信息_营业部': 'TIT_S_PUB_SALES',
#                          '交易日期': 'TRAD_TD_TDATE', '节日放假时间安排': 'CDSY_CT_HOLIDAY', '日历数据': 'CDSY_CT_CALENDER',
#                          '融资融券标的表': 'TRAD_MT_TARGET', '融资融券交易表': 'TRAD_MT_MARGIN',
#                          '相对估值指标': 'TRAD_SK_REVALUATION',
#                          '新股首日交易营业部统计': 'TRAD_SK_FISTDPLACE', '新股首日投资者类型统计': 'TRAD_SK_FISTDCOUNT',
#                          '行业市盈率表（申万发布）': 'LICO_IR_PESW', '行业市盈率表（中证发布）': 'LICO_IR_PECSI',
#
#                          '证券交易停复牌': 'TRAD_TD_SUSPEND', '证券市场交易结算资金表': 'TRAD_ST_TRADESETFUND',
#                          '证券停复牌每日情况表': 'TRAD_TD_SUSPENDDAY', '转融券交易汇总表': 'TRAD_MT_ZRQJYHZ',
#                          '转融券交易明细': 'TRAD_MT_ZRQJYMX', '转融通标的证券信息': 'TRAD_MT_ZRTTARGET',
#                          '转融通可充抵保证金证券及折算率': 'TRAD_MT_ZRTBZJJZSL', '转融通期限费率': 'TRAD_MT_ZRTQXFL',
#                          '转融资每日交易汇总表': 'TRAD_MT_ZRZMRJYHZ'})

# # table 16 基金数据
# down_dict = OrderedDict({'基金分红表': 'FUND_IA_DIVIDEND', '基金日行情': 'TRAD_FD_DAILY', '基金分类表': 'FUND_BS_ATYPE'})

# down_dict = OrderedDict({
#
#
#                          })

# down_dict = OrderedDict({'相对估值指标': 'TRAD_SK_REVALUATION', '新股首日交易营业部统计': 'TRAD_SK_FISTDPLACE',
#                          '新股首日投资者类型统计': 'TRAD_SK_FISTDCOUNT', '行业市盈率表（申万发布）': 'LICO_IR_PESW',
#                          '行业市盈率表（中证发布）': 'LICO_IR_PECSI'})

# def Mysql_select_column_data():


def Mysql_select_all_data(down_list, root_save_path, file_type='csv', except_list=None):
    if except_list is None:
        except_list = []
    down_list = list(set(down_list) - set(except_list))

    print(down_list)
    start = time.time()
    # usr_name = 'whs'
    usr_name = 'jerry'

    # pass_word = 'kj23#12!^3weghWhjqQ2rjj197'
    pass_word = 'o7ILR0WrdMN$gaqfju8!@pw9i'
    engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.33:3306/choice_fndb?charset=utf8')

    conn = engine.connect()

    path_create(root_save_path)

    for value in down_list:
        print('Table {} Start!'.format(value))
        df = pd.read_sql('SELECT * FROM choice_fndb.{}'.format(value), conn)
        print(df)
        # pd.to_pickle(df, os.path.join(root_save_path, '{}.pkl'.format(value)))
        # print('Table {} Done!'.format(value))

    end = time.time()
    print('Processing Cost:{} second'.format(end - start))


def Mysql_select_column_data(table_name, root_save_path, except_columns=None):
    if except_columns is None:
        except_columns = []
    usr_name = 'whs'
    pass_word = 'kj23#12!^3weghWhjqQ2rjj197'
    engine = create_engine(
        'mysql+pymysql://{}:{}@192.168.16.10:3306/choice_fndb?charset=utf8'.format(usr_name, pass_word))

    conn = engine.connect()
    usr_name = 'whs'
    pass_word = 'kj23#12!^3weghWhjqQ2rjj197'
    engine = create_engine(
        'mysql+pymysql://{}:{}@192.168.16.10:3306/choice_fndb?charset=utf8'.format(usr_name, pass_word))

    info_df = pd.read_sql('SHOW FULL COLUMNS FROM {}'.format(table_name), conn)['Field']
    for column in list(set(info_df.values) - set(except_columns)):
        print(column)
        column_df = pd.read_sql('SELECT {} FROM {}'.format(column, table_name), conn)
        save_path = os.path.join(root_save_path, table_name, 'split_data')
        path_create(save_path)
        column_df.to_pickle(os.path.join(save_path, column + 'pkl'))


def Mysql_select_columns_data(table_name, root_save_path):
    usr_name = 'whs'
    pass_word = 'kj23#12!^3weghWhjqQ2rjj197'
    engine = create_engine(
        'mysql+pymysql://{}:{}@192.168.16.10:3306/choice_fndb?charset=utf8'.format(usr_name, pass_word))

    conn = engine.connect()
    usr_name = 'whs'
    pass_word = 'kj23#12!^3weghWhjqQ2rjj197'
    engine = create_engine(
        'mysql+pymysql://{}:{}@192.168.16.10:3306/choice_fndb?charset=utf8'.format(usr_name, pass_word))

    info_df = pd.read_sql('SHOW FULL COLUMNS FROM {}'.format(table_name), conn)['Field']
    column_list = ['SECURITYCODE', 'TRADEDATE', 'CHG']
    print(','.join(column_list))
    column_df = pd.read_sql('SELECT {} FROM {}'.format(','.join(column_list), table_name), conn)
    save_path = os.path.join(root_save_path, table_name, 'split_data')
    path_create(save_path)
    column_df.to_pickle(os.path.join(save_path, ','.join(column_list) + 'pkl'))


if __name__ == '__main__':
    down_dict = OrderedDict({'持有其他证券情况': 'LICO_ES_HDOSEC',
                             '董监届次': 'LICO_MO_MANS',
                             '董事和监事': 'LICO_MO_DSHJS',
                             '高管持股与薪酬表': 'LICO_MO_MANHOLDRPAY',
                             '高管关联人持股': 'LICO_MO_MANRPHOLD',
                             '公司雇员': 'LICO_MO_EMPLOYEE',
                             '股东大会日期表': 'LICO_IM_GDDHDATE',
                             '股东大会召开公告': 'LICO_IM_GDDHNOTICE',
                             '股权激励基本资料': 'LICO_MO_GQJLJBZL',
                             '股权激励明细': 'LICO_MO_GQJLMX',
                             '激励获授对象明细': 'LICO_MO_JLDXHSMX',
                             '激励实施结果明细': 'LICO_MO_JLSSJGMX',
                             '监管部门调查处罚公告表': 'LICO_IM_SDPUNISHTNOTICE',
                             '经营层': 'LICO_MO_BUSILEVEL',
                             '期权各期行权时间安排': 'LICO_MO_QXQSJAP',
                             '限制性股票解锁时间安排': 'LICO_MO_LMITUNLOCKTIME',
                             '员工持股计划明细': 'LICO_ES_EMSHAREDE',
                             '员工持股计划总表': 'LICO_ES_EMSHAREPLAN',
                             '证券代码表': 'CDSY_SECUCODE',
                             '股东名单': 'LICO_ES_LISHOLD',
                             '股本结构表': 'LICO_ES_CPHSSTRUCT',
                             '违规': 'LICO_CM_ILLEGAL',
                             })

    # down_dict = OrderedDict({'证券代码表': 'CDSY_SECUCODE'})
    # down_dict = OrderedDict({'股东名单': 'LICO_ES_LISHOLD'})
    # down_dict = OrderedDict({'股本结构表': 'LICO_ES_CPHSSTRUCT'})
    # down_dict = OrderedDict({'违规': 'LICO_CM_ILLEGAL'})
    # down_dict = OrderedDict({'诉讼仲裁': 'LICO_CM_LAWARBI'})
    # down_dict = OrderedDict({'股东持股冻结情况': 'LICO_ES_SHHDFROZEN'})
    # down_dict = OrderedDict({'股东大会日期表': 'LICO_IM_GDDHDATE'})

    down_list = down_dict.values()
    root_save_path = '/mnt/mfs/dat_whs/EM_Funda'

    Mysql_select_all_data(down_list, root_save_path, file_type='pkl', except_list=None)


