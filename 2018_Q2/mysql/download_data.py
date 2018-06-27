import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
import time
from collections import OrderedDict


def path_create(target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)


# table 14
# down_dict = OrderedDict({'大宗交易': 'TRAD_BT_DAILY', '股票增仓排名表': 'TRAD_SK_RANK','个股日行情筹码分布': 'TRAD_SK_DAILYCH',
#                          '交易公开信息_股票': 'TIT_S_PUB_STOCK', '交易公开信息_营业部': 'TIT_S_PUB_SALES',
#                          '融资融券标的表': 'TRAD_MT_TARGET', '融资融券交易表': 'TRAD_MT_MARGIN',
#                          '证券交易停复牌': 'TRAD_TD_SUSPEND', '证券市场交易结算资金表': 'TRAD_ST_TRADESETFUND',
#                          '证券停复牌每日情况表': 'TRAD_TD_SUSPENDDAY', '转融券交易汇总表': 'TRAD_MT_ZRQJYHZ',
#                          '转融券交易明细': 'TRAD_MT_ZRQJYMX', '转融通标的证券信息': 'TRAD_MT_ZRTTARGET',
#                          '转融通可充抵保证金证券及折算率': 'TRAD_MT_ZRTBZJJZSL', '转融通期限费率': 'TRAD_MT_ZRTQXFL',
#                          '转融资每日交易汇总表': 'TRAD_MT_ZRZMRJYHZ'})

# down_dict = OrderedDict({'交易日期': 'TRAD_TD_TDATE', '节日放假时间安排': 'CDSY_CT_HOLIDAY',
#                          '日历数据': 'CDSY_CT_CALENDER', '相对估值指标': 'TRAD_SK_REVALUATION',
#                          '新股首日交易营业部统计': 'TRAD_SK_FISTDPLACE', '新股首日投资者类型统计': 'TRAD_SK_FISTDCOUNT',
#                          '行业市盈率表（申万发布）': 'LICO_IR_PESW', '行业市盈率表（中证发布）': 'LICO_IR_PECSI'})

down_dict = OrderedDict({'相对估值指标': 'TRAD_SK_REVALUATION', '新股首日交易营业部统计': 'TRAD_SK_FISTDPLACE',
                         '新股首日投资者类型统计': 'TRAD_SK_FISTDCOUNT', '行业市盈率表（申万发布）': 'LICO_IR_PESW',
                         '行业市盈率表（中证发布）': 'LICO_IR_PECSI'})

# # table 01
# down_dict = OrderedDict({'公司概况变动': 'LICO_BS_COPMCHG', '机构别名表': 'ORGA_BI_INSTOTHERNAME',
#                          '机构当事人': 'ORGA_BI_ORGPARTY', '机构基本资料表': 'ORGA_BI_ORGBASEINFO',
#                          '机构类型表': 'ORGA_BI_COMPTYPETEMP', '机构资格对应情况': 'ORGA_BI_QUALIFYINFO',
#                          '人物档案表': 'CDSY_PERSONINFO', '上市公司子公司基本信息表': 'INCH_CM_MINESUBINFO',
#                          '市场类型表': 'SPTM_MARKETRELATION', '系统参数表': 'CFP_PVALUE',
#                          '证券代码表': 'CDSY_SECUCODE', '证券代码对应表': 'CDSY_CORRES',
#                          '证券名称变更表': 'CDSY_CHANGEINFO', '证券上市状态变动': 'CDSY_CHANGESTATE',
#                          })

start = time.time()
usr_name = 'whs'
pass_word = 'kj23#12!^3weghWhjqQ2rjj197'
engine = create_engine('mysql+pymysql://{}:{}@192.168.16.10:3306/choice_fndb?charset=utf8'.format(usr_name, pass_word))

conn = engine.connect()

root_save_path = '/media/hdd1/whs_data/raw_data'
path_create(root_save_path)

for value in list(down_dict.values())[:1]:
    value = 'CDSY_SECUCODE'
    df = pd.read_sql('SELECT * FROM choice_fndb.{}'.format(value), conn)
    print(df)
    file_name = value
    # pd.to_pickle(df, os.path.join(root_save_path, '{}.pkl'.format(value)))
    # print(value)
end = time.time()
print('Processing Cost:{} second'.format(end - start))
