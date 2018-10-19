import pandas as pd
import numpy as np
import os

bkt = '/mnt/mfs/DAT_EQT/EM_Funda/daily'
pro = '/media/hdd1/DAT_EQT/EM_Funda/daily'

# check_data_list = ['R_ACCOUNTPAY_QYOY',
#                    'R_ACCOUNTREC_QYOY',
#                    'R_CFO_s_YOY_First',
#                    'R_CostSales_QYOY',
#                    'R_EBITDA2_QYOY',
#                    'R_ESTATEINVEST_QYOY',
#                    'R_FairVal_TotProfit_QYOY',
#                    'R_FINANCEEXP_s_QYOY',
#                    'R_GrossProfit_TTM_QYOY',
#                    'R_INVESTINCOME_s_QYOY',
#                    'R_NetAssets_s_YOY_First',
#                    'R_NetInc_s_QYOY',
#                    'R_NETPROFIT_s_QYOY',
#                    'R_OPCF_TTM_QYOY',
#                    'R_OperProfit_YOY_First',
#                    'R_SUMLIAB_QYOY',
#                    'R_WorkCapital_QYOY',
#                    'R_OTHERLASSET_QYOY',
#                    'R_NetIncRecur_QYOY']
#
check_data_list = sorted(os.listdir('/media/hdd1/DAT_EQT/EM_Funda/daily_bak'))
# check_data_list.remove('R_GSCF_sales_Y5YGR.csv')
# check_data_list.remove('R_OperProfit_s_YOY_QTTM.csv')

clear_list = []
error_list = []
bad_list = []
for file_name in check_data_list:
    print('_______________________________________________________')
    print(file_name)
    if not os.path.exists(os.path.join(bkt, file_name)):
        print(file_name+' not exist in bkt!!!!!!!!!')
        error_list += [file_name]

    else:
        data_pro = pd.read_csv(os.path.join(pro, file_name), sep='|', index_col=0, parse_dates=True).fillna(0)
        data_bkt = pd.read_csv(os.path.join(bkt, file_name), sep='|', index_col=0, parse_dates=True).fillna(0)
        xnms = sorted(list(set(data_pro.columns) & set(data_bkt.columns)))
        xinx = sorted(list(set(data_pro.index) & set(data_bkt.index)))
        xinx = xinx[200:]

        data_pro = data_pro.reindex(index=xinx, columns=xnms).fillna(0)
        data_bkt = data_bkt.reindex(index=xinx, columns=xnms).fillna(0)

        print((data_bkt != data_pro).sum().sum())
        if (data_bkt - data_pro).round(8).sum().sum() != 0:
            bad_list += [file_name]
        else:
            print('clear')
            clear_list += [file_name]
#
#
#
#
# print(q1[s][q1[s] != q2[s]].drop_duplicates());print(q2[s][q1[s] != q2[s]].drop_duplicates())
#
#
# for s in q.sum()[q.sum()!=0].index:
#     print(q1[s][q1[s] != q2[s]].drop_duplicates())
#     print(q2[s][q1[s] != q2[s]].drop_duplicates())
#     print("#" * 2 ** 5)

