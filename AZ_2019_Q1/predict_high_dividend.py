import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load.sql import conn


def split_stock(x):
    if (x.startswith('0') or x.startswith('3') or x.startswith('6')) and len(x) == 6:
        return True
    else:
        return False


def create_date_list(ei_time, target_time):
    target_time_add = target_time + timedelta(days=10)
    target_mask = ei_time >= target_time_add
    ei_time.loc[target_mask] = target_time_add[target_mask]
    return ei_time


def dividend_ratio_fun():
    data = pd.read_sql('SELECT EITIME,EISDEL,SECURITYCODE,NOTICEDATE,CASHBTAXRMB,'
                       'ASSIGNDATETYPECODE,SHAREBASE,ASSIGNEETYPE '
                       'FROM LICO_IA_ASSIGNSCHEME', conn)
    # data.to_pickle('/mnt/mfs/dat_whs/data.pkl')
    # data = pd.read_pickle('/mnt/mfs/dat_whs/data.pkl')

    data = data[data['EISDEL'] == '0']
    data = data[[split_stock(x) for x in data['SECURITYCODE']]]
    data = data[data['NOTICEDATE'] == data['NOTICEDATE']]

    # data['NOTICEDATE'] = create_date_list(data['EITIME'], data['NOTICEDATE'])

    dividend_index = data['CASHBTAXRMB'].replace(0, np.nan).dropna().index
    dividend_data = data.loc[dividend_index]

    def fun_2(x):
        # if len(x) > 1:
        #     print(x)
        return x.iloc[-1]
    # target_df.groupby(['NOTICEDATE', 'SECURITYCODE'])['‘CASHBTAXRMB'].apply(fun_2)

    deal_report_list = ['001001',
                        '001002',
                        '001003',
                        '001004',
                        '001005',
                        '001006']

    area_code_list = ['001', '004']
    MarketCap = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/LICO_YS_STOCKVALUE/MarketCap.csv')
    AmarketCap = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/LICO_YS_STOCKVALUE/AmarketCap.csv')

    result_df = pd.DataFrame()
    dividend_df = pd.DataFrame()
    for area_code in area_code_list:
        target_df = dividend_data[dividend_data['ASSIGNEETYPE'] == area_code]
        if area_code == '001':
            market_df = MarketCap
        elif area_code == '004':
            market_df = AmarketCap
        else:
            market_df = MarketCap
        for deal_report in deal_report_list:
            part_target_df = target_df[target_df['ASSIGNDATETYPECODE'] == deal_report]
            tmp_df_1 = part_target_df.groupby(['NOTICEDATE', 'SECURITYCODE'])['CASHBTAXRMB'].apply(fun_2)
            tmp_df_2 = part_target_df.groupby(['NOTICEDATE', 'SECURITYCODE'])['SHAREBASE'].apply(fun_2)
            print(area_code, deal_report, len(tmp_df_1))
            if len(tmp_df_1) > 0 or len(tmp_df_1) > 0:
                part_dividend_df = tmp_df_1.unstack()
                part_share_df = tmp_df_2.unstack()
                part_cash_df = part_dividend_df * part_share_df * 0.1
                part_cash_df.columns = bt.AZ_add_stock_suffix(part_cash_df.columns)
                dividend_df = dividend_df.add(part_dividend_df, fill_value=0)
                part_result_df = part_cash_df.div(market_df)
                result_df = result_df.add(part_result_df, fill_value=0)
    # result_df.columns = bt.AZ_add_stock_suffix(result_df.columns)
    return result_df


if __name__ == '__main__':
    dividend_ratio = dividend_ratio_fun()
    dividend_ratio.to_csv('/mnt/mfs/DAT_EQT/EM_Funda/dat_whs/dividend_ratio.csv', sep='|')
