import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load.bkt import FactorTest, get_main_model, filter_all
from work_whs.loc_lib.pre_load.sql import conn


def create_date_list(ei_time, target_time):
    target_time_add = target_time + timedelta(days=10)
    target_mask = ei_time >= target_time_add
    ei_time.loc[target_mask] = target_time_add[target_mask]
    return ei_time


def out_sample_perf_c(pnl_df_out, way=1):
    if way == 1:
        sharpe_out = bt.AZ_Sharpe_y(pnl_df_out)
    else:
        sharpe_out = bt.AZ_Sharpe_y(-pnl_df_out)
    out_condition = sharpe_out > 0.8
    return out_condition, round(sharpe_out * way, 2)


# def filter_all(cut_date, pos_df_daily, pct_n, if_return_pnl=False, if_only_long=False):
#     pnl_df = (pos_df_daily * pct_n).sum(axis=1)
#     pnl_df = pnl_df.replace(np.nan, 0)
#     # pnl_df = pd.Series(pnl_df)
#     # 样本内表现
#     return_in = pct_n[pct_n.index < cut_date]
#
#     pnl_df_in = pnl_df[pnl_df.index < cut_date]
#     asset_df_in = pnl_df_in.cumsum()
#     last_asset_in = asset_df_in.iloc[-1]
#     pos_df_daily_in = pos_df_daily[pos_df_daily.index < cut_date]
#     pot_in = bt.AZ_Pot(pos_df_daily_in, last_asset_in)
#
#     leve_ratio = bt.AZ_Leverage_ratio(asset_df_in)
#     if leve_ratio < 0:
#         leve_ratio = 100
#     sharpe_q_in_df = bt.AZ_Rolling_sharpe(pnl_df_in, roll_year=1, year_len=250, min_periods=1,
#                                           cut_point_list=[0.3, 0.5, 0.7], output=False)
#     sp_in = bt.AZ_Sharpe_y(pnl_df_in)
#     fit_ratio = bt.AZ_fit_ratio(pos_df_daily_in, return_in)
#     ic = round(bt.AZ_Normal_IC(pos_df_daily_in, pct_n, min_valids=None, lag=0).mean(), 6)
#     sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d = sharpe_q_in_df.values
#     in_condition_u = sharpe_q_in_df_u > 0.9 and leve_ratio > 1
#     in_condition_d = sharpe_q_in_df_d < -0.9 and leve_ratio > 1
#     # 分双边和只做多
#     if if_only_long:
#         in_condition = in_condition_u
#     else:
#         in_condition = in_condition_u | in_condition_d
#
#     if sharpe_q_in_df_m > 0:
#         way = 1
#     else:
#         way = -1
#
#     # 样本外表现
#     pnl_df_out = pnl_df[pnl_df.index >= cut_date]
#     out_condition, sharpe_q_out = out_sample_perf_c(pnl_df_out, way=way)
#     if if_return_pnl:
#         return in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
#                fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df
#     else:
#         return in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
#                fit_ratio, leve_ratio, sp_in, sharpe_q_out


def fun(x):
    if len(x) > 1:
        print(x)
    return x.max()


def select_astock(x):
    if len(x) == 6:
        if x[:1] in ['0', '3', '6']:
            return True
        else:
            return False
    else:
        return False


def add_suffix(x):
    if x[0] in ['0', '3']:
        return x + '.SZ'
    elif x[0] in ['6']:
        return x + '.SH'
    else:
        print('error')


def company_to_stock(map_data, company_code_df):
    print('company_to_stock')
    # stock_code_df = pd.DataFrame(columns=sorted(self.map_data_c.values))
    company_code_df = company_code_df.reindex(columns=map_data.index)
    company_code_df.columns = map_data.values
    company_code_df = company_code_df[sorted(company_code_df.columns)]
    company_code_df.columns = [add_suffix(x) for x in company_code_df.columns]
    company_code_df.dropna(how='all', inplace=True, axis='columns')
    return company_code_df


def map_data_fun(conn):
    map_data = pd.read_sql('SELECT * FROM choice_fndb.CDSY_SECUCODE', conn)
    map_data.index = map_data['COMPANYCODE']
    map_data_c = map_data['SECURITYCODE'][map_data['SECURITYTYPE'] == 'A股']
    map_data_c = map_data_c[map_data_c.apply(select_astock)]
    return map_data_c


def research_expenses_fun(map_data, conn, xinx, xnms):
    data = pd.read_sql('SELECT * FROM choice_fndb.LICO_FN_RDEXPEND', conn)
    company_code_df = data.groupby(['FIRSTNOTICEDATE', 'COMPANYCODE'])['AMTTPERIOD'].apply(fun).unstack()
    a = company_to_stock(map_data, company_code_df)
    a = a.reindex(index=xinx, columns=xnms)
    a = a.fillna(method='ffill', limit=250)
    return a


def stock_shares_fun():
    def f(df):
        df = df.drop_duplicates(subset=['SHAREHDCODE'], keep='last')
        df = df.sort_values(by=['SHAREHDRATIO'], ascending=False)
        print(df)
        if sum(df['SHAREHDRATIO']) > 60:
            return sum(df['SHAREHDRATIO'].iloc[:8])
        else:
            return np.nan

    select_col = ','.join(['EITIME', 'NOTICEDATE', 'COMPANYCODE', 'SHAREHDRATIO', 'SHAREHDTYPE', 'SHAREHDCODE'])
    data = pd.read_sql('SELECT {} FROM choice_fndb.LICO_ES_LISHOLD'.format(select_col), conn)
    data['NOTICEDATE'] = create_date_list(data['EITIME'], data['NOTICEDATE'])
    raw_df = data.groupby(['NOTICEDATE', 'COMPANYCODE'])[['SHAREHDRATIO', 'SHAREHDCODE']].apply(f).unstack()
    return raw_df


def load_daily_fun(root_path, file_name, xinx, xnms):
    OPERATEREVE_s_df = bt.AZ_Load_csv(f'{root_path}/EM_Funda/daily/{file_name}.csv')
    OPERATEREVE_s_df = OPERATEREVE_s_df.reindex(index=xinx, columns=xnms)
    return OPERATEREVE_s_df


def get_single_df():
    data = pd.read_sql('SELECT SECURITYCODE, TRADEDATE, MINFLOW, FLOWINL, FLOWOUTL, PCTFUNDL, '
                       'FLOWINXL, FLOWOUTXL, PCTFUNDXL '
                       'FROM choice_fndb.TRAD_SK_RANK', conn)
    # data.to_pickle('/mnt/mfs/dat_whs/TRAD_SK_RANK.csv')
    data = pd.read_pickle('/mnt/mfs/dat_whs/TRAD_SK_RANK.csv')

    stock_list = bt.AZ_get_stock_name()
    stock_cuted_list = bt.AZ_cut_stock_suffix(stock_list)

    PCTFUNDL = data.groupby(['TRADEDATE', 'SECURITYCODE'])['PCTFUNDL'].sum().unstack()
    PCTFUNDXL = data.groupby(['TRADEDATE', 'SECURITYCODE'])['PCTFUNDXL'].sum().unstack()

    PCTFUNDL = PCTFUNDL.reindex(columns=stock_cuted_list)
    PCTFUNDXL = PCTFUNDXL.reindex(columns=stock_cuted_list)

    PCTFUNDL.columns = bt.AZ_add_stock_suffix(PCTFUNDL.columns)
    PCTFUNDXL.columns = bt.AZ_add_stock_suffix(PCTFUNDXL.columns)

    PCTFUNDL = bt.AZ_Rolling(PCTFUNDL, 20).mean()
    PCTFUNDXL = bt.AZ_Rolling(PCTFUNDXL, 20).mean()
    return PCTFUNDL, PCTFUNDXL


def main_fun(sector_name, hold_time, if_only_long, raw_df):
    # root_path = '/media/hdd1/DAT_EQT'
    root_path = '/mnt/mfs/DAT_EQT'
    cut_date = '20180601'
    begin_date = pd.to_datetime('20140101')
    end_date = datetime.now()
    if_hedge = True
    if sector_name.startswith('market_top_300plus') or sector_name.startswith('index_000300'):
        if_weight = 1
        ic_weight = 0

    elif sector_name.startswith('market_top_300to800plus') or sector_name.startswith('index_000905'):
        if_weight = 0
        ic_weight = 1

    else:
        if_weight = 0.5
        ic_weight = 0.5
    main_model = get_main_model(root_path, begin_date, cut_date, end_date, sector_name, hold_time,
                                if_hedge, if_only_long, if_weight, ic_weight)
    raw_df = raw_df.reindex(index=main_model.xinx, columns=main_model.xnms)

    signal_df = main_model.row_extre(raw_df, main_model.sector_df, 0.2)
    pos_df = main_model.deal_mix_factor(signal_df).shift(2)

    in_condition, out_condition, ic, sharpe_q_in_df_u, sharpe_q_in_df_m, sharpe_q_in_df_d, pot_in, \
    fit_ratio, leve_ratio, sp_in, sharpe_q_out, pnl_df = filter_all(cut_date, pos_df, main_model.return_choose,
                                                                    if_return_pnl=True, if_only_long=False)
    plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df), f'single test|{sector_name}|{hold_time}|{if_only_long}',
                     text=f'pot_in={pot_in},fr={fit_ratio},lr={leve_ratio}')
    return main_model


# map_data = map_data_fun(conn)

if __name__ == '__main__':
    sector_name_list = [
        'index_000300',
        'index_000905',
        'market_top_300plus',
        'market_top_300plus_industry_10_15',
        'market_top_300plus_industry_20_25_30_35',
        'market_top_300plus_industry_40',
        'market_top_300plus_industry_45_50',
        'market_top_300plus_industry_55',

        'market_top_300to800plus',
        'market_top_300to800plus_industry_10_15',
        'market_top_300to800plus_industry_20_25_30_35',
        'market_top_300to800plus_industry_40',
        'market_top_300to800plus_industry_45_50',
        'market_top_300to800plus_industry_55'
    ]

    hold_time_list = [5, 20]
    if_only_long_list = [False, True]
    PCTFUNDL, PCTFUNDXL = get_single_df()
    INFLOWRATE = pd.read_pickle('/mnt/mfs/dat_whs/INFLOWRATE.pkl')
    for if_only_long in if_only_long_list:
        for hold_time in hold_time_list:
            for sector_name in sector_name_list:
                main_fun(sector_name, hold_time, if_only_long, INFLOWRATE)
    # data.to_csv('/mnt/mfs/dat_whs/TRAD_SK_RANK.csv')
    # if_only_long = True
    # long_short_way

