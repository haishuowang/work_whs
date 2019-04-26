import pandas as pd
import numpy as np
import os

__author__ = 'zijie.ren'


def LoadDataFrame(now_path):
    return pd.read_csv(now_path, sep='|', index_col=0)


def SaveDataFrame(now_df, now_path):
    now_df.to_csv(now_path, sep='|')


def UpdateData(now_df, save_path):
    if os.path.exists(save_path):
        old_df = LoadDataFrame(save_path)
        old_index = old_df.index.values
        new_index = now_df.index.values
        old_index = old_index[old_index < new_index[0]]
        now_df = old_df.reindex(index=old_index).append(now_df, sort=False)
    SaveDataFrame(now_df, save_path)
    return now_df


def LoadDfFromSql(sql, con, *args, **kwargs):
    return pd.read_sql(sql, con, *args, **kwargs)


def connect_sql():
    from sqlalchemy import create_engine

    # Step.1 创建数据库链接
    cfg = {
        "db_type": "mysql",
        "db_driver": "pymysql",
        "db_ip": "192.168.16.28",
        "db_port": 7777,
        "account": "yp_labman_rzj",
        "password": "17259",
    }
    cfg_as_str = "{db_type}+{db_driver}://{account}:{password}@{db_ip}:{db_port}".format(**cfg)
    engine = create_engine(cfg_as_str, connect_args={"charset": "utf8"})  # 使用utf-8字符集
    return engine


# Step.2 查询
def read_info(product: str, start=None, last=None):
    sql = f"SELECT `ContractId` as`Contract` , `StartTradeDate` as `StartDate`, " \
          f"`LastTradeDate` as `EndDate` " \
          f"FROM `crawl`.`FutureInfo_em` " \
          f"WHERE (`ContractId` REGEXP '^{product}[^A-Z].*') "
    if last:
        cond = f"AND StartTradeDate <= {last.replace('-','')} "
        sql += cond
    if start:
        cond = f"AND LastTradeDate >= {start.replace('-','')} "
        sql += cond
    return sql


def read_quote(product: list, start: str, end: str):
    """查询某个品种所有合约的量价数据"""
    # 这里使用到了模糊匹配, 如果能精确地调用需要的合约ContractId, 速度会更快
    product_str = str(tuple(product)).replace(",)", ")")
    sql = f"SELECT `ContractId` as `Contract`, `TradeDate` as `Date`, " \
          f"`Open` ,`Close`,`High`, `Low`, `Volume`, `Turnover`, `Settlement`,`OpenInterest` " \
          f"FROM `crawl`.`FutureQuote_em` " \
          f"WHERE (`ContractId` IN {product_str}) " \
          f"AND (TradeDate BETWEEN {start.replace('-','')} AND {end.replace('-','')})"
    return sql


def cut_contract(now_year, ctc_ls, ctc_start, ctc_end):
    now_product = []
    for i, x in enumerate(ctc_ls):
        now_bool = (ctc_end is None or ctc_end[i][0:4] <= now_year) or \
                   (ctc_start is None or ctc_start[i][0:4] >= now_year)
        if now_bool:
            now_product.append(x)
    return now_product


def read_member(ctc_ls: list, start: str, end: str, ctc_start=None, ctc_end=None):
    need_years = [str(x) for x in range(int(start[0:4]), int(end[0:4]) + 1)]
    sql = ''
    years_len = len(need_years)
    for i, now_year in enumerate(need_years):
        now_product = cut_contract(now_year, ctc_ls, ctc_start, ctc_end)
        now_product_str = str(tuple(now_product)).replace(",)", ")")
        now_start = start if i == 0 else now_year + '-01-01'
        now_end = end if i == years_len - 1 else now_year + '-12-31'
        now_table = '`FutureMemberRank_{}_em`'.format(now_year)
        now_sql = f"" \
                  f"SELECT `ContractId` as `Contract`, `TradeDate` as `Date`, " \
                  f"`MemberVolume` ,`MemberLong`,`MemberShort`, `Volume`, `Long`, `Short` " \
                  f"FROM `crawl`.{now_table} " \
                  f"WHERE (`ContractId` IN {now_product_str}) " \
                  f"AND (TradeDate BETWEEN {now_start.replace('-','')} AND {now_end.replace('-','')})"
        if years_len > 1:
            now_sql = '(' + now_sql + ')'
            if i > 0:
                now_sql = 'UNION' + now_sql
        sql = sql + now_sql
    return sql


def cut_member_df(engine, ins, last, start, need_years):
    years_len = len(need_years)
    tot_df = pd.DataFrame()
    inf_df = pd.DataFrame()
    for i, now_year in enumerate(need_years):
        now_start = start if i == 0 else now_year + '-01-01'
        now_last = last if i == years_len - 1 else now_year + '-12-31'
        info_sql = read_info(ins, now_start, now_last)
        info_df = LoadDfFromSql(info_sql, engine)
        inf_df = inf_df.append(info_df)
        member_sql = read_member(info_df['Contract'].tolist(), now_start, now_last)
        member_df = LoadDfFromSql(member_sql, engine)
        tot_df = tot_df.append(member_df)
    inf_df = inf_df.drop_duplicates()
    now_contract = inf_df['Contract'].tolist()
    now_df = LoadDfFromSql(read_quote(now_contract, start, last), engine)
    return tot_df, now_df


def ts_func(its_se):
    oi = its_se['Long'] + its_se['Short']
    ts = None
    if oi != 0:
        ts = (its_se['Long'] - its_se['Short']) / oi
    return ts


def one_contract_date(ins_ls, mem_df, mem_key_se, need_columns, now_key, now_se):
    now_stats = now_se['OpenInterest'] / now_se['Volume']
    now_mem = mem_df.loc[mem_key_se == now_key, :]
    now_ls = now_key.split('|')
    if now_mem.empty:
        print('{0} {1} member no data'.format(*now_ls))
        return
    now_df = pd.DataFrame()
    for now_column in need_columns:
        temp_se = now_mem[now_column].set_axis(now_mem['Member' + now_column], inplace=False)
        temp_df = temp_se.dropna().to_frame(now_column)
        now_df = now_df.join(temp_df, how='outer')
    now_sum = now_df.sum()
    if now_sum['Long'] + now_sum['Short'] <= 10:
        print('{0} {1} member oi less 10'.format(*now_ls))
        return
    mask_se = (now_df > 0).sum(1)
    other_df = now_df.loc[mask_se != 3, :].sum().to_frame('other').T
    new_df = now_df.loc[mask_se == 3, :].append(other_df)
    new_oi = new_df['Long'] + new_df['Short']
    new_se = (new_oi) / new_df['Volume']
    its_se = new_df.loc[new_se >= now_stats, :].sum()
    uts_se = new_df.loc[new_se < now_stats, :].sum()
    ins_ls.append(now_ls + [ts_func(its_se), ts_func(uts_se)])


def CobWeb(ins_ls, start, last, save_path):
    engin = connect_sql()
    need_columns = ['Volume', 'Long', 'Short']
    need_years = [str(x) for x in range(int(start[0:4]), int(last[0:4]) + 1)]
    for now_ins in ins_ls:
        mem_df, stats_df = cut_member_df(engin, now_ins, last, start, need_years)
        if mem_df.empty:
            print('{} member no data'.format(now_ins))
            continue
        now_path = save_path + now_ins + '/'
        mem_df['Date'] = mem_df['Date'].dt.strftime('%Y-%m-%d')
        stats_df['Date'] = stats_df['Date'].dt.strftime('%Y-%m-%d')
        mem_key_se = mem_df['Contract'] + '|' + mem_df['Date']
        stats_key_se = stats_df['Contract'] + '|' + stats_df['Date']
        in_key_se = np.intersect1d(mem_key_se, stats_key_se)
        stats_df = stats_df.set_axis(stats_key_se, inplace=False)
        stats_df = stats_df.reindex(index=in_key_se, columns=['OpenInterest', 'Volume'])
        stats_df[stats_df == 0] = None
        stats_df = stats_df.dropna()
        ins_r_ls = []
        for now_key, now_se in stats_df.iterrows():
            one_contract_date(ins_r_ls, mem_df, mem_key_se, need_columns, now_key, now_se)
        if len(ins_r_ls) == 0:
            continue
        ins_df = pd.DataFrame(ins_r_ls, columns=['Contract', 'Date', 'ITS', 'UTS'])
        tot_columns = ins_df.columns.drop(labels=['Date', 'Contract']).tolist()
        now_pivot = ins_df.pivot(index='Date', columns='Contract').sort_index()
        for now_column in tot_columns:
            now_df = now_pivot[now_column]
            now_df = UpdateData(now_df, now_path + now_column)


def UnionDatesData(tot_dict, save_path):
    for now_ins, now_dict in tot_dict.items():
        now_df = now_dict['trade']
        now_info = now_dict['info'].set_index('Contract')
        tot_columns = now_df.columns.drop(labels=['Date', 'Contract']).tolist()
        now_pivot = now_df.pivot(index='Date', columns='Contract').sort_index()
        now_path = save_path + now_ins + '/'
        if not os.path.exists(now_path):
            os.makedirs(now_path)
        for now_column in tot_columns + ['Mask']:
            if now_column in ['Mask']:
                now_close = now_pivot['Close']
                now_columns = now_close.columns
                now_info = now_info.reindex(index=now_columns)
                now_date_df = pd.DataFrame(None, index=now_close.index, columns=now_columns)
                now_date_df.iloc[:, 0] = now_close.index.values
                now_mask = now_date_df.fillna(method='ffill', axis=1)
                now_mask = now_mask.le(now_info['EndDate'], axis=1) & now_mask.ge(now_info['StartDate'], axis=1)
                now_df = now_mask * 1
            else:
                now_df = now_pivot[now_column]
            now_df = UpdateData(now_df, now_path + now_column)
            if now_column in ['Close']:
                now_close = now_df
                now_pre_close = now_close.shift(1)
                now_pre_close = now_pre_close[now_pre_close > 0]
                SaveDataFrame(now_df, now_path + 'PreClose')
                now_adj_r = now_close / now_pre_close - 1
                now_df = now_adj_r.fillna(0)
                SaveDataFrame(now_df, now_path + 'adj_r')


def LoadDataByDatesFromSql(startdate, enddate, instruments):
    engine = connect_sql()
    tot_dict = dict()
    for instrument in instruments:
        try:
            now_info = LoadDfFromSql(read_info(instrument, startdate, enddate), engine)
            now_contract = now_info['Contract'].tolist()
            now_df = LoadDfFromSql(read_quote(now_contract, startdate, enddate), engine)
            now_df['Date'] = now_df['Date'].dt.strftime('%Y-%m-%d')
            now_info['StartDate'] = now_info['StartDate'].dt.strftime('%Y-%m-%d')
            now_info['EndDate'] = now_info['EndDate'].dt.strftime('%Y-%m-%d')
            tot_dict[instrument] = {'info': now_info, 'trade': now_df}
        except Exception as e:
            print(instrument, ' ', e)
    return tot_dict


def rank_df(df, method, *args, **kwargs):
    if method == 'latest':
        re_df = df.iloc[:, ::-1].rank(axis='columns', method='first', *args, **kwargs).iloc[:, ::-1]
    elif method == 'time':
        now_df = (df > 0) * 1
        cumsum_df = now_df.cumsum(1)
        re_df = cumsum_df[df > 0]
    else:
        re_df = df.rank(axis='columns', method=method)
    return re_df


def reduce_rank(now_pivot, para_columns):
    '''
    out_rank = rank_1,rank_2,rank3 等 对应索引的最小值
    out_rank = rank_1,rank_2,rank3 等 对应索引的最大值
    :param now_pivot:
    :param para_columns:
    :return:
    '''
    tot_rank = None
    for now_column in para_columns:
        now_rank = rank_df(now_pivot[now_column], method='latest', ascending=False)
        if tot_rank is None:
            tot_rank = now_rank
        else:
            less_index = now_rank.gt(tot_rank)
            tot_rank[less_index] = now_rank
    return tot_rank


def no_roll_back(in_rank, para_mains, now_main=1):
    '''
    主力合约不回roll
    :param tot_rank:
    :return:
    '''
    if now_main > para_mains or in_rank.empty:
        return in_rank
    tot_rank = rank_df(in_rank, method='latest')
    if now_main == 1:
        in_rank = tot_rank
    cal_mask = tot_rank[tot_rank <= 1]
    cal_mask = cal_mask.fillna(method='ffill') > 0
    cal_rank = tot_rank[cal_mask]

    rank_cummin_col = cal_rank.cummin()
    rank_cummin_row = cal_rank.iloc[:, ::-1].cummin(1).iloc[:, ::-1]
    new_mask_col = (cal_rank - rank_cummin_col) > 0
    new_mask_row = (cal_rank - rank_cummin_row) > 0
    new_mask = new_mask_col | new_mask_row

    new_rank = cal_rank[~new_mask]
    new_rank[new_mask] = - new_rank.fillna(method='ffill')
    rer_rank = no_roll_back(in_rank[~cal_mask], para_mains, now_main + 1)
    rer_rank[cal_mask] = new_rank * now_main
    return rer_rank


def save_main(now_dict, in_rank, instrument):
    re_dict = dict()
    in_rank = in_rank[in_rank > 0]
    now_rank = in_rank.shift(1)
    rank_stack, rank_df = stack_contract(instrument, now_rank)
    re_dict['Contract'] = rank_df
    for now_column, now_df in now_dict.items():
        now_df = now_df[now_rank >= 0]
        now_se = now_df.stack().rename('new_data')
        join_df = rank_stack.join(now_se, how='outer')
        join_df = join_df.reset_index(level=1, drop=True).set_index('new_index', append=True)
        now_re = join_df.unstack()['new_data']
        re_dict[now_column] = now_re
    rank_stack, rank_df = stack_contract(instrument, in_rank)
    re_dict['ExeContract'] = rank_df
    return re_dict


def stack_contract(instrument, now_rank):
    rank_stack = now_rank.stack().astype('int')
    contract_str = [instrument + str(x).zfill(2) for x in rank_stack.values]
    rank_stack = pd.DataFrame(contract_str, index=rank_stack.index, columns=['new_index'])
    rank_df = rank_stack.reset_index(level=1).set_index('new_index', append=True).unstack()['level_1']
    return rank_stack, rank_df


def save_data(tot_ls, save_path):
    tot_dict = dict()
    for instrument, re_dict in tot_ls:
        for now_column, now_re in re_dict.items():
            if now_column in tot_dict:
                tot_dict[now_column] = tot_dict[now_column].join(now_re, how='outer')
            else:
                tot_dict[now_column] = now_re
    for now_column, now_df in tot_dict.items():
        SaveDataFrame(now_df, save_path + now_column)


def one_ins_roll(instrument, load_path, method, now_se, tot_ls):
    # print(instrument)
    para_mains = now_se['Contract_n']
    para_columns = now_se[['Volume', 'OpenInterest']]
    para_columns = para_columns[para_columns > 0].index.tolist()
    now_path = load_path + instrument + '/'
    now_columns = os.listdir(now_path)
    now_dict = dict()
    for now_column in now_columns:
        now_df = LoadDataFrame(now_path + now_column)
        if now_df.empty:
            continue
        if now_df.index.name is None:
            now_df = now_df.rename_axis('Date')
        now_dict[now_column] = now_df
    tot_rank = reduce_rank(now_dict, para_columns)
    if para_mains is not None:
        new_rank = no_roll_back(tot_rank, para_mains)
        now_rank = new_rank[new_rank <= para_mains]
    else:
        now_rank = rank_df(tot_rank, method=method)
    ins_dict = save_main(now_dict, now_rank, instrument)
    tot_ls.append([instrument, ins_dict])


def RollData(load_path, save_path, para_df, method):
    tot_ls = list()
    for instrument, now_se in para_df.iterrows():
        try:
            one_ins_roll(instrument, load_path, method, now_se, tot_ls)
        except Exception as e:
            print(instrument, ' ', e)
    save_data(tot_ls, save_path)


def main(startdate, enddate, pathO, instruments):
    dailyPX_path = pathO.dailyPX_path
    day_path = pathO.day_path
    para_df = LoadDataFrame(dailyPX_path + 'Instruments')
    para_df = para_df.reindex(index=instruments)

    tot_dict = LoadDataByDatesFromSql(startdate, enddate, instruments)
    UnionDatesData(tot_dict, day_path)
    CobWeb(instruments, startdate, enddate, day_path)

    print('roll main')
    RollData(day_path, dailyPX_path, para_df, None)
