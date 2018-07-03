import pandas as pd
import os
from collections import OrderedDict


root_path = '/mnt/mfs/dat_whs'


# 读取 sector(行业 最大市值等)
def load_sector_data(begin_date, end_date):
    market_top_n = pd.read_pickle('/mnt/mfs/DAT_EQT/STK_Groups1/market_top_1000.pkl')
    market_top_n = market_top_n[(market_top_n.index >= begin_date) & (market_top_n.index < end_date)]
    market_top_n = market_top_n[market_top_n.index >= begin_date]
    market_top_n.dropna(how='all', axis='columns', inplace=True)
    return market_top_n


# 读取 因涨跌停以及停牌等 不能变动仓位的日期信息
def load_locked_data(begin_date, end_date, sector_set):
    suspendday_df = pd.read_pickle('/mnt/mfs/DAT_EQT/EM_Tab14/adj_data/TRAD_TD_SUSPENDDAY/SUSPENDREASON_adj.pkl')
    limit_updn_df = pd.read_pickle('/mnt/mfs/dat_whs/data/locked_data/limit_updn_table.pkl')

    locked_df = limit_updn_df * suspendday_df
    locked_df = locked_df[(locked_df.index >= begin_date) & (locked_df.index < end_date)]
    locked_df = locked_df.reindex(columns=sector_set, fill_value=1)
    return locked_df


# 读取 上证50,沪深300等数据
def load_index_data(begin_date, end_date):
    # 上证50
    target_df_50 = pd.read_csv('/mnt/mfs/dat_whs/data/index_data/510050.SH_f1d.csv', index_col=0, header=None)
    # 沪深300
    target_df_300 = pd.read_csv('/mnt/mfs/dat_whs/data/index_data/510330.SH_f1d.csv', index_col=0, header=None)
    # 中证500
    target_df_500 = pd.read_csv('/mnt/mfs/dat_whs/data/index_data/512500.SH_f1d.csv', index_col=0, header=None)

    target_df = (target_df_50 + target_df_300 + target_df_500) * 1/3

    target_df.index = pd.to_datetime(target_df.index)
    target_df = target_df[(target_df.index >= begin_date) & (target_df.index < end_date)]
    return target_df[1]


# 读取return
def load_pct(begin_date, end_date, sector_set, n=5):
    load_path = os.path.join(root_path, 'data/adj_data/fnd_pct/pct_f{}d.pkl'.format(n))
    target_df = pd.read_pickle(load_path)
    target_df = target_df[(target_df.index >= begin_date) & (target_df.index < end_date)]
    target_df = target_df.reindex(columns=sector_set)
    return target_df


# 读取部分factor
def load_part_factor(begin_date, end_date, sector_df, locked_df, file_list):
    factor_set = OrderedDict()
    sector_set = sector_df.columns
    for file_name in file_list:
        load_path = os.path.join(root_path, 'data/adj_data/index_universe_f')
        target_df = pd.read_pickle(os.path.join(load_path, file_name + '.pkl'))
        target_df = target_df[(target_df.index >= begin_date) & (target_df.index < end_date)]
        target_df = target_df.reindex(columns=sector_set)
        target_df.fillna(0, inplace=True)

        # 排除涨跌停和停牌对策略的影响
        target_df = target_df * locked_df
        target_df.fillna(method='ffill', inplace=True)

        # sector筛选
        target_df = target_df * sector_df

        factor_set[file_name] = target_df
    return factor_set


#
# def load_deal_factor(begin_date, end_date, sector_df, locked_df, file_list):
#     path = '/mnt/mfs/dat_whs/data/adj_data/factor_set/'


def create_log_save_path(target_path):
    top_path = os.path.split(target_path)[0]
    if not os.path.exists(top_path):
        os.mkdir(top_path)
    if not os.path.exists(target_path):
        os.mknod(target_path)
