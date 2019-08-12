import sys

sys.path.append('/mnf/mfs')
from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load import log
from work_whs.loc_lib.pre_load.plt import savfig_send

from work_whs.test_future.FutDataLoad import FutData, FutClass
from work_whs.loc_lib.pre_load.senior_tools import SignalAnalysis
from work_whs.test_future.signal_fut_fun import FutIndex, Signal, Position

fut_data = FutData()


def plot_contract_id(con_id, begin_date, end_date, use_col=None):
    if use_col is None:
        use_col = ['Close']
    con_intra = fut_data.load_intra_data(con_id, use_col)
    part_data = con_intra.truncate(before=begin_date, after=end_date)
    # part_data
    return part_data


con_id = 'RB1910.SHF'
begin_date = pd.to_datetime('20190424')
end_date = pd.to_datetime('20190724')

part_data = plot_contract_id(con_id, begin_date, end_date)

# plt.figure(figsize=[16, 10])
# plt.plot(part_data['Close'].values)
#
# x_ticks = part_data.index.strftime("%m%d %H:%M")
# a = list(range(0, len(part_data.index), 10))
# plt.xticks(a, x_ticks[a], rotation=45)
# plt.grid()
# savfig_send(subject=f'{con_id}|{begin_date.strftime("%Y%m%d")}|{end_date.strftime("%Y%m%d")}')

file_name_list = ['全球金属网', '兰格钢铁网', '大宗内参', '海鑫钢网', '瑞达期货', '生意社', '西本新干线']

for file_name in file_name_list[:1]:
    # file_name = '生意社_spider'
    file_name = '兰格钢铁网'
    data = pd.read_csv(f'/mnt/mfs/dat_whs/{file_name}.csv', index_col=0, parse_dates=True)[
        ['mid', 'buy', 'sell', 'mid_info', 'buy_info', 'sell_info']]

    # data = data.loc[list(set(data.index) - set(part_data.index))]
    # data = data.loc[list(set(data.index) & set(part_data.index))]

    data['sum'] = data.sum(1)
    data['index'] = data.index

    tmp_df = data.groupby(by=['index'])['sum'].sum()
    tmp_df[tmp_df > 0] = 1
    tmp_df[tmp_df < 0] = -1
    part_data['return_df'] = part_data['Close'] / part_data['Close'].shift(1) - 1

    part_data['tmp_df'] = tmp_df.reindex(index=sorted(list(set(part_data.index) & set(tmp_df.index)))) \
        .reindex(index=part_data.index)

    part_data['pos_df_raw'] = tmp_df.reindex(index=sorted(list(set(part_data.index) & set(tmp_df.index)))) \
        .replace(0, np.nan).reindex(index=part_data.index).fillna(method='ffill').reindex(index=part_data.index)

    # part_data['pos_df_raw'] = tmp_df.reindex(index=part_data.index)\
    #     .replace(0, np.nan).reindex(index=part_data.index).fillna(method='ffill').reindex(index=part_data.index)

    # part_data['pos_df_raw'] = part_data['tmp_df'].replace(0, np.nan).fillna(method='ffill')

    part_data['pos_df'] = part_data['pos_df_raw'].reindex(index=part_data.index).fillna(method='ffill')
    part_data['pnl_df'] = part_data['pos_df'] * part_data['return_df']

    daily_pnl = part_data.groupby(by=['Date'])['pnl_df'].sum()
    daily_pnl.index = pd.to_datetime(daily_pnl.index)

    sp = bt.AZ_Sharpe_y(daily_pnl)
    pot = part_data['pnl_df'].sum() / part_data['pos_df'].diff().abs().sum() * 10000

    plt.figure(figsize=[16, 10])
    plt.plot(daily_pnl.cumsum())
    plt.xticks(rotation=45)
    plt.grid()

    savfig_send()
