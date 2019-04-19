import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *


def get_intra_data_table(contract_list, col_list):
    def tmp_fun(contract_id):
        fut_name = re.sub('\d', '', 'IC1901')
        contract_data = pd.read_csv(f'{fut_root_path}/{fut_name}/{contract_id}.csv', sep='|', index_col=0)
        tmp_result_list = []
        for col in col_list:
            tmp_sr = contract_data[col]
            tmp_sr.name = contract_id
            tmp_result_list.append(tmp_sr)
        return tmp_result_list

    result_list = []
    for contract_ex_id in contract_list:
        contract_id = contract_ex_id.split('.')[0]
        result_list.append(tmp_fun(contract_id))

    target_list = []
    for i in range(len(col_list)):
        tmp_list = [x[i] for x in result_list]
        target_list.append(tmp_list)
    return target_list


def get_intra_data(contract_id):
    fut_name = re.sub('\d', '', contract_id)
    contract_data = pd.read_csv(f'{fut_root_path}/{fut_name}/{contract_id}.csv', sep='|', index_col=0)
    return contract_data


def get_file_name(fut_name):
    file_list = [x[:-4] for x in os.listdir(f'{fut_root_path}/{fut_name}') if len(x) <= 10]
    return file_list


def get_intra_return(begin_time, end_time, data_r):
    data_r_filter = data_r[(data_r['Time'] > begin_time) & (data_r['Time'] < end_time)]
    daily_return = data_r_filter.groupby(['Date']).sum().abs()
    return daily_return


def get_intra_return_con(begin_time, end_time, data_r):
    data_r
    return


def main_test_fun(contract_num, diff_limit=0.0010):
    IF_name = f'IF{contract_num}'
    IC_name = f'IC{contract_num}'

    IF_data = get_intra_data(IF_name)
    IC_data = get_intra_data(IC_name)

    IF_data_r = IF_data['Close'].pct_change()
    IF_data_r.name = IF_name + '_r'

    IC_data_r = IC_data['Close'].pct_change()
    IC_data_r.name = IC_name + '_r'

    data_r = pd.concat([IF_data_r, IC_data_r], axis=1, sort=True)
    data_r['diff'] = data_r[IF_name + '_r'] - data_r[IC_name + '_r']
    data_r['Date'] = data_r.index.str.slice(0, 10)
    data_r['Time'] = data_r.index.str.slice(11, 16)
    data_r_last_30m = data_r[(data_r['Time'] > '14:30') & (data_r['Time'] < '14:57')]
    daily_diff_last_30m = data_r_last_30m.groupby(['Date'])['diff'].sum()  # .sort_values(by=['diff'])
    # 基差半个小时内为 0.003
    daily_diff_last_30m_extra = daily_diff_last_30m[daily_diff_last_30m.abs() > diff_limit]
    pos_df = (daily_diff_last_30m_extra > 0).astype(int) - (daily_diff_last_30m_extra < 0).astype(int)
    begin_time, end_time = '09:00', '09:40'

    daily_return = get_intra_return(begin_time, end_time, data_r)[[IF_name + '_r', IC_name + '_r']]
    # daily_return = get_intra_return_con(begin_time, end_time, data_r)

    daily_return_diff = daily_return[IF_name + '_r'] - daily_return[IC_name + '_r']

    pnl_df = (pos_df * daily_return_diff.shift(-1)).dropna()
    pnl_df.index = pd.to_datetime(pnl_df.index)
    return pnl_df


def test_fun(contract_num):
    IF_name = f'IF{contract_num}'
    IC_name = f'IC{contract_num}'

    IF_data = get_intra_data(IF_name)
    IC_data = get_intra_data(IC_name)

    IF_data_r = IF_data['Close'].pct_change()
    IF_data_r.name = IF_name + '_r'

    IC_data_r = IC_data['Close'].pct_change()
    IC_data_r.name = IC_name + '_r'

    data_r = pd.concat([IF_data_r, IC_data_r], axis=1, sort=True)
    data_r['diff'] = data_r[IF_name + '_r'] - data_r[IC_name + '_r']
    data_r['Date'] = data_r.index.str.slice(0, 10)
    data_r['Time'] = data_r.index.str.slice(11, 16)
    data_r_last_30m = data_r[(data_r['Time'] > '14:35') & (data_r['Time'] < '14:57')]

    daily_diff_last_30m = data_r_last_30m.groupby(['Date'])['diff'].sum()  # .sort_values(by=['diff'])
    # 基差半个小时内为 0.003
    up_df = data_r_last_30m.groupby('Date')[IC_name + '_r'].apply(lambda x: ((x - x.shift(5)) > 0).sum())
    dn_df = data_r_last_30m.groupby('Date')[IC_name + '_r'].apply(lambda x: ((x - x.shift(5)) < 0).sum())

    pos_df_up = (up_df > 11).astype(int)
    pos_df_dn = (dn_df > 11).astype(int)

    begin_time, end_time = '09:00', '09:40'
    daily_return = get_intra_return(begin_time, end_time, data_r)[[IF_name + '_r', IC_name + '_r']]
    # daily_return = get_intra_return_con(begin_time, end_time, data_r)

    daily_return_diff = daily_return[IF_name + '_r'] - daily_return[IC_name + '_r']

    pnl_df = (pos_df_dn * daily_return_diff.shift(-1)).dropna()
    pnl_df.index = pd.to_datetime(pnl_df.index)
    pnl_df = pnl_df[pnl_df.index > pd.to_datetime('20160801')]
    # plot_send_result(pnl_df, bt.AZ_Sharpe_y(pnl_df), f'{contract_num} pnl', '')
    return pnl_df


if __name__ == '__main__':
    fut_root_path = '/mnt/mfs/DAT_FUT/intraday/fut_1mbar'
    # col_list = ['Close', 'Volume']

    if_file = [x[2:-4] for x in os.listdir(f'{fut_root_path}/IF')]
    ic_file = [x[2:-4] for x in os.listdir(f'{fut_root_path}/IC')]

    target_file = sorted(list(set(if_file) & set(ic_file)))
    # 只交易活跃合约
    target_file = [x for x in target_file if x[-1] in ['3', '6', '9']]
    # pnl_df_list = []
    all_pnl_df = pd.Series()
    for contract_num in target_file:
        pnl_df = test_fun(contract_num)
        all_pnl_df = all_pnl_df.add(pnl_df, fill_value=0)
    all_pnl_df.index = pd.to_datetime(all_pnl_df.index)
    plot_send_result(all_pnl_df, bt.AZ_Sharpe_y(all_pnl_df), 'all pnl', '')
