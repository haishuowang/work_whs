import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *

# today = datetime.today().strftime('%Y_%m_%d')
today = '20190312'
#
# root_path = '/mnt/mfs/DAT_PUBLIC/FIN_FUTURE_DATA'
# file_name_list = [x for x in os.listdir(root_path) if '_'.join([today[:4], today[4:6], today[6:]]) in x]


def tmp_fun(root_path, file_name):
    data = pd.read_csv(f'{root_path}/{file_name}', encoding='GBK')
    data['更新时间'] = [':'.join(x.split(':')[:2]) for x in data['更新时间']]
    tmp_close = data.groupby(['更新时间', '合约代码'])['最新价'].apply(lambda x: x.iloc[-1]).unstack()
    tmp_close.name = 'Close'

    tmp_open = data.groupby(['更新时间', '合约代码'])['最新价'].apply(lambda x: x.iloc[0]).unstack()
    tmp_open.name = 'Open'

    tmp_high = data.groupby(['更新时间', '合约代码'])['最新价'].max().unstack()
    tmp_high.name = 'High'

    tmp_low = data.groupby(['更新时间', '合约代码'])['最新价'].min().unstack()
    tmp_low.name = 'Low'

    tmp_volume = data.groupby(['更新时间', '合约代码'])['成交量'].sum().unstack()
    tmp_volume.name = 'Volume'

    tmp_open_interest = data.groupby(['更新时间', '合约代码'])['持仓量'].sum().unstack()
    tmp_open_interest.name = 'open_interest'

    tmp_turnover = data.groupby(['更新时间', '合约代码'])['换手率'].sum().unstack()
    tmp_turnover.name = 'turnover'
    return tmp_close, tmp_open, tmp_high, tmp_low, tmp_volume, tmp_open_interest, tmp_turnover


def fut_min_daily_update(today):
    root_path = '/mnt/mfs/DAT_PUBLIC/FIN_FUTURE_DATA'
    file_name_list = [x for x in os.listdir(root_path) if '_'.join([today[:4], today[4:6], today[6:]]) in x]

    result_list = []
    for file_name in file_name_list:
        print(file_name)
        result_list.append(tmp_fun(root_path, file_name))

    save_name_list = ['Close', 'Open', 'High', 'Low', 'Volume', 'open_interest', 'turnover']
    for i in range(len(save_name_list)):
        save_file_name = save_name_list[i]
        target_df = pd.concat([res[i] for res in result_list], axis=1)
        save_path = f'/mnt/mfs/DAT_EQT/intraday/fut_1mbar/{today}'
        bt.AZ_Path_create(save_path)
        target_df.to_csv(f'{save_path}/{save_file_name}.csv', sep='|', index_label='Time')


if __name__ == '__main__':
    today = datetime.now().strftime('%Y%m%d')
    # today = '20190322'
    fut_min_daily_update(today)
