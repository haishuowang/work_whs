import sys

sys.path.append('/mnt/mfs')
from work_dmgr_fut.loc_lib.pre_load import *

time_list = np.array(['09:31', '09:32', '09:33', '09:34', '09:35', '09:36', '09:37', '09:38', '09:39', '09:40',
                      '09:41', '09:42', '09:43', '09:44', '09:45', '09:46', '09:47', '09:48', '09:49', '09:50',
                      '09:51', '09:52', '09:53', '09:54', '09:55', '09:56', '09:57', '09:58', '09:59', '10:00',
                      '10:01', '10:02', '10:03', '10:04', '10:05', '10:06', '10:07', '10:08', '10:09', '10:10',
                      '10:11', '10:12', '10:13', '10:14', '10:15', '10:16', '10:17', '10:18', '10:19', '10:20',
                      '10:21', '10:22', '10:23', '10:24', '10:25', '10:26', '10:27', '10:28', '10:29', '10:30',
                      '10:31', '10:32', '10:33', '10:34', '10:35', '10:36', '10:37', '10:38', '10:39', '10:40',
                      '10:41', '10:42', '10:43', '10:44', '10:45', '10:46', '10:47', '10:48', '10:49', '10:50',
                      '10:51', '10:52', '10:53', '10:54', '10:55', '10:56', '10:57', '10:58', '10:59', '11:00',
                      '11:01', '11:02', '11:03', '11:04', '11:05', '11:06', '11:07', '11:08', '11:09', '11:10',
                      '11:11', '11:12', '11:13', '11:14', '11:15', '11:16', '11:17', '11:18', '11:19', '11:20',
                      '11:21', '11:22', '11:23', '11:24', '11:25', '11:26', '11:27', '11:28', '11:29', '11:30',
                      '13:01', '13:02', '13:03', '13:04', '13:05', '13:06', '13:07', '13:08', '13:09', '13:10',
                      '13:11', '13:12', '13:13', '13:14', '13:15', '13:16', '13:17', '13:18', '13:19', '13:20',
                      '13:21', '13:22', '13:23', '13:24', '13:25', '13:26', '13:27', '13:28', '13:29', '13:30',
                      '13:31', '13:32', '13:33', '13:34', '13:35', '13:36', '13:37', '13:38', '13:39', '13:40',
                      '13:41', '13:42', '13:43', '13:44', '13:45', '13:46', '13:47', '13:48', '13:49', '13:50',
                      '13:51', '13:52', '13:53', '13:54', '13:55', '13:56', '13:57', '13:58', '13:59', '14:00',
                      '14:01', '14:02', '14:03', '14:04', '14:05', '14:06', '14:07', '14:08', '14:09', '14:10',
                      '14:11', '14:12', '14:13', '14:14', '14:15', '14:16', '14:17', '14:18', '14:19', '14:20',
                      '14:21', '14:22', '14:23', '14:24', '14:25', '14:26', '14:27', '14:28', '14:29', '14:30',
                      '14:31', '14:32', '14:33', '14:34', '14:35', '14:36', '14:37', '14:38', '14:39', '14:40',
                      '14:41', '14:42', '14:43', '14:44', '14:45', '14:46', '14:47', '14:48', '14:49', '14:50',
                      '14:51', '14:52', '14:53', '14:54', '14:55', '14:56', '14:57', '14:58', '14:59', '15:00'])


# date_list = ['20200108', '20200108', '20200110']


def AZ_Path_create(target_path):
    """
    添加新路径
    :param target_path:
    :return:
    """
    if not os.path.exists(target_path):
        os.makedirs(target_path)


def deal_fun(date, opt_file):
    def fun(part_tick_df):
        part_tick_df = part_tick_df.sort_values(by='MDTime')
        part_target_df = pd.Series()
        part_target_df['HTSCSecurityID'] = part_tick_df['HTSCSecurityID'].iloc[-1]
        part_target_df['MDDate'] = part_tick_df['MDDate'].iloc[-1].astype(str)
        part_target_df['MDMin'] = part_tick_df['MDMin'].iloc[-1]
        part_target_df['Open'] = part_tick_df['LastPx'].iloc[0]
        part_target_df['High'] = part_tick_df['LastPx'].max()
        part_target_df['Low'] = part_tick_df['LastPx'].min()
        part_target_df['Close'] = part_tick_df['LastPx'].iloc[-1]
        part_target_df['TotalVolumeTrade'] = part_tick_df['TotalVolumeTrade'].iloc[-1]
        part_target_df['TotalValueTrade'] = part_tick_df['TotalValueTrade'].iloc[-1]
        return part_target_df

    data = pd.read_csv(f'{tick_path}/{date}/{opt_file}')
    if os.path.exists(f'{min_path}/{opt_file}'):
        old_target_df = pd.read_csv(f'{min_path}/{opt_file}', sep='|')
        old_target_df['TradeTime'] = pd.to_datetime(old_target_df['TradeTime'])
        old_target_df.index = old_target_df['TradeTime']
        last_time = int((old_target_df.index[-1] - timedelta(minutes=1)).strftime('%H%M')) * 100000
        data = data[data['MDTime'] > last_time]

    tick_df = data[['HTSCSecurityID', 'MDDate', 'MDTime', 'PreClosePx', 'TotalVolumeTrade', 'TotalValueTrade', 'LastPx',
                    'BuyPriceQueue1', 'SellPriceQueue1']]
    tick_df['MDMin'] = tick_df['MDTime'].astype(str).str.slice(0, -5).str.zfill(4)
    tick_df['MDMin'] = tick_df['MDMin'].str.slice(0, 2) + ':' + tick_df['MDMin'].str.slice(2)

    target_df = tick_df.groupby(by='MDMin').apply(fun)
    target_df.index = target_df['MDMin']
    target_df = target_df.ffill()
    target_df['TradeTime'] = pd.to_datetime(target_df['MDDate'].astype(str) + ' ' + target_df['MDMin'])

    target_df['Volume'] = target_df['TotalVolumeTrade'] - target_df['TotalVolumeTrade'].shift(1).fillna(0)
    target_df['Turnover'] = target_df['TotalValueTrade'] - target_df['TotalValueTrade'].shift(1).fillna(0)
    target_df = target_df[['TradeTime', 'MDDate', 'MDMin', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']]
    target_df.index = target_df['TradeTime']
    AZ_Path_create(f'{min_path}/{date}')
    if os.path.exists(f'{min_path}/{opt_file}'):
        new_target_df = old_target_df.combine_first(target_df)
        new_target_df = new_target_df.sort_values(by='TradeTime') \
            .reindex(time_list[time_list <= (datetime.now() + timedelta(minutes=1)).strftime('%H:%M')])
        new_target_df.to_csv(f'{min_path}/{date}/{opt_file}', sep='|', index=False)
    else:
        target_df.to_csv(f'{min_path}/{date}/{opt_file}', sep='|', index=False)


if __name__ == '__main__':
    min_path = '/media/hdd1/DAT_OPT/Min_day'
    tick_path = '/media/hdd1/DAT_OPT/Tick'
    now_date = datetime.now().strftime('%Y%m%d')
    opt_file_list = sorted(os.listdir(f'{tick_path}/{now_date}'))
    for opt_file in opt_file_list:
        print(now_date, opt_file)
        t1 = time.time()
        deal_fun(now_date, opt_file)
        t2 = time.time()
        print(t2 - t1)
        print(datetime.now())
    if datetime.now().hour < 15:
        for opt_file in opt_file_list:
            print(now_date, opt_file)
            deal_fun(now_date, opt_file)
