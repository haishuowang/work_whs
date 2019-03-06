import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *

target_index = ['09:31',
                '09:32',
                '09:33',
                '09:34',
                '09:35',
                '09:36',
                '09:37',
                '09:38',
                '09:39',
                '09:40',
                '09:41',
                '09:42',
                '09:43',
                '09:44',
                '09:45',
                '09:46',
                '09:47',
                '09:48',
                '09:49',
                '09:50',
                '09:51',
                '09:52',
                '09:53',
                '09:54',
                '09:55',
                '09:56',
                '09:57',
                '09:58',
                '09:59',
                '10:00',
                '10:01',
                '10:02',
                '10:03',
                '10:04',
                '10:05',
                '10:06',
                '10:07',
                '10:08',
                '10:09',
                '10:10',
                '10:11',
                '10:12',
                '10:13',
                '10:14',
                '10:15',
                '10:16',
                '10:17',
                '10:18',
                '10:19',
                '10:20',
                '10:21',
                '10:22',
                '10:23',
                '10:24',
                '10:25',
                '10:26',
                '10:27',
                '10:28',
                '10:29',
                '10:30',
                '10:31',
                '10:32',
                '10:33',
                '10:34',
                '10:35',
                '10:36',
                '10:37',
                '10:38',
                '10:39',
                '10:40',
                '10:41',
                '10:42',
                '10:43',
                '10:44',
                '10:45',
                '10:46',
                '10:47',
                '10:48',
                '10:49',
                '10:50',
                '10:51',
                '10:52',
                '10:53',
                '10:54',
                '10:55',
                '10:56',
                '10:57',
                '10:58',
                '10:59',
                '11:00',
                '11:01',
                '11:02',
                '11:03',
                '11:04',
                '11:05',
                '11:06',
                '11:07',
                '11:08',
                '11:09',
                '11:10',
                '11:11',
                '11:12',
                '11:13',
                '11:14',
                '11:15',
                '11:16',
                '11:17',
                '11:18',
                '11:19',
                '11:20',
                '11:21',
                '11:22',
                '11:23',
                '11:24',
                '11:25',
                '11:26',
                '11:27',
                '11:28',
                '11:29',
                '11:30',
                '13:01',
                '13:02',
                '13:03',
                '13:04',
                '13:05',
                '13:06',
                '13:07',
                '13:08',
                '13:09',
                '13:10',
                '13:11',
                '13:12',
                '13:13',
                '13:14',
                '13:15',
                '13:16',
                '13:17',
                '13:18',
                '13:19',
                '13:20',
                '13:21',
                '13:22',
                '13:23',
                '13:24',
                '13:25',
                '13:26',
                '13:27',
                '13:28',
                '13:29',
                '13:30',
                '13:31',
                '13:32',
                '13:33',
                '13:34',
                '13:35',
                '13:36',
                '13:37',
                '13:38',
                '13:39',
                '13:40',
                '13:41',
                '13:42',
                '13:43',
                '13:44',
                '13:45',
                '13:46',
                '13:47',
                '13:48',
                '13:49',
                '13:50',
                '13:51',
                '13:52',
                '13:53',
                '13:54',
                '13:55',
                '13:56',
                '13:57',
                '13:58',
                '13:59',
                '14:00',
                '14:01',
                '14:02',
                '14:03',
                '14:04',
                '14:05',
                '14:06',
                '14:07',
                '14:08',
                '14:09',
                '14:10',
                '14:11',
                '14:12',
                '14:13',
                '14:14',
                '14:15',
                '14:16',
                '14:17',
                '14:18',
                '14:19',
                '14:20',
                '14:21',
                '14:22',
                '14:23',
                '14:24',
                '14:25',
                '14:26',
                '14:27',
                '14:28',
                '14:29',
                '14:30',
                '14:31',
                '14:32',
                '14:33',
                '14:34',
                '14:35',
                '14:36',
                '14:37',
                '14:38',
                '14:39',
                '14:40',
                '14:41',
                '14:42',
                '14:43',
                '14:44',
                '14:45',
                '14:46',
                '14:47',
                '14:48',
                '14:49',
                '14:50',
                '14:51',
                '14:52',
                '14:53',
                '14:54',
                '14:55',
                '14:56',
                '14:57',
                '14:58',
                '14:59',
                '15:00']


def part_fun(month_path, day, target_index):
    print(day)
    day_path = f'{month_path}/{day}'
    name_list = os.listdir(day_path)
    # name_list.remove('vwap.csv')
    for file_name in name_list:
        data_path = f'{day_path}/{file_name}'
        data = pd.read_csv(data_path, index_col=0)
        if list(data.index) != target_index:
            print('error')
            # send_email.send_email(file_name, ['whs@yingpei.com'], [], day)


def check_index():
    begin_str = '20180101'
    end_str = '20190225'

    begin_year, begin_month, begin_day = begin_str[:4], begin_str[:6], begin_str
    end_year, end_month, end_day = end_str[:4], end_str[:6], end_str
    intraday_path = '/mnt/mfs/DAT_EQT/intraday/eqt_1mbar'
    year_list = [x for x in os.listdir(intraday_path) if (x >= begin_year) & (x <= end_year)]
    pool = Pool(25)
    for year in sorted(year_list):
        year_path = os.path.join(intraday_path, year)
        month_list = [x for x in os.listdir(year_path) if (x >= begin_month) & (x <= end_month)]
        for month in sorted(month_list):
            month_path = os.path.join(year_path, month)
            day_list = [x for x in os.listdir(month_path) if (x >= begin_day) & (x <= end_day)]
            for day in sorted(day_list):
                args = (month_path, day, target_index)
                part_fun(*args)
    #             pool.apply_async(part_fun, args=args)
    # pool.close()
    # pool.join()


def adj_price_fun(file_name):
    price_df = bt.AZ_Load_csv(f'/mnt/mfs/DAT_EQT/intraday/{file_name}.csv')
    factor_1 = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_FACTOR1/TAFACTOR.csv')
    factor_1 = factor_1.reindex(index=price_df.index)
    price_df = price_df.reindex(columns=factor_1.columns)
    return price_df / factor_1


def load_index_weight_plus(root_path, index_name):
    index_wt_p = bt.AZ_Load_csv(f'{root_path}/EM_Funda/IDEX_YS_WEIGHT_A/SECINDEXR_{index_name}plus.csv')
    index_wt_r = bt.AZ_Load_csv(f'{root_path}/EM_Funda/IDEX_YS_WEIGHT_A/SECINDEXR_{index_name}.csv')
    index_wt = index_wt_p.combine_first(index_wt_r)
    return index_wt


def intra_vwap_price():
    file_name_list = ['09:40_10:00',
                      '14:00_14:15',
                      '14:30_14:50']
    for file_name in file_name_list:
        adj_price = adj_price_fun(file_name)
        f_name = file_name.replace(':', '')
        adj_price.to_csv(f'/mnt/mfs/DAT_EQT/intraday/vwap_return/aadj_p{f_name}.csv', sep='|')


def index_vwap_return(index_name):
    file_name_list = ['aadj_r0940_1000',
                      'aadj_r1400_1415',
                      'aadj_r1430_1450']
    for file_name in file_name_list:
        vwap_return = bt.AZ_Load_csv(f'/mnt/mfs/DAT_EQT/intraday/vwap_return/{file_name}.csv')
        index_wt = load_index_weight_plus(root_path, index_name)
        index_wt = index_wt.reindex(index=vwap_return.index)
        index_df = (vwap_return * index_wt).sum(1)
        index_df.name = index_name
        index_df.to_csv(f'/mnt/mfs/DAT_EQT/intraday/vwap_return/index_{index_name}_{file_name[6:]}.csv', sep='|')


if __name__ == '__main__':
    root_path = '/mnt/mfs/DAT_EQT'
    # index_wt_300 = load_index_weight_plus(root_path, '000300')
    # index_wt_500 = load_index_weight_plus(root_path, '000905')
    # index_vwap_return('000300')
    index_vwap_return('000905')
