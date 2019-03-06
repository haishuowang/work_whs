import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *


s_t = 60

root_path = f'/mnt/mfs/DAT_EQT/intraday/eqt_{s_t}mbar'
date_list = sorted(os.listdir(root_path))

# for i in range(int(240 / s_t)):
#     exec(f'intra_vwap_{s_t}_tab_{i+1}_df = pd.DataFrame()')
return_df = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv')


def get_daily_data(root_path, today, num):
    print(today)

    data = pd.read_csv(f'{root_path}/{today}/intra_vwap.csv', sep='|', index_col=0)
    # data.index = pd.to_datetime(today + ' ' + data.index)
    data.index = pd.to_datetime(today + ' ' + data.index)
    return data.iloc[num]


result_list = []
pool = Pool(25)
for today in date_list:
    args = (root_path, today, 3,)
    result_list.append(pool.apply_async(get_daily_data, args))
pool.close()
pool.join()

target_df = pd.DataFrame()
for res in result_list:
    target_df = target_df.append(res.get())


a = target_df.reindex(columns=return_df.columns)
a.index = pd.to_datetime([x.strftime('%Y%m%d') for x in a.index])
adj_factor = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_FACTOR1/TAFACTOR.csv')\
    .reindex(index=a.index, columns=a.columns)

b = (a.fillna(method='ffill')/adj_factor).pct_change()

b.to_csv('/mnt/mfs/dat_whs/return_60min_tab_4_df.csv', sep='|')
