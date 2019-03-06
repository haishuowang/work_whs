import sys
sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *

pnl_file_path = '/mnt/mfs/dat_whs/tmp_pnl_file'

target_file_list = ['market_top_300to800plus_industry_10_15_True_20181123_0431_hold_20__7',
                    'market_top_300to800plus_True_20181122_0952_hold_20__7',
                    'market_top_300to800plus_industry_55_True_20181123_2257_hold_20__7',
                    # 'market_top_800plus_True_20181124_0041_hold_20__7',
                    # 'market_top_800plus_industry_45_50_True_20181121_0945_hold_5__7',
                    'market_top_800plus_industry_45_50_True_20181125_1657_hold_20__7',
                    'market_top_800plus_industry_55_True_20181125_2031_hold_20__7']

all_pnl_df = pd.DataFrame()
for target_file in target_file_list:
    data = pd.read_csv(f'{pnl_file_path}/{target_file}.csv', index_col=0)
    data.columns = [target_file]
    all_pnl_df = pd.concat([all_pnl_df, data], axis=1)


'market_top_300to800plus_industry_10_15_True_20181123_0431_hold_20__7'

'market_top_300to800plus_True_20181122_0952_hold_20__7'
# 'market_top_800plus_True_20181124_0041_hold_20__7'

'market_top_300to800plus_industry_55_True_20181123_2257_hold_20__7'

# 'market_top_800plus_industry_45_50_True_20181121_0945_hold_5__7'
'market_top_800plus_industry_45_50_True_20181125_1657_hold_20__7'

'market_top_800plus_industry_55_True_20181125_2031_hold_20__7'
