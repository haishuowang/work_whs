import pandas as pd
from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load.plt import savfig_send
from open_lib.shared_paths.path import _BinFiles

mode = 'bkt'
root_path = _BinFiles(mode)


def AZ_Load_csv(target_path, parse_dates=True, sep='|', **kwargs):
    target_df = pd.read_table(target_path, sep=sep, index_col=0, low_memory=False, parse_dates=parse_dates, **kwargs)
    return target_df


def generation(index_name):
    weight_df_new = AZ_Load_csv(root_path.EM_Funda.IDEX_YS_WEIGHT_A / f'SECINDEXR_{index_name}.csv')
    weight_df_old = AZ_Load_csv(root_path.EM_Funda.IDEX_YS_WEIGHT_A / f'SECINDEXR_{index_name}plus.csv')
    weight_df = weight_df_old.combine_first(weight_df_new)

    stock_return_df = AZ_Load_csv(root_path.EM_Funda.DERIVED_14 / 'aadj_r.csv')
    index_count = (weight_df * stock_return_df).sum(1)
    print(index_count)
    plt.plot(index_count.cumsum(), label='count')
    savfig_send()
    # index_count.to_csv(root_path.EM_Funda.DERIVED_WHS / f'CHG_{index_name}.csv', sep='|')


# def update(index_name):
#     weight_df = AZ_Load_csv(root_path.EM_Funda.IDEX_YS_WEIGHT_A / f'SECINDEXR_{index_name}.csv')
#     stock_return_df = AZ_Load_csv(f'{root_path}/EM_Funda/DERIVED_14/aadj_r.csv')
#     index_count = (weight_df * stock_return_df).sum(1)
#     index_count.to_csv(f'{root_path}/EM_Funda/DERIVED_WHS/CHG_{index_name}.csv', sep='|')


# weight_df_new = AZ_Load_csv(root_path.EM_Funda.IDEX_YS_WEIGHT_A / f'SECINDEXR_{index_name}.csv')
# weight_df_old = AZ_Load_csv(root_path.EM_Funda.IDEX_YS_WEIGHT_A / f'SECINDEXR_{index_name}plus.csv')
# weight_df = weight_df_old.combine_first(weight_df_new)
#
# stock_return_df = AZ_Load_csv(root_path.EM_Funda.DERIVED_14 / 'aadj_r.csv')
# index_count = (weight_df * stock_return_df).sum(1)
# index_real = AZ_Load_csv(root_path.EM_Funda.INDEX_TD_DAILYSYS / 'CHG.csv')[index_name]
#
# index_count_c = index_count.truncate(before=pd.to_datetime('20100101'))
# index_real_c = index_real.truncate(before=pd.to_datetime('20100101')) * 0.01
# plt.plot(index_count_c.cumsum(), label='count')
# plt.plot(index_real_c.cumsum(), label='real')
# plt.plot(index_count_c.cumsum() - index_real_c.cumsum(), label='real')
# plt.legend()
# savfig_send()

def main():
    index_name_list = ['000300', '000905']
    for index_name in index_name_list:
        generation(index_name)


if __name__ == '__main__':
    main()
