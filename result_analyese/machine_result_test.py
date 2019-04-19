import sys
sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *


def load_pickle(root_path, sector_hold_ls, data_name, fun_name, file_name):
    data_fun_file_path = f'{root_path}/{sector_hold_ls}/{data_name}/{fun_name}/{file_name}'
    info_list = pd.read_pickle(data_fun_file_path)
    pnl_df = info_list[0]
    perf_list = info_list[1]
    pnl_df.name = '@'.join([sector_hold_ls, data_name, fun_name, file_name[:-4]])
    perf_df = pd.Series(perf_list, name='@'.join([data_name, fun_name, file_name[:-4]]))
    return pnl_df, perf_df


def get_all_pnl_fun():
    root_path = '/media/hdd2/dat_whs/data'
    sector_hold_ls = 'index_000300|1|False'
    data_name = 'close|volume'
    data_path = f'{root_path}/{sector_hold_ls}/{data_name}'
    result_list = []
    pool = Pool(20)
    fun_name_list = sorted(os.listdir(data_path))
    for fun_name in fun_name_list:
        file_name_list = sorted(os.listdir(f'{data_path}/{fun_name}'))
        for file_name in file_name_list:
            args = (root_path, sector_hold_ls, data_name, fun_name, file_name)
            # load_pickle(*args)
            result_list.append(pool.apply_async(load_pickle, args=args))
    pool.close()
    pool.join()
    result_list = [res.get() for res in result_list]
    all_pnl_df = pd.concat([res[0] for res in result_list], axis=1)
    all_perf_list = pd.concat([res[1] for res in result_list], axis=1)
    return all_pnl_df, all_perf_list


if __name__ == '__main__':
    all_pnl_df, perf_list = get_all_pnl_fun()
