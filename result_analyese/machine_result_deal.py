import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *
import work_whs.AZ_2019_Q1.create_tech_factor as ctf


def load_pickle(root_path, sector_hold_ls, data_name, fun_name, file_name):
    data_fun_file_path = f'{root_path}/{sector_hold_ls}/{data_name}/{fun_name}/{file_name}'
    info_list = pd.read_pickle(data_fun_file_path)
    pnl_df = info_list[0]
    perf_list = info_list[1]
    pnl_df.name = '@'.join([sector_hold_ls, data_name, fun_name, file_name[:-4]])
    perf_df = pd.Series(perf_list, name='@'.join([data_name, fun_name, file_name[:-4]]))
    return pnl_df, perf_df


def select_fun(file_name, pnl_table_c, max_num=10):
    """
    挑选corr底的函数
    :param file_name:
    :param pnl_table_c:
    :param max_num:
    :return:
    """
    i = 0
    target_pnl = pd.DataFrame(pnl_table_c[file_name])
    target_pnl.columns = ['target_pnl']
    target_sp = bt.AZ_Sharpe_y(target_pnl).values[0]
    if target_sp < 0:
        target_pnl = -target_pnl

    select_list = []
    while i < max_num:
        pnl_corr = pd.concat([pnl_table_c, target_pnl], axis=1).corr()['target_pnl']
        select_name = pnl_corr.abs().sort_values().index[0]
        select_pnl = pnl_table_c[select_name]
        pnl_table_c = pnl_table_c.drop(columns=select_name)
        select_sp = bt.AZ_Sharpe_y(select_pnl)
        if select_sp > 0:
            tmp_pnl = target_pnl.add(select_pnl, axis=0)
        else:
            tmp_pnl = target_pnl.sub(select_pnl, axis=0)
        tmp_sp = bt.AZ_Sharpe_y(tmp_pnl).values[0]

        if tmp_sp > target_sp:
            target_pnl = tmp_pnl
            select_list.append(select_name)
            i += 1
        else:
            i += 1
    return select_list


def get_all_pnl_corr(pnl_df, col_name):
    all_pnl_df = pd.read_csv('/mnt/mfs/AATST/corr_tst_pnls', sep='|', index_col=0, parse_dates=True)
    all_pnl_df_c = pd.concat([all_pnl_df, pnl_df], axis=1)
    a = all_pnl_df_c.iloc[-600:].corr()[col_name]
    print(a[a > 0.5])
    return a[a > 0.62]


def part_single_test(col_name, pnl_table, sharpe_mid, sharpe_df, tech_factor):
    # try:
    # 挑选数据
    select_list = select_fun(col_name, pnl_table[sharpe_mid.index], max_num=10)
    portfolio_index = [col_name] + list(select_list)
    buy_sell_way_df = sharpe_df[portfolio_index]
    select_pnl_df = pnl_table[portfolio_index]
    target_pnl = (select_pnl_df * buy_sell_way_df).sum(1)
    target_sharpe = bt.AZ_Sharpe_y(target_pnl)
    target_lvr = bt.AZ_Leverage_ratio(target_pnl.cumsum())
    print(target_sharpe, target_lvr)
    pnl_df, sp, pot = tech_factor.mix_test_fun(portfolio_index)
    print(sp, pot)
    # if sp > 2.4:
    # plot_send_result(pnl_df, sp, col_name.split('@')[-1], '|'.join(['|'.join(select_list), str(sp), str(pot)]))
    plot_send_result(pnl_df, sp, col_name, '|'.join(['|'.join(select_list), str(sp), str(pot)]))


def get_all_pnl_fun(sector_hold_ls, data_name):
    root_path = '/media/hdd2/dat_whs/data'
    # sector_hold_ls = 'index_000300|1|False'
    # data_name = 'close|volume'
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


def corr_filter_fun(all_pnl_df):
    print(f'pnl总体数量：{len(all_pnl_df.columns)}')
    sharpe_df = all_pnl_df.apply(bt.AZ_Sharpe_y)
    sharpe_df_sort = sharpe_df.sort_values()
    sharpe_mid = sharpe_df_sort[sharpe_df_sort.abs() > 0.7]

    part_all_pnl_df = all_pnl_df[sharpe_mid.index]
    part_all_pnl_corr = part_all_pnl_df.corr()
    a = time.time()
    part_all_pnl_high_corr = part_all_pnl_corr[part_all_pnl_corr.abs() > 0.85]
    columns_list = part_all_pnl_high_corr.columns
    for x in columns_list:
        if x in part_all_pnl_high_corr.columns:
            drop_list = list(part_all_pnl_high_corr[x].dropna().index)
            drop_list.remove(x)
            # print(drop_list)
            part_all_pnl_high_corr = part_all_pnl_high_corr.drop(index=drop_list, columns=drop_list)
        else:
            pass
    filter_col_list = part_all_pnl_high_corr.index
    b = time.time()
    print(b - a)
    print(f'过滤后pnl数量：{len(filter_col_list)}')

    return filter_col_list


def main_fun():
    root_path = '/mnt/mfs/DAT_EQT'
    sector_hold_ls = 'index_000300|1|False'
    data_name = 'close|volume'

    sector_name, hold_time, if_only_long = sector_hold_ls.split('|')
    hold_time = int(hold_time)
    if if_only_long == 'True':
        if_only_long = True
    else:
        if_only_long = False

    tech_factor = ctf.get_tech_factor_fun(root_path, sector_name, hold_time, if_only_long)
    # 获取 pnl和performance
    # all_pnl_df, all_perf_df = get_all_pnl_fun(sector_hold_ls, data_name)
    #
    #
    # filter_col_list = corr_filter_fun(all_pnl_df)
    # pd.to_pickle(filter_col_list, f'/mnt/mfs/dat_whs/{sector_hold_ls}|{data_name}|filter_col_list.pkl')

    # filter_col_list = pd.read_pickle(f'/mnt/mfs/dat_whs/{sector_hold_ls}|{data_name}|filter_col_list.pkl')
    # filter_pnl_df = all_pnl_df[filter_col_list]
    # pd.to_pickle(filter_pnl_df, f'/mnt/mfs/dat_whs/{sector_hold_ls}|{data_name}|filter_pnl_df.pkl')
    filter_pnl_df = pd.read_pickle(f'/mnt/mfs/dat_whs/{sector_hold_ls}|{data_name}|filter_pnl_df.pkl')
    # filter_pnl_df = pd.read_pickle('/mnt/mfs/dat_whs/filter_pnl_df.pkl')
    sharpe_df = filter_pnl_df.apply(bt.AZ_Sharpe_y)
    sharpe_df_sort = sharpe_df.sort_values()

    sharpe_up = sharpe_df_sort[sharpe_df_sort.abs() > 0.8]
    # print(sharpe_up)
    sharpe_mid = sharpe_df_sort[sharpe_df_sort.abs() > 0.7]
    pool = Pool(28)
    for col_name in sharpe_up.index:
        args = (col_name, filter_pnl_df, sharpe_mid, sharpe_df, tech_factor)
        # part_single_test(*args)
        pool.apply_async(part_single_test, args=args)
    pool.close()
    pool.join()


def single_back_test():
    root_path = '/mnt/mfs/DAT_EQT'
    sector_name = 'market_top_2000'
    hold_time = 5
    if_only_long = True
    tech_factor = ctf.get_tech_factor_fun(root_path, sector_name, hold_time, if_only_long)
    data_name, fun_name, file_name = ['close|volume', 'diff|ma_diff|zscore_row',
                                      'fun_path11|(5,)_([2, 10],)_[]|None||None||pnd_col_extre|5_0.3|1']
    tech_factor.single_test_fun(data_name, fun_name, file_name)


if __name__ == '__main__':
    main_fun()
    # single_back_test()
    pass
