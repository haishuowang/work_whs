import pandas as pd


def write_config(config_file, add_config_info_list):
    for config_info in add_config_info_list:
        config_file.write(config_info)


if __name__ == '__main__':
    # columns_list = 'alpha_id|alpha_calc|stock_univ|hedge_tool|hedge_type|machine|intraday|inception_dt\n'
    # WHSMIRANA01_config = 'WHSMIRANA01.py|WHSCALC01.py|TOP500|300|Long_Short|1|0|\n'
    # WHS018JUN01_config = 'WHS018JUN01.py|WHSCALC01.py|TOP2000|500|Long_Short|1|0|2018-08-28\n'
    # WHS018JUL01_config = 'WHS018JUL01.py|WHSCALC01.py|TOP2000|500|Long_Short|1|0|2018-08-28\n'
    # WHS018AUG01_config = 'WHS018AUG01.py|WHSCALC01.py|TOP2000|500|Long_Short|1|0|2018-08-28\n'

    # add_config_info_list = [WHS018JUN01_config, WHS018JUL01_config, WHS018AUG01_config]
    with open('/mnt/mfs/alpha_whs/alpha.config', 'a+') as config_file:
        write_config(config_file, add_config_info_list)

    # aadj_r_intra_vwap.loc[pd.to_datetime('2015-08-12'), '002506.SZ']

    # config1 = pd.read_pickle('/mnt/mfs/alpha_whs/config01.pkl')
    # factor_info1 = config1['factor_info']
    #
    # config2 = pd.read_pickle('/mnt/mfs/alpha_whs/018AUG.pkl')
    # factor_info2 = config2['factor_info']
    #
    # config3 = pd.read_pickle('/mnt/mfs/alpha_whs/018JUL.pkl')
    # factor_info3 = config3['factor_info']
    #
    # config4 = pd.read_pickle('/mnt/mfs/alpha_whs/018JUN.pkl')
    # factor_info4 = config4['factor_info']
    #
    # file_name_list = sorted(list(set(factor_info1[['name1', 'name2', 'name3']].values.ravel()) |
    #                              set(factor_info2[['name1', 'name2', 'name3']].values.ravel()) |
    #                              set(factor_info3[['name1', 'name2', 'name3']].values.ravel()) |
    #                              set(factor_info4[['name1', 'name2', 'name3']].values.ravel())))
    # daily_file = [x for x in file_name_list if x.startswith('R_')]
