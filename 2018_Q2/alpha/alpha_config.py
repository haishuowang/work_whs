def write_config(config_file, add_config_info_list):
    for config_info in add_config_info_list:
        config_file.write(config_info)


if __name__ == '__main__':
    columns_list = 'alpha_id|alpha_calc|stock_univ|hedge_tool|hedge_type|machine|intraday\n'
    WHSMIRANA01_config = 'WHSMIRANA01.py|WHSCALC01.py|TOP500|300|Long_Short|1|0\n'
    add_config_info_list = [columns_list, WHSMIRANA01_config]
    with open('/mnt/mfs/alpha_whs/alpha.config', 'a+') as config_file:
        write_config(config_file, add_config_info_list)
