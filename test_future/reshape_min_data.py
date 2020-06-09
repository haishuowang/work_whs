import sys

sys.path = ['/mnt/mfs'] + sys.path
from work_whs.loc_lib.pre_load import *

root_path = '/mnt/mfs/DAT_FUT/intraday/fut_1mbar'


# fut_name_list = ['CU', 'ZN', 'AL''PB', 'AU', 'RB', 'RU', 'WR', 'FU', 'AG', 'BU', 'HC', 'NI', 'SN',
#                  'CF', 'SR', 'TA', 'WH', 'RI', 'JR', 'FG', 'OI', 'RM', 'RS', 'LR', 'SF', 'SM', 'MA',
#                  'ZC', 'CY', 'AP',
#                  'A', 'B', 'C', 'J', 'L', 'M', 'P', 'V', 'Y', 'JD', 'JM', 'I', 'FB', 'BB', 'PP', 'CS', 'SC', 'EG']

# fut_name_list = ['RB', 'J', 'JM', 'RU', 'BU', 'HC', 'NI', 'I']


def fun(part_data, cut_num):
    part_data_copy = part_data.copy('deep')

    cut_vol_num = sum(part_data['Volume'].iloc[:cut_num]) - 0.01
    part_data_copy['Volume'] = part_data_copy['Volume'].cumsum()

    if cut_vol_num != -0.01:
        # print(cut_vol_num)
        print(part_data['Date'].iloc[0], cut_vol_num)
        vol_cum = part_data['Volume'].cumsum()

        cut_vol_sr = vol_cum.apply(lambda x: int(x / cut_vol_num) if x % cut_vol_num == 0 else int(x / cut_vol_num) + 1)
        cut_vol_sr = cut_vol_sr.shift(1).fillna(1)
        cut_vol_df = pd.DataFrame(cut_vol_sr)

        target_index = cut_vol_df.groupby(by=['Volume']).apply(lambda x: x.index[-1]).values
        target_index = sorted(target_index)
        target_df = part_data_copy.loc[target_index]
        # print(target_df['Volume'])

        target_df['Volume'] = target_df['Volume'] - target_df['Volume'].shift(1).fillna(0)
        return target_df
    else:
        return pd.Series()


def fun_1(part_data, cut_num):
    part_data['Volume_cumsum'] = part_data['Volume'].cumsum()
    target_df = part_data[cut_num - 1::cut_num]
    if part_data.index[-1] not in target_df.index:
        target_df = pd.concat([target_df, part_data.iloc[[-1]]], axis=0)
    target_df['Volume'] = target_df['Volume_cumsum'] - target_df['Volume_cumsum'].shift(1).fillna(0)
    target_df.drop(columns=['Volume_cumsum'], inplace=True)
    return target_df


def deal_contract_1(fut_name, con_id, cut_num, save_path):
    print(con_id)
    con_df = pd.read_csv(f'{root_path}/{fut_name}/{con_id}', sep='|', index_col=0)
    # reshape_mul_df = con_df.groupby(by=['Date']).apply(lambda x: x[cut_num-1::cut_num])
    reshape_mul_df = con_df.groupby(by=['Date']).apply(fun_1, cut_num)

    if len(reshape_mul_df) > 0:
        target_index = reshape_mul_df.index.droplevel('Date')
        reshape_df = reshape_mul_df.set_index(target_index)

        reshape_df.to_csv(f'{save_path}/{fut_name}/{con_id}', sep='|')
        return reshape_df
    else:
        return None


def deal_contract(fut_name, con_id, cut_num, save_path):
    print(con_id)
    try:
        con_df = pd.read_csv(f'{root_path}/{fut_name}/{con_id}', sep='|', index_col=0)

        reshape_mul_df = con_df.groupby(by=['Date']).apply(fun, cut_num)  # .reset_index()[0]
        if len(reshape_mul_df) > 0:
            target_index = reshape_mul_df.index.droplevel('Date')
            reshape_df = reshape_mul_df.set_index(target_index)

            reshape_df.to_csv(f'{save_path}/{fut_name}/{con_id}', sep='|')
            return reshape_df
        else:
            return None
    except Exception as error:
        print(error)


if __name__ == '__main__':
    cut_num_list = [3, 5, 10, 20, 30]

    fut_name_list = [
        'TA', 'I', 'RB', 'J', 'JM', 'BU', 'HC', 'NI', 'ZN',
        'SC',
        'JD', 'CU', 'TA', 'MA', 'M', 'AP', 'RM', 'P', 'CF', 'OI', 'ZC', 'Y',
        'SR', 'RU', 'L', 'C', 'SM', 'SN', 'AG', 'PP', 'V', 'FG', 'B', 'CS', 'EG',
        'FU', 'RU', 'PB',
    ]
    good_instruments = [
        'CU', 'ZN', 'AL', 'PB', 'AU', 'RB', 'RU', 'WR', 'FU', 'AG', 'BU', 'HC', 'NI', 'SN',
        'CF', 'SR', 'TA', 'WH', 'RI', 'JR', 'FG', 'OI', 'RM', 'RS', 'LR', 'SF', 'SM', 'MA',
        'ZC', 'CY', 'AP',
        'A', 'B', 'C', 'J', 'L', 'M', 'P', 'V', 'Y', 'JD', 'JM', 'I', 'FB', 'BB', 'PP',
        'CS', 'SC', 'EG'
    ]
    pool = Pool(15)
    for fut_name in fut_name_list:
        print(fut_name)
        for cut_num in cut_num_list:
            save_path = f'/mnt/mfs/dat_whs/DAT_FUT/intraday/fut_{cut_num}mvolbar'
            bt.AZ_Path_create(f'{save_path}/{fut_name}')
            con_id_list = sorted(os.listdir(f'{root_path}/{fut_name}'))
            # con_id_list = [x for x in con_id_list if re.sub('\D', '', x) >= datetime.now().strftime('%Y%m')[2:]]
            for con_id in con_id_list:
                # reshape_df = deal_contract(fut_name, con_id, cut_num, save_path)
                pool.apply_async(deal_contract, (fut_name, con_id, cut_num, save_path))

        for cut_num in [3, 5, 15, 30]:
            save_path_1 = f'/mnt/mfs/dat_whs/DAT_FUT/intraday/fut_{cut_num}m_bar'
            bt.AZ_Path_create(f'{save_path_1}/{fut_name}')
            con_id_list = sorted(os.listdir(f'{root_path}/{fut_name}'))
            for con_id in con_id_list:
                # deal_contract_1(fut_name, con_id, cut_num, save_path_1)
                pool.apply_async(deal_contract_1, (fut_name, con_id, cut_num, save_path_1))
    pool.close()
    pool.join()
