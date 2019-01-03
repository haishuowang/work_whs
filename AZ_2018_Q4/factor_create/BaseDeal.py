import pandas as pd
import numpy as np
import work_whs.loc_lib.shared_paths.path as pt
import work_whs.loc_lib.shared_tools.back_test as bt
import os


class BaseDeal:
    @staticmethod
    def info_dict_fun(fun, raw_data_path, args, save_path, if_replace):
        info_dict = dict()
        info_dict['fun'] = fun
        info_dict['raw_data_path'] = raw_data_path
        info_dict['args'] = args
        info_dict['if_replace'] = if_replace
        pd.to_pickle(info_dict, save_path)

    def judge_save_fun(self, target_df, file_name, save_root_path, fun, raw_data_path, args, if_filter=True,
                       if_replace=False):
        factor_to_fun = '/mnt/mfs/dat_whs/data/factor_to_fun_v2'
        if target_df.sum().sum() == 0:
            print('factor not enough!')
            return -1
        elif if_filter:
            print(f'{file_name}')
            print(target_df.iloc[-100:].abs().replace(0, np.nan).sum(axis=1).mean(), len(target_df.iloc[-100:].columns))
            print(
                target_df.iloc[-100:].abs().replace(0, np.nan).sum(axis=1).mean() / len(target_df.iloc[-100:].columns))
            target_df.to_pickle(os.path.join(save_root_path, file_name + '.pkl'))
            # 构建factor_to_fun的字典并存储
            self.info_dict_fun(fun, raw_data_path, args, os.path.join(factor_to_fun, file_name), if_replace)
            print(f'{file_name} success!')
            return 0
        else:
            target_df.to_pickle(os.path.join(save_root_path, file_name + '.pkl'))
            # 构建factor_to_fun的字典并存储
            self.info_dict_fun(fun, raw_data_path, args, os.path.join(factor_to_fun, file_name), if_replace)
            print(f'{file_name} success!')
            return 0
