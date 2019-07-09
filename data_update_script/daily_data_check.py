import sys

sys.path.append('/mnt/mfs')

from work_whs.loc_lib.pre_load import *


def find_target_file(begin_time, end_time, time_type, endswith=None):
    def time_judge(begin_time, end_time, target_time):
        # print(datetime.fromtimestamp(target_time))
        if end_time >= datetime.fromtimestamp(target_time) >= begin_time:
            return True
        else:
            return False

    result_root_path = '/mnt/mfs/dat_whs/result/result'
    raw_file_name_list = [x for x in os.listdir(result_root_path)
                          if os.path.getsize(os.path.join(result_root_path, x)) != 0]

    if endswith is not None:
        raw_file_name_list = [x for x in os.listdir(result_root_path)
                              if x[:-4].endswith(endswith)]

    if time_type == 'm':
        result_file_name_list = [x[:-4] for x in raw_file_name_list if
                                 time_judge(begin_time, end_time, os.path.getmtime(os.path.join(result_root_path, x)))]
        return sorted(result_file_name_list)
    elif time_type == 'c':
        result_file_name_list = [x[:-4] for x in raw_file_name_list if
                                 time_judge(begin_time, end_time, os.path.getctime(os.path.join(result_root_path, x)))]

        return sorted(result_file_name_list)

    elif time_type == 'a':
        result_file_name_list = [x[:-4] for x in raw_file_name_list if
                                 time_judge(begin_time, end_time, os.path.getatime(os.path.join(result_root_path, x)))]

        return sorted(result_file_name_list)

    else:
        return []


class DataCheck:
    def __init__(self, today_date):
        # self.today = datetime.now().strftime('%Y%m%d')
        self.today_date = today_date

    def EQT_intra_check(self):
        a = os.path.getctime('/mnt/mfs/DAT_EQT/intraday/eqt_1mbar/{0:.4s}/{0:.6s}/{0:.8s}/Close.csv'.format(self.today_date))
        datetime.fromtimestamp(a)

