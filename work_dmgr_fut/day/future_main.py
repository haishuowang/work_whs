from datetime import datetime, timedelta
import sys

sys.path.append('/mnt/mfs')

import roll_data as rolldata
from open_lib.shared_paths.path import _BaseDirs

_author__ = 'zijie.ren'


class _BaseDirs_Future(_BaseDirs):
    def __init__(self, mode=None, pt_mode='pro'):
        _BaseDirs.__init__(self, mode)
        BasePath = str(self.BASE_PATH)
        self.base_path = BasePath + "/DAT_FUT/"
        self.day_path = self.base_path + 'day/'
        if pt_mode == 'pro':
            self.dailyPX_path = self.base_path + 'DailyPX/'
        elif pt_mode == 'bkt':
            self.dailyPX_path = self.day_path + 'DailyPX/'


def Instruments(group='all'):
    instruments = []
    if group in ['finace', 'all']:
        finace_instruments = ['IF', 'IC', 'IH', 'TS', 'TF', 'T']
        instruments.extend(finace_instruments)
    if group in ['good', 'all']:
        good_instruments = [
            'CU', 'ZN', 'AL', 'PB', 'AU', 'RB', 'RU', 'WR', 'FU', 'AG', 'BU', 'HC', 'NI', 'SN',
            'CF', 'SR', 'TA', 'WH', 'RI', 'JR', 'FG', 'OI', 'RM', 'RS', 'LR', 'SF', 'SM', 'MA',
            'ZC', 'CY', 'AP',
            'A', 'B', 'C', 'J', 'L', 'M', 'P', 'V', 'Y', 'JD', 'JM', 'I', 'FB', 'BB', 'PP', 'CS'
            , 'SC', 'EG'
        ]
        instruments.extend(good_instruments)

    return instruments


def main():
    pathO = _BaseDirs_Future(pt_mode='pro')
    # startdate = '2018-07-01'
    # enddate = '2019-04-22'
    startdate = datetime.strftime(datetime.now() - timedelta(6), '%Y-%m-%d')
    enddate = datetime.strftime(datetime.now(), '%Y-%m-%d')

    instruments = Instruments('all')
    print('processing prepar roll data')
    rolldata.main(startdate, enddate, pathO, instruments)


if __name__ == '__main__':
    # 19点更新
    main()
