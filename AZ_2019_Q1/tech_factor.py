import sys

sys.path.append('/mnf/mfs')

from work_whs.loc_lib.pre_load import *


class TechFactor:
    def __init__(self, open_, close, high, low, volume, turnover):
        self.open, self.close, self.high, self.low, self.volume, self.turnover = \
            open_, close, high, low, volume, turnover

    def DTM_fun(self):
        def get_biger(x, y):
            x[x < y] = y[x < y]
            return x

        target_df = pd.DataFrame(index=self.open.index, columns=self.open.columns)
        open_diff = self.open - self.open.shift(1)
        target_df[open_diff <= 0] = 0
        target_df[open_diff > 0] = (self.high - self.open).combine(open_diff, get_biger)
        return target_df

    def DBM_fun(self):
        def get_biger(x, y):
            x[x < y] = y[x < y]
            return x

        target_df = pd.DataFrame(index=self.open.index, columns=self.open.columns)
        open_diff = self.open.shift(1) - self.open
        target_df[open_diff <= 0] = 0
        target_df[open_diff > 0] = (self.open - self.low).combine(open_diff, get_biger)
        return target_df

    def ADTM_fun(self, n=23):
        def get_result(x, y):
            a = 1 - y[x > y]/x[x > y]
            b = x[x < y]/y[x < y] - 1
            return a.add(b, fill_value=0)
        DTM = self.DTM_fun()
        DBM = self.DBM_fun()

        STM = bt.AZ_Rolling(DTM, n).sum()
        SBM = bt.AZ_Rolling(DBM, n).sum()

        ADTM = STM.combine(SBM, get_result)
        return ADTM

    def ADTMMA_fun(self, n=23, m=8):
        ADTM = self.ADTM_fun(n)
        ADTMMA = bt.AZ_Rolling(ADTM, m).mean()
        return ADTM - ADTMMA


if __name__ == '__main__':
    open_ = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_OPEN.csv')
    close = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv')
    high = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv')
    low = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv')
    volume = 0
    turnover = 0
    tech_factor = TechFactor(open_, close, high, low, volume, turnover)
    # ADTM = tech_factor.ADTM_fun()

    DTM = tech_factor.DTM_fun()
    DBM = tech_factor.DBM_fun()
    STM = bt.AZ_Rolling(DTM, 23).sum()
    SBM = bt.AZ_Rolling(DBM, 23).sum()
    ADTM = tech_factor.ADTM_fun(n=23)
