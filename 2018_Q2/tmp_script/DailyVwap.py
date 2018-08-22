import pandas as pd
import sys
sys.path.append("/mnt/mfs/LIB_ROOT")
import open_lib.shared_paths.path as pt


def DailyVwap(mode):
    root_path = pt._BinFiles(mode)

    tafactor_path = root_path.EM_Funda.TRAD_SK_FACTOR1 / 'TAFACTOR.csv'
    TVOL_path = root_path.EM_Funda.TRAD_SK_DAILY_JC / 'TVOL.csv'
    TVALCNY_path = root_path.EM_Funda.TRAD_SK_DAILY_JC / 'TVALCNY.csv'

    TVOL_df = pd.read_table(TVOL_path, sep='|', index_col=0, parse_dates=True)
    TVALCNY_df = pd.read_table(TVALCNY_path, sep='|', index_col=0, parse_dates=True)
    tafactor_df = pd.read_table(tafactor_path, sep='|', index_col=0, parse_dates=True) \
        .reindex(columns=TVALCNY_df.columns, index=TVALCNY_df.index)

    daily_vwap = (TVALCNY_df / TVOL_df)
    daily_vwap_adj_price = daily_vwap/tafactor_df
    daily_vwap_adj_return = daily_vwap_adj_price.pct_change().round(4)
    return daily_vwap_adj_price, daily_vwap_adj_return


def indexDailyVwap(mode):
    root_path = pt._BinFiles(mode)
    index_TVAL_path = root_path.EM_Funda.INDEX_TD_DAILY / 'TVAL.csv'
    index_TVOL_path = root_path.EM_Funda.INDEX_TD_DAILY / 'TVOL.csv'

    index_TVOL_df = pd.read_table(index_TVOL_path, sep='|', index_col=0, parse_dates=True)
    index_TVAL_df = pd.read_table(index_TVAL_path, sep='|', index_col=0, parse_dates=True)

    index_vwap = (index_TVAL_df / index_TVOL_df)
    index_vwap_return = index_vwap.pct_change().round(4)
    return index_vwap, index_vwap_return


if __name__ == '__main__':
    pass
    # a = DailyVwap('bkt')
    # aadj_p_path = '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv'
    # aadj_p = pd.read_table(aadj_p_path, sep='|', index_col=0, parse_dates=True)
    #
    # aadj_r_path = '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv'
    # aadj_r = pd.read_table(aadj_r_path, sep='|', index_col=0, parse_dates=True)
    #
    # aadj_vwap_path = '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_vwap.csv'
    # aadj_vwap = pd.read_table(aadj_r_path, sep='|', index_col=0, parse_dates=True)
    #
    # aadj_r_vwap_path = '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r_vwap.csv'
    # aadj_r_vwap = pd.read_table(aadj_r_path, sep='|', index_col=0, parse_dates=True)
    #
    # r = aadj_p/aadj_p.shift(1) - 1
