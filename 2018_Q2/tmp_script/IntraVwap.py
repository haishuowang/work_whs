import pandas as pd


def AZ_Stock_change(stock_list):
    target_list = [x[2:] + '.' + x[:2] for x in stock_list]
    return target_list


aadj_r = pd.read_csv('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv', sep='|', index_col=0, parse_dates=True)

aadj_r = aadj_r[(aadj_r.index >= pd.to_datetime('20050101')) & (aadj_r.index < pd.to_datetime('20180401'))]
aadj_r_prod = (aadj_r + 1).cumprod()
xnms = aadj_r.columns
xinx = aadj_r.index

for i in range(1):
    print('_____________________________________________')
    tafactor = pd.read_csv('/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_FACTOR1/TAFACTOR.csv', sep='|', index_col=0,
                           parse_dates=True).reindex(index=xinx, columns=xnms)

    data_raw = pd.read_pickle(f'/mnt/mfs/dat_whs/data/base_data/intra_vwap_60_tab_{i+1}_2005_2006.pkl')
    data_raw.columns = AZ_Stock_change(data_raw.columns)
    data = data_raw.reindex(index=xinx[xinx >= pd.to_datetime('20100101')], columns=xnms)

    # data_2_raw = pd.read_pickle(f'/mnt/mfs/dat_whs/data/base_data/intra_vwap_60_tab_{i+1}_2005_part.pkl')
    # data_2_raw.columns = AZ_Stock_change(data_2_raw.columns)
    # data_2 = data_2_raw.reindex(index=xinx[xinx < pd.to_datetime('20100101')], columns=xnms)

    # data = pd.concat([data_2, data_1], axis=0)

    aadj_p_intra_vwap = (data / tafactor)
    # aadj_p_intra_vwap
    aadj_r_intra_vwap = aadj_p_intra_vwap / aadj_p_intra_vwap.shift(1) - 1  # aadj_p_intra_vwap.pct_change()
    #
    aadj_r_intra_vwap_prod = (aadj_r_intra_vwap + 1).cumprod()

    a = aadj_r_prod['603999.SH']
    b = aadj_r_intra_vwap_prod['603999.SH']
    c = aadj_r_intra_vwap.abs().max(axis=1)
    print(c[c > 0.3])

    # print(a.iloc[-1], b.iloc[-1])
    # aadj_p_intra_vwap.to_csv(f'/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_intra_{i+1}_vwap.csv', sep='|')
    # aadj_r_intra_vwap.to_csv(f'/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r_intra_{i+1}_vwap.csv', sep='|')

# target_time = '20150709'
# stock_code = 'SZ000001'
# tafactor_i = tafactor.loc[pd.to_datetime(target_time), stock_code[2:] + '.'+stock_code[:2]]
# close = pd.read_csv(f'/mnt/mfs/DAT_PUBLIC/intraday/eqt_1mbar/'
#                     f'{target_time[:4]}/{target_time[:6]}/{target_time}/Close.csv'
#                     , index_col=0)[stock_code][:60]
# volume = pd.read_csv(f'/mnt/mfs/DAT_PUBLIC/intraday/eqt_1mbar/'
#                      f'{target_time[:4]}/{target_time[:6]}/{target_time}/Volume.csv'
#                      , index_col=0)[stock_code][:60]
# turnover = pd.read_csv(f'/mnt/mfs/DAT_PUBLIC/intraday/eqt_1mbar/'
#                        f'{target_time[:4]}/{target_time[:6]}/{target_time}/Turnover.csv'
#                        , index_col=0)[stock_code][:60]
date = '20150422'
data = pd.read_csv(f'/mnt/mfs/DAT_PUBLIC/intraday/eqt_1mbar/{date[:4]}/{date[:6]}/{date}/Close.csv')

x = ['000661.SZ',
     '002153.SZ',
     '002568.SZ',
     '002739.SZ',
     '300033.SZ',
     '300226.SZ',
     '300244.SZ',
     '300295.SZ',
     '300359.SZ',
     '300369.SZ',
     '300377.SZ',
     '300380.SZ',
     '300382.SZ',
     '300383.SZ',
     '300384.SZ',
     '300386.SZ',
     '300389.SZ',
     '300392.SZ',
     '300399.SZ',
     '300418.SZ',
     '300419.SZ',
     '300433.SZ',
     '600436.SH',
     '600446.SH',
     '600519.SH',
     '600570.SH',
     '601021.SH',
     '603168.SH']
