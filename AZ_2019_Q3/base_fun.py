import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as lr

from AZ_send_mail import SendFigures
from matplotlib import pyplot as plt


class BackTest:
    @staticmethod
    def AZ_Load_csv(target_path, parse_dates=True, index_col=0, sep='|', **kwargs):
        target_df = pd.read_table(target_path, sep=sep, index_col=index_col, low_memory=False,
                                  parse_dates=parse_dates, **kwargs)
        return target_df

    @staticmethod
    def AZ_Sharpe_y(pnl_df):
        """
        :param pnl_df:收益
        :return: sharpe
        """
        return round((np.sqrt(250) * pnl_df.mean()) / pnl_df.std(), 4)

    @staticmethod
    def AZ_Normal_IC(signal, pct_n, min_valids=None, lag=0):
        """

        :param signal:因子
        :param pct_n: 未来一段时间的收益率
        :param min_valids:
        :param lag:
        :return:
        """
        signal = signal.shift(lag)
        signal = signal.replace(0, np.nan)
        corr_df = signal.corrwith(pct_n, axis=1).dropna()

        if min_valids is not None:
            signal_valid = signal.count(axis=1)
            signal_valid[signal_valid < min_valids] = np.nan
            signal_valid[signal_valid >= min_valids] = 1
            corr_signal = corr_df * signal_valid
        else:
            corr_signal = corr_df
        return round(corr_signal, 6)

    def AZ_Normal_IR(self, signal, pct_n, min_valids=None, lag=0):
        """

        :param self:
        :param signal:
        :param pct_n:
        :param min_valids:
        :param lag:
        :return:
        """
        corr_signal = self.AZ_Normal_IC(signal, pct_n, min_valids, lag)
        ic_mean = corr_signal.mean()
        ic_std = corr_signal.std()
        ir = ic_mean / ic_std
        return ir, corr_signal


bt = BackTest()


def get_st_new_stock_info(root_path):
    """

    :param root_path:
    :return: 0,1的dataframe
             st_info:剔除st股票, new_info：剔除上市不足250天新股
    """
    return_df = bt.AZ_Load_csv(f'{root_path}/EM_Funda/DERIVED_14/aadj_r.csv').astype(float)
    trade_index = return_df.index

    LISTSTATE_df = bt.AZ_Load_csv(f'{root_path}/EM_Tab01/CDSY_SECUCODE/LISTSTATE.csv')
    new_info = LISTSTATE_df.reindex(index=trade_index).fillna(method='ffill').shift(250).notnull().astype(int)

    st_info = bt.AZ_Load_csv(f'{root_path}/EM_Funda/DERIVED_01/StAndPtStock.csv')
    st_info.fillna(method='ffill', inplace=True)
    return st_info, new_info


# pandas说明文档
# pandas http://pandas.pydata.org/pandas-docs/stable/reference/frame.html
# st_info * new_info * index_500
# 数据路径：
# 1/PE /mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_REVALUATION/PE_TTM.csv
# 1/PS /mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_REVALUATION/PS_TTM.csv
# 1/PB /mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_REVALUATION/PBLatestQuater.csv
# ROE /mnt/mfs/DAT_EQT/EM_Funda/daily/R_ROE_TTM_First.csv
# ROA /mnt/mfs/DAT_EQT/EM_Funda/daily/R_ROA_TTM_First.csv
# market /mnt/mfs/DAT_EQT/EM_Funda/LICO_YS_STOCKVALUE/AmarketCapExStri.csv
# industry /mnt/mfs/DAT_EQT/EM_Funda/LICO_IM_INCHG/ZhongZheng_Level1_00.csv
#                          ..................
#          /mnt/mfs/DAT_EQT/EM_Funda/LICO_IM_INCHG/ZhongZheng_Level1_09.csv
#
# def tmp_fun(x):
#     dn = x.quantile(0.02)
#     up = x.quantile(0.98)
#     x[x<dn] = dn
#     x[x>up] = up
#     return x
# b = a.apply(tmp_fun, axis=1)

# industry

level_1 = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/LICO_IM_INCHG/ZhongZheng_Level1_00.csv')
level_2 = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/LICO_IM_INCHG/ZhongZheng_Level1_01.csv')
level_3 = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/LICO_IM_INCHG/ZhongZheng_Level1_02.csv')
level_4 = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/LICO_IM_INCHG/ZhongZheng_Level1_03.csv')
level_5 = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/LICO_IM_INCHG/ZhongZheng_Level1_04.csv')
level_6 = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/LICO_IM_INCHG/ZhongZheng_Level1_05.csv')
level_7 = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/LICO_IM_INCHG/ZhongZheng_Level1_06.csv')
level_8 = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/LICO_IM_INCHG/ZhongZheng_Level1_07.csv')
level_9 = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/LICO_IM_INCHG/ZhongZheng_Level1_08.csv')
level_10 = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/LICO_IM_INCHG/ZhongZheng_Level1_09.csv')

ori_stk_rtn = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv')
stk_rtn = ori_stk_rtn.resample('M').sum()
colnms = stk_rtn.columns.tolist()
date_list = ori_stk_rtn.index.tolist()
date_list_month = stk_rtn.index.tolist()

# 1 load value factor
ep = 1 / bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_REVALUATION/PE_TTM.csv')
sp = 1 / bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_REVALUATION/PS_TTM.csv')
bp = 1 / bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_REVALUATION/PBLatestQuater.csv')

ep = ep.reindex(index=ori_stk_rtn.index)
sp = sp.reindex(index=ori_stk_rtn.index)
bp = bp.reindex(index=ori_stk_rtn.index)

# 2 load growth factor

sale_g = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/daily/R_Revenue_YOY_First.csv')
pro_g = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/daily/R_NetMargin_s_YOY_First.csv')

# 3 load profit factor

roe = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/daily/R_ROE_TTM_First.csv')
roa = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/daily/R_ROA_TTM_First.csv')
grossPmargin = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/daily/R_SalesGrossMGN_TTM_First.csv')
pmargin = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/daily/R_SalesNetMGN_TTM_First.csv')

# 4 load leverage factor

currentratio = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/daily/R_CurrentRatio_First.csv')
cashratio = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/daily/R_ConsQuickRatio_First.csv')

# 5 load Market facotor

mkt = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/LICO_YS_STOCKVALUE/AmarketCapExStri.csv')
mkt = mkt.replace(0, np.nan)
ln_mkt = np.log(mkt)
ln_mkt = ln_mkt.reindex(index=ori_stk_rtn.index.tolist())
ori_ln_mkt = ln_mkt.copy()
# 6 reverse factor

data = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv').resample('M').sum()
index_500_r = data['000001'] * 0.01


def tmp_fun(stk_rtn, index_500_r):
    res_alpha = stk_rtn * np.nan
    beta = stk_rtn * np.nan

    index_500_r = np.array(index_500_r).reshape(-1, 1)

    for i in range(stk_rtn.shape[0] - 12):
        x = index_500_r[i:i + 12]
        for j in range(stk_rtn.shape[1]):
            model = lr()
            y = stk_rtn.iloc[i:i + 12, j]
            model.fit(x, y)

            a, b = model.coef_, model.intercept_

            beta.iloc[i + 12, j] = a

            y_pre = a * index_500_r[i + 12] + b

            res_alpha.iloc[i + 12, j] = stk_rtn.iloc[i + 12, j] - y_pre

    res_alpha = res_alpha.reindex(index=ori_stk_rtn.index.tolist())
    res_alpha = res_alpha.fillna(method='ffill')

    beta = beta.reindex(index=ori_stk_rtn.index.tolist())
    beta = beta.fillna(method='ffill')
    return res_alpha, beta


res_alpha, beta = tmp_fun(stk_rtn, index_500_r)

# price change ratio in last three months


price = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv')

return_nm = price / price.shift(75) - 1
return_nm = return_nm.reindex(index=ori_stk_rtn.index.tolist())

# 7

std_nm = ori_stk_rtn.rolling(window=150).std()

# index_500 & not_index_500 universe
index_500 = bt.AZ_Load_csv(f'/mnt/mfs/DAT_EQT/EM_Funda/IDEX_YS_WEIGHT_A/SECURITYNAME_000905.csv')
index_500 = index_500.where(index_500 != index_500, other=1)

index_500 = index_500.reindex(index=date_list)
index_500 = index_500.reindex(columns=colnms)

st_info, new_info = get_st_new_stock_info('/mnt/mfs/DAT_EQT')

index_500 = index_500 * st_info * new_info.resample('M').last()
# not_index_500 = (index_500 != index_500).astype(int)
not_index_500 = index_500.replace(1, 0)
not_index_500 = not_index_500.fillna(1)
not_index_500 = not_index_500.replace(0, np.nan).resample('M').last()


# extreme processing


def mad_truncate(factor):
    down = factor.quantile(0.02, axis=1)
    up = factor.quantile(0.98, axis=1)

    ncol = factor.columns.size
    nrow = factor.index.size
    # 3sigma
    # down = factor.mean(axis = 1)-3*factor.std(axis = 1)
    # up = factor.mean(axis = 1)+3*factor.std(axis = 1)

    down = np.array(down.values.tolist() * ncol).reshape(ncol, nrow).T
    down = pd.DataFrame(down, index=factor.index.tolist(), columns=factor.columns.tolist())

    up = np.array(up.values.tolist() * ncol).reshape(ncol, nrow).T
    up = pd.DataFrame(up, index=factor.index.tolist(), columns=factor.columns.tolist())

    factor[factor < down] = down
    factor[factor > up] = up

    return factor


factors = [ep, sp, bp, sale_g, pro_g, roe, roa, grossPmargin, pmargin, currentratio, cashratio, ln_mkt, res_alpha,
           return_nm, std_nm]

for i in range(len(factors)):
    factors[i] = mad_truncate(factors[i])

[ep, sp, bp, sale_g, pro_g, roe, roa, grossPmargin, pmargin, currentratio, cashratio, ln_mkt, res_alpha, return_nm,
 std_nm] = factors

ep = ep[colnms]
sp = sp[colnms]
bp = bp[colnms]

sale_g = sale_g[colnms]
pro_g = pro_g[colnms]

roe = roe[colnms]
roa = roa[colnms]
grossPmargin = grossPmargin[colnms]
pmargin = pmargin[colnms]

currentratio = currentratio[colnms]
cashratio = cashratio[colnms]

ln_mkt = ln_mkt[colnms]

res_alpha = res_alpha[colnms]
return_nm = return_nm[colnms]

std_nm = std_nm[colnms]


# linear regression to neutralize factor


def neutralize(factor):
    global ln_mkt
    global date_list

    ind_1 = level_1.reindex(columns=colnms).iloc[-1, :].fillna(0)
    ind_1.name = 'ind_1'
    ind_2 = level_2.reindex(columns=colnms).iloc[-1, :].fillna(0)
    ind_2.name = 'ind_2'
    ind_3 = level_3.reindex(columns=colnms).iloc[-1, :].fillna(0)
    ind_3.name = 'ind_3'
    ind_4 = level_4.reindex(columns=colnms).iloc[-1, :].fillna(0)
    ind_4.name = 'ind_4'
    ind_5 = level_5.reindex(columns=colnms).iloc[-1, :].fillna(0)
    ind_5.name = 'ind_5'
    ind_6 = level_6.reindex(columns=colnms).iloc[-1, :].fillna(0)
    ind_6.name = 'ind_6'
    ind_7 = level_7.reindex(columns=colnms).iloc[-1, :].fillna(0)
    ind_7.name = 'ind_7'
    ind_8 = level_8.reindex(columns=colnms).iloc[-1, :].fillna(0)
    ind_8.name = 'ind_8'
    ind_9 = level_9.reindex(columns=colnms).iloc[-1, :].fillna(0)
    ind_9.name = 'ind_9'
    ind_10 = level_10.reindex(columns=colnms).iloc[-1, :].fillna(0)
    ind_10.name = 'ind_10'

    # ind_1 = level_1.reindex(columns=colnms).loc[target_date].fillna(0)
    # ind_1.name = 'ind_1'
    # ind_2 = level_2.reindex(columns=colnms).loc[target_date].fillna(0)
    # ind_2.name = 'ind_2'
    # ind_3 = level_3.reindex(columns=colnms).loc[target_date].fillna(0)
    # ind_3.name = 'ind_3'
    # ind_4 = level_4.reindex(columns=colnms).loc[target_date].fillna(0)
    # ind_4.name = 'ind_4'
    # ind_5 = level_5.reindex(columns=colnms).loc[target_date].fillna(0)
    # ind_5.name = 'ind_5'
    # ind_6 = level_6.reindex(columns=colnms).loc[target_date].fillna(0)
    # ind_6.name = 'ind_6'
    # ind_7 = level_7.reindex(columns=colnms).loc[target_date].fillna(0)
    # ind_7.name = 'ind_7'
    # ind_8 = level_8.reindex(columns=colnms).loc[target_date].fillna(0)
    # ind_8.name = 'ind_8'
    # ind_9 = level_9.reindex(columns=colnms).loc[target_date].fillna(0)
    # ind_9.name = 'ind_9'
    # ind_10 = level_10.reindex(columns=colnms).loc[target_date].fillna(0)
    # ind_10.name = 'ind_10'

    # xnms = list(set(factor.columns) & set(ln_mkt.columns))

    for target_date in date_list:
        # roe.iloc[-1]
        # target_date = pd.to_datetime('20190531')
        # factor = factor[xnms]

        y = factor.loc[target_date].fillna(0)

        # y = y[xnms]
        # ln_mkt = ln_mkt[xnms]
        mt_log_sr = ln_mkt.loc[target_date].fillna(0)

        X = pd.concat([ind_1, ind_2, ind_3, ind_4, ind_5, ind_6, ind_7, ind_8, ind_9, ind_10, mt_log_sr], axis=1)

        model = lr()
        model.fit(X, y)
        a, b = model.coef_, model.intercept_
        y_predict = (a * X).sum(1) + b
        es = y - y_predict
        factor.loc[target_date] = es

    return factor


factors = [ep, sp, bp, sale_g, pro_g, roe, roa, grossPmargin, pmargin, currentratio, cashratio, ln_mkt, res_alpha,
           return_nm, std_nm]

for i in range(len(factors)):
    factors[i] = neutralize(factors[i])

[ep, sp, bp, sale_g, pro_g, roe, roa, grossPmargin, pmargin, currentratio, cashratio, ln_mkt, res_alpha, return_nm,
 std_nm] = factors

# end of basic processing of sub factors


# resample factors for 1st FoF
factors = [ep, sp, bp, sale_g, pro_g, roe, roa, grossPmargin, pmargin, currentratio, cashratio, ln_mkt, res_alpha,
           return_nm, std_nm]

for i in range(len(factors)):
    factors[i] = factors[i].resample('M').last()
[ep1, sp1, bp1, sale_g1, pro_g1, roe1, roa1, grossPmargin1, \
 pmargin1, currentratio1, cashratio1, ln_mkt1, res_alpha1, return_nm1, \
 std_nm1] = factors


def standardize(factor):
    mean = factor.mean(axis=1)
    std = factor.std(axis=1)

    factor = factor.sub(mean, axis=0).div(std, axis=0)
    return factor


factors1 = [ep1, sp1, bp1, sale_g1, pro_g1, roe1, roa1, grossPmargin1, pmargin1, currentratio1, cashratio1, ln_mkt1,
            res_alpha1, return_nm1, std_nm1]
for i in range(len(factors1)):
    factors1[i] = standardize(factors1[i])
[ep1, sp1, bp1, sale_g1, pro_g1, roe1, roa1, grossPmargin1, pmargin1, currentratio1, cashratio1, ln_mkt1, res_alpha1,
 return_nm1, std_nm1] = factors1

# synthetic major factor

val1 = (ep1 + bp1 + sp1) / 3
grow1 = (sale_g1 + pro_g1) / 2
profit1 = (roe1 + roa1 + grossPmargin1 + pmargin1) / 4
lev1 = (currentratio1 + cashratio1) / 2
mkt1 = -ln_mkt1
reverse1 = -(res_alpha1 + return_nm1) / 2
fluc1 = -std_nm1

# equal weighted

ew_factor = val1 + grow1 + profit1 + lev1 + mkt1 + reverse1 + fluc1

ew_factor_500_r = (ew_factor * index_500).rank(axis=1, ascending=False)

ew_factor_not_500_r = (ew_factor * not_index_500).rank(axis=1, ascending=False)

ew_factor_pos = stk_rtn * 0

ew_factor_pos[ew_factor_500_r.fillna(51) <= 50] = 1
ew_factor_pos[ew_factor_not_500_r.fillna(51) <= 50] = 1

ew_pnl = (ew_factor_pos.shift(1) * stk_rtn).sum(axis=1).cumsum()

plt.plot(ew_pnl)
plt.savefig("/mnt/mfs/work_mia/eq.png")
plt.close()

# previous N periods

N = 6

# mean of ic in previous N periods

val_ic = bt.AZ_Normal_IC(val1, stk_rtn, lag=1).rolling(window=N).mean()
grow_ic = bt.AZ_Normal_IC(grow1, stk_rtn, lag=1).rolling(window=N).mean()
profit_ic = bt.AZ_Normal_IC(profit1, stk_rtn, lag=1).rolling(window=N).mean()
lev_ic = bt.AZ_Normal_IC(lev1, stk_rtn, lag=1).rolling(window=N).mean()
mkt_ic = bt.AZ_Normal_IC(mkt1, stk_rtn, lag=1).rolling(window=N).mean()
reverse_ic = bt.AZ_Normal_IC(reverse1, stk_rtn, lag=1).rolling(window=N).mean()
fluc_ic = bt.AZ_Normal_IC(fluc1, stk_rtn, lag=1).rolling(window=N).mean()

# time series factor momentum

val_ic[val_ic < 0] = 0
grow_ic[grow_ic < 0] = 0
profit_ic[profit_ic < 0] = 0
lev_ic[lev_ic < 0] = 0
mkt_ic[mkt_ic < 0] = 0
reverse_ic[reverse_ic < 0] = 0
fluc_ic[fluc_ic < 0] = 0

# ic weight

ic_sum = val_ic + grow_ic + profit_ic + lev_ic + mkt_ic + reverse_ic + fluc_ic

ic_factor = val1.mul(val_ic / ic_sum, axis=0) + grow1.mul(grow_ic / ic_sum, axis=0) \
            + profit1.mul(profit_ic / ic_sum, axis=0) + lev1.mul(lev_ic / ic_sum, axis=0) \
            + mkt1.mul(mkt_ic / ic_sum, axis=0) + reverse1.mul(reverse_ic / ic_sum, axis=0) \
            + fluc1.mul(fluc_ic / ic_sum, axis=0)

ic_factor_500_r = (ic_factor * index_500).rank(axis=1, ascending=False)

ic_factor_not_500_r = (ic_factor * not_index_500).rank(axis=1, ascending=False)

ic_factor_pos = stk_rtn * 0

ic_factor_pos[ic_factor_500_r.fillna(51) <= 50] = 1
ic_factor_pos[ic_factor_not_500_r.fillna(51) <= 50] = 1

ic_pnl = (ic_factor_pos.shift(1) * stk_rtn).sum(axis=1).cumsum()

plt.plot(ic_pnl)
plt.savefig("/mnt/mfs/work_mia/ic.png")
plt.close()

# std of ic in previous N periods


val_ic_s = bt.AZ_Normal_IC(val1, stk_rtn, lag=1).rolling(window=N).std()
grow_ic_s = bt.AZ_Normal_IC(grow1, stk_rtn, lag=1).rolling(window=N).std()
profit_ic_s = bt.AZ_Normal_IC(profit1, stk_rtn, lag=1).rolling(window=N).std()
lev_ic_s = bt.AZ_Normal_IC(lev1, stk_rtn, lag=1).rolling(window=N).std()
mkt_ic_s = bt.AZ_Normal_IC(mkt1, stk_rtn, lag=1).rolling(window=N).std()
reverse_ic_s = bt.AZ_Normal_IC(reverse1, stk_rtn, lag=1).rolling(window=N).std()
fluc_ic_s = bt.AZ_Normal_IC(fluc1, stk_rtn, lag=1).rolling(window=N).std()

# icir weight

val_icir = val_ic / val_ic_s
grow_icir = grow_ic / grow_ic_s
profit_icir = profit_ic / profit_ic_s
lev_icir = lev_ic / lev_ic_s
mkt_icir = mkt_ic / mkt_ic_s
reverse_icir = reverse_ic / reverse_ic_s
fluc_icir = fluc_ic / fluc_ic_s

# time series factor momentum

val_icir[val_icir < 0] = 0
grow_icir[grow_icir < 0] = 0
profit_icir[profit_icir < 0] = 0
lev_icir[lev_icir < 0] = 0
mkt_icir[mkt_icir < 0] = 0
reverse_icir[reverse_icir < 0] = 0
fluc_icir[fluc_icir < 0] = 0

icir_sum = val_icir + grow_icir + profit_icir + lev_icir + mkt_icir + reverse_icir + fluc_icir

icir_factor = val1.mul(val_icir / icir_sum, axis=0) + grow1.mul(grow_icir / icir_sum, axis=0) \
              + profit1.mul(profit_icir / icir_sum, axis=0) + lev1.mul(lev_icir / icir_sum, axis=0) \
              + mkt1.mul(mkt_icir / icir_sum, axis=0) + reverse1.mul(reverse_icir / icir_sum, axis=0) \
              + fluc1.mul(fluc_icir / icir_sum, axis=0)

icir_factor_500_r = (icir_factor * index_500).rank(axis=1, ascending=False)

icir_factor_not_500_r = (icir_factor * not_index_500).rank(axis=1, ascending=False)

icir_factor_pos = stk_rtn * 0

icir_factor_pos[icir_factor_500_r.fillna(51) <= 50] = 1
icir_factor_pos[icir_factor_not_500_r.fillna(51) <= 50] = 1

icir_pnl = (icir_factor_pos.shift(1) * stk_rtn).sum(axis=1).cumsum()

plt.plot(icir_pnl)
plt.savefig("/mnt/mfs/work_mia/icir.png")
plt.close()

# icwr weight

val_icwr = bt.AZ_Normal_IC(val1, stk_rtn, lag=1)
grow_icwr = bt.AZ_Normal_IC(grow1, stk_rtn, lag=1)
profit_icwr = bt.AZ_Normal_IC(profit1, stk_rtn, lag=1)
lev_icwr = bt.AZ_Normal_IC(lev1, stk_rtn, lag=1)
mkt_icwr = bt.AZ_Normal_IC(mkt1, stk_rtn, lag=1)
reverse_icwr = bt.AZ_Normal_IC(reverse1, stk_rtn, lag=1)
fluc_icwr = bt.AZ_Normal_IC(fluc1, stk_rtn, lag=1)

val_icwr[val_icwr <= 0] = 0
val_icwr[val_icwr > 0] = 1

grow_icwr[grow_icwr <= 0] = 0
grow_icwr[grow_icwr > 0] = 1

profit_icwr[profit_icwr <= 0] = 0
profit_icwr[profit_icwr > 0] = 1

lev_icwr[lev_icwr <= 0] = 0
lev_icwr[lev_icwr > 0] = 1

mkt_icwr[mkt_icwr <= 0] = 0
mkt_icwr[mkt_icwr > 0] = 1

reverse_icwr[reverse_icwr <= 0] = 0
reverse_icwr[reverse_icwr > 0] = 1

fluc_icwr[fluc_icwr <= 0] = 0
fluc_icwr[fluc_icwr > 0] = 1

val_icwr = val_icwr.rolling(window=N).sum() / N
grow_icwr = grow_icwr.rolling(window=N).sum() / N
profit_icwr = profit_icwr.rolling(window=N).sum() / N
lev_icwr = lev_icwr.rolling(window=N).sum() / N
mkt_icwr = mkt_icwr.rolling(window=N).sum() / N
reverse_icwr = reverse_icwr.rolling(window=N).sum() / N
fluc_icwr = fluc_icwr.rolling(window=N).sum() / N

# time series factor momentum

val_icwr[val_icwr < 0.5] = 0
grow_icwr[grow_icwr < 0.5] = 0
profit_icwr[profit_icwr < 0.5] = 0
lev_icwr[lev_icwr < 0.5] = 0
mkt_icwr[mkt_icwr < 0.5] = 0
reverse_icwr[reverse_icwr < 0.5] = 0
fluc_icwr[fluc_icwr < 0.5] = 0

icwr_sum = val_icwr + grow_icwr + profit_icwr + lev_icwr + mkt_icwr + reverse_icwr + fluc_icwr

icwr_factor = val1.mul(val_icwr / icwr_sum, axis=0) + grow1.mul(grow_icwr / icwr_sum, axis=0) \
              + profit1.mul(profit_icwr / icwr_sum, axis=0) + lev1.mul(lev_icwr / icwr_sum, axis=0) \
              + mkt1.mul(mkt_icwr / icwr_sum, axis=0) + reverse1.mul(reverse_icwr / icwr_sum, axis=0) \
              + fluc1.mul(fluc_icwr / icwr_sum, axis=0)

icwr_factor_500_r = (icwr_factor * index_500).rank(axis=1, ascending=False)

icwr_factor_not_500_r = (icwr_factor * not_index_500).rank(axis=1, ascending=False)

icwr_factor_pos = stk_rtn * 0

icwr_factor_pos[icwr_factor_500_r.fillna(51) <= 50] = 1
icwr_factor_pos[icwr_factor_not_500_r.fillna(51) <= 50] = 1

icwr_pnl = (icwr_factor_pos.shift(1) * stk_rtn).sum(axis=1).cumsum()

plt.plot(icwr_pnl)
plt.savefig("/mnt/mfs/work_mia/icwr.png")
plt.close()

date_list_month = ew_pnl.index.tolist()
plt.plot(date_list_month, ew_pnl, date_list_month, ic_pnl, date_list_month, icir_pnl, date_list_month, icwr_pnl)
plt.legend(['ew', 'ic', 'icir', 'icwr'])
plt.savefig("/mnt/mfs/work_mia/N=12.png")
plt.close()

plt.plot(date_list_month, ic_pnl - ew_pnl, date_list_month, icir_pnl - ew_pnl, date_list_month, icwr_pnl - ew_pnl)
plt.legend(['ic', 'icir', 'icwr'])
plt.savefig("/mnt/mfs/work_mia/N=12_ewbase.png")
plt.close()


# discrete factor


def numofcol(ind):
    allcol = ind.shape[1]
    not_na = allcol - ind.isna().sum(axis=1)

    num = pd.DataFrame(np.array(not_na.tolist() * allcol).reshape(allcol, len(date_list)).T, index=ind.index.tolist(),
                       columns=ind.columns.tolist())

    return num


def industry(ind):
    ind_r = ind.rank(axis=1)
    ncol = numofcol(ind)
    ind_top = ind[ind_r >= (ncol * 4 / 5)].median(axis=1)
    ind_bot = ind[ind_r <= (ncol / 5)].median(axis=1)
    ind_disper = ind_top - ind_bot
    return ind_disper


def discrete(factor):
    ind1 = factor[level_1.columns.tolist()]
    ind2 = factor[level_2.columns.tolist()]
    ind3 = factor[level_3.columns.tolist()]
    ind4 = factor[level_4.columns.tolist()]
    ind5 = factor[level_5.columns.tolist()]
    ind6 = factor[level_6.columns.tolist()]
    ind7 = factor[level_7.columns.tolist()]
    ind8 = factor[level_8.columns.tolist()]
    ind9 = factor[level_9.columns.tolist()]
    ind10 = factor[level_10.columns.tolist()]
    ind = [ind1, ind2, ind3, ind4, ind5, ind6, ind7, ind8, ind9, ind10]
    s = 0
    for inds in ind:
        t = industry(inds)
        s = s + t

    return s / 10


def standardize1(factor):
    mean = factor.rolling(window=150).mean()
    std = factor.rolling(window=150).std()

    factor = (factor - mean) / std
    return factor


factors = [ep, sp, bp, sale_g, pro_g, roe, roa, grossPmargin, pmargin, currentratio, cashratio, ori_ln_mkt, res_alpha,
           return_nm, std_nm]
for i in range(len(factors)):
    factors[i] = standardize(factors[i])
[ep_dis, sp_dis, bp_dis, sale_g_dis, pro_g_dis, roe_dis, roa_dis, grossPmargin_dis, pmargin_dis, currentratio_dis,
 cashratio_dis, ln_mkt_dis, res_alpha_dis, return_nm_dis, std_nm_dis] = factors

factors = [ep, sp, bp, sale_g, pro_g, roe, roa, grossPmargin, pmargin, currentratio, cashratio, ori_ln_mkt, res_alpha,
           return_nm, std_nm]
for i in range(len(factors)):
    factors[i] = discrete(factors[i])
[ep2, sp2, bp2, sale_g2, pro_g2, roe2, roa2, grossPmargin2, pmargin2, currentratio2, cashratio2, ln_mkt2, res_alpha2,
 return_nm2, std_nm2] = factors

factors = [ep2, sp2, bp2, sale_g2, pro_g2, roe2, roa2, grossPmargin2, pmargin2, currentratio2, cashratio2, ln_mkt2,
           res_alpha2, return_nm2, std_nm2]
for i in range(len(factors)):
    factors[i] = standardize1(factors[i])
[ep2, sp2, bp2, sale_g2, pro_g2, roe2, roa2, grossPmargin2, pmargin2, currentratio2, cashratio2, ln_mkt2, res_alpha2,
 return_nm2, std_nm2] = factors

# synthetic major factor

val2 = (ep2 + bp2 + sp2) / 3
grow2 = (sale_g2 + pro_g2) / 2
profit2 = (roe2 + roa2 + grossPmargin2 + pmargin2) / 4
lev2 = (currentratio2 + cashratio2) / 2
mkt2 = -ln_mkt2
reverse2 = -(res_alpha2 + return_nm2) / 2
fluc2 = -std_nm2

val2 = val2.resample('M').last()
grow2 = grow2.resample('M').last()
profit2 = profit2.resample('M').last()
lev2 = lev2.resample('M').last()
mkt2 = mkt2.resample('M').last()
reverse2 = reverse2.resample('M').last()
fluc2 = fluc2.resample('M').last()

val_dis = (ep_dis + bp_dis + sp_dis) / 3
grow_dis = (sale_g_dis + pro_g_dis) / 2
profit_dis = (roe_dis + roa_dis + grossPmargin_dis + pmargin_dis) / 4
lev_dis = (currentratio_dis + cashratio_dis) / 2
mkt_dis = -ln_mkt_dis
reverse_dis = -(res_alpha_dis + return_nm_dis) / 2
fluc_dis = -std_nm_dis

val_dis = val_dis.resample('M').last().fillna(0)
grow_dis = grow_dis.resample('M').last().fillna(0)
profit_dis = profit_dis.resample('M').last().fillna(0)
lev_dis = lev_dis.resample('M').last().fillna(0)
mkt_dis = mkt_dis.resample('M').last().fillna(0)
reverse_dis = reverse_dis.resample('M').last().fillna(0)
fluc_dis = fluc_dis.resample('M').last().fillna(0)

dis_factor_pos = pd.concat([val2, grow2, profit2, lev2, mkt2, reverse2, fluc2], axis=1)
dis_factor_pos_r = dis_factor_pos.rank(axis=1, ascending=False)
dis_factor_pos = dis_factor_pos * 0

dis_factor_pos[dis_factor_pos_r <= 4] = 1 / 4

dis_factor = stk_rtn * 0

date = dis_factor.index.tolist()
for d in date:
    slice = dis_factor_pos.loc[d].fillna(0)
    dis_factor.loc[d] = (
                val_dis * slice[0] + grow_dis * slice[1] + profit_dis * slice[2] + lev_dis * slice[3] + mkt_dis * slice[
            4] + reverse_dis * slice[5] + fluc_dis * slice[6]).loc[d]

dis_factor_500_r = (dis_factor * index_500).rank(axis=1, ascending=False)

dis_factor_not_500_r = (dis_factor * not_index_500).rank(axis=1, ascending=False)

dis_pos = stk_rtn * 0

dis_pos[dis_factor_500_r.fillna(51) <= 50] = 1
dis_pos[dis_factor_not_500_r.fillna(51) <= 50] = 1

dis_pnl = (dis_pos.shift(1) * stk_rtn).sum(axis=1).cumsum()
dis_rtn = (dis_pos.shift(1) * stk_rtn).sum(axis=1)
dis_sr = np.sqrt(250) * dis_rtn.rolling(window=6).mean() / dis_rtn.rolling(window=6).std()
plt.plot(dis_sr)
plt.savefig("/mnt/mfs/work_mia/dis_sr.png")
plt.close()

# congestion degree

turnrate = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TURNRATE.csv')
turnrate = turnrate.reindex(index=date_list)
turnrate = turnrate.rolling(window=75).mean()

std_3m = ori_stk_rtn.rolling(window=75).std()


# beta = ori_stk_rtn*0
# for i in range(beta.shape[0]-75):
#     x = index_500_r[i:i+75]
#     for j in range(beta.shape[1]):
#
#         model = lr()
#         y = ori_stk_rtn.iloc[i:i+75,j]
#         beta.iloc[]
#         res_alpha.iloc[i+12,j] = stk_rtn.iloc[i+12,j]-y_pre
#
#


def congest(factor):
    factor_r_a = factor.rank(axis=1)
    factor_r_d = factor.rank(axis=1, ascending=False)
    turn = (turnrate[factor_r_d < 50].mean(axis=1) / turnrate[factor_r_a < 50].mean(axis=1).replace(0, np.nan)).fillna(
        0)
    fluct = (std_3m[factor_r_d < 50].mean(axis=1) / std_3m[factor_r_a < 50].mean(axis=1).replace(0, np.nan)).fillna(0)
    beta_ratio = (beta[factor_r_d < 50].mean(axis=1) / beta[factor_r_a < 50].mean(axis=1).replace(0, np.nan)).fillna(0)

    cong = turn + fluct + beta_ratio
    return cong


factors = [ep, sp, bp, sale_g, pro_g, roe, roa, grossPmargin, pmargin, currentratio, cashratio, ori_ln_mkt, res_alpha,
           return_nm, std_nm]
for i in range(len(factors)):
    factors[i] = congest(factors[i])
[ep3, sp3, bp3, sale_g3, pro_g3, roe3, roa3, grossPmargin3, pmargin3, currentratio3, cashratio3, ln_mkt3, res_alpha3,
 return_nm3, std_nm3] = factors

val3 = (ep3 + bp3 + sp3) / 3
grow3 = (sale_g3 + pro_g3) / 2
profit3 = (roe3 + roa3 + grossPmargin3 + pmargin3) / 4
lev3 = (currentratio3 + cashratio3) / 2
mkt3 = -ln_mkt3
reverse3 = -(res_alpha3 + return_nm3) / 2
fluc3 = -std_nm3

val3 = val3.resample('M').last()
grow3 = grow3.resample('M').last()
profit3 = profit3.resample('M').last()
lev3 = lev3.resample('M').last()
mkt3 = mkt3.resample('M').last()
reverse3 = reverse3.resample('M').last()
fluc3 = fluc3.resample('M').last()

con_factor_pos = pd.concat([val3, grow3, profit3, lev3, mkt3, reverse3, fluc3], axis=1).replace(0, np.nan)
con_factor_pos_r = con_factor_pos.rank(axis=1)
con_factor_pos = con_factor_pos * 0

con_factor_pos[con_factor_pos_r <= 4] = 1 / 4

con_factor = stk_rtn * 0

date = con_factor.index.tolist()
for d in date:
    sliceofc = con_factor_pos.loc[d].fillna(0)
    con_factor.loc[d] = (val_dis * sliceofc[0] + grow_dis * sliceofc[1] + profit_dis * sliceofc[2] + lev_dis * sliceofc[
        3] + mkt_dis * sliceofc[4] + reverse_dis * sliceofc[5] + fluc_dis * sliceofc[6]).loc[d]

con_factor_500_r = (con_factor * index_500).rank(axis=1, ascending=False)

con_factor_not_500_r = (con_factor * not_index_500).rank(axis=1, ascending=False)

con_pos = stk_rtn * 0

con_pos[con_factor_500_r.fillna(51) <= 50] = 1
con_pos[con_factor_not_500_r.fillna(51) <= 50] = 1

con_pnl = (con_pos.shift(1) * stk_rtn).sum(axis=1).cumsum()
# dis_rtn = (dis_pos.shift(1)*stk_rtn).sum(axis=1)
# dis_sr = np.sqrt(250)*dis_rtn.rolling(window=6).mean()/dis_rtn.rolling(window=6).std()
plt.plot(con_pnl)
plt.savefig("/mnt/mfs/work_mia/con.png")
plt.close()

# bottom_up


bot_factor = con_factor + icwr_factor + dis_factor

bot_factor_500_r = (bot_factor * index_500).rank(axis=1, ascending=False)

bot_factor_not_500_r = (bot_factor * not_index_500).rank(axis=1, ascending=False)

bot_pos = stk_rtn * 0

bot_pos[bot_factor_500_r.fillna(51) <= 50] = 1
bot_pos[bot_factor_not_500_r.fillna(51) <= 50] = 1

bot_pnl = (bot_pos.shift(1) * stk_rtn).sum(axis=1).cumsum()

# top_down

top1_pos = stk_rtn * 0
top2_pos = stk_rtn * 0
top3_pos = stk_rtn * 0

top1_pos[icwr_factor_500_r <= 17] = 1
top1_pos[icwr_factor_not_500_r <= 17] = 1

top2_pos[dis_factor_500_r <= 17] = 1
top2_pos[dis_factor_not_500_r <= 17] = 1

top3_pos[con_factor_500_r <= 17] = 1
top3_pos[con_factor_not_500_r <= 17] = 1

top_pos = top1_pos + top2_pos + top3_pos

top_pnl = (top_pos.shift(1) * stk_rtn).sum(axis=1).cumsum()

plt.plot(date_list_month, ew_pnl, date_list_month, icwr_pnl, date_list_month, dis_pnl, date_list_month, con_pnl,
         date_list_month, bot_pnl, date_list_month, top_pnl)
plt.legend(['ew', 'icwr', 'dis', 'con', 'bot', 'top'])
plt.savefig("/mnt/mfs/work_mia/coalesce.png")
plt.close()

if __name__ == '__main__':
    root_path = '/mnt/mfs/DAT_EQT'
    # st_stock_info, new_stock_info = get_st_new_stock_info(root_path)
    a = bt.AZ_Load_csv('/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_REVALUATION/PE_TTM.csv')

import time

start = time.clock()
# for i in range(stk_rtn.shape[0]-12):
#     x = index_500_r[i:i+12]
#     for j in range(stk_rtn.shape[1]):
#
#         model = lr()
#         y = stk_rtn.iloc[i:i+12,j]
#
#         y_pre = model.fit(x,y).coef_*index_500_r[i+12]+model.fit(x,y).intercept_
#         res_alpha.iloc[i+12,j] = stk_rtn.iloc[i+12,j]-y_pre
#
#
neutralize(ep)

end = time.clock()
print('running time: %s sec' % (end - start))

from multiprocessing import Pool
import os, time, random


def long_time_task(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))


if __name__ == '__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(4)
    for i in range(5):
        p.apply_async(long_time_task, args=(i,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')

from multiprocessing import Pool, Lock


def fun(x):
    return x ** 2


if __name__ == '__main__':
    pool = Pool(15)
    result_list = []
    for i in factors:
        result_list.append(pool.apply_async(neutralize, args=(i,)))
    pool.close()

    pool.join()
    a = [x.get() for x in result_list]
    print(a)
