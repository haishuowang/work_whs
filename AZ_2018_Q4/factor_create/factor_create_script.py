import funda_data.funda_data_deal as fdd
import talib as ta
import loc_lib.shared_paths.path as pt
import loc_lib.shared_tools.back_test as bt
import loc_lib.shared_paths.path as pt
import pandas as pd
import numpy as np
import sklearn
from datetime import datetime, timedelta
import time

BaseDeal = fdd.BaseDeal
FundaBaseDeal = fdd.FundaBaseDeal
SectorData = fdd.SectorData
TechBaseDeal = fdd.TechBaseDeal


class AZ_Factor_Momentum:
    @staticmethod
    def ADX(High, Low, Close, timeperiod=14):
        adx = pd.DataFrame()
        for i in High.columns:
            adx[i] = ta.ADX(High[i], Low[i], Close[i], timeperiod)
        return adx

    @staticmethod
    def ADXR(High, Low, Close, timeperiod=14):
        adxr = pd.DataFrame()
        for i in High.columns:
            adxr[i] = ta.ADXR(High[i], Low[i], Close[i], timeperiod)
        return adxr

    @staticmethod
    def APO(Close, fastperiods=12, lowperiod=26, matype=0):  # default
        return Close.apply(lambda col: ta.APO(col, fastperiods, lowperiod, matype), axis=0)

    @staticmethod
    def AROON(High, Low, timeperiod=14):
        aroondown, aroonup = pd.DataFrame(), pd.DataFrame()
        for i in High.columns:
            aroondown[i], aroonup[i] = ta.AROON(High[i], Low[i], timeperiod)
        return aroondown, aroonup

    @staticmethod
    def AROONOSC(High, Low, timeperiod=14):
        aroonosc = pd.DataFrame()
        for i in High.columns:
            aroonosc[i] = ta.AROONOSC(High[i], Low[i], timeperiod)
        return aroonosc

    @staticmethod
    def BOP(Open, High, Low, Close):
        bop = pd.DataFrame()
        for i in High.columns:
            bop[i] = ta.BOP(Open[i], High[i], Low[i], Close[i])
        return bop

    @staticmethod
    def CCI(High, Low, Close, timeperiod=14):
        cci = pd.DataFrame()
        for i in High.columns:
            cci[i] = ta.CCI(High[i], Low[i], Close[i], timeperiod)
        return cci

    @staticmethod
    def CMO(Close, timeperiod=14):
        return Close.apply(lambda col: ta.CMO(col, timeperiod), axis=0)

    @staticmethod
    def DX(High, Low, Close, timeperiod=14):
        dx = pd.DataFrame()
        for i in High.columns:
            dx[i] = ta.DX(High[i], Low[i], Close[i], timeperiod)
        return dx

    @staticmethod
    def MACD(Close, fastperiod=12, slowperiod=26, signalperiod=9):
        macd, macdsignal, macdhist = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for i in Close.columns:
            macd[i], macdsignal[i], macdhist[i] = ta.MACD(Close[i], fastperiod, slowperiod, signalperiod)
        return macd, macdsignal, macdhist

    @staticmethod
    def MACDEXT(Close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0):
        macd, macdsignal, macdhist = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for i in Close.columns:
            macd[i], macdsignal[i], macdhist[i] = ta.MACDEXT(Close[i], fastperiod, fastmatype, slowperiod, slowmatype,
                                                             signalperiod, signalmatype)
        return macd, macdsignal, macdhist

    @staticmethod
    def MACDFIX(Close, signalperiod=9):
        macd, macdsignal, macdhist = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for i in Close.columns:
            macd[i], macdsignal[i], macdhist[i] = ta.MACDFIX(Close[i], signalperiod)
        return macd, macdsignal, macdhist

    @staticmethod
    def MFI(High, Low, Close, Volume, timeperiod=14):
        mfi = pd.DataFrame()
        for i in High.columns:
            mfi[i] = ta.MFI(High[i], Low[i], Close[i], Volume[i], timeperiod)
        return mfi

    @staticmethod
    def MINUS_DI(High, Low, Close, timeperiod=14):
        minus_di = pd.DataFrame()
        for i in High.columns:
            minus_di[i] = ta.MINUS_DI(High[i], Low[i], Close[i], timeperiod)
        return minus_di

    @staticmethod
    def MINUS_DM(High, Low, timeperiod=14):
        minus_dm = pd.DataFrame()
        for i in High.columns:
            minus_dm[i] = ta.MINUS_DM(High[i], Low[i], timeperiod)
        return minus_dm

    @staticmethod
    def MOM(Close, timeperiod=10):
        return Close.apply(lambda col: ta.MOM(col, timeperiod), axis=0)

    @staticmethod
    def PLUS_DI(High, Low, Close, timeperiod=14):
        plus_di = pd.DataFrame()
        for i in High.columns:
            plus_di[i] = ta.PLUS_DI(High[i], Low[i], Close[i], timeperiod)
        return plus_di

    @staticmethod
    def PLUS_DM(High, Low, timeperiod=14):
        plus_dm = pd.DataFrame()
        for i in High.columns:
            plus_dm[i] = ta.PLUS_DM(High[i], Low[i], timeperiod)
        return plus_dm

    @staticmethod
    def PPO(Close, fastperiod=12, slowperiod=26, matype=0):
        return Close.apply(lambda col: ta.PPO(col, fastperiod, slowperiod, matype), axis=0)

    @staticmethod
    def ROC(Close, timeperiod=10):
        return Close.apply(lambda col: ta.ROC(col, timeperiod), axis=0)

    @staticmethod
    def ROCP(Close, timeperiod=10):
        return Close.apply(lambda col: ta.ROCP(col, timeperiod), axis=0)

    @staticmethod
    def ROCR(Close, timeperiod=10):
        return Close.apply(lambda col: ta.ROCR(col, timeperiod), axis=0)

    @staticmethod
    def ROCR100(Close, timeperiod=10):
        return Close.apply(lambda col: ta.ROCR100(col, timeperiod), axis=0)

    @staticmethod
    def RSI(Close, timeperiod=14):
        return Close.apply(lambda col: ta.RSI(col, timeperiod), axis=0)

    @staticmethod
    def STOCH(High, Low, Close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):
        slowk, slowd = pd.DataFrame(), pd.DataFrame()
        for i in High.columns:
            slowk[i], slowd[i] = ta.STOCH(High[i], Low[i], Close[i], fastk_period, slowk_period, slowk_matype,
                                          slowd_period, slowd_matype)
        return slowk, slowd

    @staticmethod
    def STOCHF(High, Low, Close, fastk_period=5, fastd_period=3, fastd_matype=0):
        fastk, fastd = pd.DataFrame(), pd.DataFrame()
        for i in High.columns:
            fastk[i], fastd[i] = ta.STOCHF(High[i], Low[i], Close[i], fastk_period, fastd_period, fastd_matype)
        return fastk, fastd

    @staticmethod
    def STOCHRSI(Close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0):
        fastk, fastd = pd.DataFrame(), pd.DataFrame()
        for i in Close.columns:
            fastk[i], fastd[i] = ta.STOCHRSI(Close[i], timeperiod, fastk_period, fastd_period, fastd_matype)
        return fastk, fastd

    @staticmethod
    def TRIX(Close, timeperiod=30):
        return Close.apply(lambda col: ta.TRIX(col, timeperiod), axis=0)

    @staticmethod
    def ULTOSC(High, Low, Close, timeperiod1=7, timeperiod2=14, timeperiod3=28):
        ultosc = pd.DataFrame()
        for i in Close.columns:
            ultosc[i] = ta.ULTOSC(High[i], Low[i], Close[i], timeperiod1, timeperiod2, timeperiod3)
        return ultosc

    @staticmethod
    def WILLR(High, Low, Close, timeperiod=14):
        willr = pd.DataFrame()
        for i in High.columns:
            willr[i] = ta.WILLR(High[i], Low[i], Close[i], timeperiod)
        return willr


class AZ_Factor_Overlap:

    @staticmethod
    def BBANDS(Close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0):
        upperband, middleband, lowerband = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for i in Close.columns:
            upperband[i], middleband[i], lowerband[i] = ta.BBANDS(Close[i], timeperiod, nbdevup, nbdevdn, matype)
        upperband.replace(np.nan, 0, inplace=True)
        middleband.replace(np.nan, 0, inplace=True)
        lowerband.replace(np.nan, 0, inplace=True)
        return upperband, middleband, lowerband

    @staticmethod
    def DEMA(Close, timeperiod=30):
        return Close.apply(lambda col: ta.DEMA(col, timeperiod), axis=0)

    @staticmethod
    def EMA(Close, timeperiod=30):
        return Close.apply(lambda col: ta.EMA(col, timeperiod), axis=0)

    @staticmethod
    def HT_TRENDLINE(Close):
        return Close.apply(lambda col: ta.HT_TRENDLINE(col), axis=0)

    @staticmethod
    def KAMA(Close, timeperiod=30):
        return Close.apply(lambda col: ta.KAMA(col, timeperiod), axis=0)

    @staticmethod
    def MA(Close, timeperiod=30, matype=0):
        return Close.apply(lambda col: ta.MA(col, timeperiod, matype), axis=0)

    @staticmethod
    def MAMA(Close, fastlimit=0, slowlimit=0):
        mama, fama = pd.DataFrame(), pd.DataFrame()
        for i in Close.columns:
            mama[i], fama[i] = ta.MAMA(Close[i], fastlimit, slowlimit)
        return mama, fama

    @staticmethod
    def MAVP(Close, periods, minperiod=2, maxperiod=30, matype=0):  # periods　should be array
        return Close.apply(lambda col: ta.MAVP(col, periods, minperiod, maxperiod, matype), axis=0)

    @staticmethod
    def MIDPOINT(Close, timeperiod=14):
        return Close.apply(lambda col: ta.MIDPOINT(col, timeperiod), axis=0)

    @staticmethod
    def MIDPRICE(High, Low, timeperiod=14):
        midprice = pd.DataFrame()
        for i in High.columns:
            midprice[i] = ta.MIDPRICE(High[i], Low[i], timeperiod)
        return midprice

    @staticmethod
    def SAR(High, Low, acceleration=0, maximum=0):
        sar = pd.DataFrame()
        for i in High.columns:
            sar[i] = ta.SAR(High[i], Low[i], acceleration=0, maximum=0)
        return sar

    @staticmethod
    def SAREXT(High, Low, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0,
               accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0):
        sarext = pd.DataFrame()
        for i in High.columns:
            sarext[i] = ta.SAREXT(High[i], Low[i], startvalue, offsetonreverse, accelerationinitlong, accelerationlong,
                                  accelerationmaxlong, accelerationinitshort, accelerationshort, accelerationmaxshort)
        return sarext

    @staticmethod
    def SMA(Close, timeperiod=30):
        return Close.apply(lambda col: ta.SMA(col, timeperiod), axis=0)

    @staticmethod
    def T3(Close, timeperiod=5, vfactor=0):
        return Close.apply(lambda col: ta.T3(col, timeperiod, vfactor), axis=0)

    @staticmethod
    def TEMA(Close, timeperiod=30):
        return Close.apply(lambda col: ta.TEMA(col, timeperiod), axis=0)

    @staticmethod
    def TRIMA(Close, timeperiod=30):
        return Close.apply(lambda col: ta.TRIMA(col, timeperiod), axis=0)

    @staticmethod
    def WMA(Close, timeperiod=30):
        return Close.apply(lambda col: ta.WMA(col, timeperiod), axis=0)


class AZ_Factor_Volume:

    @staticmethod
    def AD(High, Low, Close, Volume):
        ad = pd.DataFrame()
        for i in High.columns:
            ad[i] = ta.AD(High[i], Low[i], Close[i], Volume[i])
        return ad

    @staticmethod
    def ADOSC(High, Low, Close, Volume, fastperiod=3, slowperiod=10):
        adosc = pd.DataFrame()
        for i in High.columns:
            adosc[i] = ta.ADOSC(High[i], Low[i], Close[i], Volume[i], fastperiod, slowperiod)
        return adosc

    @staticmethod
    def OBV(Close, Volume):
        obv = pd.DataFrame()
        for i in Close.columns:
            obv[i] = ta.OBV(Close[i], Volume[i])
        return obv


class AZ_Factor_Volatility:

    @staticmethod
    def ATR(High, Low, Close, timeperiod=14):
        atr = pd.DataFrame()
        for i in High.columns:
            atr[i] = ta.ATR(High[i], Low[i], Close[i], timeperiod)
        return atr

    @staticmethod
    def NATR(High, Low, Close, timeperiod=14):
        natr = pd.DataFrame()
        for i in High.columns:
            natr[i] = ta.NATR(High[i], Low[i], Close[i], timeperiod)
        return natr

    @staticmethod
    def TRANGE(High, Low, Close):
        trange = pd.DataFrame()
        for i in High.columns:
            trange[i] = ta.TRANGE(High[i], Low[i], Close[i])
        return trange


class AZ_Factor_Price:

    @staticmethod
    def AVGPRICE(Open, High, Low, Close):
        avgprice = pd.DataFrame()
        for i in High.columns:
            avgprice[i] = ta.AVGPRICE(Open[i], High[i], Low[i], Close[i])
        return avgprice

    @staticmethod
    def MEDPRICE(High, Low):
        medprice = pd.DataFrame()
        for i in High.columns:
            medprice[i] = ta.MEDPRICE(High[i], Low[i])
        return medprice

    @staticmethod
    def TYPPRICE(High, Low, Close):
        typprice = pd.DataFrame()
        for i in High.columns:
            typprice[i] = ta.TYPPRICE(High[i], Low[i], Close[i])
        return typprice

    @staticmethod
    def WCLPRICE(High, Low, Close):
        wclprice = pd.DataFrame()
        for i in High.columns:
            wclprice[i] = ta.WCLPRICE(High[i], Low[i], Close[i])
        return wclprice


class AZ_Factor_Cycle:

    @staticmethod
    def HT_DCPERIOD(Close):
        return Close.apply(lambda col: ta.HT_DCPERIOD(col), axis=0)

    @staticmethod
    def HT_DCPHASE(Close):
        return Close.apply(lambda col: ta.HT_DCPHASE(col), axis=0)

    @staticmethod
    def HT_PHASOR(Close):
        inphase, quadrature = pd.DataFrame(), pd.DataFrame()
        for i in Close.columns:
            inphase[i], quadrature[i] = ta.HT_PHASOR(Close[i])
        return inphase, quadrature

    @staticmethod
    def HT_SINE(Close):
        sine, leadsine = pd.DataFrame(), pd.DataFrame()
        for i in Close.columns:
            sine[i], leadsine[i] = ta.HT_SINE(Close[i])
        return sine, leadsine

    @staticmethod
    def HT_TRENDMODE(Close):
        return Close.apply(lambda col: ta.HT_TRENDMODE(col), axis=0)


class AZ_Factor_Pattern:

    @staticmethod
    def CDL2CROWS(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDL2CROWS(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDL3BLACKCROWS(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDL3BLACKCROWS(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDL3INSIDE(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDL3INSIDE(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDL3LINESTRIKE(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDL3LINESTRIKE(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDL3OUTSIDE(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDL3OUTSIDE(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDL3STARSINSOUTH(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDL3STARSINSOUTH(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDL3WHITESOLDIERS(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDL3WHITESOLDIERS(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLABANDONEDBABY(Open, High, Low, Close, penetration=0):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLABANDONEDBABY(Open[i], High[i], Low[i], Close[i], penetration)
        return integer

    @staticmethod
    def CDLADVANCEBLOCK(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLADVANCEBLOCK(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLBELTHOLD(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLBELTHOLD(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLBREAKAWAY(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLBREAKAWAY(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLCLOSINGMARUBOZU(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLCLOSINGMARUBOZU(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLCONCEALBABYSWALL(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLCONCEALBABYSWALL(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLCOUNTERATTACK(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLCOUNTERATTACK(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLDARKCLOUDCOVER(Open, High, Low, Close, penetration=0):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLDARKCLOUDCOVER(Open[i], High[i], Low[i], Close[i], penetration)
        return integer

    @staticmethod
    def CDLDOJI(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLDOJI(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLDOJISTAR(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLDOJISTAR(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLDRAGONFLYDOJI(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLDRAGONFLYDOJI(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLENGULFING(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLENGULFING(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLEVENINGDOJISTAR(Open, High, Low, Close, penetration=0):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLEVENINGDOJISTAR(Open[i], High[i], Low[i], Close[i], penetration)
        return integer

    @staticmethod
    def CDLEVENINGSTAR(Open, High, Low, Close, penetration=0):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLEVENINGSTAR(Open[i], High[i], Low[i], Close[i], penetration)
        return integer

    @staticmethod
    def CDLGAPSIDESIDEWHITE(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLGAPSIDESIDEWHITE(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLGRAVESTONEDOJI(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLGRAVESTONEDOJI(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLHAMMER(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLHAMMER(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLHANGINGMAN(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLHANGINGMAN(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLHARAMI(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLHARAMI(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLHARAMICROSS(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLHARAMICROSS(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLHIGHWAVE(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLHIGHWAVE(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLHIKKAKE(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLHIKKAKE(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLHIKKAKEMOD(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLHIKKAKEMOD(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLHOMINGPIGEON(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLHOMINGPIGEON(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLIDENTICAL3CROWS(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLIDENTICAL3CROWS(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLINNECK(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLINNECK(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLINVERTEDHAMMER(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLINVERTEDHAMMER(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLKICKING(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLKICKING(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLKICKINGBYLENGTH(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLKICKINGBYLENGTH(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLLADDERBOTTOM(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLLADDERBOTTOM(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLLONGLEGGEDDOJI(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLLONGLEGGEDDOJI(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLLONGLINE(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLLONGLINE(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLMARUBOZU(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLMARUBOZU(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLMATHOLD(Open, High, Low, Close, penetration=0):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLMATHOLD(Open[i], High[i], Low[i], Close[i], penetration)
        return integer

    @staticmethod
    def CDLMORNINGDOJISTAR(Open, High, Low, Close, penetration=0):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLMORNINGDOJISTAR(Open[i], High[i], Low[i], Close[i], penetration)
        return integer

    @staticmethod
    def CDLONNECK(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLONNECK(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLPIERCING(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLPIERCING(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLRICKSHAWMAN(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLRICKSHAWMAN(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLRISEFALL3METHODS(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLRISEFALL3METHODS(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLSEPARATINGLINES(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLSEPARATINGLINES(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLSHOOTINGSTAR(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLSHOOTINGSTAR(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLSHORTLINE(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLSHORTLINE(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLSPINNINGTOP(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLSPINNINGTOP(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLSTALLEDPATTERN(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLSTALLEDPATTERN(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLSTICKSANDWICH(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLSTICKSANDWICH(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLTAKURI(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLTAKURI(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLTASUKIGAP(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLTASUKIGAP(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLTHRUSTING(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLTHRUSTING(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLTRISTAR(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLTRISTAR(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLUNIQUE3RIVER(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLUNIQUE3RIVER(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLUPSIDEGAP2CROWS(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLUPSIDEGAP2CROWS(Open[i], High[i], Low[i], Close[i])
        return integer

    @staticmethod
    def CDLXSIDEGAP3METHODS(Open, High, Low, Close):
        integer = pd.DataFrame()
        for i in High.columns:
            integer[i] = ta.CDLXSIDEGAP3METHODS(Open[i], High[i], Low[i], Close[i])
        return integer


class AZ_Factor_Statistic:

    @staticmethod
    def BETA(High, Low, timeperiod=5):
        real = pd.DataFrame()
        for i in High.columns:
            real[i] = ta.BETA(High[i], Low[i], timeperiod)
        return real

    @staticmethod
    def CORREL(High, Low, timeperiod=30):
        real = pd.DataFrame()
        for i in High.columns:
            real[i] = ta.CORREL(High[i], Low[i], timeperiod)
        return real

    @staticmethod
    def LINEARREG(Close, timeperiod=14):
        return Close.apply(lambda col: ta.LINEARREG(col, timeperiod), axis=0)

    @staticmethod
    def LINEARREG_ANGLE(Close, timeperiod=14):
        return Close.apply(lambda col: ta.LINEARREG_ANGLE(col, timeperiod), axis=0)

    @staticmethod
    def LINEARREG_INTERCEPT(Close, timeperiod=14):
        return Close.apply(lambda col: ta.LINEARREG_INTERCEPT(col, timeperiod), axis=0)

    @staticmethod
    def LINEARREG_SLOPE(Close, timeperiod=14):
        return Close.apply(lambda col: ta.LINEARREG_SLOPE(col, timeperiod), axis=0)

    @staticmethod
    def STDDEV(Close, timeperiod=5, nbdev=1):
        return Close.apply(lambda col: ta.STDDEV(col, timeperiod, nbdev), axis=0)

    @staticmethod
    def TSF(Close, timeperiod=14):
        return Close.apply(lambda col: ta.TSF(col, timeperiod), axis=0)

    @staticmethod
    def VAR(Close, timeperiod=5, nbdev=1):
        return Close.apply(lambda col: ta.VAR(col, timeperiod, nbdev), axis=0)


class AZ_Factor_Math:

    @staticmethod
    def ACOS(Close):
        return Close.apply(lambda col: ta.ACOS(col), axis=0)

    @staticmethod
    def ASIN(Close):
        return Close.apply(lambda col: ta.ASIN(col), axis=0)

    @staticmethod
    def ATAN(Close):
        return Close.apply(lambda col: ta.ATAN(col), axis=0)

    @staticmethod
    def CEIL(Close):
        return Close.apply(lambda col: ta.CEIL(col), axis=0)

    @staticmethod
    def COS(Close):
        return Close.apply(lambda col: ta.COS(col), axis=0)

    @staticmethod
    def COSH(Close):
        return Close.apply(lambda col: ta.COSH(col), axis=0)

    @staticmethod
    def EXP(Close):
        return Close.apply(lambda col: ta.EXP(col), axis=0)

    @staticmethod
    def FLOOR(Close):
        return Close.apply(lambda col: ta.FLOOR(col), axis=0)

    @staticmethod
    def LN(Close):
        return Close.apply(lambda col: ta.LN(col), axis=0)

    @staticmethod
    def LOG10(Close):
        return Close.apply(lambda col: ta.LOG10(col), axis=0)

    @staticmethod
    def SIN(Close):
        return Close.apply(lambda col: ta.SIN(col), axis=0)

    @staticmethod
    def SINH(Close):
        return Close.apply(lambda col: ta.SINH(col), axis=0)

    @staticmethod
    def SQRT(Close):
        return Close.apply(lambda col: ta.SQRT(col), axis=0)

    @staticmethod
    def TAN(Close):
        return Close.apply(lambda col: ta.TAN(col), axis=0)

    @staticmethod
    def TANH(Close):
        return Close.apply(lambda col: ta.TANH(col), axis=0)

    @staticmethod
    def ADD(High, Low):
        real = pd.DataFrame()
        for i in High.columns:
            real[i] = ta.ADD(High[i], Low[i])
        return real

    @staticmethod
    def DIV(High, Low):
        real = pd.DataFrame()
        for i in High.columns:
            real[i] = ta.DIV(High[i], Low[i])
        return real

    @staticmethod
    def MAX(Close, timeperiod=30):
        return Close.apply(lambda col: ta.TSF(col, timeperiod), axis=0)

    @staticmethod
    def MAXINDEX(Close, timeperiod=30):
        return Close.apply(lambda col: ta.TSF(col, timeperiod), axis=0)

    @staticmethod
    def MIN(Close, timeperiod=30):
        return Close.apply(lambda col: ta.TSF(col, timeperiod), axis=0)

    @staticmethod
    def MININDEX(Close, timeperiod=30):
        return Close.apply(lambda col: ta.TSF(col, timeperiod), axis=0)

    @staticmethod
    def MINMAX(Close, timeperiod=30):
        min, max = pd.DataFrame(), pd.DataFrame()
        for i in Close.columns:
            min[i], max[i] = ta.MINMAX(Close[i], timeperiod)
        return min, max

    @staticmethod
    def MINMAXINDEX(Close, timeperiod=30):
        minidx, maxidx = pd.DataFrame(), pd.DataFrame()
        for i in Close.columns:
            minidx[i], maxidx[i] = ta.MINMAXINDEX(Close[i], timeperiod)
        return minidx, maxidx

    @staticmethod
    def MULT(High, Close):
        real = pd.DataFrame()
        for i in Close.columns:
            real[i] = ta.MULT(High[i], Close[i])
        return real

    @staticmethod
    def SUB(High, Close):
        real = pd.DataFrame()
        for i in Close.columns:
            real[i] = ta.SUB(High[i], Close[i])
        return real

    @staticmethod
    def SUM(Close, timeperiod=30):
        return Close.apply(lambda col: ta.SUM(col, timeperiod), axis=0)


class FactorMomentum:
    @staticmethod
    def ADX(High, Low, Close, sector_df, timeperiod, limit_up=40, limit_dn=20):
        """趋势强弱指标"""
        real = AZ_Factor_Momentum.ADX(High, Low, Close, timeperiod)
        target_df = (real > limit_up).astype(int) - (real < limit_dn).astype(int)
        return target_df * sector_df

    @staticmethod
    def ADXR(High, Low, Close, sector_df, timeperiod, limit_up=40, limit_dn=20):
        real = AZ_Factor_Momentum.ADXR(High, Low, Close, timeperiod)
        target_df = (real > limit_up).astype(int) - (real < limit_dn).astype(int)
        return target_df * sector_df

    @staticmethod
    def APO(Close, sector_df, fastperiods=12, lowperiod=26, matype=0):  # default
        real = AZ_Factor_Momentum.APO(Close, fastperiods, lowperiod, matype)
        target_df = (real > 0).astype(int) - (real < 0).astype(int)
        return target_df * sector_df

    @staticmethod
    def AROON(High, Low, sector_df, timeperiod, limit=80):
        aroondn, aroonup = AZ_Factor_Momentum.AROON(High, Low, timeperiod)
        target_df = (aroondn > limit).astype(int) - (aroonup > limit).astype(int)
        return target_df * sector_df

    @staticmethod
    def AROONSC(High, Low, sector_df, timeperiod, limit=80):
        aroondn, aroonup = AZ_Factor_Momentum.AROONOSC(High, Low, timeperiod)
        target_df = (aroondn > limit).astype(int) - (aroonup > limit).astype(int)
        return target_df * sector_df

    @staticmethod
    def CMO(Close, sector_df, timeperiod, limit=0):
        real = AZ_Factor_Momentum.CMO(Close, timeperiod)
        target_df = (real > limit).astype(int) - (real < limit).astype(int)
        return target_df * sector_df

    @staticmethod
    def MACD(Close, sector_df, fastperiod=12, slowperiod=26, signalperiod=9):
        macd, macdsignal, macdhist = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for i in Close.columns:
            macd[i], macdsignal[i], macdhist[i] = ta.MACD(Close[i], fastperiod, slowperiod, signalperiod)
        macdhist_copy = macdhist.copy()
        macdhist_copy.replace(np.nan, 0)
        macdhist_copy[macdhist > 0] = 1
        macdhist_copy[macdhist < 0] = 0
        target_df = macdhist_copy - macdhist_copy.shift(1)
        return target_df * sector_df

    @staticmethod
    def MFI(High, Low, Close, Volume, sector_df, timeperiod, limit_up=80, limit_dn=20):
        real = AZ_Factor_Momentum.MFI(High, Low, Close, Volume, timeperiod)
        target_df = (real > limit_up).astype(int) - (real > limit_dn).astype(int)
        return target_df * sector_df

    @staticmethod
    def RSI(Close, sector_df, timeperiod=14, limit_up_dn=30):
        real = AZ_Factor_Momentum.RSI(Close, timeperiod) - 50
        target_df = (real > limit_up_dn).astype(int) - (real < limit_up_dn).astype(int)
        return target_df * sector_df

    @staticmethod
    def MA_LINE(Close, sector_df, slowperiod, fastperiod):
        slow_line = Close.rolling(slowperiod, min_periods=0).mean()
        fast_line = Close.rolling(fastperiod, min_periods=0).mean()
        MA_diff = fast_line - slow_line
        MA_diff_copy = MA_diff.copy()
        MA_diff_copy[MA_diff > 0] = 1
        MA_diff_copy[MA_diff < 0] = 0
        target_df = MA_diff_copy - MA_diff_copy.shift(1)
        return target_df * sector_df


class FactorVolume:
    # @staticmethod
    # def AD(High, Low, Close, Volume):
    #     ad = pd.DataFrame()
    #     for i in High.columns:
    #         ad[i] = ta.AD(High[i], Low[i], Close[i], Volume[i])
    #     return ad

    @staticmethod
    def ADOSC(High, Low, Close, Volume, sector_df, fastperiod, slowperiod, limit_up_dn=0):
        real = AZ_Factor_Volume.ADOSC(High, Low, Close, Volume, fastperiod, slowperiod)
        target_df = (real > limit_up_dn).astype(int) - (real < -limit_up_dn).astype(int)
        return target_df * sector_df


class FactorOverlap:
    @staticmethod
    def BBANDS(Close, sector_df, timeperiod, limit_up_down):
        up_line, mid_line, down_line = AZ_Factor_Overlap.BBANDS(Close, timeperiod, nbdevup=limit_up_down,
                                                                nbdevdn=limit_up_down, matype=0)
        target_df = Close.copy()
        target_df.iloc[:, :] = 0
        target_df[(Close <= up_line) & (Close >= down_line)] = 0
        target_df[Close > up_line] = 1
        target_df[Close < down_line] = -1
        return target_df * sector_df


class FactorVolatility(BaseDeal):
    def ATR(self, High, Low, Close, sector_df, timeperiod, percent):
        real = AZ_Factor_Volatility.ATR(High, Low, Close, timeperiod)
        tmp_df = bt.AZ_Row_zscore(real)
        target_df = self.row_extre(tmp_df, sector_df, percent)
        return target_df


class TechFactor(FactorVolume, FactorOverlap, FactorMomentum, FactorVolatility, BaseDeal):
    def __init__(self, root_path, sector_df, save_root_path):
        self.sector_df = sector_df
        xnms = sector_df.columns
        # xinx = sector_df.index
        # load_path = '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14'
        self.aadj_p_path = root_path.EM_Funda.DERIVED_14 / 'aadj_p.csv'
        self.aadj_p = bt.AZ_Load_csv(self.aadj_p_path).reindex(columns=xnms)

        self.aadj_p_HIGH_path = root_path.EM_Funda.DERIVED_14 / 'aadj_p_HIGH.csv'
        self.aadj_p_HIGH = bt.AZ_Load_csv(self.aadj_p_HIGH_path).reindex(columns=xnms)

        self.aadj_p_LOW_path = root_path.EM_Funda.DERIVED_14 / 'aadj_p_LOW.csv'
        self.aadj_p_LOW = bt.AZ_Load_csv(self.aadj_p_LOW_path).reindex(columns=xnms)

        self.aadj_p_OPEN_path = root_path.EM_Funda.DERIVED_14 / 'aadj_p_OPEN.csv'
        self.aadj_p_OPEN = bt.AZ_Load_csv(self.aadj_p_OPEN_path).reindex(columns=xnms)

        self.TVOL_path = root_path.EM_Funda.TRAD_SK_DAILY_JC / 'TVOL.csv'
        self.TVOL = bt.AZ_Load_csv(self.TVOL_path).reindex(columns=xnms).replace(np.nan, 0)
        self.save_root_path = save_root_path

    def ADX_(self, n_list, limit_up, limit_dn):
        for n in n_list:
            target_df = self.ADX(self.aadj_p_HIGH, self.aadj_p_LOW, self.aadj_p, self.sector_df,
                                 n, limit_up, limit_dn)
            file_name = 'ADX_{}_{}_{}'.format(n, limit_up, limit_dn)
            fun = 'Tech_Factor.FactorMomentum.ADX'
            raw_data_path = (self.aadj_p_HIGH_path, self.aadj_p_LOW_path, self.aadj_p_path)
            args = (n, limit_up, limit_dn)
            self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)

    def AROON_(self, n_list, limit):
        for n in n_list:
            target_df = self.AROON(self.aadj_p_HIGH, self.aadj_p_LOW, self.sector_df, n, limit)
            file_name = 'AROON_{}_{}'.format(n, limit)
            fun = 'Tech_Factor.FactorMomentum.AROON'
            raw_data_path = (self.aadj_p_HIGH_path, self.aadj_p_LOW_path)
            args = (n, limit)
            self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)

    def CMO_(self, n_list, limit):
        for n in n_list:
            target_df = self.CMO(self.aadj_p, self.sector_df, n, limit)
            file_name = 'CMO_{}_{}'.format(n, limit)
            fun = 'Tech_Factor.FactorMomentum.CMO'
            raw_data_path = (self.aadj_p_path,)
            args = (n, limit)
            self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)

    def MFI_(self, n_list, limit_up=80, limit_dn=20):
        for n in n_list:
            target_df = self.MFI(self.aadj_p_HIGH, self.aadj_p_LOW, self.aadj_p, self.TVOL, self.sector_df, n, limit_up,
                                 limit_dn)
            file_name = 'MFI_{}_{}_{}'.format(n, limit_up, limit_dn)
            fun = 'Tech_Factor.FactorMomentum.MFI'
            raw_data_path = (self.aadj_p_HIGH_path, self.aadj_p_LOW_path, self.aadj_p_path, self.TVOL_path)
            args = (n, limit_up, limit_dn)
            self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)

    def RSI_(self, n_list, limit_up_dn=30):
        for n in n_list:
            target_df = self.RSI(self.aadj_p, self.sector_df, n, limit_up_dn)
            file_name = 'RSI_{}_{}'.format(n, limit_up_dn)
            fun = 'Tech_Factor.FactorMomentum.RSI'
            raw_data_path = (self.aadj_p_path,)
            args = (n, limit_up_dn)
            self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)

    def ADOSC_(self, long_short_list, limit_up_dn=0):
        for long, short in long_short_list:
            target_df = self.ADOSC(self.aadj_p_HIGH, self.aadj_p_LOW, self.aadj_p, self.TVOL, self.sector_df,
                                   long, short, limit_up_dn)
            file_name = 'ADOSC_{}_{}_{}'.format(long, short, limit_up_dn)
            fun = 'Tech_Factor.FactorVolume.ADOSC'
            raw_data_path = (self.aadj_p_HIGH_path, self.aadj_p_LOW_path, self.aadj_p_path, self.TVOL_path)
            args = (long, short, limit_up_dn)
            self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)

    def ATR_(self, n_list, percent):
        for n in n_list:
            target_df = self.ATR(self.aadj_p_HIGH, self.aadj_p_LOW, self.aadj_p, self.sector_df, n, percent)
            file_name = 'ATR_{}_{}'.format(n, percent)
            fun = 'Tech_Factor.FactorVolatility.ATR'
            raw_data_path = (self.aadj_p_HIGH_path, self.aadj_p_LOW_path, self.aadj_p_path)
            args = (n, percent)
            self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)

    def MACD_(self, fastperiod, slowperiod, signalperiod):
        target_df = self.MACD(self.aadj_p, self.sector_df, fastperiod, slowperiod, signalperiod)
        file_name = f'MACD_{fastperiod}_{slowperiod}_{signalperiod}'
        fun = 'Tech_Factor.FactorMomentum.MACD'
        raw_data_path = (self.aadj_p_path, )
        args = (fastperiod, slowperiod, signalperiod)
        self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)

    def MA_LINE_(self, long_short_list):
        for slowperiod, fastperiod in long_short_list:
            target_df = self.MA_LINE(self.aadj_p, self.sector_df, slowperiod, fastperiod)
            file_name = f'MA_LINE_{fastperiod}_{slowperiod}'
            fun = 'Tech_Factor.FactorMomentum.MA_LINE'
            raw_data_path = (self.aadj_p_path,)
            args = (fastperiod, slowperiod)
            self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)

    def BBANDS_(self, n_list, limit_list):
        for n in n_list:
            for limit_up_dn in limit_list:
                target_df = self.BBANDS(self.aadj_p, self.sector_df, n, limit_up_dn)
                # print(target_df)
                file_name = f'BBANDS_{n}_{limit_up_dn}'
                fun = 'Tech_Factor.FactorOverlap.BBANDS'
                raw_data_path = (self.aadj_p_path,)
                args = (n, limit_up_dn)
                self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)
