import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import time
from collections import OrderedDict
import talib as ta

sys.path.append("/mnt/mfs/LIB_ROOT")
# import funda_data as fd
# from funda_data.funda_data_deal import SectorData
import open_lib.shared_paths.path as pt
from open_lib.shared_tools import send_email
from multiprocessing import Pool


# import warnings
# warnings.filterwarnings('ignore')
# import loc_lib.shared_tools.back_test as bt


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
    def MACDEXT(Close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9,
                signalmatype=0):
        macd, macdsignal, macdhist = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for i in Close.columns:
            macd[i], macdsignal[i], macdhist[i] = ta.MACDEXT(Close[i], fastperiod, fastmatype, slowperiod,
                                                             slowmatype,
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
            if (High[i] == High[i]).sum() != 0 and \
                    (Low[i] == Low[i]).sum() != 0 and \
                    (Close[i] == Close[i]).sum() != 0 and \
                    (Volume[i] == Volume[i]).sum() != 0:
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
            upperband[i], middleband[i], lowerband[i] = ta.BBANDS(Close[i], timeperiod, nbdevup, nbdevdn,
                                                                  matype)
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
            sarext[i] = ta.SAREXT(High[i], Low[i], startvalue, offsetonreverse, accelerationinitlong,
                                  accelerationlong,
                                  accelerationmaxlong, accelerationinitshort, accelerationshort,
                                  accelerationmaxshort)
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
            try:
                adosc[i] = ta.ADOSC(High[i], Low[i], Close[i], Volume[i], fastperiod, slowperiod)
            except:
                adosc[i] = len(High[i]) * np.nan
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


class bt:
    @staticmethod
    def AZ_Load_csv(target_path, index_time_type=True):
        target_df = pd.read_table(target_path, sep='|', index_col=0, low_memory=False).round(8)
        if index_time_type:
            target_df.index = pd.to_datetime(target_df.index)
        return target_df

    @staticmethod
    def AZ_Rolling(df, n, min_periods=1):
        return df.rolling(window=n, min_periods=min_periods)

    @staticmethod
    def AZ_Rolling_mean(df, n, min_periods=1):
        target = df.rolling(window=n, min_periods=min_periods).mean()
        target.iloc[:n - 1] = np.nan
        return target

    @staticmethod
    def AZ_Path_create(target_path):
        """
        添加新路径
        :param target_path:
        :return:
        """
        if not os.path.exists(target_path):
            os.makedirs(target_path)

    @staticmethod
    def AZ_Row_zscore(df, cap=None):
        df_mean = df.mean(axis=1)
        df_std = df.std(axis=1)
        target = df.sub(df_mean, axis=0).div(df_std, axis=0)
        if cap is not None:
            target[target > cap] = cap
            target[target < -cap] = -cap
        return target


class BaseDeal:
    @staticmethod
    def signal_mean_fun(signal_df):
        return signal_df.abs().sum(axis=1).replace(0, np.nan).dropna() / len(signal_df) > 0.1

    @staticmethod
    def pnd_continue_ud(raw_df, sector_df, n_list):
        def fun(df, n):
            df_pct = df.diff()
            up_df = (df_pct > 0)
            dn_df = (df_pct < 0)
            target_up_df = up_df.copy()
            target_dn_df = dn_df.copy()

            for i in range(n - 1):
                target_up_df = target_up_df * up_df.shift(i + 1)
                target_dn_df = target_dn_df * dn_df.shift(i + 1)
            target_df = target_up_df.fillna(0).astype(int) - target_dn_df.fillna(0).astype(int)
            return target_df

        all_target_df = pd.DataFrame()
        for n in n_list:
            target_df = fun(raw_df, n)
            target_df = target_df * sector_df
            all_target_df = all_target_df.add(target_df, fill_value=0)
        return all_target_df

    @staticmethod
    def pnd_continue_ud_pct(raw_df, sector_df, n_list):
        all_target_df = pd.DataFrame()
        for n in n_list:
            target_df = raw_df.rolling(window=n).apply(lambda x: 1 if (x >= 0).all() and sum(x) > 0
            else (-1 if (x <= 0).all() and sum(x) < 0 else 0), raw=True)
            target_df = target_df * sector_df
            all_target_df = all_target_df.add(target_df, fill_value=0)
        return all_target_df

    @staticmethod
    def row_extre(raw_df, sector_df, percent):
        raw_df = raw_df * sector_df
        target_df = raw_df.rank(axis=1, pct=True)
        target_df[target_df >= 1 - percent] = 1
        target_df[target_df <= percent] = -1
        target_df[(target_df > percent) & (target_df < 1 - percent)] = 0
        return target_df

    @staticmethod
    def pnd_col_extre(raw_df, sector_df, window, percent, min_periods=1):
        dn_df = raw_df.rolling(window=window, min_periods=min_periods).quantile(percent)
        up_df = raw_df.rolling(window=window, min_periods=min_periods).quantile(1 - percent)
        dn_target = -(raw_df < dn_df).astype(int)
        up_target = (raw_df > up_df).astype(int)
        target_df = dn_target + up_target
        return target_df * sector_df

    @staticmethod
    def info_dict_fun(fun, raw_data_path, args, save_path, if_replace):
        info_dict = dict()
        info_dict['fun'] = fun
        info_dict['raw_data_path'] = raw_data_path
        info_dict['args'] = args
        info_dict['if_replace'] = if_replace
        pd.to_pickle(info_dict, save_path)

    @staticmethod
    def pnd_hl(high, low, close, sector_df, n):
        high_n = high.rolling(window=n, min_periods=1).max().shift(1)
        low_n = low.rolling(window=n, min_periods=1).min().shift(1)
        h_diff = (close - high_n)
        l_diff = (close - low_n)

        h_diff[h_diff > 0] = 1
        h_diff[h_diff <= 0] = 0

        l_diff[l_diff >= 0] = 0
        l_diff[l_diff < 0] = -1

        pos = h_diff + l_diff
        return pos * sector_df

    @staticmethod
    def pnd_volume(volume, sector_df, n):
        volume_roll_mean = bt.AZ_Rolling_mean(volume, n) * sector_df
        volume_df_count_down = 1 / (volume_roll_mean.replace(0, np.nan))
        return volume_df_count_down

    @staticmethod
    def pnd_volitality(adj_r, sector_df, n):
        vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5)
        vol_df[vol_df < 0.08] = 0.08
        return vol_df

    @staticmethod
    def pnd_volitality_count_down(adj_r, sector_df, n):
        vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5) * sector_df
        vol_df[vol_df < 0.08] = 0.08
        return 1 / vol_df.replace(0, np.nan)

    @staticmethod
    def pnd_evol(adj_r, sector_df, n):
        vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5)
        vol_df[vol_df < 0.08] = 0.08
        evol_df = bt.AZ_Rolling(vol_df, 30).apply(lambda x: 1 if x[-1] > 2 * x.mean() else 0, raw=True)
        return evol_df * sector_df

    def pnd_vol_continue_ud(self, adj_r, sector_df, n):
        vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5)
        vol_df[vol_df < 0.08] = 0.08
        vol_continue_ud_df = self.pnd_continue_ud(vol_df, sector_df, n_list=[3, 4, 5])
        return vol_continue_ud_df

    @staticmethod
    def pnnd_moment(df, sector_df, n_short=10, n_long=60):
        ma_long = df.rolling(window=n_long, min_periods=1).mean()
        ma_short = df.rolling(window=n_short, min_periods=1).mean()
        ma_dif = ma_short - ma_long
        ma_dif[ma_dif == 0] = 0
        ma_dif[ma_dif > 0] = 1
        ma_dif[ma_dif < 0] = -1
        return ma_dif * sector_df

    @staticmethod
    def p1d_jump_hl(close, open_, sector_df, split_float_list):
        target_df = pd.DataFrame()
        for split_float in split_float_list:
            jump_df = open_ / close.shift(1) - 1
            tmp_df = pd.DataFrame(index=jump_df.index, columns=jump_df.columns)
            tmp_df[(jump_df > 0.101) | (jump_df < -0.101)] = 0
            tmp_df[(split_float >= jump_df) & (jump_df >= -split_float)] = 0
            tmp_df[jump_df > split_float] = 1
            tmp_df[jump_df < -split_float] = -1
            target_df = target_df.add(tmp_df, fill_value=0)
        return target_df * sector_df

    def judge_save_fun(self, target_df, file_name, save_root_path, fun, raw_data_path, args, if_filter=True,
                       if_replace=False):
        factor_to_fun = '/mnt/mfs/dat_whs/data/factor_to_fun'
        if if_filter:
            target_df.to_pickle(os.path.join(save_root_path, file_name + '.pkl'))
            # 构建factor_to_fun的字典并存储
            self.info_dict_fun(fun, raw_data_path, args, os.path.join(factor_to_fun, file_name), if_replace)
            print(f'{file_name} success!')
        else:
            target_df.to_pickle(os.path.join(save_root_path, file_name + '.pkl'))
            # 构建factor_to_fun的字典并存储
            self.info_dict_fun(fun, raw_data_path, args, os.path.join(factor_to_fun, file_name), if_replace)
            print(f'{file_name} success!')


class FD:
    class funda_data_deal:
        class BaseDeal:
            @staticmethod
            def signal_mean_fun(signal_df):
                return signal_df.abs().sum(axis=1).replace(0, np.nan).dropna() / len(signal_df) > 0.1

            @staticmethod
            def pnd_continue_ud(raw_df, sector_df, n_list):
                def fun(df, n):
                    df_pct = df.diff()
                    up_df = (df_pct > 0)
                    dn_df = (df_pct < 0)
                    target_up_df = up_df.copy()
                    target_dn_df = dn_df.copy()

                    for i in range(n - 1):
                        target_up_df = target_up_df * up_df.shift(i + 1)
                        target_dn_df = target_dn_df * dn_df.shift(i + 1)
                    target_df = target_up_df.fillna(0).astype(int) - target_dn_df.fillna(0).astype(int)
                    return target_df

                all_target_df = pd.DataFrame()
                for n in n_list:
                    target_df = fun(raw_df, n)
                    target_df = target_df * sector_df
                    all_target_df = all_target_df.add(target_df, fill_value=0)
                return all_target_df

            @staticmethod
            def pnd_continue_ud_pct(raw_df, sector_df, n_list):
                all_target_df = pd.DataFrame()
                for n in n_list:
                    target_df = raw_df.rolling(window=n).apply(lambda x: 1 if (x >= 0).all() and sum(x) > 0
                    else (-1 if (x <= 0).all() and sum(x) < 0 else 0), raw=True)
                    target_df = target_df * sector_df
                    all_target_df = all_target_df.add(target_df, fill_value=0)
                return all_target_df

            @staticmethod
            def row_extre(raw_df, sector_df, percent):
                raw_df = raw_df * sector_df
                target_df = raw_df.rank(axis=1, pct=True)
                target_df[target_df >= 1 - percent] = 1
                target_df[target_df <= percent] = -1
                target_df[(target_df > percent) & (target_df < 1 - percent)] = 0
                return target_df

            @staticmethod
            def pnd_col_extre(raw_df, sector_df, window, percent, min_periods=1):
                dn_df = raw_df.rolling(window=window, min_periods=min_periods).quantile(percent)
                up_df = raw_df.rolling(window=window, min_periods=min_periods).quantile(1 - percent)
                dn_target = -(raw_df < dn_df).astype(int)
                up_target = (raw_df > up_df).astype(int)
                target_df = dn_target + up_target
                return target_df * sector_df

            @staticmethod
            def info_dict_fun(fun, raw_data_path, args, save_path, if_replace):
                info_dict = dict()
                info_dict['fun'] = fun
                info_dict['raw_data_path'] = raw_data_path
                info_dict['args'] = args
                info_dict['if_replace'] = if_replace
                pd.to_pickle(info_dict, save_path)

            @staticmethod
            def pnd_hl(high, low, close, sector_df, n):
                high_n = high.rolling(window=n, min_periods=1).max().shift(1)
                low_n = low.rolling(window=n, min_periods=1).min().shift(1)
                h_diff = (close - high_n)
                l_diff = (close - low_n)

                h_diff[h_diff > 0] = 1
                h_diff[h_diff <= 0] = 0

                l_diff[l_diff >= 0] = 0
                l_diff[l_diff < 0] = -1

                pos = h_diff + l_diff
                return pos * sector_df

            @staticmethod
            def pnd_volume(volume, sector_df, n):
                volume_roll_mean = bt.AZ_Rolling_mean(volume, n) * sector_df
                volume_df_count_down = 1 / (volume_roll_mean.replace(0, np.nan))
                return volume_df_count_down

            @staticmethod
            def pnd_volitality(adj_r, sector_df, n):
                vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5)
                vol_df[vol_df < 0.08] = 0.08
                return vol_df

            @staticmethod
            def pnd_volitality_count_down(adj_r, sector_df, n):
                vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5) * sector_df
                vol_df[vol_df < 0.08] = 0.08
                return 1 / vol_df.replace(0, np.nan)

            @staticmethod
            def pnd_evol(adj_r, sector_df, n):
                vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5)
                vol_df[vol_df < 0.08] = 0.08
                evol_df = bt.AZ_Rolling(vol_df, 30).apply(lambda x: 1 if x[-1] > 2 * x.mean() else 0, raw=True)
                return evol_df * sector_df

            def pnd_vol_continue_ud(self, adj_r, sector_df, n):
                vol_df = bt.AZ_Rolling(adj_r, n).std() * (250 ** 0.5)
                vol_df[vol_df < 0.08] = 0.08
                vol_continue_ud_df = self.pnd_continue_ud(vol_df, sector_df, n_list=[3, 4, 5])
                return vol_continue_ud_df

            @staticmethod
            def pnnd_moment(df, sector_df, n_short=10, n_long=60):
                ma_long = df.rolling(window=n_long, min_periods=1).mean()
                ma_short = df.rolling(window=n_short, min_periods=1).mean()
                ma_dif = ma_short - ma_long
                ma_dif[ma_dif == 0] = 0
                ma_dif[ma_dif > 0] = 1
                ma_dif[ma_dif < 0] = -1
                return ma_dif * sector_df

            @staticmethod
            def p1d_jump_hl(close, open_, sector_df, split_float_list):
                target_df = pd.DataFrame()
                for split_float in split_float_list:
                    jump_df = open_ / close.shift(1) - 1
                    tmp_df = pd.DataFrame(index=jump_df.index, columns=jump_df.columns)
                    tmp_df[(jump_df > 0.101) | (jump_df < -0.101)] = 0
                    tmp_df[(split_float >= jump_df) & (jump_df >= -split_float)] = 0
                    tmp_df[jump_df > split_float] = 1
                    tmp_df[jump_df < -split_float] = -1
                    target_df = target_df.add(tmp_df, fill_value=0)
                return target_df * sector_df

            def judge_save_fun(self, target_df, file_name, save_root_path, fun, raw_data_path, args, if_filter=True,
                               if_replace=False):
                factor_to_fun = '/mnt/mfs/dat_whs/data/factor_to_fun'
                if if_filter:
                    target_df.to_pickle(os.path.join(save_root_path, file_name + '.pkl'))
                    # 构建factor_to_fun的字典并存储
                    self.info_dict_fun(fun, raw_data_path, args, os.path.join(factor_to_fun, file_name), if_replace)
                    print(f'{file_name} success!')
                else:
                    target_df.to_pickle(os.path.join(save_root_path, file_name + '.pkl'))
                    # 构建factor_to_fun的字典并存储
                    self.info_dict_fun(fun, raw_data_path, args, os.path.join(factor_to_fun, file_name), if_replace)
                    print(f'{file_name} success!')

    class EM_Funda:
        class EM_Funda_Deal(BaseDeal):
            def dev_row_extre(self, data1, data2, sector_df, percent):
                target_df = self.row_extre(data1 / data2, sector_df, percent)
                return target_df

            def mix_factor_mcap(self, df, denom, grow_1, grow_2, std, sector_df, persent):
                factor_1 = (df / denom) * sector_df
                factor_2 = (((grow_1 + grow_2) / 2) / std) * sector_df
                a = bt.AZ_Row_zscore(factor_1)
                b = bt.AZ_Row_zscore(factor_2)
                c = a + b
                target_df = self.row_extre(c, sector_df, persent)
                return target_df

            def mix_factor_asset(self, df, asset, grow_1, grow_2, std, sector_df, persent):
                factor_1 = (df / asset) * sector_df
                factor_2 = (((grow_1 + grow_2) / 2) / std) * sector_df
                a = bt.AZ_Row_zscore(factor_1)
                b = bt.AZ_Row_zscore(factor_2)
                c = a + b
                target_df = self.row_extre(c, sector_df, persent)
                return target_df

            def mix_factor_mcap_intdebt(self, df, mcap, intdebt, grow_1, grow_2, std, sector_df, persent):
                factor_1 = (df / (mcap + intdebt)) * sector_df
                factor_2 = (((grow_1 + grow_2) / 2) / std) * sector_df
                a = bt.AZ_Row_zscore(factor_1, 5)
                b = bt.AZ_Row_zscore(factor_2, 5)
                c = a + b
                target_df = self.row_extre(c, sector_df, persent)
                return target_df

    class EM_Tab14:
        class EM_Tab14_Deal(BaseDeal):
            def return_pnd(self, aadj_r, sector_df, n, percent):
                return_pnd_df = bt.AZ_Rolling(aadj_r, n).sum()
                target_df = self.row_extre(return_pnd_df, sector_df, percent)
                return target_df

            def wgt_return_pnd(self, aadj_r, turnratio, sector_df, n, percent):
                aadj_r_c = (aadj_r * turnratio)
                wgt_return_pnd_df = bt.AZ_Rolling(aadj_r_c, n).sum()
                target_df = self.row_extre(wgt_return_pnd_df, sector_df, percent)
                return target_df

            def log_price(self, close, sector_df, percent):
                target_df = self.row_extre(np.log(close), sector_df, percent)
                return target_df

            def turn_pnd(self, turnratio, sector_df, n, percent):
                turnratio_mean = bt.AZ_Rolling(turnratio, n).mean()
                target_df = self.row_extre(turnratio_mean, sector_df, percent)
                return target_df * sector_df

            @staticmethod
            def bias_turn_pnd(turnratio, sector_df, n):
                bias_turnratio = bt.AZ_Rolling(turnratio, n).mean() / bt.AZ_Rolling(turnratio, 480).mean() - 1
                bias_turnratio_up = (bias_turnratio > 0).astype(int)
                bias_turnratio_dn = (bias_turnratio < 0).astype(int)
                target_df = bias_turnratio_up - bias_turnratio_dn
                return target_df * sector_df

            @staticmethod
            def MACD(close, sector_df, n_fast, n_slow):
                EMAfast = close.ewm(span=n_fast, min_periods=n_slow - 1).mean()
                EMAslow = close.ewm(span=n_slow, min_periods=n_slow - 1).mean()
                MACD = EMAfast - EMAslow
                MACDsign = MACD.ewm(span=9, min_periods=8).mean()
                MACDdiff = MACD - MACDsign
                target_df_up = (MACDdiff > 0).astype(int)
                target_df_dn = (MACDdiff < 0).astype(int)
                target_df = target_df_up - target_df_dn
                return target_df * sector_df

            @staticmethod
            def CCI(high, low, close, sector_df, n, limit_list):
                PP = (high + low + close) / 3
                CCI_signal = (PP - bt.AZ_Rolling(PP, n).mean()) / bt.AZ_Rolling(PP, n).std()
                all_target_df = pd.DataFrame()
                for limit in limit_list:
                    CCI_up = (CCI_signal >= limit).astype(int)
                    CCI_dn = -(CCI_signal <= -limit).astype(int)
                    CCI = CCI_up + CCI_dn
                    all_target_df = all_target_df.add(CCI, fill_value=0)
                return all_target_df * sector_df

    class Tech_Factor:
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
            @staticmethod
            def ADOSC(High, Low, Close, Volume, sector_df, fastperiod, slowperiod, limit_up_dn=0):
                real = AZ_Factor_Volume.ADOSC(High, Low, Close, Volume, fastperiod, slowperiod)
                target_df = (real > limit_up_dn).astype(int) - (real < -limit_up_dn).astype(int)
                return target_df * sector_df

        class FactorVolatility(BaseDeal):

            def ATR(self, High, Low, Close, sector_df, timeperiod, percent):
                real = AZ_Factor_Volatility.ATR(High, Low, Close, timeperiod)
                tmp_df = bt.AZ_Row_zscore(real)
                target_df = self.row_extre(tmp_df, sector_df, percent)
                return target_df

        class FactorAlpha(BaseDeal):
            @staticmethod
            def MACD_alpha(aadj_r, all_index_df, sector_df, fastperiod, slowperiod, signalperiod, index_weight):
                index_df = all_index_df[['000300', '000905']].mul(index_weight).sum(1).reindex(aadj_r.index)
                Close = aadj_r.sub(index_df, axis=0).cumsum()
                macd, macdsignal, macdhist = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                for i in Close.columns:
                    if Close[i].isnull().prod() != 1:
                        macd[i], macdsignal[i], macdhist[i] = ta.MACD(Close[i], fastperiod, slowperiod, signalperiod)
                    else:
                        macd[i], macdsignal[i], macdhist[i] = [np.nan] * len(Close[i]), \
                                                              [np.nan] * len(Close[i]), \
                                                              [np.nan] * len(Close[i])
                macdhist_copy = macdhist.copy()
                macdhist_copy.replace(np.nan, 0)
                macdhist_copy[macdhist > 0] = 1
                macdhist_copy[macdhist < 0] = 0
                target_df = macdhist_copy - macdhist_copy.shift(1)
                return target_df * sector_df

            @staticmethod
            def MA_LINE_alpha(aadj_r, all_index_df, sector_df, slowperiod, fastperiod, index_weight):
                index_df = all_index_df[['000300', '000905']].mul(index_weight).sum(1).reindex(aadj_r.index)
                Close = aadj_r.sub(index_df, axis=0).cumsum()
                slow_line = Close.rolling(slowperiod, min_periods=0).mean()
                fast_line = Close.rolling(fastperiod, min_periods=0).mean()
                MA_diff = fast_line - slow_line
                MA_diff_copy = MA_diff.copy()
                MA_diff_copy[MA_diff > 0] = 1
                MA_diff_copy[MA_diff < 0] = 0
                target_df = MA_diff_copy - MA_diff_copy.shift(1)
                return target_df * sector_df

            @staticmethod
            def BBANDS_alpha(aadj_r, all_index_df, sector_df, timeperiod, limit_up_down, index_weight):
                index_df = all_index_df[['000300', '000905']].mul(index_weight).sum(1).reindex(aadj_r.index)
                Close = aadj_r.sub(index_df, axis=0).cumsum()
                up_line, mid_line, down_line = AZ_Factor_Overlap.BBANDS(Close, timeperiod, nbdevup=limit_up_down,
                                                                        nbdevdn=limit_up_down, matype=0)
                target_df = Close.copy()
                target_df.iloc[:, :] = 0
                target_df[(Close <= up_line) & (Close >= down_line)] = 0
                target_df[Close > up_line] = 1
                target_df[Close < down_line] = -1
                return target_df * sector_df

            def pnd_continue_ud_alpha(self, aadj_r, all_index_df, sector_df, n_list, index_weight):
                index_df = all_index_df[['000300', '000905']].mul(index_weight).sum(1).reindex(aadj_r.index)
                Close = aadj_r.sub(index_df, axis=0).cumsum()
                target_df = self.pnd_continue_ud(Close, sector_df, n_list)
                return target_df

            def pnd_continue_pct_ud_alpha(self, aadj_r, all_index_df, sector_df, n_list, index_weight):
                index_df = all_index_df[['000300', '000905']].mul(index_weight).sum(1).reindex(aadj_r.index)
                aadj_r_alpha = aadj_r.sub(index_df, axis=0)
                target_df = self.pnd_continue_ud(aadj_r_alpha, sector_df, n_list)
                return target_df

            @staticmethod
            def RSI_alpha(aadj_r, all_index_df, sector_df, timeperiod, limit_up_dn, index_weight):
                index_df = all_index_df[['000300', '000905']].mul(index_weight).sum(1).reindex(aadj_r.index)
                Close = aadj_r.sub(index_df, axis=0).cumsum()
                real = AZ_Factor_Momentum.RSI(Close, timeperiod) - 50
                target_df = (real > limit_up_dn).astype(int) - (real < limit_up_dn).astype(int)
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

        class TechFactor(FactorVolume, FactorMomentum, FactorVolatility, BaseDeal):
            def __init__(self, root_path, sector_df, save_root_path):
                self.sector_df = sector_df
                xnms = sector_df.columns
                xinx = sector_df.index
                # load_path = '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14'
                self.aadj_p_path = root_path.EM_Funda.DERIVED_14 / 'aadj_p.csv'
                self.aadj_p = bt.AZ_Load_csv(self.aadj_p_path).reindex(columns=xnms, index=xinx)

                self.aadj_p_HIGH_path = root_path.EM_Funda.DERIVED_14 / 'aadj_p_HIGH.csv'
                self.aadj_p_HIGH = bt.AZ_Load_csv(self.aadj_p_HIGH_path).reindex(columns=xnms, index=xinx)

                self.aadj_p_LOW_path = root_path.EM_Funda.DERIVED_14 / 'aadj_p_LOW.csv'
                self.aadj_p_LOW = bt.AZ_Load_csv(self.aadj_p_LOW_path).reindex(columns=xnms, index=xinx)

                self.aadj_p_OPEN_path = root_path.EM_Funda.DERIVED_14 / 'aadj_p_OPEN.csv'
                self.aadj_p_OPEN = bt.AZ_Load_csv(self.aadj_p_OPEN_path).reindex(columns=xnms, index=xinx)

                self.TVOL_path = root_path.EM_Funda.TRAD_SK_DAILY_JC / 'TVOL.csv'
                self.TVOL = bt.AZ_Load_csv(self.TVOL_path).reindex(columns=xnms, index=xinx).replace(np.nan, 0)
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
                    target_df = self.MFI(self.aadj_p_HIGH, self.aadj_p_LOW, self.aadj_p, self.TVOL, self.sector_df, n,
                                         limit_up,
                                         limit_dn)
                    file_name = 'MFI_{}_{}_{}'.format(n, limit_up, limit_dn)
                    fun = 'Tech_Factor.FactorMomentum.MFI'
                    raw_data_path = (self.aadj_p_HIGH_path, self.aadj_p_LOW_path, self.aadj_p_path, self.TVOL_path)
                    args = (n, limit_up, limit_dn)
                    self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)

            def RSI_(self, n_list, limit_up_dn=30):
                for n in n_list:
                    target_df = self.RSI(self.aadj_p, self.sector_df, n, limit_up_dn=30)
                    file_name = 'RSI_{}_{}'.format(n, limit_up_dn)
                    fun = 'Tech_Factor.FactorMomentum.RSI'
                    raw_data_path = (self.aadj_p,)
                    args = (n, limit_up_dn)
                    self.judge_save_fun(target_df, file_name, self.save_root_path, fun, raw_data_path, args)

            def ADOSC_(self, long_short_list, limit_up_dn=0):
                for long, short in long_short_list:
                    target_df = self.ADOSC(self.aadj_p_HIGH, self.aadj_p_LOW, self.aadj_p, self.TVOL, self.sector_df,
                                           long, short, limit_up_dn=0)
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

    class EM_Funda_test:
        class EM_Funda_test_Deal(BaseDeal):
            pass


class SectorData(object):
    def __init__(self, root_path):
        self.root_path = root_path

    # 获取剔除新股的矩阵
    def get_new_stock_info(self, xnms, xinx):
        new_stock_data = bt.AZ_Load_csv(self.root_path.EM_Tab01.CDSY_SECUCODE / 'LISTSTATE.csv')
        new_stock_data.fillna(method='ffill', inplace=True)
        # 获取交易日信息
        return_df = bt.AZ_Load_csv(self.root_path.EM_Funda.DERIVED_14 / 'aadj_r.csv').astype(float)
        trade_time = return_df.index
        new_stock_data = new_stock_data.reindex(index=trade_time).fillna(method='ffill')
        target_df = new_stock_data.shift(40).notnull().astype(int)
        target_df = target_df.reindex(columns=xnms, index=xinx)
        return target_df

    # 获取剔除st股票的矩阵
    def get_st_stock_info(self, xnms, xinx):
        data = bt.AZ_Load_csv(self.root_path.EM_Tab01.CDSY_CHANGEINFO / 'CHANGEA.csv')
        data = data.reindex(columns=xnms, index=xinx)
        data.fillna(method='ffill', inplace=True)

        data = data.astype(str)
        target_df = data.applymap(lambda x: 0 if 'ST' in x or 'PT' in x else 1)
        return target_df

    # 读取 sector(行业 最大市值等)
    def load_sector_data(self, begin_date, end_date, sector_name):
        if sector_name.startswith('index'):
            index_name = sector_name.split('_')[-1]
            market_top_n = bt.AZ_Load_csv(self.root_path.EM_Funda.IDEX_YS_WEIGHT_A / f'SECURITYNAME_{index_name}.csv')
            # market_top_n[market_top_n != market_top_n] = 1.
            market_top_n = market_top_n.where(market_top_n != market_top_n, other=1)
        else:
            market_top_n = bt.AZ_Load_csv(self.root_path.EM_Funda.DERIVED_10 / (sector_name + '.csv'))

        market_top_n = market_top_n[(market_top_n.index >= begin_date) & (market_top_n.index < end_date)]
        market_top_n.dropna(how='all', axis='columns', inplace=True)
        xnms = market_top_n.columns
        xinx = market_top_n.index

        new_stock_df = self.get_new_stock_info(xnms, xinx)
        st_stock_df = self.get_st_stock_info(xnms, xinx)
        sector_df = market_top_n * new_stock_df * st_stock_df
        sector_df.replace(0, np.nan, inplace=True)
        return sector_df


def find_fun(fun_list):
    target_class = FD
    # print(fun_list)
    for a in fun_list[:-1]:
        target_class = getattr(target_class, a)
    target_fun = getattr(target_class(), fun_list[-1])
    return target_fun


def load_raw_data(root_path, raw_data_path, xnms, xinx, if_replace, target_date):
    raw_data_list = []
    for target_path in raw_data_path:

        if target_path == 'EM_Funda/INDEX_TD_DAILYSYS/CHG.csv':
            tmp_data = bt.AZ_Load_csv(os.path.join('/media/hdd1/DAT_EQT', target_path))
        else:
            tmp_data = bt.AZ_Load_csv(os.path.join('/media/hdd1/DAT_EQT', target_path)) \
                .reindex(columns=xnms, index=xinx)

        if tmp_data.index[-1] != target_date:
            send_email.send_email(target_path + ' Data Error!',
                                  ['whs@yingpei.com'],
                                  [],
                                  '[{}]'.format(target_date.strftime('%Y%m%d')))
        if if_replace:
            tmp_data = tmp_data.replace(0, np.nan)
        raw_data_list += [tmp_data]
    return raw_data_list


def create_data_fun(root_path, file_name, sector_df, xnms, xinx, target_date):
    info = info_dict[file_name]
    args = info['args']
    fun_list = info['fun'].split('.')
    # raw_data_path = [str(x)[16:] for x in info['raw_data_path'] if x.startswith('/mnt/mfs/DAT_EQT') else x]
    raw_data_path = list(map(lambda x: str(x)[17:] if str(x).startswith('/mnt/mfs/DAT_EQT') else x,
                             info['raw_data_path']))
    if_replace = info['if_replace']
    raw_data_list = load_raw_data(root_path, raw_data_path, xnms, xinx, if_replace, target_date)

    target_fun = find_fun(fun_list)
    target_df = target_fun(*raw_data_list, sector_df, *args)
    if (target_df.iloc[-1] != 0).sum() == 0:
        send_email.send_email(file_name, ['whs@yingpei.com'], [], 'Data Update Warning')
    return target_df


class PrecalcDataCreate:
    def __init__(self, sector_name, file_name_list):
        self.sector_name = sector_name
        self.file_name_list = file_name_list
        mode = 'pro'

        begin_date = pd.to_datetime('20120101')
        end_date = datetime.now()

        self.save_root_path = f'/media/hdd1/DAT_PreCalc/PreCalc_whs/{sector_name}'
        bt.AZ_Path_create(self.save_root_path)

        self.root_path = pt._BinFiles(mode)
        sector_data_class = SectorData(self.root_path)
        self.sector_df = sector_data_class.load_sector_data(begin_date, end_date, sector_name)

        self.target_date = self.sector_df.index[-1]

        self.xnms = self.sector_df.columns
        self.xinx = self.sector_df.index

    def data_create(self):
        # try:
        for file_name in self.file_name_list:
            file_save_path = os.path.join(self.save_root_path, f'{file_name}.pkl')
            # if os.path.exists(file_save_path):
            #     cut_date = self.xinx[-5]
            #     create_data = pd.read_pickle(file_save_path)
            #     create_data = create_data[(create_data.index <= cut_date)]
            #     part_create_data = create_data_fun(self.root_path, file_name, self.sector_df, self.xnms,
            #                                        self.xinx[-300:], self.target_date)
            #     part_create_data = part_create_data[(part_create_data.index > cut_date)]
            #     create_data = create_data.append(part_create_data, sort=False)
            #
            # else:
            print(file_name)
            create_data = create_data_fun(self.root_path, file_name, self.sector_df, self.xnms, self.xinx,
                                          self.target_date)

            create_data.to_pickle(file_save_path)
        # except:
        #     send_email.send_email(','.join(self.file_name_list), ['whs@yingpei.com'], [],
        #                           f'[pro update]{self.sector_name}')
        #     print('error', file_name)


class SectorSplit:
    def __init__(self, sector_name):
        begin_date = pd.to_datetime('20050505')
        end_date = datetime.now()
        # sector_name = 'market_top_2000'
        self.sector_name = sector_name
        market_top_n = bt.AZ_Load_csv(f'/media/hdd1/DAT_EQT/EM_Funda/DERIVED_10/{sector_name}.csv')
        market_top_n = market_top_n[(market_top_n.index >= begin_date) & (market_top_n.index < end_date)]
        self.sector_df = market_top_n
        self.xinx = self.sector_df.index
        self.xnms = self.sector_df.columns

    def industry(self, file_list):
        industry_df_sum = pd.DataFrame()
        for file_name in file_list:
            industry_df = bt.AZ_Load_csv(f'/media/hdd1/DAT_EQT/EM_Funda/LICO_IM_INCHG/Global_Level1_{file_name}.csv') \
                .reindex(index=self.xinx, columns=self.xnms)
            industry_df_sum = industry_df_sum.add(industry_df, fill_value=0)
        industry_df_sum = industry_df_sum.reindex(index=self.xinx, columns=self.xnms)

        industry_df_sum = self.sector_df.mul(industry_df_sum, fill_value=0).replace(0, np.nan) \
            .dropna(how='all', axis='columns')
        industry_df_sum.to_csv('/media/hdd1/DAT_EQT/EM_Funda/DERIVED_10/{}_industry_{}.csv'
                               .format(self.sector_name, '_'.join([str(x) for x in file_list])), sep='|')
        return industry_df_sum


def create_sector():
    sector_split = SectorSplit('market_top_2000')
    for file_list in [[10, 15], [20, 25, 30, 35], [40], [45, 50], [55]]:
        sector_split.industry(file_list)

    sector_split = SectorSplit('market_top_1000')
    for file_list in [[10, 15], [20, 25, 30, 35], [40], [45, 50], [55]]:
        sector_split.industry(file_list)

    sector_split = SectorSplit('market_top_800plus')
    for file_list in [[10, 15], [20, 25, 30, 35], [40], [45, 50], [55]]:
        sector_split.industry(file_list)

    sector_split = SectorSplit('market_top_300')
    for file_list in [[10, 15], [20, 25, 30, 35], [40], [45, 50], [55]]:
        sector_split.industry(file_list)

    sector_split = SectorSplit('market_top_300plus')
    for file_list in [[10, 15], [20, 25, 30, 35], [40], [45, 50], [55]]:
        sector_split.industry(file_list)

    sector_split = SectorSplit('market_top_300to800plus')
    for file_list in [[10, 15], [20, 25, 30, 35], [40], [45, 50], [55]]:
        sector_split.industry(file_list)


def main(config_name_dict, config_info):
    config_path = '/media/hdd1/DAT_PreCalc/PreCalc_whs/config_file'
    up_date_dict = OrderedDict()
    for config_name in config_name_dict.keys():
        # print(config_name)
        col_list = config_name_dict[config_name]
        config1 = pd.read_pickle(f'{config_path}/{config_name}.pkl')
        factor_info = config1['factor_info']
        sector_name = config1['sector_name']
        if sector_name in up_date_dict.keys():
            up_date_dict[sector_name] = up_date_dict[sector_name] | set(factor_info[col_list].values.ravel())
        else:
            up_date_dict[sector_name] = set(factor_info[col_list].values.ravel())
    pool = Pool(4)
    for sector_name in list(up_date_dict.keys()) + list(config_info.keys()):
        if sector_name in config_info.keys() and sector_name in up_date_dict.keys():
            file_name_list = sorted(list(up_date_dict[sector_name] | set(config_info[sector_name])))
        elif sector_name in config_info.keys() and sector_name not in up_date_dict.keys():
            file_name_list = sorted(list(config_info[sector_name]))
        else:
            file_name_list = sorted(list(up_date_dict[sector_name]))

        precalc_data_create = PrecalcDataCreate(sector_name, file_name_list)
        precalc_data_create.data_create()
    #     pool.apply_async(precalc_data_create.data_create)
    # pool.close()
    # pool.join()


info_dict = OrderedDict([
    ('ADOSC_20_60_0',
     {'args': (20, 60, 0),
      'fun': 'Tech_Factor.FactorVolume.ADOSC',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('ADOSC_40_100_0',
     {'args': (40, 100, 0),
      'fun': 'Tech_Factor.FactorVolume.ADOSC',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('ADOSC_5_10_0',
     {'args': (5, 10, 0),
      'fun': 'Tech_Factor.FactorVolume.ADOSC',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('ADOSC_60_120_0',
     {'args': (60, 120, 0),
      'fun': 'Tech_Factor.FactorVolume.ADOSC',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('ADOSC_60_160_0',
     {'args': (60, 160, 0),
      'fun': 'Tech_Factor.FactorVolume.ADOSC',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('ADX_100_20_10',
     {'args': (100, 20, 10),
      'fun': 'Tech_Factor.FactorMomentum.ADX',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('ADX_100_40_20',
     {'args': (100, 40, 20),
      'fun': 'Tech_Factor.FactorMomentum.ADX',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('ADX_100_80',
     {'args': (100, 80),
      'fun': 'Tech_Factor.FactorMomentum.AROONSC',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv']}),
    ('ADX_10_20_10',
     {'args': (10, 20, 10),
      'fun': 'Tech_Factor.FactorMomentum.ADX',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('ADX_140_20_10',
     {'args': (140, 20, 10),
      'fun': 'Tech_Factor.FactorMomentum.ADX',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('ADX_140_40_20',
     {'args': (140, 40, 20),
      'fun': 'Tech_Factor.FactorMomentum.ADX',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('ADX_200_20_10',
     {'args': (200, 20, 10),
      'fun': 'Tech_Factor.FactorMomentum.ADX',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('ADX_200_40_20',
     {'args': (200, 40, 20),
      'fun': 'Tech_Factor.FactorMomentum.ADX',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('ADX_20_20_10',
     {'args': (20, 20, 10),
      'fun': 'Tech_Factor.FactorMomentum.ADX',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('ADX_40_20_10',
     {'args': (40, 20, 10),
      'fun': 'Tech_Factor.FactorMomentum.ADX',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('ADX_40_40_20',
     {'args': (40, 40, 20),
      'fun': 'Tech_Factor.FactorMomentum.ADX',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('ADX_40_80',
     {'args': (40, 80),
      'fun': 'Tech_Factor.FactorMomentum.AROONSC',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv']}),
    ('AROON_100_80',
     {'args': (100, 80),
      'fun': 'Tech_Factor.FactorMomentum.AROON',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv']}),
    ('AROON_10_80',
     {'args': (10, 80),
      'fun': 'Tech_Factor.FactorMomentum.AROON',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv']}),
    ('AROON_140_80',
     {'args': (140, 80),
      'fun': 'Tech_Factor.FactorMomentum.AROON',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv']}),
    ('AROON_200_80',
     {'args': (200, 80),
      'fun': 'Tech_Factor.FactorMomentum.AROON',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv']}),
    ('AROON_20_80',
     {'args': (20, 80),
      'fun': 'Tech_Factor.FactorMomentum.AROON',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv']}),
    ('AROON_40_80',
     {'args': (40, 80),
      'fun': 'Tech_Factor.FactorMomentum.AROON',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv']}),
    ('ATR_100_0.2',
     {'args': (100, 0.2),
      'fun': 'Tech_Factor.FactorVolatility.ATR',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('ATR_10_0.2',
     {'args': (10, 0.2),
      'fun': 'Tech_Factor.FactorVolatility.ATR',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('ATR_140_0.2',
     {'args': (140, 0.2),
      'fun': 'Tech_Factor.FactorVolatility.ATR',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('ATR_200_0.2',
     {'args': (200, 0.2),
      'fun': 'Tech_Factor.FactorVolatility.ATR',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('ATR_20_0.2',
     {'args': (20, 0.2),
      'fun': 'Tech_Factor.FactorVolatility.ATR',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('ATR_40_0.2',
     {'args': (40, 0.2),
      'fun': 'Tech_Factor.FactorVolatility.ATR',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('BBANDS_100_1',
     {'args': (100, 1),
      'fun': 'Tech_Factor.FactorOverlap.BBANDS',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('BBANDS_100_1.5',
     {'args': (100, 1.5),
      'fun': 'Tech_Factor.FactorOverlap.BBANDS',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('BBANDS_100_2',
     {'args': (100, 2),
      'fun': 'Tech_Factor.FactorOverlap.BBANDS',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('BBANDS_10_1',
     {'args': (10, 1),
      'fun': 'Tech_Factor.FactorOverlap.BBANDS',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('BBANDS_10_1.5',
     {'args': (10, 1.5),
      'fun': 'Tech_Factor.FactorOverlap.BBANDS',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('BBANDS_10_2',
     {'args': (10, 2),
      'fun': 'Tech_Factor.FactorOverlap.BBANDS',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('BBANDS_140_1',
     {'args': (140, 1),
      'fun': 'Tech_Factor.FactorOverlap.BBANDS',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('BBANDS_140_1.5',
     {'args': (140, 1.5),
      'fun': 'Tech_Factor.FactorOverlap.BBANDS',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('BBANDS_140_2',
     {'args': (140, 2),
      'fun': 'Tech_Factor.FactorOverlap.BBANDS',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('BBANDS_200_1',
     {'args': (200, 1),
      'fun': 'Tech_Factor.FactorOverlap.BBANDS',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('BBANDS_200_1.5',
     {'args': (200, 1.5),
      'fun': 'Tech_Factor.FactorOverlap.BBANDS',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('BBANDS_200_2',
     {'args': (200, 2),
      'fun': 'Tech_Factor.FactorOverlap.BBANDS',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('BBANDS_20_1',
     {'args': (20, 1),
      'fun': 'Tech_Factor.FactorOverlap.BBANDS',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('BBANDS_20_1.5',
     {'args': (20, 1.5),
      'fun': 'Tech_Factor.FactorOverlap.BBANDS',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('BBANDS_20_2',
     {'args': (20, 2),
      'fun': 'Tech_Factor.FactorOverlap.BBANDS',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('BBANDS_40_1',
     {'args': (40, 1),
      'fun': 'Tech_Factor.FactorOverlap.BBANDS',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('BBANDS_40_1.5',
     {'args': (40, 1.5),
      'fun': 'Tech_Factor.FactorOverlap.BBANDS',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('BBANDS_40_2',
     {'args': (40, 2),
      'fun': 'Tech_Factor.FactorOverlap.BBANDS',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('BBANDS_alpha_100_1.5_0.5_0.5',
     {'args': (100, 1.5, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_100_1.5_0_1',
     {'args': (100, 1.5, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_100_1.5_1_0',
     {'args': (100, 1.5, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_100_1_0.5_0.5',
     {'args': (100, 1, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_100_1_0_1',
     {'args': (100, 1, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_100_1_1_0',
     {'args': (100, 1, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_100_2_0.5_0.5',
     {'args': (100, 2, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_100_2_0_1',
     {'args': (100, 2, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_100_2_1_0',
     {'args': (100, 2, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_10_1.5_0.5_0.5',
     {'args': (10, 1.5, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_10_1.5_0_1',
     {'args': (10, 1.5, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_10_1.5_1_0',
     {'args': (10, 1.5, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_10_1_0.5_0.5',
     {'args': (10, 1, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_10_1_0_1',
     {'args': (10, 1, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_10_1_1_0',
     {'args': (10, 1, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_10_2_0.5_0.5',
     {'args': (10, 2, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_10_2_0_1',
     {'args': (10, 2, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_10_2_1_0',
     {'args': (10, 2, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_140_1.5_0.5_0.5',
     {'args': (140, 1.5, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_140_1.5_0_1',
     {'args': (140, 1.5, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_140_1.5_1_0',
     {'args': (140, 1.5, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_140_1_0.5_0.5',
     {'args': (140, 1, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_140_1_0_1',
     {'args': (140, 1, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_140_1_1_0',
     {'args': (140, 1, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_140_2_0.5_0.5',
     {'args': (140, 2, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_140_2_0_1',
     {'args': (140, 2, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_140_2_1_0',
     {'args': (140, 2, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_200_1.5_0.5_0.5',
     {'args': (200, 1.5, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_200_1.5_0_1',
     {'args': (200, 1.5, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_200_1.5_1_0',
     {'args': (200, 1.5, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_200_1_0.5_0.5',
     {'args': (200, 1, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_200_1_0_1',
     {'args': (200, 1, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_200_1_1_0',
     {'args': (200, 1, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_200_2_0.5_0.5',
     {'args': (200, 2, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_200_2_0_1',
     {'args': (200, 2, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_200_2_1_0',
     {'args': (200, 2, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_20_1.5_0.5_0.5',
     {'args': (20, 1.5, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_20_1.5_0_1',
     {'args': (20, 1.5, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_20_1.5_1_0',
     {'args': (20, 1.5, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_20_1_0.5_0.5',
     {'args': (20, 1, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_20_1_0_1',
     {'args': (20, 1, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_20_1_1_0',
     {'args': (20, 1, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_20_2_0.5_0.5',
     {'args': (20, 2, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_20_2_0_1',
     {'args': (20, 2, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_20_2_1_0',
     {'args': (20, 2, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_40_1.5_0.5_0.5',
     {'args': (40, 1.5, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_40_1.5_0_1',
     {'args': (40, 1.5, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_40_1.5_1_0',
     {'args': (40, 1.5, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_40_1_0.5_0.5',
     {'args': (40, 1, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_40_1_0_1',
     {'args': (40, 1, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_40_1_1_0',
     {'args': (40, 1, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_40_2_0.5_0.5',
     {'args': (40, 2, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_40_2_0_1',
     {'args': (40, 2, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('BBANDS_alpha_40_2_1_0',
     {'args': (40, 2, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.BBANDS_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('CCI_p120d_limit_12',
     {'args': (120, [1, 2]),
      'fun': 'EM_Tab14.EM_Tab14_Deal.CCI',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('CCI_p150d_limit_12',
     {'args': (150, [1, 2]),
      'fun': 'EM_Tab14.EM_Tab14_Deal.CCI',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('CCI_p20d_limit_12',
     {'args': (20, [1, 2]),
      'fun': 'EM_Tab14.EM_Tab14_Deal.CCI',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('CCI_p60d_limit_12',
     {'args': (60, [1, 2]),
      'fun': 'EM_Tab14.EM_Tab14_Deal.CCI',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('CMO_100_0',
     {'args': (100, 0),
      'fun': 'Tech_Factor.FactorMomentum.CMO',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('CMO_10_0',
     {'args': (10, 0),
      'fun': 'Tech_Factor.FactorMomentum.CMO',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('CMO_140_0',
     {'args': (140, 0),
      'fun': 'Tech_Factor.FactorMomentum.CMO',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('CMO_200_0',
     {'args': (200, 0),
      'fun': 'Tech_Factor.FactorMomentum.CMO',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('CMO_20_0',
     {'args': (20, 0),
      'fun': 'Tech_Factor.FactorMomentum.CMO',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('CMO_40_0',
     {'args': (40, 0),
      'fun': 'Tech_Factor.FactorMomentum.CMO',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('EBIT_and_asset_QYOY_Y3YGR_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.mix_factor_asset',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_EBIT_TTM_First.csv',
                        'EM_Funda/daily/R_SUMASSET_First.csv',
                        'EM_Funda/daily/R_EBIT_TTM_QYOY.csv',
                        'EM_Funda/daily/R_EBIT_TTM_Y3YGR.csv',
                        'EM_Funda/daily/R_EBIT_TTM_QSD4Y.csv']}),
    ('EBIT_and_asset_Y3YGR_Y5YGR_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.mix_factor_asset',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_EBIT_TTM_First.csv',
                        'EM_Funda/daily/R_SUMASSET_First.csv',
                        'EM_Funda/daily/R_EBIT_TTM_Y3YGR.csv',
                        'EM_Funda/daily/R_EBIT_TTM_Y5YGR.csv',
                        'EM_Funda/daily/R_EBIT_TTM_QSD4Y.csv']}),
    ('EBIT_and_mcap_QYOY_Y3YGR_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.mix_factor_mcap',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_EBIT_TTM_First.csv',
                        'EM_Funda/LICO_YS_STOCKVALUE/AmarketCap.csv',
                        'EM_Funda/daily/R_EBIT_TTM_QYOY.csv',
                        'EM_Funda/daily/R_EBIT_TTM_Y3YGR.csv',
                        'EM_Funda/daily/R_EBIT_TTM_QSD4Y.csv']}),
    ('EBIT_and_mcap_Y3YGR_Y5YGR_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.mix_factor_mcap',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_EBIT_TTM_First.csv',
                        'EM_Funda/LICO_YS_STOCKVALUE/AmarketCap.csv',
                        'EM_Funda/daily/R_EBIT_TTM_Y3YGR.csv',
                        'EM_Funda/daily/R_EBIT_TTM_Y5YGR.csv',
                        'EM_Funda/daily/R_EBIT_TTM_QSD4Y.csv']}),
    ('EBIT_and_mcap_intdebt_QYOY_Y3YGR_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.mix_factor_mcap_intdebt',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_EBIT_TTM_First.csv',
                        'EM_Funda/LICO_YS_STOCKVALUE/AmarketCap.csv',
                        'EM_Funda/daily/R_IntDebt_First.csv',
                        'EM_Funda/daily/R_EBIT_TTM_QYOY.csv',
                        'EM_Funda/daily/R_EBIT_TTM_Y3YGR.csv',
                        'EM_Funda/daily/R_EBIT_TTM_QSD4Y.csv']}),
    ('EBIT_and_mcap_intdebt_Y3YGR_Y5YGR_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.mix_factor_mcap_intdebt',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_EBIT_TTM_First.csv',
                        'EM_Funda/LICO_YS_STOCKVALUE/AmarketCap.csv',
                        'EM_Funda/daily/R_IntDebt_First.csv',
                        'EM_Funda/daily/R_EBIT_TTM_Y3YGR.csv',
                        'EM_Funda/daily/R_EBIT_TTM_Y5YGR.csv',
                        'EM_Funda/daily/R_EBIT_TTM_QSD4Y.csv']}),
    ('MACD_10_30',
     {'args': (10, 30),
      'fun': 'EM_Tab14.EM_Tab14_Deal.MACD',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('MACD_12_26_9',
     {'args': (12, 26, 9),
      'fun': 'Tech_Factor.FactorMomentum.MACD',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('MACD_20_100',
     {'args': (20, 100),
      'fun': 'EM_Tab14.EM_Tab14_Deal.MACD',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('MACD_20_200',
     {'args': (20, 200),
      'fun': 'EM_Tab14.EM_Tab14_Deal.MACD',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('MACD_20_60_18',
     {'args': (20, 60, 18),
      'fun': 'Tech_Factor.FactorMomentum.MACD',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('MACD_40_160',
     {'args': (40, 160),
      'fun': 'EM_Tab14.EM_Tab14_Deal.MACD',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('MACD_40_200',
     {'args': (40, 200),
      'fun': 'EM_Tab14.EM_Tab14_Deal.MACD',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('MACD_alpha_12_26_9_0.5_0.5',
     {'args': (12, 26, 9, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.MACD_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('MACD_alpha_12_26_9_0_1',
     {'args': (12, 26, 9, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.MACD_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('MACD_alpha_12_26_9_1_0',
     {'args': (12, 26, 9, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.MACD_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('MACD_alpha_20_60_18_0.5_0.5',
     {'args': (20, 60, 18, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.MACD_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('MACD_alpha_20_60_18_0_1',
     {'args': (20, 60, 18, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.MACD_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('MACD_alpha_20_60_18_1_0',
     {'args': (20, 60, 18, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.MACD_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('MA_LINE_100_40',
     {'args': (100, 40),
      'fun': 'Tech_Factor.FactorMomentum.MA_LINE',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('MA_LINE_10_5',
     {'args': (10, 5),
      'fun': 'Tech_Factor.FactorMomentum.MA_LINE',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('MA_LINE_120_60',
     {'args': (120, 60),
      'fun': 'Tech_Factor.FactorMomentum.MA_LINE',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('MA_LINE_160_60',
     {'args': (160, 60),
      'fun': 'Tech_Factor.FactorMomentum.MA_LINE',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('MA_LINE_60_20',
     {'args': (60, 20),
      'fun': 'Tech_Factor.FactorMomentum.MA_LINE',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('MA_LINE_alpha_100_40_0.5_0.5',
     {'args': (100, 40, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.MA_LINE_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('MA_LINE_alpha_100_40_0_1',
     {'args': (100, 40, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.MA_LINE_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('MA_LINE_alpha_100_40_1_0',
     {'args': (100, 40, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.MA_LINE_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('MA_LINE_alpha_10_5_0.5_0.5',
     {'args': (10, 5, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.MA_LINE_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('MA_LINE_alpha_10_5_0_1',
     {'args': (10, 5, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.MA_LINE_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('MA_LINE_alpha_10_5_1_0',
     {'args': (10, 5, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.MA_LINE_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('MA_LINE_alpha_120_60_0.5_0.5',
     {'args': (120, 60, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.MA_LINE_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('MA_LINE_alpha_120_60_0_1',
     {'args': (120, 60, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.MA_LINE_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('MA_LINE_alpha_120_60_1_0',
     {'args': (120, 60, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.MA_LINE_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('MA_LINE_alpha_160_60_0.5_0.5',
     {'args': (160, 60, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.MA_LINE_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('MA_LINE_alpha_160_60_0_1',
     {'args': (160, 60, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.MA_LINE_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('MA_LINE_alpha_160_60_1_0',
     {'args': (160, 60, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.MA_LINE_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('MA_LINE_alpha_60_20_0.5_0.5',
     {'args': (60, 20, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.MA_LINE_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('MA_LINE_alpha_60_20_0_1',
     {'args': (60, 20, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.MA_LINE_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('MA_LINE_alpha_60_20_1_0',
     {'args': (60, 20, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.MA_LINE_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('MFI_100_70_30',
     {'args': (100, 70, 30),
      'fun': 'Tech_Factor.FactorMomentum.MFI',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('MFI_100_80_20',
     {'args': (100, 80, 20),
      'fun': 'Tech_Factor.FactorMomentum.MFI',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('MFI_10_70_30',
     {'args': (10, 70, 30),
      'fun': 'Tech_Factor.FactorMomentum.MFI',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('MFI_140_70_30',
     {'args': (140, 70, 30),
      'fun': 'Tech_Factor.FactorMomentum.MFI',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('MFI_140_80_20',
     {'args': (140, 80, 20),
      'fun': 'Tech_Factor.FactorMomentum.MFI',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('MFI_200_70_30',
     {'args': (200, 70, 30),
      'fun': 'Tech_Factor.FactorMomentum.MFI',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('MFI_200_80_20',
     {'args': (200, 80, 20),
      'fun': 'Tech_Factor.FactorMomentum.MFI',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('MFI_20_70_30',
     {'args': (20, 70, 30),
      'fun': 'Tech_Factor.FactorMomentum.MFI',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('MFI_40_70_30',
     {'args': (40, 70, 30),
      'fun': 'Tech_Factor.FactorMomentum.MFI',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('MFI_40_80_20',
     {'args': (40, 80, 20),
      'fun': 'Tech_Factor.FactorMomentum.MFI',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('NetProfit_and_asset_QYOY_Y3YGR_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.mix_factor_asset',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NetProfit_TTM_First.csv',
                        'EM_Funda/daily/R_SUMASSET_First.csv',
                        'EM_Funda/daily/R_NetProfit_TTM_QYOY.csv',
                        'EM_Funda/daily/R_NetProfit_TTM_Y3YGR.csv',
                        'EM_Funda/daily/R_NetProfit_TTM_QSD4Y.csv']}),
    ('NetProfit_and_asset_Y3YGR_Y5YGR_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.mix_factor_asset',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NetProfit_TTM_First.csv',
                        'EM_Funda/daily/R_SUMASSET_First.csv',
                        'EM_Funda/daily/R_NetProfit_TTM_Y3YGR.csv',
                        'EM_Funda/daily/R_NetProfit_TTM_Y5YGR.csv',
                        'EM_Funda/daily/R_NetProfit_TTM_QSD4Y.csv']}),
    ('NetProfit_and_mcap_QYOY_Y3YGR_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.mix_factor_mcap',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NetProfit_TTM_First.csv',
                        'EM_Funda/LICO_YS_STOCKVALUE/AmarketCap.csv',
                        'EM_Funda/daily/R_NetProfit_TTM_QYOY.csv',
                        'EM_Funda/daily/R_NetProfit_TTM_Y3YGR.csv',
                        'EM_Funda/daily/R_NetProfit_TTM_QSD4Y.csv']}),
    ('NetProfit_and_mcap_Y3YGR_Y5YGR_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.mix_factor_mcap',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NetProfit_TTM_First.csv',
                        'EM_Funda/LICO_YS_STOCKVALUE/AmarketCap.csv',
                        'EM_Funda/daily/R_NetProfit_TTM_Y3YGR.csv',
                        'EM_Funda/daily/R_NetProfit_TTM_Y5YGR.csv',
                        'EM_Funda/daily/R_NetProfit_TTM_QSD4Y.csv']}),
    ('NetProfit_and_mcap_intdebt_QYOY_Y3YGR_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.mix_factor_mcap_intdebt',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NetProfit_TTM_First.csv',
                        'EM_Funda/LICO_YS_STOCKVALUE/AmarketCap.csv',
                        'EM_Funda/daily/R_IntDebt_First.csv',
                        'EM_Funda/daily/R_NetProfit_TTM_QYOY.csv',
                        'EM_Funda/daily/R_NetProfit_TTM_Y3YGR.csv',
                        'EM_Funda/daily/R_NetProfit_TTM_QSD4Y.csv']}),
    ('NetProfit_and_mcap_intdebt_Y3YGR_Y5YGR_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.mix_factor_mcap_intdebt',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NetProfit_TTM_First.csv',
                        'EM_Funda/LICO_YS_STOCKVALUE/AmarketCap.csv',
                        'EM_Funda/daily/R_IntDebt_First.csv',
                        'EM_Funda/daily/R_NetProfit_TTM_Y3YGR.csv',
                        'EM_Funda/daily/R_NetProfit_TTM_Y5YGR.csv',
                        'EM_Funda/daily/R_NetProfit_TTM_QSD4Y.csv']}),
    ('OPCF_and_asset_QYOY_Y3YGR_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.mix_factor_asset',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OPCF_TTM_First.csv',
                        'EM_Funda/daily/R_SUMASSET_First.csv',
                        'EM_Funda/daily/R_OPCF_TTM_QYOY.csv',
                        'EM_Funda/daily/R_OPCF_TTM_Y3YGR.csv',
                        'EM_Funda/daily/R_OPCF_TTM_QSD4Y.csv']}),
    ('OPCF_and_asset_Y3YGR_Y5YGR_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.mix_factor_asset',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OPCF_TTM_First.csv',
                        'EM_Funda/daily/R_SUMASSET_First.csv',
                        'EM_Funda/daily/R_OPCF_TTM_Y3YGR.csv',
                        'EM_Funda/daily/R_OPCF_TTM_Y5YGR.csv',
                        'EM_Funda/daily/R_OPCF_TTM_QSD4Y.csv']}),
    ('OPCF_and_mcap_QYOY_Y3YGR_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.mix_factor_mcap',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OPCF_TTM_First.csv',
                        'EM_Funda/LICO_YS_STOCKVALUE/AmarketCap.csv',
                        'EM_Funda/daily/R_OPCF_TTM_QYOY.csv',
                        'EM_Funda/daily/R_OPCF_TTM_Y3YGR.csv',
                        'EM_Funda/daily/R_OPCF_TTM_QSD4Y.csv']}),
    ('OPCF_and_mcap_Y3YGR_Y5YGR_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.mix_factor_mcap',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OPCF_TTM_First.csv',
                        'EM_Funda/LICO_YS_STOCKVALUE/AmarketCap.csv',
                        'EM_Funda/daily/R_OPCF_TTM_Y3YGR.csv',
                        'EM_Funda/daily/R_OPCF_TTM_Y5YGR.csv',
                        'EM_Funda/daily/R_OPCF_TTM_QSD4Y.csv']}),
    ('OPCF_and_mcap_intdebt_QYOY_Y3YGR_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.mix_factor_mcap_intdebt',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OPCF_TTM_First.csv',
                        'EM_Funda/LICO_YS_STOCKVALUE/AmarketCap.csv',
                        'EM_Funda/daily/R_IntDebt_First.csv',
                        'EM_Funda/daily/R_OPCF_TTM_QYOY.csv',
                        'EM_Funda/daily/R_OPCF_TTM_Y3YGR.csv',
                        'EM_Funda/daily/R_OPCF_TTM_QSD4Y.csv']}),
    ('OPCF_and_mcap_intdebt_Y3YGR_Y5YGR_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.mix_factor_mcap_intdebt',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OPCF_TTM_First.csv',
                        'EM_Funda/LICO_YS_STOCKVALUE/AmarketCap.csv',
                        'EM_Funda/daily/R_IntDebt_First.csv',
                        'EM_Funda/daily/R_OPCF_TTM_Y3YGR.csv',
                        'EM_Funda/daily/R_OPCF_TTM_Y5YGR.csv',
                        'EM_Funda/daily/R_OPCF_TTM_QSD4Y.csv']}),
    ('PBLast_p10d_col_extre_0.2',
     {'args': (10, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_REVALUATION/PBLast.csv']}),
    ('PBLast_p120d_col_extre_0.2',
     {'args': (120, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_REVALUATION/PBLast.csv']}),
    ('PBLast_p20d_col_extre_0.2',
     {'args': (20, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_REVALUATION/PBLast.csv']}),
    ('PBLast_p345d_continue_ud',
     {'args': ([3, 4, 5],),
      'fun': 'funda_data_deal.BaseDeal.pnd_continue_ud',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_REVALUATION/PBLast.csv']}),
    ('PBLast_p60d_col_extre_0.2',
     {'args': (60, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_REVALUATION/PBLast.csv']}),
    ('PBLast_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_REVALUATION/PBLast.csv']}),
    ('PE_TTM_p10d_col_extre_0.2',
     {'args': (10, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_REVALUATION/PE_TTM.csv']}),
    ('PE_TTM_p120d_col_extre_0.2',
     {'args': (120, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_REVALUATION/PE_TTM.csv']}),
    ('PE_TTM_p20d_col_extre_0.2',
     {'args': (20, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_REVALUATION/PE_TTM.csv']}),
    ('PE_TTM_p345d_continue_ud',
     {'args': ([3, 4, 5],),
      'fun': 'funda_data_deal.BaseDeal.pnd_continue_ud',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_REVALUATION/PE_TTM.csv']}),
    ('PE_TTM_p60d_col_extre_0.2',
     {'args': (60, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_REVALUATION/PE_TTM.csv']}),
    ('PE_TTM_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_REVALUATION/PE_TTM.csv']}),
    ('PS_TTM_p10d_col_extre_0.2',
     {'args': (10, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_REVALUATION/PS_TTM.csv']}),
    ('PS_TTM_p120d_col_extre_0.2',
     {'args': (120, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_REVALUATION/PS_TTM.csv']}),
    ('PS_TTM_p20d_col_extre_0.2',
     {'args': (20, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_REVALUATION/PS_TTM.csv']}),
    ('PS_TTM_p345d_continue_ud',
     {'args': ([3, 4, 5],),
      'fun': 'funda_data_deal.BaseDeal.pnd_continue_ud',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_REVALUATION/PS_TTM.csv']}),
    ('PS_TTM_p60d_col_extre_0.2',
     {'args': (60, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_REVALUATION/PS_TTM.csv']}),
    ('PS_TTM_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_REVALUATION/PS_TTM.csv']}),
    ('RQCHL_p10d_col_extre_0.2',
     {'args': (10, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RQCHL.csv']}),
    ('RQCHL_p120d_col_extre_0.2',
     {'args': (120, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RQCHL.csv']}),
    ('RQCHL_p20d_col_extre_0.2',
     {'args': (20, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RQCHL.csv']}),
    ('RQCHL_p345d_continue_ud',
     {'args': ([3, 4, 5],),
      'fun': 'funda_data_deal.BaseDeal.pnd_continue_ud',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RQCHL.csv']}),
    ('RQCHL_p60d_col_extre_0.2',
     {'args': (60, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RQCHL.csv']}),
    ('RQCHL_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RQCHL.csv']}),
    ('RQMCL_p10d_col_extre_0.2',
     {'args': (10, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RQMCL.csv']}),
    ('RQMCL_p120d_col_extre_0.2',
     {'args': (120, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RQMCL.csv']}),
    ('RQMCL_p20d_col_extre_0.2',
     {'args': (20, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RQMCL.csv']}),
    ('RQMCL_p345d_continue_ud',
     {'args': ([3, 4, 5],),
      'fun': 'funda_data_deal.BaseDeal.pnd_continue_ud',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RQMCL.csv']}),
    ('RQMCL_p60d_col_extre_0.2',
     {'args': (60, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RQMCL.csv']}),
    ('RQMCL_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RQMCL.csv']}),
    ('RQYE_p10d_col_extre_0.2',
     {'args': (10, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RQYE.csv']}),
    ('RQYE_p120d_col_extre_0.2',
     {'args': (120, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RQYE.csv']}),
    ('RQYE_p20d_col_extre_0.2',
     {'args': (20, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RQYE.csv']}),
    ('RQYE_p345d_continue_ud',
     {'args': ([3, 4, 5],),
      'fun': 'funda_data_deal.BaseDeal.pnd_continue_ud',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RQYE.csv']}),
    ('RQYE_p60d_col_extre_0.2',
     {'args': (60, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RQYE.csv']}),
    ('RQYE_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RQYE.csv']}),
    ('RQYL_p10d_col_extre_0.2',
     {'args': (10, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RQYL.csv']}),
    ('RQYL_p120d_col_extre_0.2',
     {'args': (120, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RQYL.csv']}),
    ('RQYL_p20d_col_extre_0.2',
     {'args': (20, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RQYL.csv']}),
    ('RQYL_p345d_continue_ud',
     {'args': ([3, 4, 5],),
      'fun': 'funda_data_deal.BaseDeal.pnd_continue_ud',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RQYL.csv']}),
    ('RQYL_p60d_col_extre_0.2',
     {'args': (60, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RQYL.csv']}),
    ('RQYL_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RQYL.csv']}),
    ('RSI_100_10',
     {'args': (100, 10),
      'fun': 'Tech_Factor.FactorMomentum.RSI',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('RSI_100_30',
     {'args': (100, 30),
      'fun': 'Tech_Factor.FactorMomentum.RSI',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('RSI_10_10',
     {'args': (10, 10),
      'fun': 'Tech_Factor.FactorMomentum.RSI',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('RSI_10_30',
     {'args': (10, 30),
      'fun': 'Tech_Factor.FactorMomentum.RSI',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('RSI_140_10',
     {'args': (140, 10),
      'fun': 'Tech_Factor.FactorMomentum.RSI',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('RSI_140_30',
     {'args': (140, 30),
      'fun': 'Tech_Factor.FactorMomentum.RSI',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('RSI_200_10',
     {'args': (200, 10),
      'fun': 'Tech_Factor.FactorMomentum.RSI',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('RSI_200_30',
     {'args': (200, 30),
      'fun': 'Tech_Factor.FactorMomentum.RSI',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('RSI_20_10',
     {'args': (20, 10),
      'fun': 'Tech_Factor.FactorMomentum.RSI',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('RSI_20_30',
     {'args': (20, 30),
      'fun': 'Tech_Factor.FactorMomentum.RSI',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('RSI_40_10',
     {'args': (40, 10),
      'fun': 'Tech_Factor.FactorMomentum.RSI',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('RSI_40_30',
     {'args': (40, 30),
      'fun': 'Tech_Factor.FactorMomentum.RSI',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('RSI_alpha100_1.5_0.5_0.5',
     {'args': (100, 1.5, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha100_1.5_0_1',
     {'args': (100, 1.5, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha100_1.5_1_0',
     {'args': (100, 1.5, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha100_1_0.5_0.5',
     {'args': (100, 1, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha100_1_0_1',
     {'args': (100, 1, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha100_1_1_0',
     {'args': (100, 1, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha100_2_0.5_0.5',
     {'args': (100, 2, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha100_2_0_1',
     {'args': (100, 2, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha100_2_1_0',
     {'args': (100, 2, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha10_1.5_0.5_0.5',
     {'args': (10, 1.5, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha10_1.5_0_1',
     {'args': (10, 1.5, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha10_1.5_1_0',
     {'args': (10, 1.5, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha10_1_0.5_0.5',
     {'args': (10, 1, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha10_1_0_1',
     {'args': (10, 1, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha10_1_1_0',
     {'args': (10, 1, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha10_2_0.5_0.5',
     {'args': (10, 2, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha10_2_0_1',
     {'args': (10, 2, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha10_2_1_0',
     {'args': (10, 2, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha140_1.5_0.5_0.5',
     {'args': (140, 1.5, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha140_1.5_0_1',
     {'args': (140, 1.5, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha140_1.5_1_0',
     {'args': (140, 1.5, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha140_1_0.5_0.5',
     {'args': (140, 1, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha140_1_0_1',
     {'args': (140, 1, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha140_1_1_0',
     {'args': (140, 1, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha140_2_0.5_0.5',
     {'args': (140, 2, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha140_2_0_1',
     {'args': (140, 2, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha140_2_1_0',
     {'args': (140, 2, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha200_1.5_0.5_0.5',
     {'args': (200, 1.5, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha200_1.5_0_1',
     {'args': (200, 1.5, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha200_1.5_1_0',
     {'args': (200, 1.5, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha200_1_0.5_0.5',
     {'args': (200, 1, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha200_1_0_1',
     {'args': (200, 1, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha200_1_1_0',
     {'args': (200, 1, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha200_2_0.5_0.5',
     {'args': (200, 2, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha200_2_0_1',
     {'args': (200, 2, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha200_2_1_0',
     {'args': (200, 2, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha20_1.5_0.5_0.5',
     {'args': (20, 1.5, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha20_1.5_0_1',
     {'args': (20, 1.5, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha20_1.5_1_0',
     {'args': (20, 1.5, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha20_1_0.5_0.5',
     {'args': (20, 1, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha20_1_0_1',
     {'args': (20, 1, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha20_1_1_0',
     {'args': (20, 1, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha20_2_0.5_0.5',
     {'args': (20, 2, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha20_2_0_1',
     {'args': (20, 2, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha20_2_1_0',
     {'args': (20, 2, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha40_1.5_0.5_0.5',
     {'args': (40, 1.5, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha40_1.5_0_1',
     {'args': (40, 1.5, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha40_1.5_1_0',
     {'args': (40, 1.5, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha40_1_0.5_0.5',
     {'args': (40, 1, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha40_1_0_1',
     {'args': (40, 1, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha40_1_1_0',
     {'args': (40, 1, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha40_2_0.5_0.5',
     {'args': (40, 2, [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha40_2_0_1',
     {'args': (40, 2, [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RSI_alpha40_2_1_0',
     {'args': (40, 2, [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.RSI_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('RZCHE_p10d_col_extre_0.2',
     {'args': (10, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RZCHE.csv']}),
    ('RZCHE_p120d_col_extre_0.2',
     {'args': (120, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RZCHE.csv']}),
    ('RZCHE_p20d_col_extre_0.2',
     {'args': (20, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RZCHE.csv']}),
    ('RZCHE_p345d_continue_ud',
     {'args': ([3, 4, 5],),
      'fun': 'funda_data_deal.BaseDeal.pnd_continue_ud',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RZCHE.csv']}),
    ('RZCHE_p60d_col_extre_0.2',
     {'args': (60, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RZCHE.csv']}),
    ('RZCHE_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RZCHE.csv']}),
    ('RZMRE_p10d_col_extre_0.2',
     {'args': (10, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RZMRE.csv']}),
    ('RZMRE_p120d_col_extre_0.2',
     {'args': (120, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RZMRE.csv']}),
    ('RZMRE_p20d_col_extre_0.2',
     {'args': (20, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RZMRE.csv']}),
    ('RZMRE_p345d_continue_ud',
     {'args': ([3, 4, 5],),
      'fun': 'funda_data_deal.BaseDeal.pnd_continue_ud',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RZMRE.csv']}),
    ('RZMRE_p60d_col_extre_0.2',
     {'args': (60, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RZMRE.csv']}),
    ('RZMRE_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RZMRE.csv']}),
    ('RZRQYE_p10d_col_extre_0.2',
     {'args': (10, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RZRQYE.csv']}),
    ('RZRQYE_p120d_col_extre_0.2',
     {'args': (120, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RZRQYE.csv']}),
    ('RZRQYE_p20d_col_extre_0.2',
     {'args': (20, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RZRQYE.csv']}),
    ('RZRQYE_p345d_continue_ud',
     {'args': ([3, 4, 5],),
      'fun': 'funda_data_deal.BaseDeal.pnd_continue_ud',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RZRQYE.csv']}),
    ('RZRQYE_p60d_col_extre_0.2',
     {'args': (60, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RZRQYE.csv']}),
    ('RZRQYE_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RZRQYE.csv']}),
    ('RZYE_p10d_col_extre_0.2',
     {'args': (10, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RZYE.csv']}),
    ('RZYE_p120d_col_extre_0.2',
     {'args': (120, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RZYE.csv']}),
    ('RZYE_p20d_col_extre_0.2',
     {'args': (20, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RZYE.csv']}),
    ('RZYE_p345d_continue_ud',
     {'args': ([3, 4, 5],),
      'fun': 'funda_data_deal.BaseDeal.pnd_continue_ud',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RZYE.csv']}),
    ('RZYE_p60d_col_extre_0.2',
     {'args': (60, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RZYE.csv']}),
    ('RZYE_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': True,
      'raw_data_path': ['EM_Funda/TRAD_MT_MARGIN/RZYE.csv']}),
    ('R_ACCOUNTPAY_QYOY_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_ACCOUNTPAY_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_ACCOUNTPAY_QTTM.csv']}),
    ('R_ACCOUNTPAY_QYOY_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_ACCOUNTPAY_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_ACCOUNTPAY_QTTM.csv']}),
    ('R_ACCOUNTREC_QYOY_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_ACCOUNTREC_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_ACCOUNTREC_QTTM.csv']}),
    ('R_ACCOUNTREC_QYOY_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_ACCOUNTREC_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_ACCOUNTREC_QTTM.csv']}),
    ('R_ASSETDEVALUELOSS_s_QYOY_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_ASSETDEVALUELOSS_s_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_ASSETDEVALUELOSS_s_QTTM.csv']}),
    ('R_ASSETDEVALUELOSS_s_QYOY_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_ASSETDEVALUELOSS_s_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_ASSETDEVALUELOSS_s_QTTM.csv']}),
    ('R_AssetDepSales_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_AssetDepSales_s_First.csv']}),
    ('R_AssetDepSales_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_AssetDepSales_s_First.csv']}),
    ('R_BusinessCycle_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_BusinessCycle_First.csv']}),
    ('R_CFOPS_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_CFOPS_s_First.csv']}),
    ('R_CFOPS_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_CFOPS_s_First.csv']}),
    ('R_CFO_TotRev_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_CFO_TotRev_s_First.csv']}),
    ('R_CFO_TotRev_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_CFO_TotRev_s_First.csv']}),
    ('R_CFO_s_YOY_First_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_CFO_s_YOY_First.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_CFO_s_YOY_QTTM.csv']}),
    ('R_CFO_s_YOY_First_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_CFO_s_YOY_First.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_CFO_s_YOY_QTTM.csv']}),
    ('R_CFO_s_YOY_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_CFO_s_YOY_First.csv']}),
    ('R_CFO_s_YOY_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_CFO_s_YOY_First.csv']}),
    ('R_COMPANYCODE_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_COMPANYCODE_First.csv']}),
    ('R_Cashflow_s_YOY_First_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_Cashflow_s_YOY_First.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_Cashflow_s_YOY_QTTM.csv']}),
    ('R_Cashflow_s_YOY_First_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_Cashflow_s_YOY_First.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_Cashflow_s_YOY_QTTM.csv']}),
    ('R_Cashflow_s_YOY_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_Cashflow_s_YOY_First.csv']}),
    ('R_Cashflow_s_YOY_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_Cashflow_s_YOY_First.csv']}),
    ('R_CostSales_QYOY_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_CostSales_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_CostSales_QTTM.csv']}),
    ('R_CostSales_QYOY_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_CostSales_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_CostSales_QTTM.csv']}),
    ('R_CostSales_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_CostSales_s_First.csv']}),
    ('R_CostSales_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_CostSales_s_First.csv']}),
    ('R_CurrentAssetsTurnover_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_CurrentAssetsTurnover_First.csv']}),
    ('R_CurrentAssetsTurnover_QTTM_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_CurrentAssetsTurnover_QTTM.csv']}),
    ('R_CurrentAssetsTurnover_QTTM_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_CurrentAssetsTurnover_QTTM.csv']}),
    ('R_CurrentAssets_TotAssets_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_CurrentAssets_TotAssets_First.csv']}),
    ('R_DebtAssets_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_DebtAssets_First.csv']}),
    ('R_DebtAssets_QTTM_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_DebtAssets_QTTM.csv']}),
    ('R_DebtAssets_QTTM_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_DebtAssets_QTTM.csv']}),
    ('R_DebtEqt_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_DebtEqt_First.csv']}),
    ('R_EBITDA2_QYOY_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_EBITDA2_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_EBITDA2_QTTM.csv']}),
    ('R_EBITDA2_QYOY_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_EBITDA2_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_EBITDA2_QTTM.csv']}),
    ('R_EBITDA_IntDebt_QTTM_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_EBITDA_IntDebt_QTTM.csv']}),
    ('R_EBITDA_IntDebt_QTTM_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_EBITDA_IntDebt_QTTM.csv']}),
    ('R_EBITDA_QTTM_and_MCAP_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_EBITDA_QTTM.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/LICO_YS_STOCKVALUE/AmarketCapExStri.csv']}),
    ('R_EBITDA_QTTM_and_R_SUMASSET_First_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_EBITDA_QTTM.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_SUMASSET_First.csv']}),
    ('R_EBITDA_QYOY_and_MCAP_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_IntDebt_Y3YGR.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/LICO_YS_STOCKVALUE/AmarketCapExStri.csv']}),
    ('R_EBITDA_sales_TTM_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_EBITDA_sales_TTM_First.csv']}),
    ('R_EBITDA_sales_TTM_QTTM_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_EBITDA_sales_TTM_QTTM.csv']}),
    ('R_EBIT_Y3YGR_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_EBIT_Y3YGR.csv']}),
    ('R_EBIT_Y3YGR_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_EBIT_Y3YGR.csv']}),
    ('R_EBIT_sales_QTTM_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_EBIT_sales_QTTM.csv']}),
    ('R_EBIT_sales_QTTM_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_EBIT_sales_QTTM.csv']}),
    ('R_EPSDiluted_YOY_First_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_EPSDiluted_YOY_First.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_EPSDiluted_YOY_QTTM.csv']}),
    ('R_EPSDiluted_YOY_First_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_EPSDiluted_YOY_First.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_EPSDiluted_YOY_QTTM.csv']}),
    ('R_EPS_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_EPS_s_First.csv']}),
    ('R_EPS_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_EPS_s_First.csv']}),
    ('R_EPS_s_YOY_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_EPS_s_YOY_First.csv']}),
    ('R_EPS_s_YOY_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_EPS_s_YOY_First.csv']}),
    ('R_ESTATEINVEST_QYOY_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_ESTATEINVEST_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_ESTATEINVEST_QTTM.csv']}),
    ('R_ESTATEINVEST_QYOY_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_ESTATEINVEST_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_ESTATEINVEST_QTTM.csv']}),
    ('R_FCFTot_Y3YGR_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_FCFTot_Y3YGR.csv']}),
    ('R_FCFTot_Y3YGR_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_FCFTot_Y3YGR.csv']}),
    ('R_FINANCEEXP_s_QYOY_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_FINANCEEXP_s_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_FINANCEEXP_s_QTTM.csv']}),
    ('R_FINANCEEXP_s_QYOY_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_FINANCEEXP_s_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_FINANCEEXP_s_QTTM.csv']}),
    ('R_FairValChgPnL_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_FairValChgPnL_s_First.csv']}),
    ('R_FairValChgPnL_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_FairValChgPnL_s_First.csv']}),
    ('R_FairValChg_TotProfit_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_FairValChg_TotProfit_s_First.csv']}),
    ('R_FairValChg_TotProfit_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_FairValChg_TotProfit_s_First.csv']}),
    ('R_FairVal_TotProfit_QYOY_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_FairVal_TotProfit_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_FairVal_TotProfit_QTTM.csv']}),
    ('R_FairVal_TotProfit_QYOY_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_FairVal_TotProfit_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_FairVal_TotProfit_QTTM.csv']}),
    ('R_FinExp_sales_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_FinExp_sales_s_First.csv']}),
    ('R_FinExp_sales_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_FinExp_sales_s_First.csv']}),
    ('R_GSCF_sales_QYOY_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_GSCF_sales_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_GSCF_sales_QTTM.csv']}),
    ('R_GSCF_sales_QYOY_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_GSCF_sales_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_GSCF_sales_QTTM.csv']}),
    ('R_GSCF_sales_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_GSCF_sales_s_First.csv']}),
    ('R_GSCF_sales_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_GSCF_sales_s_First.csv']}),
    ('R_GrossProfit_TTM_QYOY_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_GrossProfit_TTM_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_GrossProfit_TTM_QTTM.csv']}),
    ('R_GrossProfit_TTM_QYOY_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_GrossProfit_TTM_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_GrossProfit_TTM_QTTM.csv']}),
    ('R_INVESTINCOME_s_QYOY_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_INVESTINCOME_s_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_INVESTINCOME_s_QTTM.csv']}),
    ('R_INVESTINCOME_s_QYOY_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_INVESTINCOME_s_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_INVESTINCOME_s_QTTM.csv']}),
    ('R_IntDebt_Mcap_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_IntDebt_Mcap_First.csv']}),
    ('R_IntDebt_Mcap_QYOY_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_IntDebt_Mcap_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_IntDebt_Mcap_QTTM.csv']}),
    ('R_IntDebt_Mcap_QYOY_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_IntDebt_Mcap_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_IntDebt_Mcap_QTTM.csv']}),
    ('R_IntDebt_Y3YGR_and_R_SUMASSET_First_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_IntDebt_Y3YGR.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_SUMASSET_First.csv']}),
    ('R_LOANREC_s_QYOY_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_LOANREC_s_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_LOANREC_s_QTTM.csv']}),
    ('R_LOANREC_s_QYOY_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_LOANREC_s_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_LOANREC_s_QTTM.csv']}),
    ('R_LTDebt_WorkCap_QTTM_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_LTDebt_WorkCap_QTTM.csv']}),
    ('R_LTDebt_WorkCap_QTTM_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_LTDebt_WorkCap_QTTM.csv']}),
    ('R_LTDebt_WorkCap_QYOY_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_LTDebt_WorkCap_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_LTDebt_WorkCap_QTTM.csv']}),
    ('R_LTDebt_WorkCap_QYOY_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_LTDebt_WorkCap_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_LTDebt_WorkCap_QTTM.csv']}),
    ('R_MgtExp_sales_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_MgtExp_sales_s_First.csv']}),
    ('R_MgtExp_sales_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_MgtExp_sales_s_First.csv']}),
    ('R_NETPROFIT_s_QYOY_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_NETPROFIT_s_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_NETPROFIT_s_QTTM.csv']}),
    ('R_NETPROFIT_s_QYOY_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_NETPROFIT_s_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_NETPROFIT_s_QTTM.csv']}),
    ('R_NOTICEDATE_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NOTICEDATE_First.csv']}),
    ('R_NOTICEDATE_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NOTICEDATE_First.csv']}),
    ('R_NetAssets_s_POP_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NetAssets_s_POP_First.csv']}),
    ('R_NetAssets_s_POP_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NetAssets_s_POP_First.csv']}),
    ('R_NetAssets_s_YOY_First_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_NetAssets_s_YOY_First.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_NetAssets_s_YOY_QTTM.csv']}),
    ('R_NetAssets_s_YOY_First_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_NetAssets_s_YOY_First.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_NetAssets_s_YOY_QTTM.csv']}),
    ('R_NetAssets_s_YOY_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NetAssets_s_YOY_First.csv']}),
    ('R_NetAssets_s_YOY_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NetAssets_s_YOY_First.csv']}),
    ('R_NetCashflowPS_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NetCashflowPS_s_First.csv']}),
    ('R_NetCashflowPS_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NetCashflowPS_s_First.csv']}),
    ('R_NetIncRecur_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NetIncRecur_s_First.csv']}),
    ('R_NetIncRecur_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NetIncRecur_s_First.csv']}),
    ('R_NetInc_TotProfit_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NetInc_TotProfit_s_First.csv']}),
    ('R_NetInc_TotProfit_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NetInc_TotProfit_s_First.csv']}),
    ('R_NetInc_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NetInc_s_First.csv']}),
    ('R_NetInc_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NetInc_s_First.csv']}),
    ('R_NetInc_s_QYOY_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_NetInc_s_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_NetInc_s_QTTM.csv']}),
    ('R_NetInc_s_QYOY_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_NetInc_s_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_NetInc_s_QTTM.csv']}),
    ('R_NetMargin_s_YOY_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NetMargin_s_YOY_First.csv']}),
    ('R_NetMargin_s_YOY_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NetMargin_s_YOY_First.csv']}),
    ('R_NetProfit_sales_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NetProfit_sales_s_First.csv']}),
    ('R_NetProfit_sales_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NetProfit_sales_s_First.csv']}),
    ('R_NetROA_TTM_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NetROA_TTM_First.csv']}),
    ('R_NetROA_TTM_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NetROA_TTM_First.csv']}),
    ('R_NetROA_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NetROA_s_First.csv']}),
    ('R_NetROA_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NetROA_s_First.csv']}),
    ('R_NonOperProft_TotProfit_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NonOperProft_TotProfit_s_First.csv']}),
    ('R_NonOperProft_TotProfit_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_NonOperProft_TotProfit_s_First.csv']}),
    ('R_OPCF_CurrentLiab_First_and_MCAP_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_OPCF_CurrentLiab_First.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/LICO_YS_STOCKVALUE/AmarketCapExStri.csv']}),
    ('R_OPCF_IntDebt_QTTM_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OPCF_IntDebt_QTTM.csv']}),
    ('R_OPCF_NetInc_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OPCF_NetInc_s_First.csv']}),
    ('R_OPCF_NetInc_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OPCF_NetInc_s_First.csv']}),
    ('R_OPCF_TTM_QYOY_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_OPCF_TTM_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_OPCF_TTM_QTTM.csv']}),
    ('R_OPCF_TTM_QYOY_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_OPCF_TTM_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_OPCF_TTM_QTTM.csv']}),
    ('R_OPCF_TotDebt_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OPCF_TotDebt_First.csv']}),
    ('R_OPCF_TotDebt_QTTM_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OPCF_TotDebt_QTTM.csv']}),
    ('R_OPCF_TotDebt_QTTM_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OPCF_TotDebt_QTTM.csv']}),
    ('R_OPCF_TotDebt_QYOY_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OPCF_TotDebt_QYOY.csv']}),
    ('R_OPCF_sales_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OPCF_sales_First.csv']}),
    ('R_OPCF_sales_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OPCF_sales_s_First.csv']}),
    ('R_OPCF_sales_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OPCF_sales_s_First.csv']}),
    ('R_OPEX_sales_QYOY_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_OPEX_sales_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_OPEX_sales_QTTM.csv']}),
    ('R_OPEX_sales_QYOY_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_OPEX_sales_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_OPEX_sales_QTTM.csv']}),
    ('R_OPEX_sales_TTM_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OPEX_sales_TTM_First.csv']}),
    ('R_OPEX_sales_TTM_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OPEX_sales_TTM_First.csv']}),
    ('R_OPEX_sales_TTM_Y3YGR_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OPEX_sales_TTM_Y3YGR.csv']}),
    ('R_OPEX_sales_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OPEX_sales_s_First.csv']}),
    ('R_OPEX_sales_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OPEX_sales_s_First.csv']}),
    ('R_OTHERLASSET_QYOY_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_OTHERLASSET_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_OTHERLASSET_QTTM.csv']}),
    ('R_OTHERLASSET_QYOY_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_OTHERLASSET_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_OTHERLASSET_QTTM.csv']}),
    ('R_OperCost_sales_QYOY_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_OperCost_sales_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_OperCost_sales_QTTM.csv']}),
    ('R_OperCost_sales_QYOY_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_OperCost_sales_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_OperCost_sales_QTTM.csv']}),
    ('R_OperCost_sales_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OperCost_sales_s_First.csv']}),
    ('R_OperCost_sales_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OperCost_sales_s_First.csv']}),
    ('R_OperProfit_YOY_First_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_OperProfit_YOY_First.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_OperProfit_YOY_QTTM.csv']}),
    ('R_OperProfit_YOY_First_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_OperProfit_YOY_First.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_OperProfit_YOY_QTTM.csv']}),
    ('R_OperProfit_s_POP_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OperProfit_s_POP_First.csv']}),
    ('R_OperProfit_s_POP_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OperProfit_s_POP_First.csv']}),
    ('R_OperProfit_s_YOY_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OperProfit_s_YOY_First.csv']}),
    ('R_OperProfit_s_YOY_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OperProfit_s_YOY_First.csv']}),
    ('R_OperProfit_sales_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OperProfit_sales_s_First.csv']}),
    ('R_OperProfit_sales_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_OperProfit_sales_s_First.csv']}),
    ('R_ParentProfit_s_POP_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_ParentProfit_s_POP_First.csv']}),
    ('R_ParentProfit_s_POP_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_ParentProfit_s_POP_First.csv']}),
    ('R_ParentProfit_s_YOY_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_ParentProfit_s_YOY_First.csv']}),
    ('R_ParentProfit_s_YOY_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_ParentProfit_s_YOY_First.csv']}),
    ('R_ROA_TTM_Y3YGR_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_ROA_TTM_Y3YGR.csv']}),
    ('R_ROE1_QYOY_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_ROE1_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_ROE1_QTTM.csv']}),
    ('R_ROE1_QYOY_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_ROE1_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_ROE1_QTTM.csv']}),
    ('R_ROENetIncRecur_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_ROENetIncRecur_s_First.csv']}),
    ('R_ROENetIncRecur_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_ROENetIncRecur_s_First.csv']}),
    ('R_ROE_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_ROE_s_First.csv']}),
    ('R_ROE_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_ROE_s_First.csv']}),
    ('R_RecurNetProft_NetProfit_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_RecurNetProft_NetProfit_s_First.csv']}),
    ('R_RecurNetProft_NetProfit_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_RecurNetProft_NetProfit_s_First.csv']}),
    ('R_RevenuePS_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_RevenuePS_s_First.csv']}),
    ('R_RevenuePS_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_RevenuePS_s_First.csv']}),
    ('R_RevenueTotPS_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_RevenueTotPS_s_First.csv']}),
    ('R_RevenueTotPS_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_RevenueTotPS_s_First.csv']}),
    ('R_Revenue_s_POP_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_Revenue_s_POP_First.csv']}),
    ('R_Revenue_s_POP_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_Revenue_s_POP_First.csv']}),
    ('R_Revenue_s_YOY_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_Revenue_s_YOY_First.csv']}),
    ('R_Revenue_s_YOY_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_Revenue_s_YOY_First.csv']}),
    ('R_SUMLIAB_QYOY_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_SUMLIAB_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_SUMLIAB_QTTM.csv']}),
    ('R_SUMLIAB_QYOY_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_SUMLIAB_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_SUMLIAB_QTTM.csv']}),
    ('R_SUMLIAB_Y3YGR_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_SUMLIAB_Y3YGR.csv']}),
    ('R_SUMLIAB_Y3YGR_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_SUMLIAB_Y3YGR.csv']}),
    ('R_SalesCost_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_SalesCost_s_First.csv']}),
    ('R_SalesCost_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_SalesCost_s_First.csv']}),
    ('R_SalesGrossMGN_QTTM_and_MCAP_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_SalesGrossMGN_QTTM.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/LICO_YS_STOCKVALUE/AmarketCapExStri.csv']}),
    ('R_SalesGrossMGN_QTTM_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_SalesGrossMGN_QTTM.csv']}),
    ('R_SalesGrossMGN_QTTM_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_SalesGrossMGN_QTTM.csv']}),
    ('R_SalesGrossMGN_QYOY_and_MCAP_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_SalesGrossMGN_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/LICO_YS_STOCKVALUE/AmarketCapExStri.csv']}),
    ('R_SalesGrossMGN_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_SalesGrossMGN_s_First.csv']}),
    ('R_SalesGrossMGN_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_SalesGrossMGN_s_First.csv']}),
    ('R_SalesNetMGN_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_SalesNetMGN_s_First.csv']}),
    ('R_SalesNetMGN_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_SalesNetMGN_s_First.csv']}),
    ('R_TangAssets_IntDebt_QYOY_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_TangAssets_IntDebt_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_TangAssets_IntDebt_QTTM.csv']}),
    ('R_TangAssets_IntDebt_QYOY_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_TangAssets_IntDebt_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_TangAssets_IntDebt_QTTM.csv']}),
    ('R_TangAssets_TotLiab_QTTM_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_TangAssets_TotLiab_QTTM.csv']}),
    ('R_TangAssets_TotLiab_QTTM_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_TangAssets_TotLiab_QTTM.csv']}),
    ('R_Tax_TotProfit_QTTM_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_Tax_TotProfit_QTTM.csv']}),
    ('R_Tax_TotProfit_QTTM_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_Tax_TotProfit_QTTM.csv']}),
    ('R_Tax_TotProfit_s_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_Tax_TotProfit_s_First.csv']}),
    ('R_Tax_TotProfit_s_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_Tax_TotProfit_s_First.csv']}),
    ('R_TotAssets_s_YOY_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_TotAssets_s_YOY_First.csv']}),
    ('R_TotAssets_s_YOY_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_TotAssets_s_YOY_First.csv']}),
    ('R_TotLiab_s_YOY_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_TotLiab_s_YOY_First.csv']}),
    ('R_TotLiab_s_YOY_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_TotLiab_s_YOY_First.csv']}),
    ('R_TotRev_TTM_Y3YGR_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_TotRev_TTM_Y3YGR.csv']}),
    ('R_TotRev_TTM_Y3YGR_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_TotRev_TTM_Y3YGR.csv']}),
    ('R_TotRev_s_POP_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_TotRev_s_POP_First.csv']}),
    ('R_TotRev_s_POP_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_TotRev_s_POP_First.csv']}),
    ('R_TotRev_s_YOY_First_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_TotRev_s_YOY_First.csv']}),
    ('R_TotRev_s_YOY_First_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_TotRev_s_YOY_First.csv']}),
    ('R_WorkCapital_QYOY_and_QTTM_0.2',
     {'args': (0.2,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_WorkCapital_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_WorkCapital_QTTM.csv']}),
    ('R_WorkCapital_QYOY_and_QTTM_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.dev_row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/daily/R_WorkCapital_QYOY.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_WorkCapital_QTTM.csv']}),
    ('TVOL_p10d_col_extre_0.2',
     {'args': (10, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('TVOL_p120d_col_extre_0.2',
     {'args': (120, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('TVOL_p20d_col_extre_0.2',
     {'args': (20, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('TVOL_p30d_col_extre_0.2',
     {'args': (30, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('TVOL_p345d_continue_ud',
     {'args': ([3, 4, 5],),
      'fun': 'funda_data_deal.BaseDeal.pnd_continue_ud',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('TVOL_p60d_col_extre_0.2',
     {'args': (60, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('TVOL_p90d_col_extre_0.2',
     {'args': (90, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('TVOL_pd_continue_ud',
     {'args': ([],),
      'fun': 'funda_data_deal.BaseDeal.pnd_continue_ud',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('TVOL_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('TotRev_and_asset_QYOY_Y3YGR_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.mix_factor_asset',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_TotRev_TTM_First.csv',
                        'EM_Funda/daily/R_SUMASSET_First.csv',
                        'EM_Funda/daily/R_TotRev_TTM_QYOY.csv',
                        'EM_Funda/daily/R_TotRev_TTM_Y3YGR.csv',
                        'EM_Funda/daily/R_TotRev_TTM_QSD4Y.csv']}),
    ('TotRev_and_asset_Y3YGR_Y5YGR_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.mix_factor_asset',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_TotRev_TTM_First.csv',
                        'EM_Funda/daily/R_SUMASSET_First.csv',
                        'EM_Funda/daily/R_TotRev_TTM_Y3YGR.csv',
                        'EM_Funda/daily/R_TotRev_TTM_Y5YGR.csv',
                        'EM_Funda/daily/R_TotRev_TTM_QSD4Y.csv']}),
    ('TotRev_and_mcap_QYOY_Y3YGR_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.mix_factor_mcap',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_TotRev_TTM_First.csv',
                        'EM_Funda/LICO_YS_STOCKVALUE/AmarketCap.csv',
                        'EM_Funda/daily/R_TotRev_TTM_QYOY.csv',
                        'EM_Funda/daily/R_TotRev_TTM_Y3YGR.csv',
                        'EM_Funda/daily/R_TotRev_TTM_QSD4Y.csv']}),
    ('TotRev_and_mcap_Y3YGR_Y5YGR_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.mix_factor_mcap',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_TotRev_TTM_First.csv',
                        'EM_Funda/LICO_YS_STOCKVALUE/AmarketCap.csv',
                        'EM_Funda/daily/R_TotRev_TTM_Y3YGR.csv',
                        'EM_Funda/daily/R_TotRev_TTM_Y5YGR.csv',
                        'EM_Funda/daily/R_TotRev_TTM_QSD4Y.csv']}),
    ('TotRev_and_mcap_intdebt_QYOY_Y3YGR_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.mix_factor_mcap_intdebt',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_TotRev_TTM_First.csv',
                        'EM_Funda/LICO_YS_STOCKVALUE/AmarketCap.csv',
                        'EM_Funda/daily/R_IntDebt_First.csv',
                        'EM_Funda/daily/R_TotRev_TTM_QYOY.csv',
                        'EM_Funda/daily/R_TotRev_TTM_Y3YGR.csv',
                        'EM_Funda/daily/R_TotRev_TTM_QSD4Y.csv']}),
    ('TotRev_and_mcap_intdebt_Y3YGR_Y5YGR_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda.EM_Funda_Deal.mix_factor_mcap_intdebt',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/daily/R_TotRev_TTM_First.csv',
                        'EM_Funda/LICO_YS_STOCKVALUE/AmarketCap.csv',
                        'EM_Funda/daily/R_IntDebt_First.csv',
                        'EM_Funda/daily/R_TotRev_TTM_Y3YGR.csv',
                        'EM_Funda/daily/R_TotRev_TTM_Y5YGR.csv',
                        'EM_Funda/daily/R_TotRev_TTM_QSD4Y.csv']}),
    ('aadj_r_p10d_col_extre_0.2',
     {'args': (10, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('aadj_r_p120d_col_extre_0.2',
     {'args': (120, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('aadj_r_p20d_col_extre_0.2',
     {'args': (20, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('aadj_r_p345d_continue_ud',
     {'args': ([3, 4, 5],),
      'fun': 'funda_data_deal.BaseDeal.pnd_continue_ud',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('aadj_r_p345d_continue_ud_pct',
     {'args': ([3, 4, 5],),
      'fun': 'funda_data_deal.BaseDeal.pnd_continue_ud_pct',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('aadj_r_p60d_col_extre_0.2',
     {'args': (60, 0.2),
      'fun': 'funda_data_deal.BaseDeal.pnd_col_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('aadj_r_row_extre_0.2',
     {'args': (0.2,),
      'fun': 'funda_data_deal.BaseDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('bias_turn_p120d',
     {'args': (120,),
      'fun': 'EM_Tab14.EM_Tab14_Deal.bias_turn_pnd',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TURNRATE.csv']}),
    ('bias_turn_p20d',
     {'args': (20,),
      'fun': 'EM_Tab14.EM_Tab14_Deal.bias_turn_pnd',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TURNRATE.csv']}),
    ('bias_turn_p60d',
     {'args': (60,),
      'fun': 'EM_Tab14.EM_Tab14_Deal.bias_turn_pnd',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TURNRATE.csv']}),
    ('continue_ud_p100d',
     {'args': (100,),
      'fun': 'funda_data_deal.BaseDeal.pnd_vol_continue_ud',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('continue_ud_p10d',
     {'args': (10,),
      'fun': 'funda_data_deal.BaseDeal.pnd_vol_continue_ud',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('continue_ud_p120d',
     {'args': (120,),
      'fun': 'funda_data_deal.BaseDeal.pnd_vol_continue_ud',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('continue_ud_p200d',
     {'args': (200,),
      'fun': 'funda_data_deal.BaseDeal.pnd_vol_continue_ud',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('continue_ud_p20d',
     {'args': (20,),
      'fun': 'funda_data_deal.BaseDeal.pnd_vol_continue_ud',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('continue_ud_p30d',
     {'args': (30,),
      'fun': 'funda_data_deal.BaseDeal.pnd_vol_continue_ud',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('continue_ud_p50d',
     {'args': (50,),
      'fun': 'funda_data_deal.BaseDeal.pnd_vol_continue_ud',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('continue_ud_p60d',
     {'args': (60,),
      'fun': 'funda_data_deal.BaseDeal.pnd_vol_continue_ud',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('continue_ud_p90d',
     {'args': (90,),
      'fun': 'funda_data_deal.BaseDeal.pnd_vol_continue_ud',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('evol_p100d',
     {'args': (100,),
      'fun': 'funda_data_deal.BaseDeal.pnd_evol',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('evol_p10d',
     {'args': (10,),
      'fun': 'funda_data_deal.BaseDeal.pnd_evol',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('evol_p120d',
     {'args': (120,),
      'fun': 'funda_data_deal.BaseDeal.pnd_evol',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('evol_p200d',
     {'args': (200,),
      'fun': 'funda_data_deal.BaseDeal.pnd_evol',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('evol_p20d',
     {'args': (20,),
      'fun': 'funda_data_deal.BaseDeal.pnd_evol',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('evol_p30d',
     {'args': (30,),
      'fun': 'funda_data_deal.BaseDeal.pnd_evol',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('evol_p50d',
     {'args': (50,),
      'fun': 'funda_data_deal.BaseDeal.pnd_evol',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('evol_p60d',
     {'args': (60,),
      'fun': 'funda_data_deal.BaseDeal.pnd_evol',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('evol_p90d',
     {'args': (90,),
      'fun': 'funda_data_deal.BaseDeal.pnd_evol',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('intra_dn_15_bar_div_daily_col_score_row_extre_0.3',
     {'args': (100, 0.3),
      'fun': 'Intraday.IntradayDeal.col_score_row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_dn_15_bar_div_daily']}),
    ('intra_dn_15_bar_div_daily_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'Intraday.IntradayDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_dn_15_bar_div_daily']}),
    ('intra_dn_15_bar_vol_col_score_row_extre_0.3',
     {'args': (100, 0.3),
      'fun': 'Intraday.IntradayDeal.col_score_row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_dn_15_bar_vol']}),
    ('intra_dn_15_bar_vol_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'Intraday.IntradayDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_dn_15_bar_vol']}),
    ('intra_dn_15_bar_vwap_col_score_row_extre_0.3',
     {'args': (100, 0.3),
      'fun': 'Intraday.IntradayDeal.col_score_row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_dn_15_bar_vwap']}),
    ('intra_dn_15_bar_vwap_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'Intraday.IntradayDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_dn_15_bar_vwap']}),
    ('intra_dn_div_daily_col_score_row_extre_0.3',
     {'args': (100, 0.3),
      'fun': 'Intraday.IntradayDeal.col_score_row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_dn_div_daily']}),
    ('intra_dn_div_daily_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'Intraday.IntradayDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_dn_div_daily']}),
    ('intra_dn_vol_col_score_row_extre_0.3',
     {'args': (100, 0.3),
      'fun': 'Intraday.IntradayDeal.col_score_row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_dn_vol']}),
    ('intra_dn_vol_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'Intraday.IntradayDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_dn_vol']}),
    ('intra_dn_vwap_col_score_row_extre_0.3',
     {'args': (100, 0.3),
      'fun': 'Intraday.IntradayDeal.col_score_row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_dn_vwap']}),
    ('intra_dn_vwap_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'Intraday.IntradayDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_dn_vwap']}),
    ('intra_up_15_bar_div_daily_col_score_row_extre_0.3',
     {'args': (100, 0.3),
      'fun': 'Intraday.IntradayDeal.col_score_row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_up_15_bar_div_daily']}),
    ('intra_up_15_bar_div_daily_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'Intraday.IntradayDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_up_15_bar_div_daily']}),
    ('intra_up_15_bar_div_dn_15_bar_col_score_row_extre_0.3',
     {'args': (100, 0.3),
      'fun': 'Intraday.IntradayDeal.col_score_row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_up_15_bar_div_dn_15_bar']}),
    ('intra_up_15_bar_div_dn_15_bar_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'Intraday.IntradayDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_up_15_bar_div_dn_15_bar']}),
    ('intra_up_15_bar_vol_col_score_row_extre_0.3',
     {'args': (100, 0.3),
      'fun': 'Intraday.IntradayDeal.col_score_row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_up_15_bar_vol']}),
    ('intra_up_15_bar_vol_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'Intraday.IntradayDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_up_15_bar_vol']}),
    ('intra_up_15_bar_vwap_col_score_row_extre_0.3',
     {'args': (100, 0.3),
      'fun': 'Intraday.IntradayDeal.col_score_row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_up_15_bar_vwap']}),
    ('intra_up_15_bar_vwap_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'Intraday.IntradayDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_up_15_bar_vwap']}),
    ('intra_up_div_daily_col_score_row_extre_0.3',
     {'args': (100, 0.3),
      'fun': 'Intraday.IntradayDeal.col_score_row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_up_div_daily']}),
    ('intra_up_div_daily_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'Intraday.IntradayDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_up_div_daily']}),
    ('intra_up_div_dn_col_score_row_extre_0.3',
     {'args': (100, 0.3),
      'fun': 'Intraday.IntradayDeal.col_score_row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_up_div_dn']}),
    ('intra_up_div_dn_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'Intraday.IntradayDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_up_div_dn']}),
    ('intra_up_vol_col_score_row_extre_0.3',
     {'args': (100, 0.3),
      'fun': 'Intraday.IntradayDeal.col_score_row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_up_vol']}),
    ('intra_up_vol_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'Intraday.IntradayDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_up_vol']}),
    ('intra_up_vwap_col_score_row_extre_0.3',
     {'args': (100, 0.3),
      'fun': 'Intraday.IntradayDeal.col_score_row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_up_vwap']}),
    ('intra_up_vwap_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'Intraday.IntradayDeal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/my_data/intra_up_vwap']}),
    ('log_price_0.2',
     {'args': (0.2,),
      'fun': 'EM_Tab14.EM_Tab14_Deal.log_price',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('moment_p10100d',
     {'args': (10, 100),
      'fun': 'funda_data_deal.BaseDeal.pnnd_moment',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('moment_p1060d',
     {'args': (10, 60),
      'fun': 'funda_data_deal.BaseDeal.pnnd_moment',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('moment_p20100d',
     {'args': (20, 100),
      'fun': 'funda_data_deal.BaseDeal.pnnd_moment',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('moment_p20200d',
     {'args': (20, 200),
      'fun': 'funda_data_deal.BaseDeal.pnnd_moment',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('moment_p30200d',
     {'args': (30, 200),
      'fun': 'funda_data_deal.BaseDeal.pnnd_moment',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('moment_p40200d',
     {'args': (40, 200),
      'fun': 'funda_data_deal.BaseDeal.pnnd_moment',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('moment_p50300d',
     {'args': (50, 300),
      'fun': 'funda_data_deal.BaseDeal.pnnd_moment',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('moment_p510d',
     {'args': (5, 10),
      'fun': 'funda_data_deal.BaseDeal.pnnd_moment',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('p1d_jump_hl0.030.020.01',
     {'args': ([0.03, 0.02, 0.01],),
      'fun': 'funda_data_deal.BaseDeal.p1d_jump_hl',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/NEW.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/OPEN.csv']}),
    ('pnd_continue_pct_ud_alpha345_0.5_0.5',
     {'args': ([3, 4, 5], [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.pnd_continue_pct_ud_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('pnd_continue_pct_ud_alpha345_0_1',
     {'args': ([3, 4, 5], [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.pnd_continue_pct_ud_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('pnd_continue_pct_ud_alpha345_1_0',
     {'args': ([3, 4, 5], [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.pnd_continue_pct_ud_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('pnd_continue_ud_alpha345_0.5_0.5',
     {'args': ([3, 4, 5], [0.5, 0.5]),
      'fun': 'Tech_Factor.FactorAlpha.pnd_continue_ud_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('pnd_continue_ud_alpha345_0_1',
     {'args': ([3, 4, 5], [0, 1]),
      'fun': 'Tech_Factor.FactorAlpha.pnd_continue_ud_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('pnd_continue_ud_alpha345_1_0',
     {'args': ([3, 4, 5], [1, 0]),
      'fun': 'Tech_Factor.FactorAlpha.pnd_continue_ud_alpha',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/INDEX_TD_DAILYSYS/CHG.csv']}),
    ('price_p10d_hl',
     {'args': (10,),
      'fun': 'funda_data_deal.BaseDeal.pnd_hl',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        'EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        'EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('price_p120d_hl',
     {'args': (120,),
      'fun': 'funda_data_deal.BaseDeal.pnd_hl',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        'EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        'EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('price_p20d_hl',
     {'args': (20,),
      'fun': 'funda_data_deal.BaseDeal.pnd_hl',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        'EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        'EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('price_p60d_hl',
     {'args': (60,),
      'fun': 'funda_data_deal.BaseDeal.pnd_hl',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_p_HIGH.csv',
                        'EM_Funda/DERIVED_14/aadj_p_LOW.csv',
                        'EM_Funda/DERIVED_14/aadj_p.csv']}),
    ('return_p120d_0.2',
     {'args': (120, 0.2),
      'fun': 'EM_Tab14.EM_Tab14_Deal.return_pnd',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('return_p20d_0.2',
     {'args': (20, 0.2),
      'fun': 'EM_Tab14.EM_Tab14_Deal.return_pnd',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('return_p30d_0.2',
     {'args': (30, 0.2),
      'fun': 'EM_Tab14.EM_Tab14_Deal.return_pnd',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('return_p60d_0.2',
     {'args': (60, 0.2),
      'fun': 'EM_Tab14.EM_Tab14_Deal.return_pnd',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('return_p90d_0.2',
     {'args': (90, 0.2),
      'fun': 'EM_Tab14.EM_Tab14_Deal.return_pnd',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('tab1_1_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda_test.EM_Funda_test_Deal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/dat_whs/stock_code_df_tab1_1']}),
    ('tab1_2_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda_test.EM_Funda_test_Deal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/dat_whs/stock_code_df_tab1_2']}),
    ('tab1_5_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda_test.EM_Funda_test_Deal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/dat_whs/stock_code_df_tab1_5']}),
    ('tab1_7_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda_test.EM_Funda_test_Deal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/dat_whs/stock_code_df_tab1_7']}),
    ('tab1_8_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda_test.EM_Funda_test_Deal.row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/dat_whs/data/new_factor_data/market_top_800plus_industry_55/.csv']}),
    ('tab1_9_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda_test.EM_Funda_test_Deal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/dat_whs/stock_code_df_tab1_9']}),
    ('tab2_11_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda_test.EM_Funda_test_Deal.row_extre',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/dat_whs/data/new_factor_data/market_top_800plus_industry_55/.csv']}),
    ('tab2_1_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda_test.EM_Funda_test_Deal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/dat_whs/stock_code_df_tab2_1']}),
    ('tab2_4_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda_test.EM_Funda_test_Deal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/dat_whs/stock_code_df_tab2_4']}),
    ('tab2_7_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda_test.EM_Funda_test_Deal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/dat_whs/stock_code_df_tab2_7']}),
    ('tab2_8_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda_test.EM_Funda_test_Deal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/dat_whs/stock_code_df_tab2_8']}),
    ('tab2_9_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda_test.EM_Funda_test_Deal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/dat_whs/stock_code_df_tab2_9']}),
    ('tab4_1_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda_test.EM_Funda_test_Deal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/dat_whs/stock_code_df_tab4_1']}),
    ('tab4_2_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda_test.EM_Funda_test_Deal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/dat_whs/stock_code_df_tab4_2']}),
    ('tab4_5_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda_test.EM_Funda_test_Deal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/dat_whs/stock_code_df_tab4_5']}),
    ('tab5_13_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda_test.EM_Funda_test_Deal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/dat_whs/stock_code_df_tab5_13']}),
    ('tab5_14_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda_test.EM_Funda_test_Deal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/dat_whs/stock_code_df_tab5_14']}),
    ('tab5_15_row_extre_0.3',
     {'args': (0.3,),
      'fun': 'EM_Funda_test.EM_Funda_test_Deal.row_extre',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/dat_whs/stock_code_df_tab5_15']}),
    ('turn_p120d_0.2',
     {'args': (120, 0.2),
      'fun': 'EM_Tab14.EM_Tab14_Deal.turn_pnd',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TURNRATE.csv']}),
    ('turn_p150d_0.18',
     {'args': (150, 0.18),
      'fun': 'EM_Tab14.EM_Tab14_Deal.turn_pnd',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TURNRATE.csv']}),
    ('turn_p20d_0.2',
     {'args': (20, 0.2),
      'fun': 'EM_Tab14.EM_Tab14_Deal.turn_pnd',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TURNRATE.csv']}),
    ('turn_p30d_0.24',
     {'args': (30, 0.24),
      'fun': 'EM_Tab14.EM_Tab14_Deal.turn_pnd',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TURNRATE.csv']}),
    ('turn_p60d_0.2',
     {'args': (60, 0.2),
      'fun': 'EM_Tab14.EM_Tab14_Deal.turn_pnd',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TURNRATE.csv']}),
    ('vol_count_down_p100d',
     {'args': (100,),
      'fun': 'funda_data_deal.BaseDeal.pnd_volitality_count_down',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('vol_count_down_p10d',
     {'args': (10,),
      'fun': 'funda_data_deal.BaseDeal.pnd_volitality_count_down',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('vol_count_down_p120d',
     {'args': (120,),
      'fun': 'funda_data_deal.BaseDeal.pnd_volitality_count_down',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('vol_count_down_p200d',
     {'args': (200,),
      'fun': 'funda_data_deal.BaseDeal.pnd_volitality_count_down',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('vol_count_down_p20d',
     {'args': (20,),
      'fun': 'funda_data_deal.BaseDeal.pnd_volitality_count_down',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('vol_count_down_p30d',
     {'args': (30,),
      'fun': 'funda_data_deal.BaseDeal.pnd_volitality_count_down',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('vol_count_down_p50d',
     {'args': (50,),
      'fun': 'funda_data_deal.BaseDeal.pnd_volitality_count_down',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('vol_count_down_p60d',
     {'args': (60,),
      'fun': 'funda_data_deal.BaseDeal.pnd_volitality_count_down',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('vol_count_down_p90d',
     {'args': (90,),
      'fun': 'funda_data_deal.BaseDeal.pnd_volitality_count_down',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('vol_p100d',
     {'args': (100,),
      'fun': 'funda_data_deal.BaseDeal.pnd_volitality',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('vol_p10d',
     {'args': (10,),
      'fun': 'funda_data_deal.BaseDeal.pnd_volitality',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('vol_p120d',
     {'args': (120,),
      'fun': 'funda_data_deal.BaseDeal.pnd_volitality',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('vol_p200d',
     {'args': (200,),
      'fun': 'funda_data_deal.BaseDeal.pnd_volitality',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('vol_p20d',
     {'args': (20,),
      'fun': 'funda_data_deal.BaseDeal.pnd_volitality',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('vol_p30d',
     {'args': (30,),
      'fun': 'funda_data_deal.BaseDeal.pnd_volitality',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('vol_p50d',
     {'args': (50,),
      'fun': 'funda_data_deal.BaseDeal.pnd_volitality',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('vol_p60d',
     {'args': (60,),
      'fun': 'funda_data_deal.BaseDeal.pnd_volitality',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('vol_p90d',
     {'args': (90,),
      'fun': 'funda_data_deal.BaseDeal.pnd_volitality',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/DERIVED_14/aadj_r.csv']}),
    ('volume_count_down_p10d',
     {'args': (10,),
      'fun': 'funda_data_deal.BaseDeal.pnd_volume',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('volume_count_down_p120d',
     {'args': (120,),
      'fun': 'funda_data_deal.BaseDeal.pnd_volume',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('volume_count_down_p20d',
     {'args': (20,),
      'fun': 'funda_data_deal.BaseDeal.pnd_volume',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('volume_count_down_p60d',
     {'args': (60,),
      'fun': 'funda_data_deal.BaseDeal.pnd_volume',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('volume_moment_p1040d',
     {'args': (10, 40),
      'fun': 'funda_data_deal.BaseDeal.pnnd_moment',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('volume_moment_p20120d',
     {'args': (20, 120),
      'fun': 'funda_data_deal.BaseDeal.pnnd_moment',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('volume_moment_p530d',
     {'args': (5, 30),
      'fun': 'funda_data_deal.BaseDeal.pnnd_moment',
      'if_replace': False,
      'raw_data_path': ['EM_Funda/TRAD_SK_DAILY_JC/TVOL.csv']}),
    ('wgt_return_p120d_0.2',
     {'args': (120, 0.2),
      'fun': 'EM_Tab14.EM_Tab14_Deal.wgt_return_pnd',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TURNRATE.csv']}),
    ('wgt_return_p20d_0.2',
     {'args': (20, 0.2),
      'fun': 'EM_Tab14.EM_Tab14_Deal.wgt_return_pnd',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TURNRATE.csv']}),
    ('wgt_return_p60d_0.2',
     {'args': (60, 0.2),
      'fun': 'EM_Tab14.EM_Tab14_Deal.wgt_return_pnd',
      'if_replace': False,
      'raw_data_path': ['/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv',
                        '/mnt/mfs/DAT_EQT/EM_Funda/TRAD_SK_DAILY_JC/TURNRATE.csv']})])

config_info = dict()

# config_info['index_000905'] = [
#     'evol_p10d', 'RSI_200_30', 'BBANDS_alpha_140_2_0.5_0.5', 'MACD_alpha_12_26_9_0.5_0.5', 'MACD_20_60_18',
#     'BBANDS_10_2', 'RSI_alpha40_2_0.5_0.5', 'volume_count_down_p10d', 'evol_p60d', 'wgt_return_p20d_0.2',
#     'volume_count_down_p20d'
# ]

config_info['market_top_300plus_industry_10_15'] = [
    # 'turn_p120d_0.2', 'RQMCL_p345d_continue_ud', 'RQYE_p20d_col_extre_0.2', 'MA_LINE_10_5',
    #
    # 'turn_p150d_0.18', 'RZCHE_row_extre_0.2', 'RQMCL_p345d_continue_ud', 'evol_p10d',
    # 'TVOL_row_extre_0.2', 'RZYE_p10d_col_extre_0.2',
    #
    # 'RQYE_row_extre_0.2', 'evol_p10d', 'wgt_return_p60d_0.2', 'R_OPEX_sales_QYOY_and_QTTM_0.3',
    # 'RQYE_p20d_col_extre_0.2',

]

config_info['market_top_300plus_industry_20_25_30_35'] = [

]

config_info['market_top_300plus_industry_45_50'] = [
    # 'continue_ud_p100d', 'MA_LINE_alpha_120_60_1_0', 'OPCF_and_asset_Y3YGR_Y5YGR_0.3',
    # 'aadj_r_p20d_col_extre_0.2', 'R_OTHERLASSET_QYOY_and_QTTM_0.3'
]

config_info['market_top_300to800plus'] = [

]

config_info['market_top_300to800plus_industry_20_25_30_35'] = [

]

config_info['market_top_300to800plus_industry_45_50'] = [
    # 'vol_count_down_p20d', 'evol_p100d', 'evol_p90d', 'evol_p60d', 'evol_p50d', 'evol_p30d', 'evol_p20d',
    # 'MA_LINE_alpha_120_60_0_1', 'evol_p10d', 'MA_LINE_160_60', 'price_p60d_hl',
]

config_info['market_top_300to800plus_industry_55'] = [

]
config_info['index_000905'] = [
    'vol_p10d', 'price_p20d_hl', 'wgt_return_p60d_0.2', 'vol_p10d', 'aadj_r_p120d_col_extre_0.2', 'evol_p10d',
    'vol_p20d', 'aadj_r_p345d_continue_ud', 'aadj_r_p10d_col_extre_0.2', 'MACD_20_60_18', 'continue_ud_p200d'
]

if __name__ == '__main__':
    a = time.time()
    # create_sector()
    config_name_dict = {
        # 'market_top_300plus_False_20181101_1135_hold_5__8':
        #     ['name1', 'name2', 'name3'],
        # 'market_top_300plus_industry_20_25_30_35_False_20181102_2318_hold_5__8':
        #     ['name1', 'name2', 'name3'],
        # 'market_top_300plus_industry_40_False_20181103_1403_hold_5__8':
        #     ['name1', 'name2', 'name3'],
        # 'market_top_300plus_False_20181109_1634_hold_20__8':
        #     ['name1', 'name2', 'name3'],
        # 'market_top_300to800plus_industry_20_25_30_35_False_20181107_0943_hold_5__8':
        #     ['name1', 'name2', 'name3'],
        # 'market_top_300to800plus_industry_10_15_True_20181117_2314_hold_5__7':
        #     ['name1', 'name3'],
        # 'market_top_300plus_True_20181115_1919_hold_5__7':
        #     ['name1', 'name3'],
        # 'market_top_800plus_True_20181119_0453_hold_5__7':
        #     ['name1', 'name3'],
        # 'market_top_800plus_industry_45_50_True_20181125_1657_hold_20__7':
        #     ['name1', 'name3'],
        # 'market_top_300to800plus_industry_10_15_True_20181123_0431_hold_20__7':
        #     ['name1', 'name3'],
        # 'market_top_300to800plus_True_20181122_0952_hold_20__7':
        #     ['name1', 'name3'],
        # 'market_top_300plus_True_20181204_0930_hold_5__11':
        #     ['name3'],
        # 'market_top_300plus_industry_45_50_True_20181207_0934_hold_20__11':
        #     ['name3'],
        # 'market_top_300plus_True_20181205_0955_hold_5__7_long':
        #     ['name1', 'name3'],
        # 'market_top_300to800plus_industry_45_50_True_20181205_1859_hold_5__11':
        #     ['name3'],
        # 'market_top_800plus_industry_45_50_True_20181206_2040_hold_5__11':
        #     ['name3'],
        # 'market_top_800plus_industry_55_True_20181206_2253_hold_5__11':
        #     ['name3'],
        # 'market_top_800plus_True_20181202_1830_hold_20__7':
        #     ['name1', 'name3'],
        # 'market_top_300to800plus_industry_10_15_True_20190106_0028_hold_20__19':
        #     ['name3'],
        # 'market_top_300to800plus_industry_10_15_True_20190105_0015_hold_5__19':
        #     ['name3'],
        # 'market_top_300to800plus_industry_45_50_True_20181202_1423_hold_20__7':
        #     ['name1', 'name3'],
        # 'market_top_300to800plus_True_20181227_0632_hold_20__17':
        #     ['name3'],
    }

    main(config_name_dict, config_info)
    b = time.time()
    print('pre cal cost time:{} s'.format(b - a))
