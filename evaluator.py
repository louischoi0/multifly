import functools
import numpy as np
import math
from datetime import datetime
import sys

def MOM(time_series,delta)  :
    series = time_series.iloc[-delta:].values
    return series[-1] / series[0]

def RSI(time_series,delta) :
    xseries = time_series.iloc[-delta:]
    yseries = time_series.iloc[-delta-1:]

    AU = [*map(lambda x,y : x - y if x > y else np.nan, xseries,yseries)]
    DU = [*map(lambda x,y : y - x if x < y else np.nan, xseries,yseries)]

    AU = np.array([*filter( lambda x : ~np.isnan(x) , AU )])
    DU = np.array([*filter( lambda x : ~np.isnan(x) , DU )])

    AU = AU.mean() if AU.size != 0 else 0
    DU = DU.mean() if DU.size != 0 else 0

    return AU / (AU+DU) if (AU+DU) != 0 else 0

def MOVEV_CLOSE(time_series,delta) :
    series = time_series.iloc[-delta:]
    return series.mean()

def ATR(time_series,delta) :
    series = time_series.iloc[-delta:]
    atrm = np.array([*map(lambda x : series.iloc[-delta+x]/ series.iloc[-delta+x-1], range(1,delta))])
    return atrm.mean(axis=0)

class volDash :
    def __init__(self,time_series,volume_series) :
        self.time_series = time_series
        self.volume_series = volume_series
        self.delta = 10
        self.buy_time = None
        self.time_cut = 96
        self.buy_price = None
        self.volume_delta = 10

        self.net_upper = 1.12
        self.net_lower = 0.96

    def eval_buy_condition(self,stime_series,volume_series,code,date) :
        time_series = stime_series.loc[:,code]
        volume_series = volume_series.loc[:,code]

        atr = ATR(time_series,self.delta)
        now = time_series.iloc[-1]
        before = time_series.iloc[-2]

        vol_diff = volume_series.iloc[-self.delta:].mean() / volume_series.iloc[-self.delta]
        signal = now > before + 0.5 * atr and vol_diff > 1.26

        if signal :
            self.buy_time = datetime.strptime(date,"%Y-%m-%d-%H-%M")
            self.buy_price = now

        return signal

    def eval_sell_condition(self,time_series,volume_series,code,date) :
        now_time = datetime.strptime(date,"%Y-%m-%d-%H-%M")
        diff = now_time - self.buy_time
        now = time_series.iloc[-1].loc[code]

        signal = ((diff.days * 24) + diff.seconds / 3600 ) > self.time_cut or now > self.buy_price * self.net_upper or now < self.buy_price * self.net_lower
        return  signal

def k_control(props,code,time_series,volume_series,date) :
    b = time_series.iloc[-1,:].loc[code]
    a = time_series.iloc[-2,:].loc[code]

    c = (b - a) / a

    k = props * 0.5 / props / (1-c)

    return props * 0.5 / (1-c)

class unMomentum :
    def __init__(self,time_series,v) :
        self.time_series = time_series
        self.rsi_delta = 10
        self.me0_delta = 200
        self.me1_delta = 5
        self.rsi_alpha = 5

    def eval_buy_condition(self,time_series,v,code,date) :
        time_series = time_series.loc[:,code]
        mvv_long = MOVEV_CLOSE(time_series,self.me0_delta)
        mvv_short = MOVEV_CLOSE(time_series,self.me1_delta)

        rsiv = RSI(time_series,self.rsi_delta)
        now = time_series.loc[date]

        return now > mvv_long and now < mvv_short and rsiv < self.rsi_alpha

    def eval_sell_condition(self,time_series,v,code,date) :
        time_series = time_series.loc[:,code]
        mvv_short = MOVEV_CLOSE(time_series,self.me1_delta)
        now = time_series.loc[date]

        return now > mvv_short

class Momentum :
    def __init__(self,time_series,v) :
        self.time_series = time_series
        self.mom_delta = 100

    def eval_buy_condition(self,time_series,v,code,date):
        time_series = time_series.loc[:,code]
        mom_score = MOM(time_series,self.mom_delta)
        return mom_score > 1.03

    def eval_sell_condition(self,time_series,v,code,date) :
        time_series = time_series.loc[:,code]
        mom_score = MOM(time_series,self.mom_delta)
        return mom_score < 0.94

    def get_weights(self,stime_series) :
        time_series = time_series.loc[:,code]
        MOMS = [*map(lambda x : MOM(stime_series,x*10), range(1,4) )]
        MOM_SCORE = np.sum(MOMS,axis=0)
        MOM_SCORE /= np.sum(MOM_SCORE)

class signal :
    def __init__(self,time_series) :
        pass
