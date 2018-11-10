from evaluator import *
from routine import *
import sys
import matplotlib.pyplot as plt
import evaluator
from copy import deepcopy
import sqlite3


CONNECTION = sqlite3.connect("BIT3.db")

CODES = ["BTC","EOS","ETC","BCH","XRP","ZRX","LTC"]
CODES0 = ["BTC","EOS","ZRX","BCH"]
CODES1 = ["XRP","EOS","ZRX","ETC","LTC"]
CODES2 = ["BCH","LTC","ETH","ETC"]
CODES3 = ["BCH","BTC","LTC","ETC"]
CODES4 = ["ETH","LTC","ETC","BCH","BTC"]
CODES5 = ["ETH","LTC","ETC","EOS"]

CODES = CODES

def MDD(series) :
    p = 0
    mdd = 0

    for i,t in enumerate(series) :
        p = p if t < p else t
        tmdd = t / p - 1
        mdd = tmdd if tmdd < mdd else mdd

    return mdd * -100

def get_max_up_point(series) :
    p = series[0]
    mud = 0

    in_loc = 0
    out_loc = 0

    tmud = mud
    res = [(0,0,0)]

    for i,t in enumerate(series) :
        in_loc = i if t < p else in_loc
        p = p if t > p else t

        tmud = (t / p) - 1

        out_loc = i if tmud > mud else out_loc
        mud = tmud if tmud > mud else mud

        if res[-1][0] < mud and (res[-1][1] != in_loc or res[-1][2] != out_loc) :
            res.append([mud,in_loc,out_loc])

    res = np.array(res)

    out_loc = int(res[-1][2])
    in_loc = int(res[-1][1])

    return in_loc , out_loc

class vinfo :
    def __init__(self,atrr,vdr,nl,nu) :
        self.atrr = atrr
        self.vdr = vdr
        self.nl = nl
        self.nu = nu
        self.term = 0

def get_vol_max_point(time_series,volume_series,delta=10) :
    net_upper = 1.1
    net_lower = 97
    vol_sig = 1.3

    volumes = volume_series.values
    index = time_series.index.values

    series = time_series.values
    i , o = get_max_up_point(series[delta:])

    in_date = index[i]
    out_date = index[o]

    #vdr = volumes[i-1] / volumes[i-10:i-1].mean()
    vdr = 0

    nu = series[o] / series[i] - 1
    atr = ATR(time_series,delta)
    nl = 0

    return atr,nu,i,o

def insert_signals(con,name,sig) :
    sig.to_sql(con=con,name=name,index_label="time",if_exists="append")

if __name__ == "__main__" :
    strat = sys.argv[1]
    bit_code = sys.argv[2] if len(sys.argv) > 2 else None

    time_series = get_time_series("BIT2.db","CLOSE_4H")
    volume_series = get_volume_series("BIT2.db","VOLUME_4H")

    evaluator_instance = getattr(evaluator,strat)(time_series,volume_series)
    simulator_instance = simulator(time_series.loc[:,CODES],volume_series,evaluator_instance)

    history = simulator_instance.invest() if bit_code is None else simulator_instance.invest_single_coin(bit_code)
    #insert_signals(CONNECTION,"standard_irina",history)
    history = history.values

    print(history[-1])
    mdd = MDD(history)
    print(mdd)

    hsize = len(history)
    x = range(hsize)

    fig = plt.figure()

    ax = fig.add_subplot(211)

    stan = time_series.iloc[-len(history):,:].pct_change().fillna(0).values + 1
    stan = np.cumprod(stan,axis=0)
    stan *= 1000
    c = time_series.columns.values

    #ax.plot(x,stan[:,0],label=c[0])
    #ax.plot(x,stan[:,1],label=c[1])
    #ax.plot(x,stan[:,2],label=c[2])
    #ax.plot(x,stan[:,3],label=c[3])
    #ax.plot(x,stan[:,4],label=c[4])
    #ax.plot(x,stan[:,5],label=c[5])
    #ax.plot(x,stan[:,6],label=c[6])
    ax.plot(x,history,label="portfolio")
    ax.legend(loc="upper left")
    plt.show()
