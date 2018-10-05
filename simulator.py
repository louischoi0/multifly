from evaluator import *
from routine import *
import sys
import matplotlib.pyplot as plt
import evaluator
from copy import deepcopy

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

class simulator :
    def __init__(self,time_series,volume_series,valuator_instance) :
        self.time_series = time_series
        self.volume_series = volume_series
        self.evaluator_instance = evaluator_instance
        self.standard_value = 1000
        self.position = 0
        self.coin = 0
        self.rdelta = 100
        self.history = []
        self.mdelta = 10
        self.ventry = [0.]

    def rebalance(self,props,stime_series) :
        idx, = np.where( self.position == 0 )
        weights = stime_series.iloc[-1,idx].values / stime_series.iloc[-self.mdelta,idx].values
        weights -= 1

        if np.any( weights < 0 ) :
            weights += np.abs(np.min(weights))

        if np.sum(weights) == 0 :
            self.ventry[:] = 0
            return

        weights = weights / np.sum(weights)

        self.ventry[:] = 0
        self.ventry[idx] = weights * props

    def invest(self) :
        history = []
        index = self.time_series.index.values[200:]
        time_series = self.time_series.loc[:,CODES]
        volume_series = self.volume_series
        props = self.standard_value
        code_count = len(CODES)

        self.ventry = np.array([0.] * code_count)
        self.position = np.array([0.] * code_count)
        self.coins = np.array([0.] * code_count)

        evaluator_instance = [*map(lambda x : deepcopy(self.evaluator_instance), range(code_count))]

        for idx,date in enumerate(index) :
            stime_series = time_series.loc[:date,:]
            svolume_series = volume_series.loc[:date,:]
            nows = time_series.loc[date,:].values

            self.rebalance(props,stime_series)

            def routine(ei,position,props,coins,code):
                now = time_series.loc[date,code]
                _position = position
                _props = props

                if position == 0:
                    signal = ei.eval_buy_condition(stime_series,svolume_series,code,date)
                else :
                    signal = ei.eval_sell_condition(stime_series,svolume_series,code,date)

                if signal and position == 0 :
                    invest_prop = props

                    if props <= 0 :
                        return position,props,coins

                    coins += invest_prop * 0.997 / now

                    _props -= invest_prop
                    _position = 1
                    print("{} Buy {} {} at {}".format(date,coins,code,now))

                elif signal and position == 1 :

                    if coins <= 0 :
                        return position,props,coins

                    _props += coins * now
                    _position = 0
                    print("{} Cell {} {} at {}".format(date,coins,code,now))
                    coins = 0

                return _position,_props,coins

            for idx, code in enumerate(CODES) :
                po,p,c = routine(evaluator_instance[idx],self.position[idx],self.ventry[idx],self.coins[idx],code)
                self.position[idx] = po
                self.ventry[idx] = p
                self.coins[idx] = c

            props = np.sum(self.ventry)

            self.standard_value = props + np.sum(self.coins * nows)
            history.append(self.standard_value)

        return history

    def invest_single_coin(self,code) :
        index = self.time_series.index.values[200:]
        time_series = self.time_series
        volume_series = self.volume_series

        for idx,date in enumerate(index) :
            stime_series = time_series.loc[:date,:]
            svolume_series = volume_series.loc[:date,:]

            now = time_series.loc[date,code]
            res = -1

            if self.position == 0 :
                signal = evaluator_instance.eval_buy_condition(stime_series,svolume_series,code,date)

            else :
                signal = evaluator_instance.eval_sell_condition(stime_series,svolume_series,code,date)

            if signal and self.position == 0 :
                #invest_prop = k_control(self.standard_value,code,stime_series,svolume_series,date)
                invest_prop = self.standard_value

                if invest_prop < 100 :
                    continue

                self.coin = invest_prop * 0.997 / now
                self.standard_value -= invest_prop
                self.position = 1

                print("{} Buy {} coins at {}".format(date,self.coin,now))

            elif signal and self.position == 1 :
                self.standard_value += self.coin * now
                self.position = 0
                print("{} Cell {} coins at {}".format(date,self.coin,now))
                self.coin = 0

            eval = now * self.coin + self.standard_value
            self.history.append(eval)

        return np.array(self.history)

if __name__ == "__main__" :
    strat = sys.argv[1]
    bit_code = sys.argv[2] if len(sys.argv) > 2 else None

    time_series = get_time_series("BIT2.db","CLOSE_4H")
    volume_series = get_volume_series("BIT2.db","VOLUME_4H")

    evaluator_instance = getattr(evaluator,strat)(time_series,volume_series)
    simulator_instance = simulator(time_series.loc[:,CODES],volume_series,evaluator_instance)

    history = simulator_instance.invest() if bit_code is None else simulator_instance.invest_single_coin(bit_code)

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
