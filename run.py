import sqlite3
from evaluator import volDash
from datetime import datetime,timedelta
import numpy as np
import pandas as pd
import numpy as np
import tensorflow as tf
import sys
from simulator import *

FORMAT = "%Y-%m-%d-%H-%M"

class dataware :
    def __init__(self,db_name) :
        self.con = sqlite3.connect(db_name)

    def get_series(self,table_name,start) :
        return pd.read_sql(sql="select * from {} where time >= \"{}\"; ".format(table_name,start),con=self.con,index_col="time")

class maestro :
    def __init__(self,group_count) :
        self.num_code = group_count
        self.score = np.repeat(0. , group_count)
        self.puts = np.repeat(0. , group_count)
        self.metaw = np.repeat(1. , group_count)
        self.coins = np.repeat(0. , group_count)
        self.props = 1000

    def arrange_puts(self) :
        self.props += np.sum(self.puts)
        self.puts[:] = np.sum(self.props) / self.num_code
        self.props = 0

class nueros :
    def __init__(self) :
        pass

    def run(self,batch,labels):
        vseries = np.diff(batch[0])
        vvolume = np.diff(batch[1])

        labels = np.array(labels)

        labels_ = tf.placeholder(shape=(1),dtype=tf.float32)
        input_ = tf.placeholder(shape=(1,1,9), dtype=tf.float32)

        cell = tf.contrib.rnn.BasicLSTMCell(num_units=1)
        outputs, state = tf.nn.dynamic_rnn(cell,input_,dtype=tf.float32)


        softmax_x = tf.reshape(outputs,[1,1])
        softmax_w = tf.get_variable("softmax_w", [1,1])
        softmax_b = tf.get_variable("softmax_b", [1])

        outputs = tf.matmul(softmax_x,softmax_w) + softmax_b
        outputs = tf.reshape(outputs,[1])

        result = tf.losses.mean_squared_error(labels_,outputs)

        loss = tf.reduce_mean(result)
        opt = tf.train.AdamOptimizer(0.01).minimize(loss)

        sess = tf.Session()
        init = tf.global_variables_initializer()

        sess.run(init)
        record = []

        for i in range(0,100) :
            for k in range(0,15) :
                input = vseries[k, :]
                input = np.reshape(input,[1,1,9])
                label = labels[k]

                output_ , state_, loss_ , _ = sess.run([outputs,state,loss,opt],feed_dict={labels_:[label],input_:input})

                if i == 99 :
                    record.append(output_[0])

        print(loss_)
        print(labels)
        print(record)


CODES = ["BTC","EOS","ETC","BCH","XRP","ZRX","LTC"]
CODES = ["EOS","XRP"]

class runner :
    def __init__(self,dw,instance_name,codes) :
        self.dw = dw
        self.mi = maestro(len(CODES))
        self.codes = codes
        self.ob = 36
        self.nueros = nueros(codes)

    def run(self,eval_class,start,unit,tick) :
        surfix = "H" if unit == "hour" else "M"
        table_name_s = "CLOSE_{}{}".format(str(tick),surfix)
        table_name_v = "VOLUME_{}{}".format(str(tick),surfix)

        rstart = datetime.strptime(start,FORMAT)
        rstart = rstart - timedelta(hours=self.ob)
        rstart = rstart.strftime(FORMAT)

        self.target_series = self.dw.get_series(table_name_s,rstart).loc[:,self.codes]
        self.volume_series = self.dw.get_series(table_name_v,rstart).loc[:,self.codes]

        targets = [*map(lambda x : self.target_series.loc[:,x], self.codes ) ]
        volumes = [*map(lambda x : self.target_series.loc[:,x], self.codes ) ]
        self.evaluators = [*map(lambda t,v,c : eval_class(t,v,c) , targets,volumes,self.codes )]

        date_array = self.target_series.loc[start:,:].index.values

        for idx,date in enumerate(date_array) :
            self.day_routine(date)

    def day_routine(self,date) :
        print(date)
        self.mi.arrange_puts()
        stime_series = self.target_series.loc[:date, :]
        svolume_series = self.volume_series.loc[:date,:]
        evaluators = self.evaluators

        results = [*map( lambda x,c,i : x.eval_condition(stime_series.loc[:,c],svolume_series.loc[:,c],date,self.mi.puts[i]), evaluators, self.codes,range(len(self.codes)))]
        self.eval_results(results,date)
        print(self.mi.standard)

    def eval_results(self, results,date) :
        now_v = self.target_series.loc[date, : ].values

        for idx,result in enumerate(results) :
            signal = result[0]
            action = result[1]

            self.eval_routine(signal,action,idx,now_v)

        self.mi.standard = np.sum(self.mi.puts) + self.mi.props + np.sum(self.mi.coins * now_v)

    def eval_routine(self,signal,action,idx,now_v) :
        if action == "sell" and signal :
            print("Sell {} Coin {} at {}".format(self.codes[idx],self.mi.coins[idx],now_v[idx]))
            self.mi.props += self.mi.coins[idx] * now_v[idx]
            self.mi.coins[idx] = 0

        elif action == "buy" and signal :
            now_c = self.mi.puts[idx] / now_v[idx]
            print("Buy {} Coin {} at {}".format(self.codes[idx],now_c,now_v[idx]))
            self.mi.coins[idx] += now_c
            self.mi.puts[idx] = 0

def slice_from_signal(rlist, series, vseries,delta) :
    rlist = np.array(rlist)
    signal_nums, = np.where( rlist[:,0] > 10 )
    signal_nums = len(signal_nums)

    batch = np.zeros([2,signal_nums,delta])
    rlist = rlist[ len(rlist) - signal_nums :]

    for idx,v in enumerate(rlist) :
        if v[0] > 10 :
            batch[0,idx,:] = series[v[0]-delta:v[0]]
            batch[1,idx,:] = vseries[v[0]-delta:v[0]]

    return batch

def make_batch(series,vseries) :
    r = 1000
    rr = 1000
    rlist = []
    rrlist = []
    labels = []

    for i in range(0,200) :
        res = get_vol_max_point(series[i*2:i*2+25],vseries[i*2:i*2+25],15)
        r *= res[1] + 1

        if res[1] > 0.05 :
            rr *= res[1] + 1
            rlist.append((res[2] + i*2, res[3] + i*2))
            labels.append(res[1])

        rrlist.append((res[2] + i*2, res[3] + i*2))

    return rlist,rrlist,rr,r,labels

if __name__ == "__main__" :
    dw = dataware("BIT3.db")

    series = dw.get_series("CLOSE_4H","2018-08-05-00").loc[:,"XRP"]
    vseries = dw.get_series("VOLUME_4H","2018-08-05-00").loc[:,"XRP"]

    r,rr,r0,r1,labels  = make_batch(series,vseries)
    batch = slice_from_signal(r, series, vseries,10)

    n = nueros()
    n.run(batch,labels)

    sys.exit(-1)

    runner_instance = runner(dw,"irina",CODES)
    runner_instance.run(volDash,"2018-09-05-00-00","hour",4)
