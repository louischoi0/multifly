import numpy as np


class agent :
    def __init__(self) :
        pass


CODE_COUNT = 1
input_nodes = np.repeat(0.,CODE_COUNT * 3 + 1)

# action : 0 1 2
# codes : "BTC" "XRP" ..

output_nodes = np.array([0,0,0])
COIN_OFF = slice(1,2)

def eval(input_nodes,time_series,date,yester_day) :
    value = time_series.loc[date,:].values
    standard_value = input_nodes[-1] + np.sum(input_nodes[COIN_OFF] * value)

    return  (standard_value - yester_day) * -1, standard_value

def inference(time_series,volume_series) :
    dates = time_series.index.values
    vtime_series = time_series.pct_change().fillna(0)
    vvolume_series = volume_series.pct_change().fillna(0)

    coins = np.repeat(0., CODE_COUNT)
    props = 1000

    for idx,date in enumerate(dates):
        input_nodes[0:CODE_COUNT] = vtime_series[date,:].values
        input_nodes[CODE_COUNT:CODE_COUNT*2] = vvolume_series[date,:].values
        input_nodes[CODE_COUNT*2:CODE_COUNT*3] = coins
        input_nodes[-1] = props

        loss , now = eval(input_nodes,time_series,date,now)
        pass
