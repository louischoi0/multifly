import requests
import sqlite3
import pandas as pd
import sys
import json
from datetime import datetime
import sys

FORMAT = "%Y-%m-%d-%H-%M"

def get_limit(last_time,tick="hour") :
    last = datetime.strptime(last_time,FORMAT)
    now = datetime.now().strftime(FORMAT)
    now = datetime.strptime(now,FORMAT)
    delta = now - last
    hour_delta = int((delta.days * 24) + (delta.seconds / 3600))

    return hour_delta

def get_from_to(bit_code,unit,limit=2000,tick=1,xchange="UPBIT") :
    url = "https://min-api.cryptocompare.com/data/histo{}?fsym={}&tsym=KRW&limit={}&aggregate={}&e=CCCAGG".format(str(unit),bit_code,str(limit),str(tick))
    response = requests.get(url)
    values = response.text
    values = json.loads(values)
    values = values["Data"]

    table = pd.DataFrame(columns=["close","volume"])

    for v in values :
        date = datetime.fromtimestamp(v["time"]).strftime("%Y-%m-%d-%H-%M")
        values = [ v["close"],v["volumeto"] ]
        table.loc[date] = values

    print(table.index.size)
    return table


def get_append_to(con,unit,tick,init=False) :
    if not init :
        surfix = "H" if unit == "hour" else "M"
        last_time = pd.read_sql(con=con,sql="select * from close_{}{} order by \"time\" desc limit 1".format(str(tick),surfix))
        last_time = last_time.iloc[-1,:].loc["time"]
        delta = int(get_limit(last_time))

        if tick == "minute" :
            delta *= 60

        delta /= tick
        delta = int(delta) - 1

    else:
        delta = 2000

    codes = ["BTC","EOS","ZRX","BCH","XRP","ETC","LTC","OMG"]

    if delta < 1 :
        return

    tables = [ *map(lambda x : get_from_to(x,unit,delta,tick) , codes ) ]

    close_table = pd.DataFrame(columns=codes)
    volume_table = pd.DataFrame(columns=codes)

    for idx,table in enumerate(tables) :
        close_table[codes[idx]] = table["close"]
        volume_table[codes[idx]] = table["volume"]

    unit_str = "H" if unit == "hour" else "M"

    print(close_table)

    close_table.to_sql(name="CLOSE_{}{}".format(str(tick),unit_str),con=con,index="time",if_exists="append")
    volume_table.to_sql(name="VOLUME_{}{}".format(str(tick),unit_str),con=con,if_exists="append",index="time")

if __name__ == "__main__" :
    CONNECTION = sqlite3.connect("BIT3.db")
    unit = sys.argv[1]
    tick = int(sys.argv[2])
    init = True if len(sys.argv) > 3 and sys.argv[3] == "init" else False

    get_append_to(CONNECTION,unit,tick,init)
