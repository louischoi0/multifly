import requests
import sqlite3
import pandas as pd
import sys
import json
from datetime import datetime
import sys

def get_from_to(connection,bit_code,exchange="UPBIT") :
    url = "https://min-api.cryptocompare.com/data/histohour?fsym={}&tsym=KRW&limit=2000&aggregate=4&e=CCCAGG".format(bit_code)
    response = requests.get(url)
    values = response.text
    values = json.loads(values)
    values = values["Data"]

    table = pd.DataFrame(columns=["close","volume"])

    for v in values :
        date = datetime.fromtimestamp(v["time"]).strftime("%Y-%m-%d-%H-%M")
        values = [ v["close"],v["volumeto"] ]
        table.loc[date] = values
        print(values)

    return table


if __name__ == "__main__" :
    CONNECTION = sqlite3.connect("BIT2.db")

    codes = ["BTC","EOS","ZRX","BCH","XRP","ETC","LTC"]
    tables = [ *map(lambda x : get_from_to(CONNECTION,x) , codes ) ]

    close_table = pd.DataFrame(columns=codes)
    volume_table = pd.DataFrame(columns=codes)

    for idx,table in enumerate(tables) :
        close_table[codes[idx]] = table["close"]
        volume_table[codes[idx]] = table["volume"]

    close_table.to_sql(name="CLOSE_4H",con=CONNECTION,if_exists="replace")
    volume_table.to_sql(name="VOLUME_4H",con=CONNECTION,if_exists="replace")
