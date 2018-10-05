import pandas as pd
import sqlite3

def get_time_series(db,table) :
    con = sqlite3.connect(db)
    return pd.read_sql(sql="select * from {}".format(table) , con=con,index_col="index")

def get_volume_series(db,table) :
    con = sqlite3.connect(db)
    return pd.read_sql(sql="select * from {}".format(table) , con=con,index_col="index")
