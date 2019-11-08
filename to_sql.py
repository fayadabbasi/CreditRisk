import psycopg2 as pg2
import pandas as pd
import numpy as np

from sqlalchemy import create_engine
import time

engine = create_engine('postgresql+psycopg2://postgres:docker@localhost:5432/creditrisk')

def to_sql(df):
    df.to_sql('creditrisk',engine, if_exists='append',index=False)
    time.sleep(15)