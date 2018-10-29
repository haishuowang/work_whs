import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine


usr_name = 'whs'
pass_word = 'kj23#12!^3weghWhjqQ2rjj197'
engine = create_engine(
    'mysql+pymysql://{}:{}@192.168.16.10:3306/choice_fndb?charset=utf8'.format(usr_name, pass_word))

conn = engine.connect()

df = pd.read_sql('SELECT * FROM TRAD_FD_DAILY WHERE SECINNERCODE=510330', conn)

