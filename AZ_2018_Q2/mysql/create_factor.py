import sys
sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *
from sqlalchemy import create_engine

usr_name = 'whs'
pass_word = 'kj23#12!^3weghWhjqQ2rjj197'
engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.10:3306/choice_fndb?charset=utf8')

conn = engine.connect()

data = pd.read_sql('SELECT * FROM LICO_FN_FIXEDASSET', conn)
data.to_csv('/mnt/mfs/dat_whs/data.csv')
