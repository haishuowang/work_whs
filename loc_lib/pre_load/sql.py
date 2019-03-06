from sqlalchemy import create_engine
usr_name = 'whs'
pass_word = 'kj23#12!^3weghWhjqQ2rjj197'
engine = create_engine(f'mysql+pymysql://{usr_name}:{pass_word}@192.168.16.10:3306/choice_fndb?charset=utf8')

conn = engine.connect()
