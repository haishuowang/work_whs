import os
# 数据根目录路径
DATA_ROOT_PATH =    '/media/hdd0/data'
DATA_EQT_ROOT =     os.path.join(DATA_ROOT_PATH, 'EQT')
DATA_FUT_ROOT =     os.path.join(DATA_ROOT_PATH, 'EQT')
DATA_OPT_ROOT =     os.path.join(DATA_ROOT_PATH, 'EQT')

# # 处理过的数据原始数据的路径
adj_path = os.path.join(DATA_ROOT_PATH, 'adj_data')
raw_path = os.path.join(DATA_ROOT_PATH, 'raw_data')

# # # 股票与期货路径
adj_eqt_path = os.path.join(adj_path, 'equity')
adj_fut_path = os.path.join(adj_path, 'future')

raw_eqt_path = os.path.join(raw_path, 'equity')
raw_fut_path = os.path.join(raw_path, 'future')

# # # # 日内日间数据
adj_eqt_extra_path = os.path.join(adj_eqt_path, 'extraday')
adj_eqt_intra_path = os.path.join(adj_eqt_path, 'intraday')

extra_fut_adj_path = os.path.join(adj_fut_path, 'extraday')
intra_fut_adj_path = os.path.join(adj_fut_path, 'intraday')

raw_eqt_extra_path = os.path.join(raw_eqt_path, 'extraday')
raw_eqt_intra_path = os.path.join(raw_eqt_path, 'intraday')

raw_fut_extra_path = os.path.join(raw_fut_path, 'extraday')
raw_fut_intra_path = os.path.join(raw_fut_path, 'intraday')

# # # # # 具体数据
adj_eqt_indicator_path = os.path.join(adj_eqt_extra_path, 'indicator')
adj_eqt_1mbar_path = os.path.join(adj_eqt_extra_path, 'eqt_1mbar')

raw_eqt_extra_day_path = os.path.join(raw_eqt_extra_path, 'choice', 'day')
raw_eqt_extra_index_path = os.path.join(raw_eqt_extra_path, 'choice', 'index')

raw_eqt_1mbar_path = os.path.join(raw_eqt_intra_path, 'eqt_1mbar_raw')
raw_fut_day_path = os.path.join(raw_fut_extra_path, 'day')
