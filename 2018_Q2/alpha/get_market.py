import pandas as pd
import os
import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4])
plt.show()

# load_path = r'/media/hdd0/data/raw_data/equity/extraday/taobao'
#
# stock_list = os.listdir(load_path)
# target = pd.DataFrame()
# for stock in stock_list:
#     print(stock)
#     if '.csv' in stock:
#         data = pd.read_csv(os.path.join(load_path, stock), index_col=2, encoding='gbk')['流通市值']
#         data.name = stock.split('.')[0]
#         target = pd.concat([target, data], axis=1)
# target.index.name = None
# target.to_csv('/media/hdd0/data/raw_data/equity/extraday/choice/index/AllStock/all_market.csv')
