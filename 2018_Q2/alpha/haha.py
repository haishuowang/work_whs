from multiprocessing import Pool

import copy
#
# def parameter_fun():
#     for vol_num in iter([2, 3, 4]):
#         for hold_time in iter([6, 8, 10]):
#             for n in iter([5, 10, 15]):
#                 for limit in iter([1.8, 2, 2.2, 2.4]):
#                     for stock_num in iter([50, 100, 200]):
#                         yield vol_num, hold_time, n, limit, stock_num
#
#
# pool = Pool(4)
#
#
# def fu2(price, vol_num, hold_time, n, limit, stock_num):
#     print(price, vol_num, hold_time, n, limit, stock_num)
#
#
# price = 'hahaha'
#
# for vol_num, hold_time, n, limit, stock_num in parameter_fun():
#     pool.apply_async(fu2, args=(price, vol_num, hold_time, n, limit, stock_num))
# pool.close()
# pool.join()

# haha = lambda x: x**2


def haha(x):
    print(x)
    return x**2


pool = Pool(4)
a = [0] * 10
for i in range(10):
    a[i] = pool.apply_async(haha, (i,)).get()
pool.close()
pool.join()

