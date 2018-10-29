from multiprocessing import Pool
import time


def func(i):
    print(111111111111111111)
    return i


result_list = []
pool = Pool(4)
for i in range(10):
    result_list.append(pool.apply_async(func, args=(i,)))
pool.close()
pool.join()

for i in result_list:
    print(i.get())