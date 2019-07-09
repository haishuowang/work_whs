import sys

sys.path.append('/mnt/mfs')
import string
from work_whs.loc_lib.pre_load import *


def fun(x):
    return x ** 2


if __name__ == '__main__':
    pool = Pool(15)
    result_list = []
    for i in range(10):
        result_list.append(pool.apply_async(fun, args=(i,)))
    pool.close()

    pool.join()
    a = [x.get() for x in result_list]
    print(a)
