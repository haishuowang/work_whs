from datetime import datetime
import time
import pandas as pd
import numpy as np
from multiprocessing import Pool, Manager
import time


def fun(x):
    print(1)
    x.value * x.value
    # x * x
    print(2)
    time.sleep(3)


manager = Manager()

df = manager.Value(pd.DataFrame, pd.DataFrame([[1] * 10000, ] * 1000), lock=False)
# df = pd.DataFrame([[1] * 10000, ] * 1000)
t1 = time.time()
pool = Pool(5)
for i in range(10):
    pool.apply_async(fun, args=(df,))
pool.close()
pool.join()
t2 = time.time()
print(t2-t1)