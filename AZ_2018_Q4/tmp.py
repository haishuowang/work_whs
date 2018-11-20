import dask.dataframe as dd
import pandas as pd
import time

data_1 = pd.DataFrame([[1] * 2500] * 3000)


def fun(x):
    time.sleep(0.000000001)
    return sum(x)


a1 = time.time()
result_1 = data_1.rolling(100, min_periods=0).apply(fun)
a2 = time.time()
print(a2 - a1)

for ncore in [2, 4, 6, 8, 10, 12, 14, 16]:
    b1 = time.time()
    result_2_t = dd.from_pandas(data_1, npartitions=ncore).map_partitions(
        lambda df: (df.T.rolling(100, min_periods=0).apply(fun)).T).compute()
    result_2 = result_2_t.T
    b2 = time.time()
    print(f'{ncore}:{b2 - b1}')


def AZ_Rolling_mean_multi(data, window, func, ncore=4):
    result_t = dd.from_pandas(data, npartitions=ncore).map_partitions(
        lambda df: (df.T.rolling(window, min_periods=0).apply(func)).T).compute()
    result = result_t.T
    return result
