from multiprocessing import Pool
import time


def test(x):
    print('a1', x)
    time.sleep(1 / 2)


def main(pool):
    for i in range(10):
        pool.apply_async(test, args=(i,))
