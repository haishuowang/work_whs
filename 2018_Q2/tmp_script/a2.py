from multiprocessing import Pool
import time


def test(x):
    print('a2', x)
    time.sleep(1/2)


def main_multy(pool):
    for i in range(10):
        pool.apply_async(test, args=(i,))


def main():
    for i in range(10):
        test(i,)
