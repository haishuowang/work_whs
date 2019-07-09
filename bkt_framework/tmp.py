import sys

sys.path.append('/mnt/mfs')
import string
from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load.plt import plot_send_result
from work_whs.bkt_factor_create.raw_data_path import base_data_dict
import warnings
warnings.filterwarnings("ignore")


class B:
    def __init__(self):
        self.test_list = {}

    def haha(self):
        k = random.random()
        self.test_list[str(k)] = k
        print(self.test_list)

    def x(self):
        pool = Pool(20)
        for i in range(10):
            pool.apply_async(self.haha)
        pool.close()
        pool.join()

    def run(self):
        self.x()


class FactorTest:
    def __init__(self):
        self.test_list = {}

    def haha(self):
        k = random.random()
        self.test_list[str(k)] = k
        print(self.test_list)

    def x(self):
        pool = Pool(20)
        for _ in range(20):
            pool.apply_async(self.haha)
        pool.close()
        pool.join()

    def run(self):
        self.x()


class C:
    def __init__(self):
        import multiprocessing
        mgr = multiprocessing.Manager()
        self.test_list = mgr.dict()

    def haha(self):
        k = random.random()
        self.test_list[str(k)] = k
        print(self.test_list)

    def x(self):
        pool = Pool(20)
        for i in range(10):
            pool.apply_async(self.haha)
        pool.close()
        pool.join()

    def run(self):
        self.x()


c = C()
c.run()
for i in c.test_list.items():
    print(i)


# factor_test = FactorTest()
# factor_test.run()
# print(factor_test.test_list)
