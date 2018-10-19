from tmp_script import a1, a2
from multiprocessing import Pool
pool = Pool(20)
for fun in [a1, a2]:
    print(fun.main(pool))
pool.close()
pool.join()
