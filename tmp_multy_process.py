from multiprocessing import Pool


def fun(x):
    return x ** 2


# 创建6个进程池
pool = Pool(6)
for i in range(10):
    # 异步的加入进程池
    pool.apply_async(fun, args=(i,))
# 进程池关闭
pool.close()
# 在等待所有子进程运行完毕
pool.join()

