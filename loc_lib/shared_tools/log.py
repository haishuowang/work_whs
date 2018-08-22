import time


def AZ_use_time(fn):
    """
    计算程序运行时间
    :param fn:
    :return:
    """
    def tmp_fun(*args, **wkargs):
        start_time = time.clock()
        target = fn(*args, **wkargs)
        end_time = time.clock()
        print('processing:{}'.format(end_time - start_time))
        return target
    return tmp_fun


def pass_error(fn):
    """
    跳过出错程序
    :param fn:
    :return:
    """
    def tmp_fun(*args, **wkargs):
        try:
            target = fn(*args, **wkargs)
            return target

        except BaseException:
            print('some error!')
    return tmp_fun
