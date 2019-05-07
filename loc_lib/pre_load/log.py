import time

# decoration


def use_time(func):
    def _deco(*args, **wkargs):
        start_time = time.clock()
        target = func(*args, **wkargs)
        end_time = time.clock()
        print('processing:{}'.format(end_time - start_time))
        return target
    return _deco


def try_catch(func):
    def _deco(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
            return res
        except Exception as error:
            print(error)
            return error
    return _deco

# def log(func):
