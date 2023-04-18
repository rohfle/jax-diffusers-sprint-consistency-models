import time
# a decorator to time functions
def timeit(func):
    def runit(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        print("time ", func.__name__, ":", duration)
        return result
    return runit
