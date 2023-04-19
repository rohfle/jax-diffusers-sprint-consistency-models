import time
import jax
# a decorator to time functions
def timeit(func):
    def runit(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        print("time ", func.__name__, ":", duration)
        return result
    return runit

def timediter(iterable, label):
    it = iter(iterable)
    while True:
        try:
            start = time.time()
            value = next(it)
            end = time.time()
            print(f"time {label} yield: {end - start:.6f} seconds")
            yield value
        except StopIteration:
            break

def profile(name, limit=3):
    def outer(func):
        left = limit
        def runit(*args, **kwargs):
            if left > 0:
                with jax.profiler.trace(name, create_perfetto_link=True):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return runit
    return outer

def profileiter(iterable, label):
    it = iter(iterable)
    try:
        with jax.profiler.trace(label, create_perfetto_link=True):
            yield next(it)

        while True:
            yield next(it)
    except StopIteration:
        pass
