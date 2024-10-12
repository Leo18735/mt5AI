import time


def time_it(func):
    def wrapper(*args, **kwargs):
        time0 = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__}: {time.time() - time0}")
        return result
    return wrapper
