import time


def trace(f, name):
    a = time.time()
    res = f()
    b = time.time()
    print(name, b - a)
    return res
