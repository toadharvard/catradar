import time


def trace(f, name):
    a = time.time()
    f()
    b = time.time()
    print(name, b - a)
