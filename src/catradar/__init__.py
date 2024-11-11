import taichi as ti

ti.init(arch=ti.gpu)


def hello() -> str:
    return "Hello from catradar!"
