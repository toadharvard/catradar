import taichi as ti


def hello() -> str:
    return "Hello from catradar!"


ti.init(arch=ti.gpu)
