import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

X, Y = 1000, 1000
N = 20000
r_0 = 5
R_0 = 10
tau = 0.5
radius = 2

positions = ti.Vector.field(2, dtype=ti.f32, shape=N)
velocities = ti.Vector.field(2, dtype=ti.f32, shape=N)
intersects = ti.Vector.field(1, dtype=ti.u1, shape=N)
angry = ti.Vector.field(1, dtype=ti.u1, shape=N)


gui = ti.GUI("Cat Simulation", res=X)


@ti.kernel
def initialize():
    for i in range(N):
        positions[i] = [ti.random() * X, ti.random() * Y]
        velocities[i] = [ti.random() * 2 - 1, ti.random() * 2 - 1]


@ti.func
def distance(p1, p2):
    return ti.sqrt((p1 - p2).dot(p1 - p2))


@ti.kernel
def update_positions():
    for i in range(N):
        positions[i] += velocities[i]
        for d in ti.static(range(2)):
            if positions[i][d] < 0 or positions[i][d] > (X if d == 0 else Y):
                velocities[i][d] = -velocities[i][d]


@ti.kernel
def update_interactions():
    angry.fill(0)
    intersects.fill(0)

    for i in range(N):
        for j in range(i + 1, N):
            dist = distance(positions[i], positions[j])
            if dist <= r_0:
                intersects[i] = ti.u1(True)
                intersects[j] = ti.u1(True)
            if dist <= R_0:
                # Шипение с вероятностью обратно пропорциональной квадрату расстояния
                prob = 1 / (dist**2)
                if ti.random(float) > prob:
                    angry[i] = ti.u1(True)
                    angry[j] = ti.u1(True)


def render():
    # Преобразование позиций в numpy массив и нормализация в диапазон [0, 1] для рендеринга
    positions_np = positions.to_numpy() / np.array([X, Y])

    gui.circles(positions_np, radius=radius, color=0x0000FF)

    mask = np.nonzero(angry.to_numpy())[0]
    filtered_positions = positions_np[mask]
    gui.circles(pos=filtered_positions, radius=radius, color=0x00FF00)

    mask = np.nonzero(intersects.to_numpy())[0]
    filtered_positions = positions_np[mask]
    gui.circles(pos=filtered_positions, radius=radius, color=0xFF0000)

    gui.show()


initialize()
gui.fps_limit = 60 / tau  # TODO: абсолютно неверное использование tau
while gui.running:
    update_positions()
    update_interactions()
    render()
