import taichi as ti

__all__ = ["setup_positions_data", "initialize_positions", "update_positions"]

velocities = NotImplemented

X: ti.f32
Y: ti.f32
N: ti.i32


def setup_positions_data(aX, aY, aN):
    global X, Y, N
    X = aX
    Y = aY
    N = aN

    global velocities
    velocities = ti.Vector.field(2, dtype=ti.f32, shape=N)


# Random initialization of positions and velocities
@ti.kernel
def initialize_positions(positions: ti.template(), opt: ti.i32):
    if opt == 0:
        for i in range(N):
            positions[i] = ti.Vector([50 + ti.random() * 10, 50 + ti.random()])
            velocities[i] = ti.Vector([10 + ti.random(), 10 + ti.random()]) * 0.5
    if opt == 1:
        for i in range(N):
            positions[i] = ti.Vector([ti.random() * X, ti.random() * Y])
            velocities[i] = (
                ti.Vector([ti.random() * 100 - 50, ti.random() * 100 - 50]) * 0.01
            )


@ti.kernel
def update_positions(positions: ti.template()):
    for i in range(N):
        positions[i] += velocities[i]
        # Boundary conditions
        if positions[i].x < 0:
            positions[i].x = 0
            velocities[i].x *= -1
        if positions[i].x > X:
            positions[i].x = X
            velocities[i].x *= -1
        if positions[i].y < 0:
            positions[i].y = 0
            velocities[i].y *= -1
        if positions[i].y > Y:
            positions[i].y = Y
            velocities[i].y *= -1
