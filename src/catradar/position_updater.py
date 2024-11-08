import taichi as ti
from catradar.common import *

velocities = ti.Vector.field(2, dtype=ti.f32, shape=N)


@ti.func
def initialize_positions():
    for i in range(N):
        positions[i] = ti.Vector([50 + ti.random() * 10, 50 + ti.random()])
        velocities[i] = ti.Vector([10 + ti.random(), 10 + ti.random()]) * 10


# First module: Updates positions and writes to shared memory
@ti.kernel
def update_positions():
    for i in range(N):
        positions[i] += velocities[i] * dt
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
