import taichi as ti
from math import pi

__all__ = ["setup_positions_data", "initialize_positions", "update_positions"]

# movement pattern 1 global vars
p1_speeds = NotImplemented
p1_angles = NotImplemented

# movement pattern 2 global vars
p2_resistance = 0.05

velocities = NotImplemented

X: ti.f32
Y: ti.f32
N: ti.i32


def setup_positions_data(aX, aY, aN):
    global X, Y, N
    X = aX
    Y = aY
    N = aN

    global velocities, p1_angles, p1_speeds
    velocities = ti.Vector.field(2, dtype=ti.f32, shape=N)
    p1_angles = ti.field(dtype=ti.f32, shape=N)
    p1_speeds = ti.field(dtype=ti.f32, shape=N)


@ti.func
def initialize_data_for_pos_updaters():
    for i in range(N):
        p1_speeds[i] = 3 + (ti.random() * 2 - 1.0) * 1
        p1_angles[i] = ti.random() * 2 * pi


# Random initialization of positions and velocities
@ti.kernel
def initialize_positions(positions: ti.template(), opt: ti.i32):
    initialize_data_for_pos_updaters()
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


# Free movement
@ti.func
def movement_pattern_0(positions):
    pass


# Carousel
@ti.func
def movement_pattern_1(positions):
    for i in range(N):
        p1_angles[i] = ti.raw_mod(p1_angles[i] + 0.05, 2 * pi)
        velocities[i][0] = ti.cos(p1_angles[i]) * p1_speeds[i]
        velocities[i][1] = ti.sin(p1_angles[i]) * p1_speeds[i]


# Colliding
@ti.func
def movement_pattern_2(positions, intesections):
    for i in range(N):
        self_pos = positions[i]
        force = ti.math.vec2(0.0, 0.0)
        if velocities[i].norm() > 1:
            force = -(velocities[i] * p2_resistance)

        intersect_len = intesections[i, 0]
        for j in range(1, intersect_len + 1):
            interact_pos = positions[intesections[i, j]]
            vec_interact_to_self = self_pos - interact_pos
            dist = ti.max((vec_interact_to_self).norm(), 1)
            force += (vec_interact_to_self / ti.pow(dist, 3)) * 10

        velocities[i] += force


@ti.func
def cursor_push(positions, cursor_pos):
    for i in range(N):
        vec_cursor_to_self = positions[i] - cursor_pos
        cursor_dist = vec_cursor_to_self.norm()
        if cursor_dist < 100:
            velocities[i] += (vec_cursor_to_self / ti.pow(cursor_dist, 2)) * 100


@ti.kernel
def update_positions(
    positions: ti.template(),
    intesections: ti.template(),
    cursor_pos: ti.math.vec2,
    cursor_push_on: ti.i8,
    speed_mult: ti.f32,
    opt: ti.i32,
    dt: ti.f32,
):
    if opt == 0:
        movement_pattern_0(positions)
    if opt == 1:
        movement_pattern_1(positions)
    if opt == 2:
        movement_pattern_2(positions, intesections)

    if cursor_push_on:
        cursor_push(positions, cursor_pos)

    for i in range(N):
        positions[i] += speed_mult * velocities[i] * dt * 60
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
