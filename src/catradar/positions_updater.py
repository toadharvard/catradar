import taichi as ti
from math import pi

from catradar.common import (
    MOVE_PATTERN_CAROUSEL,
    MOVE_PATTERN_COLLIDING,
    MOVE_PATTERN_FREE,
)

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
    for i in range(N):
        if opt == 0:
            positions[i] = ti.Vector([ti.random() * X, ti.random() * Y])
            velocities[i] = (
                ti.Vector([ti.random() * 100 - 50, ti.random() * 100 - 50]) * 0.01
            )
        else:
            positions[i] = ti.Vector([50 + ti.random() * 10, 50 + ti.random()])
            velocities[i] = ti.Vector([10 + ti.random(), 10 + ti.random()]) * 0.5


@ti.kernel
def movement_patter_free(
    positions: ti.template(),  # Need to pass temp argument because taichi does not evaluate this otherwise
):
    pass


@ti.kernel
def movement_pattern_carousel(
    positions: ti.template(),  # Need to pass temp argument because taichi does not evaluate this otherwise
):
    for i in range(N):
        p1_angles[i] = ti.raw_mod(p1_angles[i] + 0.05, 2 * pi)
        velocities[i][0] = ti.cos(p1_angles[i]) * p1_speeds[i]
        velocities[i][1] = ti.sin(p1_angles[i]) * p1_speeds[i]


@ti.kernel
def movement_pattern_colliding(positions: ti.template(), intersections: ti.template()):
    for i in range(N):
        self_pos = positions[i]
        force = ti.math.vec2(0.0, 0.0)
        if velocities[i].norm() > 1:
            force = -(velocities[i] * p2_resistance)

        intersect_len = intersections[i, 0]
        for j in range(1, intersect_len + 1):
            interact_pos = positions[intersections[i, j]]
            vec_interact_to_self = self_pos - interact_pos
            dist = ti.max(vec_interact_to_self.norm(), 1)
            force += (vec_interact_to_self / ti.pow(dist, 3)) * 10

        velocities[i] += force


@ti.kernel
def cursor_push(positions: ti.template(), cursor_pos: ti.math.vec2):
    for i in range(N):
        vec_cursor_to_self = positions[i] - cursor_pos
        cursor_dist = vec_cursor_to_self.norm()
        if cursor_dist < 100:
            velocities[i] += (vec_cursor_to_self / ti.pow(cursor_dist, 2)) * 100


@ti.kernel
def update_pos_on_velocity(
    positions: ti.template(),
    speed_mult: ti.f32,
    dt: ti.f32,
):
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


def update_positions(
    positions,
    intersections,
    cursor_pos: ti.math.vec2,
    cursor_push_on: ti.i8,
    speed_mult: ti.f32,
    opt: ti.i32,
    dt: ti.f32,
):
    if opt == MOVE_PATTERN_FREE:
        movement_patter_free(positions)
    if opt == MOVE_PATTERN_CAROUSEL:
        movement_pattern_carousel(positions)
    if opt == MOVE_PATTERN_COLLIDING:
        movement_pattern_colliding(positions, intersections)

    if cursor_push_on:
        cursor_push(positions, cursor_pos)

    update_pos_on_velocity(positions, speed_mult, dt)
