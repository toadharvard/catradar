import taichi as ti
from math import pi

from catradar.borders_processor import is_segment_intersect, get_rotated_vector
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
last_positions = NotImplemented

X: ti.f32
Y: ti.f32
N: ti.i32
INF: ti.f32 = 1e9


def setup_positions_data(aX, aY, aN):
    global X, Y, N
    X = aX
    Y = aY
    N = aN

    global velocities, p1_angles, p1_speeds, last_positions
    velocities = ti.Vector.field(2, dtype=ti.f32, shape=N)
    last_positions = ti.Vector.field(2, dtype=ti.f32, shape=N)
    p1_angles = ti.field(dtype=ti.f32, shape=N)
    p1_speeds = ti.field(dtype=ti.f32, shape=N)


@ti.func
def _initialize_data_for_pos_updaters():
    for i in range(N):
        p1_speeds[i] = ti.random() * 2 + 2
        p1_angles[i] = ti.random() * 2 * pi


# Random initialization of positions and velocities
@ti.kernel
def initialize_positions(positions: ti.template(), opt: ti.i32):
    _initialize_data_for_pos_updaters()
    for i in range(N):
        if opt == 0:
            positions[i] = ti.Vector([ti.random() * X, ti.random() * Y])
            velocities[i] = (
                ti.Vector([ti.random() * 100 - 50, ti.random() * 100 - 50]) * 0.01
            )
        else:
            positions[i] = ti.Vector([50 + ti.random() * 10, 50 + ti.random()])
            velocities[i] = ti.Vector([10 + ti.random(), 10 + ti.random()]) * 0.5

        last_positions[i].x = positions[i].x
        last_positions[i].y = positions[i].y


@ti.kernel
def _movement_patter_free():
    pass


@ti.kernel
def _movement_pattern_carousel():
    for i in range(N):
        p1_angles[i] += 0.05
        if p1_angles[i] >= 2 * pi:
            p1_angles[i] -= 2 * pi
        velocities[i][0] = ti.cos(p1_angles[i]) * p1_speeds[i]
        velocities[i][1] = ti.sin(p1_angles[i]) * p1_speeds[i]


@ti.kernel
def _movement_pattern_colliding(positions: ti.template(), intersections: ti.template()):
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
def _cursor_push(positions: ti.template(), cursor_pos: ti.math.vec2):
    for i in range(N):
        vec_cursor_to_self = positions[i] - cursor_pos
        cursor_dist = vec_cursor_to_self.norm()
        if cursor_dist < 100:
            velocities[i] += (vec_cursor_to_self / ti.pow(cursor_dist, 2)) * 100


@ti.kernel
def _update_pos_on_velocity(
    positions: ti.template(),
    speed_mult: ti.f32,
    dt: ti.f32,
):
    for i in range(N):
        last_positions[i] = positions[i]
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


@ti.kernel
def _check_borders(
    positions: ti.template(),
    borders_count: ti.i32,
    borders: ti.template(),
):
    for i in range(N):
        for j in range(borders_count):
            if is_segment_intersect(
                last_positions[i],
                positions[i],
                ti.math.vec2(borders[2 * j].x, borders[2 * j].y),
                ti.math.vec2(borders[2 * j + 1].x, borders[2 * j + 1].y),
            ):
                velocities[i] = get_rotated_vector(
                    last_positions[i],
                    positions[i],
                    ti.math.vec2(borders[2 * j].x, borders[2 * j].y),
                    ti.math.vec2(borders[2 * j + 1].x, borders[2 * j + 1].y),
                    velocities[i],
                )
                positions[i] = last_positions[i]


def update_positions(
    positions,
    borders,
    borders_count,
    intersections,
    cursor_pos: ti.math.vec2,
    cursor_push_on: ti.i8,
    speed_mult: ti.f32,
    opt: ti.i32,
    dt: ti.f32,
):
    if opt == MOVE_PATTERN_FREE:
        _movement_patter_free()
    if opt == MOVE_PATTERN_CAROUSEL:
        _movement_pattern_carousel()
    if opt == MOVE_PATTERN_COLLIDING:
        _movement_pattern_colliding(positions, intersections)

    if cursor_push_on:
        _cursor_push(positions, cursor_pos)

    _update_pos_on_velocity(positions, speed_mult, dt)

    _check_borders(positions, borders_count // 2, borders)
