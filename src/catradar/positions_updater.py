import math

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

        last_positions[i].x = positions[i].x
        last_positions[i].y = positions[i].y


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


@ti.func
def line_intersection(x1, y1, x2, y2, x3, y3, x4, y4) -> ti.math.vec2:
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1

    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    D = A1 * B2 - A2 * B1

    res = ti.math.vec2(0.0, 0.0)
    if D == 0:
        res = ti.math.vec2(INF, INF)
    else:
        Dx = C1 * B2 - C2 * B1
        Dy = A1 * C2 - A2 * C1

        x = Dx / D
        y = Dy / D
        res = ti.math.vec2(x, y)

    return res


@ti.func
def point_in_rect(px, py, sx1, sy1, sx2, sy2) -> bool:
    return ti.min(sx1, sx2) <= px <= ti.max(sx1, sx2) and ti.min(
        sy1, sy2
    ) <= py <= ti.max(sy1, sy2)


@ti.func
def calc_angel(a: ti.math.vec2, b: ti.math.vec2) -> ti.f32:
    dot = ti.math.dot(a, b)
    print(dot)
    u = ti.math.length(a)
    v = ti.math.length(b)

    res: ti.f32 = 0
    if u == 0 or v == 0:
        res = INF
    else:
        res = ti.math.acos(ti.max(ti.min(dot / (u * v), 1), -1))

    return res


@ti.func
def rotate_vector(v, alpha):
    cos_a = ti.math.cos(alpha)
    sin_a = ti.math.sin(alpha)

    px_new = v.x * cos_a - v.y * sin_a
    py_new = v.x * sin_a + v.y * cos_a

    return ti.math.vec2(px_new, py_new)


@ti.func
def process_point_in_segment(
    point_id: ti.i32,
    positions: ti.template(),
    sx1: ti.f32,
    sy1: ti.f32,
    sx2: ti.f32,
    sy2: ti.f32,
    vel: ti.template(),
):
    px1 = last_positions[point_id].x
    py1 = last_positions[point_id].y
    px2 = positions[point_id].x
    py2 = positions[point_id].y

    inter = line_intersection(px1, py1, px2, py2, sx1, sy1, sx2, sy2)
    if inter.x != INF or inter.y != INF:
        if point_in_rect(inter.x, inter.y, px1, py1, px2, py2) and point_in_rect(
            inter.x, inter.y, sx1, sy1, sx2, sy2
        ):
            line = ti.math.vec2(sx1 - sx2, sy1 - sy2)
            perp = ti.math.vec2(-line.y, line.x)
            s1_p = ti.math.vec2(sx1 - px1, sy1 - py1)
            if ti.math.dot(perp, s1_p) < 0:
                perp = -perp

            p_vec = ti.math.vec2(px2 - px1, py2 - py1)
            angel: ti.f32 = calc_angel(p_vec, perp)
            if p_vec.x * perp.y - p_vec.y * perp.x < 0:
                angel *= -1

            vel *= -1
            vel = rotate_vector(vel, angel * 2)
            positions[point_id] = last_positions[point_id]
            print(angel * 180 / math.pi)


@ti.kernel
def check_borders(
    positions: ti.template(),
    borders_count: ti.i32,
    borders: ti.template(),
):
    for i in range(N):
        for j in range(borders_count):
            process_point_in_segment(
                i,
                positions,
                borders[2 * j].x,
                borders[2 * j].y,
                borders[2 * j + 1].x,
                borders[2 * j + 1].y,
                velocities[i],
            )


def update_positions(
    positions,
    borders_lst,
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

    count = len(borders_lst)
    if count > 0:
        borders = ti.Vector.field(3, dtype=ti.f32, shape=count)
        for i in range(count):
            borders[i] = borders_lst[i]
        check_borders(positions, count // 2, borders)
