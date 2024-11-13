import taichi as ti

__all__ = [
    "setup_grid_data",
    "compute_states",
    "update_logs",
]

from catradar.common import (
    EUCLIDEAN_NORM,
    MANHATTAN_NORM,
    MAX_NORM,
    STATE_IDLE,
    STATE_INTERSECTION,
    STATE_INTERACT,
    state_to_str,
)

X: ti.f32
Y: ti.f32
N: ti.i32
R0: ti.f32
R1: ti.f32
LIMIT_PER_CELL: ti.i32
grid_cell_size: ti.i32
cell_count_x = ti.i32
cell_count_y = ti.i32
INTERSECTION_NUM: ti.i32

# Grid data structures for collision detection
grid_num_circles = NotImplemented
grid_circles = NotImplemented


def setup_grid_data(
    aX: ti.f32,
    aY: ti.f32,
    aN: ti.i32,
    aR0: ti.f32,
    aR1: ti.f32,
    aLIMIT_PER_CELL: ti.i32,
    aINTERSECTION_NUM: ti.i32,
):
    global X, Y, N, R0, R1, LIMIT_PER_CELL, INTERSECTION_NUM
    X = aX
    Y = aY
    N = aN
    R0 = aR0
    R1 = aR1
    LIMIT_PER_CELL = aLIMIT_PER_CELL
    INTERSECTION_NUM = aINTERSECTION_NUM

    global grid_cell_size, cell_count_x, cell_count_y
    grid_cell_size = R1
    cell_count_x = int(X / grid_cell_size) + 1
    cell_count_y = int(Y / grid_cell_size) + 1

    global grid_num_circles, grid_circles
    grid_num_circles = ti.field(dtype=ti.i32, shape=(cell_count_x, cell_count_y))
    grid_circles = ti.field(
        dtype=ti.i32,
        shape=(cell_count_x, cell_count_y, LIMIT_PER_CELL),
    )


@ti.func
def _calc_dist(
    pos_i: ti.types.vector(2, dtype=float),
    pos_j: ti.types.vector(2, dtype=float),
    norm_func: ti.i32,
) -> ti.f32:
    res = 0.0
    if norm_func == EUCLIDEAN_NORM:
        res = (pos_i - pos_j).norm()
    if norm_func == MANHATTAN_NORM:
        res = ti.abs(pos_i.x - pos_j.x) + ti.abs(pos_i.y - pos_j.y)
    if norm_func == MAX_NORM:
        res = ti.max(ti.abs(pos_i.x - pos_j.x), ti.abs(pos_i.y - pos_j.y))
    return res


logs_prev_state = ti.field(ti.i32, shape=())
logs_new_state = ti.field(ti.i32, shape=())
logs_who_changed_id = ti.field(ti.i32, shape=())


@ti.kernel
def compute_states(
    positions: ti.template(),
    states: ti.template(),
    intesections: ti.template(),
    norm_func: ti.i32,
    logged_id: ti.i32,
):
    grid_num_circles.fill(0)

    # Insert circles into grid
    for idx in range(N):
        pos = positions[idx]
        cell_x = int(pos.x / grid_cell_size)
        cell_y = int(pos.y / grid_cell_size)
        num = ti.atomic_add(grid_num_circles[cell_x, cell_y], 1)
        if num < LIMIT_PER_CELL:
            grid_circles[cell_x, cell_y, num] = idx

    for idx in range(N):
        pos_i = positions[idx]
        cell_x = int(pos_i.x / grid_cell_size)
        cell_y = int(pos_i.y / grid_cell_size)

        state = STATE_IDLE
        intersect_len = 0

        for offset_x in range(-1, 2):
            for offset_y in range(-1, 2):
                other_cell_x = cell_x + offset_x
                other_cell_y = cell_y + offset_y
                if (
                    0 <= other_cell_x < cell_count_x
                    and 0 <= other_cell_y < cell_count_y
                ):
                    num = ti.min(
                        grid_num_circles[other_cell_x, other_cell_y], LIMIT_PER_CELL
                    )
                    for k in range(num):
                        jdx = grid_circles[other_cell_x, other_cell_y, k]
                        if jdx != idx:
                            pos_j = positions[jdx]
                            dist = _calc_dist(pos_i, pos_j, norm_func)
                            if dist <= R0:
                                state = STATE_INTERSECTION
                                if logged_id == idx:
                                    logs_who_changed_id[None] = jdx

                                intesections[idx, intersect_len + 1] = jdx
                                intersect_len += 1

                                if intersect_len == INTERSECTION_NUM:
                                    break  # Exit early for performance
                            elif dist <= R1:
                                prob = 1.0 / (dist * dist)
                                if ti.random() < prob:
                                    state = STATE_INTERACT
                                    if logged_id == idx:
                                        logs_who_changed_id[None] = jdx

                    if state == STATE_INTERSECTION:
                        if intersect_len == INTERSECTION_NUM:
                            break  # Exit early if state is determined
            if state == STATE_INTERSECTION:
                if intersect_len == INTERSECTION_NUM:
                    break

        if logged_id == idx:
            logs_prev_state[None] = states[idx]
            logs_new_state[None] = state
        states[idx] = state
        intesections[idx, 0] = intersect_len


def update_logs(logged_id, logs):
    if logs_new_state[None] == logs_prev_state[None]:
        return
    if logs_who_changed_id[None] == -1:
        logs.append(
            "State of {} id changed: {} -> {}".format(
                logged_id,
                state_to_str[logs_prev_state[None]],
                state_to_str[logs_new_state[None]],
            )
        )
        return
    logs.append(
        "State of {} id changed: {} -> {} by {} id".format(
            logged_id,
            state_to_str[logs_prev_state[None]],
            state_to_str[logs_new_state[None]],
            logs_who_changed_id[None],
        )
    )
