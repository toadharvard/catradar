# implementation of this module was taken from https://docs.taichi-lang.org/blog/acclerate-collision-detection-with-taichi
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
    TESTING_MODE,
)

EPS: ti.f32 = 1e-8
X: ti.f32
Y: ti.f32
N: ti.i32
R0: ti.f32
R1: ti.f32
LIMIT_PER_CELL: ti.i32
grid_cell_size: ti.i32  # the length and height of each grid cell (cells are squares)
cell_count_x = ti.i32  # count of cells by X coordinate
cell_count_y = ti.i32  # count of cells by Y coordinate
INTERSECTION_NUM: ti.i32
MODE: ti.i32

circles_per_cell = NotImplemented  # count of circles per cell of grid
column_sum = NotImplemented  # prefix sums for each column of circles_per_cell
prefix_sum = NotImplemented  # prefix sums for circles_per_cell
list_head = NotImplemented
list_cur = NotImplemented
list_tail = NotImplemented
circles_id = NotImplemented

new_state = ti.field(ti.i32, shape=())
prev_state = ti.field(ti.i32, shape=())
who_changed_id = ti.field(ti.i32, shape=())


def setup_grid_data(
    aX: ti.f32,
    aY: ti.f32,
    aN: ti.i32,
    aR0: ti.f32,
    aR1: ti.f32,
    aLIMIT_PER_CELL: ti.i32,
    aINTERSECTION_NUM: ti.i32,
    aMODE: ti.i32,
):
    global X, Y, N, R0, R1, LIMIT_PER_CELL, INTERSECTION_NUM, MODE
    X = aX
    Y = aY
    N = aN
    R0 = aR0
    R1 = aR1
    LIMIT_PER_CELL = aLIMIT_PER_CELL
    INTERSECTION_NUM = aINTERSECTION_NUM
    MODE = aMODE

    global grid_cell_size, cell_count_x, cell_count_y
    grid_cell_size = R1
    cell_count_x = int(X / grid_cell_size) + 1
    cell_count_y = int(Y / grid_cell_size) + 1

    global \
        circles_per_cell, \
        column_sum, \
        prefix_sum, \
        list_head, \
        list_cur, \
        list_tail, \
        circles_id
    circles_per_cell = ti.field(dtype=ti.i32, shape=(cell_count_x, cell_count_y))
    column_sum = ti.field(dtype=ti.i32, shape=cell_count_x)
    prefix_sum = ti.field(dtype=ti.i32, shape=(cell_count_x, cell_count_y))
    list_head = ti.field(dtype=ti.i32, shape=cell_count_x * cell_count_y)
    list_cur = ti.field(dtype=ti.i32, shape=cell_count_x * cell_count_y)
    list_tail = ti.field(dtype=ti.i32, shape=cell_count_x * cell_count_y)
    circles_id = ti.field(dtype=ti.i32, shape=N)


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


@ti.kernel
def compute_states(
    positions: ti.template(),
    states: ti.template(),
    intersections: ti.template(),
    update_intersections: ti.i8,
    norm_func: ti.i32,
    logged_id: ti.i32,
):
    # Compute count of circles per cell
    circles_per_cell.fill(0)
    for i in range(N):
        cell_idx = ti.floor(positions[i] / grid_cell_size, int)
        ti.atomic_add(circles_per_cell[cell_idx], 1)

    # Compute prefix sum for each column
    for i in range(cell_count_x):
        cur_sum = 0
        for j in range(cell_count_y):
            cur_sum += circles_per_cell[i, j]
        column_sum[i] = cur_sum

    # Compute prefix sum for all grid
    prefix_sum[0, 0] = 0
    ti.loop_config(serialize=True)
    for i in range(1, cell_count_x):
        prefix_sum[i, 0] = prefix_sum[i - 1, 0] + column_sum[i - 1]
    for i in range(cell_count_x):
        for j in range(cell_count_y):
            if j == 0:
                prefix_sum[i, j] += circles_per_cell[i, j]
            else:
                prefix_sum[i, j] = prefix_sum[i, j - 1] + circles_per_cell[i, j]

            linear_idx = i * cell_count_y + j
            list_head[linear_idx] = prefix_sum[i, j] - circles_per_cell[i, j]
            list_cur[linear_idx] = list_head[linear_idx]
            list_tail[linear_idx] = prefix_sum[i, j]

    # Place the id of the circles in the right places of circles_id
    for i in range(N):
        grid_idx = ti.floor(positions[i] / grid_cell_size, int)
        linear_idx = grid_idx[0] * cell_count_y + grid_idx[1]
        cell_location = ti.atomic_add(list_cur[linear_idx], 1)
        circles_id[cell_location] = i

    # Compute state of each circle after filling grid
    for i in range(N):
        grid_idx = ti.floor(positions[i] / grid_cell_size, int)
        x_begin = max(grid_idx[0] - 1, 0)
        x_end = min(grid_idx[0] + 2, cell_count_x)
        y_begin = max(grid_idx[1] - 1, 0)
        y_end = min(grid_idx[1] + 2, cell_count_y)

        state = STATE_IDLE
        intersect_len = 0

        if logged_id == i:
            who_changed_id[None] = -1  # Initially, no one changed state of idx

        for neigh_x in range(x_begin, x_end):
            for neigh_y in range(y_begin, y_end):
                neigh_linear_idx = neigh_x * cell_count_y + neigh_y
                processed_neigh_num = 0
                for p in range(
                    list_head[neigh_linear_idx], list_tail[neigh_linear_idx]
                ):
                    if processed_neigh_num > LIMIT_PER_CELL:
                        break
                    processed_neigh_num += 1
                    j = circles_id[p]
                    if i != j:
                        dist = _calc_dist(positions[i], positions[j], norm_func)
                        if dist <= R0:
                            state = STATE_INTERSECTION
                            if logged_id == i:
                                who_changed_id[None] = j

                            if not update_intersections:
                                break
                            else:
                                intersections[i, intersect_len + 1] = j
                                intersect_len += 1

                                if intersect_len == INTERSECTION_NUM:
                                    break  # Exit early for performance
                        elif dist <= R1 and state != STATE_INTERSECTION:
                            temp = dist - R0 * 0.75 + EPS
                            prob = 1 if MODE == TESTING_MODE else 1.0 / (temp * temp)
                            if ti.random() <= prob:
                                state = STATE_INTERACT
                                if logged_id == i:
                                    who_changed_id[None] = j

                if state == STATE_INTERSECTION and (
                    not update_intersections or intersect_len == INTERSECTION_NUM
                ):
                    break

            if state == STATE_INTERSECTION and (
                not update_intersections or intersect_len == INTERSECTION_NUM
            ):
                break

        if logged_id == i:
            prev_state[None] = states[i]
            new_state[None] = state
        states[i] = state
        intersections[i, 0] = intersect_len


@ti.kernel
def update_logs(
    new_state_vec: ti.template(),
    prev_state_vec: ti.template(),
    who_changed_id_vec: ti.template(),
    cur_logs_ptr: ti.template(),
    max_sz: ti.i32,
) -> bool:
    overflow = False
    if new_state[None] != prev_state[None]:
        if cur_logs_ptr[None] == max_sz:
            overflow = True
            cur_logs_ptr[None] = 0

        new_state_vec[cur_logs_ptr[None]] = new_state[None]
        prev_state_vec[cur_logs_ptr[None]] = prev_state[None]
        who_changed_id_vec[cur_logs_ptr[None]] = who_changed_id[None]
        cur_logs_ptr[None] += 1

    return overflow
