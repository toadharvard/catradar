import pytest
import taichi as ti

from catradar.common import STATE_INTERSECTION, STATE_IDLE
from catradar.grid_manager import _calc_dist, setup_grid_data, compute_states
from catradar.positions_updater import setup_positions_data, initialize_positions


# Compute only STATE_INTERSECTION states because state STATE_INTERACT is randomized and will not be tested
@ti.kernel
def naive_algo(
    N: ti.i32,
    RO: ti.i32,
    positions: ti.template(),
    states: ti.template(),
    norm_func: ti.i32,
):
    for i in range(N):
        states[i] = STATE_IDLE

    for i in range(N):
        for j in range(i + 1, N):
            if _calc_dist(positions[i], positions[j], norm_func) <= RO:
                states[i] = STATE_INTERSECTION
                states[j] = STATE_INTERSECTION


THRESHOLD = 0.05


@pytest.mark.parametrize(
    "N,X,Y,R0,R1,LIMIT_PER_CELL,INTERSECTION_NUM,update_intersections",
    [
        pytest.param(20, 100, 100, 5, 10, 250, 10, True),  # small grid, few cats
        pytest.param(100, 100, 100, 5, 10, 250, 10, True),  # small grid, many cats
        pytest.param(500, 1000, 1000, 5, 20, 250, 10, True),  # normal grid
        pytest.param(
            10000, 1000, 1000, 5, 20, 250, 10, True
        ),  # normal grid, many cats
        pytest.param(
            10000, 10000, 10000, 25, 50, 250, 10, True
        ),  # big grid, many cats
        pytest.param(
            5000, 4000, 7000, 5, 20, 250, 10, False
        ),  # regular grid with false as update_intersections, different X and Y
        pytest.param(
            5000, 6000, 3000, 5, 20, 250, 10, False
        ),  # regular grid with false as update_intersections, different X and Y
    ],
)
def test_compute_states(
    N: ti.i32,
    X: ti.f32,
    Y: ti.f32,
    R0: ti.f32,
    R1: ti.f32,
    LIMIT_PER_CELL: ti.i32,
    INTERSECTION_NUM: ti.i32,
    update_intersections: bool,
):
    assert R0 <= R1
    positions = ti.Vector.field(2, dtype=ti.f32, shape=N)
    states_expected = ti.field(dtype=ti.i32, shape=N)
    states_actual = ti.field(dtype=ti.i32, shape=N)

    setup_positions_data(X, Y, N)
    setup_grid_data(X, Y, N, R0, R1, LIMIT_PER_CELL, INTERSECTION_NUM)

    intersections_mock = ti.field(
        dtype=ti.i32,
        shape=(N, INTERSECTION_NUM + 1),
    )
    logged_id_mock = 0

    wrong_count = 0
    for init_opt in range(2):
        initialize_positions(positions, init_opt)
        for norm_func in range(3):
            naive_algo(N, R0, positions, states_expected, norm_func)

            compute_states(
                positions,
                states_actual,
                intersections_mock,
                update_intersections,
                norm_func,
                logged_id_mock,
            )

            for i in range(N):
                if states_expected[i] == STATE_INTERSECTION:
                    if states_actual[i] != STATE_INTERSECTION:
                        wrong_count += 1
                if states_expected[i] == STATE_IDLE:
                    if states_actual[i] == STATE_INTERSECTION:
                        wrong_count += 1

            assert wrong_count <= int(THRESHOLD * N)
