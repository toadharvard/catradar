import pytest
import taichi as ti

from catradar.common import STATE_INTERSECTION, STATE_IDLE, TESTING_MODE, STATE_INTERACT
from catradar.grid_manager import _calc_dist, setup_grid_data, compute_states
from catradar.positions_updater import setup_positions_data, initialize_positions


@ti.kernel
def naive_algo(
    N: ti.i32,
    RO: ti.i32,
    R1: ti.i32,
    positions: ti.template(),
    states: ti.template(),
    norm_func: ti.i32,
):
    for i in range(N):
        states[i] = STATE_IDLE

    for i in range(N):
        for j in range(i + 1, N):
            dist = _calc_dist(positions[i], positions[j], norm_func)
            if dist <= RO:
                states[i] = STATE_INTERSECTION
                states[j] = STATE_INTERSECTION
            elif dist <= R1:
                if states[i] != STATE_INTERSECTION:
                    states[i] = STATE_INTERACT
                if states[j] != STATE_INTERSECTION:
                    states[j] = STATE_INTERACT


LIM = 100
INTR = 10
THRESHOLD = 0.01


@pytest.mark.parametrize(
    "N,X,Y,R0,R1,LIMIT_PER_CELL,INTERSECTION_NUM,update_intersections",
    [
        pytest.param(20, 100, 100, 1, 10, LIM, INTR, True),  # small grid, few cats
        pytest.param(20, 100, 100, 5, 15, LIM, INTR, True),  # small grid, few cats
        pytest.param(500, 100, 100, 5, 20, LIM, INTR, True),  # small grid, many cats
        pytest.param(500, 1000, 1000, 1, 10, LIM, INTR, True),  # small grid, many cats
        pytest.param(500, 1000, 1000, 5, 20, LIM, INTR, True),  # small grid, many cats
        pytest.param(500, 1000, 1000, 10, 50, LIM, INTR, True),  # normal grid
        pytest.param(
            10000, 1000, 1000, 5, 20, LIM, INTR, True
        ),  # normal grid, many cats
        pytest.param(
            10000, 10000, 10000, 25, 50, LIM, INTR, True
        ),  # big grid, many cats
        pytest.param(
            10000, 4000, 7000, 5, 20, LIM, INTR, False
        ),  # regular grid with false as update_intersections, different X and Y
        pytest.param(
            10000, 6000, 3000, 5, 20, LIM, INTR, False
        ),  # regular grid with false as update_intersections, different X and Y
        pytest.param(
            100000, 1000, 1000, 5, 20, LIM, INTR, True
        ),  # A lot of cats, small grid
        pytest.param(
            100000, 10000, 10000, 1, 10, LIM, INTR, True
        ),  # A lot of cats, big grid
        pytest.param(
            100000, 10000, 10000, 10, 50, LIM, INTR, True
        ),  # A lot of cats, big grid
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
    setup_grid_data(X, Y, N, R0, R1, LIMIT_PER_CELL, INTERSECTION_NUM, TESTING_MODE)

    intersections_mock = ti.field(
        dtype=ti.i32,
        shape=(N, INTERSECTION_NUM + 1),
    )
    logged_id_mock = 0

    wrong_count = 0
    for init_opt in range(2):
        initialize_positions(positions, init_opt)
        for norm_func in range(3):
            naive_algo(N, R0, R1, positions, states_expected, norm_func)

            compute_states(
                positions,
                states_actual,
                intersections_mock,
                update_intersections,
                norm_func,
                logged_id_mock,
            )

            for i in range(N):
                if states_expected[i] != states_actual[i]:
                    wrong_count += 1

            assert wrong_count <= int(THRESHOLD * N)
