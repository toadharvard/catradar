import pytest
import taichi as ti

from catradar import positions_updater

X = 100
Y = 100
mock_dt = 0.1
EPS = 1e-8


@pytest.mark.parametrize(
    "N,init_pos,velocities_list,speed_mult, expected_pos",
    [
        pytest.param(1, [(10, 10)], [(1, 1)], 2, [(22, 22)]),
        pytest.param(1, [(5, 5)], [(2, 2)], 2, [(29, 29)]),
        pytest.param(2, [(3, 3), (5, 5)], [(2, 2), (1, 1)], 2, [(27, 27), (17, 17)]),
        pytest.param(
            4,
            [(3, 3), (5, 5), (80, 80), (90, 90)],
            [(2, 2), (-1, 1), (7, -8), (9, -2)],
            2,
            [(27, 27), (0, 17), (100, 0), (100, 66)],
        ),
        pytest.param(
            4,
            [(1, 1), (64, 37), (43, 20), (87, 95)],
            [(-2, 2), (5, 10), (-2, 0), (29, 6)],
            2,
            [(0, 25), (100, 100), (19, 20), (100, 100)],
        ),
    ],
)
def test_positions_updater(
    N: ti.i32,
    init_pos: list[tuple[int, int]],
    velocities_list: list[tuple[int, int]],
    speed_mult: ti.f32,
    expected_pos: list[tuple[int, int]],
):
    positions_updater.setup_positions_data(X, Y, N)

    positions = ti.Vector.field(2, dtype=ti.f32, shape=N)
    for index, pos in enumerate(init_pos):
        positions[index] = ti.Vector([pos[0], pos[1]])

    for index, vel in enumerate(velocities_list):
        positions_updater.velocities[index] = ti.Vector([vel[0], vel[1]])

    positions_updater.update_pos_on_velocity(positions, speed_mult, mock_dt)

    for i in range(N):
        res = ti.abs(positions[i] - expected_pos[i]) < EPS
        assert res[0] and res[1]
