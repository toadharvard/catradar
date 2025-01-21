import pytest
import taichi as ti

from catradar import positions_updater

EPS = 1e-8


@pytest.mark.parametrize(
    "N,init_pos,velocities_list,speed_mult, expected_pos",
    [
        pytest.param(1, [(10, 10)], [(1, 1)], 2, [(22, 22)]),
        # pytest.param(1, None, 20),
        # pytest.param(2, None),
        # pytest.param(2, None),
        # pytest.param(5, None),
        # pytest.param(5, None),
        # pytest.param(10, None),
        # pytest.param(10, None),
    ],
)
def test_compute_states(
    N: ti.i32,
    init_pos: list[tuple[int, int]],
    velocities_list: list[tuple[int, int]],
    speed_mult: ti.f32,
    expected_pos: list[tuple[int, int]],
):
    positions_updater.setup_positions_data(100, 100, N)

    positions = ti.Vector.field(2, dtype=ti.f32, shape=N)
    for index, pos in enumerate(init_pos):
        positions[index] = ti.Vector([pos[0], pos[1]])

    velocities = ti.Vector.field(2, dtype=ti.f32, shape=N)
    for index, vel in enumerate(velocities_list):
        velocities[index] = ti.Vector([vel[0], vel[1]])

    positions_updater.velocities = velocities
    positions_updater.update_pos_on_velocity(positions, speed_mult, 0.1)

    for i in range(N):
        res = ti.abs(positions[i] - expected_pos[i]) < EPS
        assert res[0] and res[1]
