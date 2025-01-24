import pytest
import taichi as ti

from catradar import canvas
from catradar.common import STATE_INTERSECTION, STATE_IDLE, STATE_INTERACT

# Values that do not affect the test result
render_rate_mock = 100
norm_ration_mock = 1000
X_mock = 1000
Y_mock = 1000
R0_mock = 10


@pytest.mark.parametrize(
    "N,logged_id,states_list",
    [
        pytest.param(3, 1, [STATE_IDLE, STATE_IDLE, STATE_IDLE]),
        pytest.param(3, 1, [STATE_INTERACT, STATE_INTERACT, STATE_INTERACT]),
        pytest.param(
            3, 1, [STATE_INTERSECTION, STATE_INTERSECTION, STATE_INTERSECTION]
        ),
        pytest.param(
            5,
            0,
            [
                STATE_INTERSECTION,
                STATE_INTERSECTION,
                STATE_INTERACT,
                STATE_INTERACT,
                STATE_INTERSECTION,
            ],
        ),
        pytest.param(
            5,
            1,
            [
                STATE_IDLE,
                STATE_INTERSECTION,
                STATE_INTERACT,
                STATE_INTERACT,
                STATE_INTERSECTION,
            ],
        ),
        pytest.param(
            5,
            2,
            [
                STATE_IDLE,
                STATE_IDLE,
                STATE_INTERACT,
                STATE_INTERACT,
                STATE_INTERACT,
            ],
        ),
    ],
)
def test_update_colors(
    N: ti.i32,
    logged_id: ti.i32,
    states_list,
):
    assert 0 <= logged_id < N
    canvas.setup_data_for_scene(X_mock, Y_mock, N, R0_mock, norm_ration_mock)
    positions_mock = ti.Vector.field(2, dtype=ti.f32, shape=N)
    states = ti.field(dtype=ti.i32, shape=N)
    for i in range(N):
        states[i] = states_list[i]

    canvas.update_colors(
        positions_mock, states, logged_id, render_rate_mock, norm_ration_mock
    )

    # Check that cats with different states have different colors
    for i in range(N):
        for j in range(N):
            if states[i] == states[j]:
                assert (
                    all(
                        (
                            canvas.colors_to_draw[i] == canvas.colors_to_draw[j]
                        ).to_numpy()
                    )
                    or i == logged_id
                    or j == logged_id
                )
            else:
                assert any(
                    (canvas.colors_to_draw[i] != canvas.colors_to_draw[j]).to_numpy()
                )

    # Check that cat with logged_id has a different color from the others
    for i in range(N):
        if i == logged_id:
            continue
        assert any(
            (canvas.colors_to_draw[i] != canvas.colors_to_draw[logged_id]).to_numpy()
        )
