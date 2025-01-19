import pytest
import taichi as ti

from catradar.borders_processor import get_rotated_vector

EPS = 1e-3


# Need this wrapper because we can not call `ti.func` out of taichi scope
@ti.kernel
def _get_rotated_vector_kernel(
    last_pos: ti.math.vec2,
    new_pos: ti.math.vec2,
    border1: ti.math.vec2,
    border2: ti.math.vec2,
    to_rotate: ti.math.vec2,
) -> ti.math.vec2:
    return get_rotated_vector(last_pos, new_pos, border1, border2, to_rotate)


@pytest.mark.parametrize(
    "last_pos,new_pos,border1,border2,to_rotate, expected_vec",
    [
        pytest.param(
            ti.math.vec2(2, 1),
            ti.math.vec2(2, -1),
            ti.math.vec2(-100, 0),
            ti.math.vec2(100, 0),
            ti.math.vec2(0, -2),
            ti.math.vec2(0, 2),
        ),
        pytest.param(
            ti.math.vec2(5, 6),
            ti.math.vec2(7, 4),
            ti.math.vec2(-50, 5),
            ti.math.vec2(50, 5),
            ti.math.vec2(2, -2),
            ti.math.vec2(2, 2),
        ),
        pytest.param(
            ti.math.vec2(-1, 5),
            ti.math.vec2(1, 5),
            ti.math.vec2(0, 0),
            ti.math.vec2(0, 10),
            ti.math.vec2(2, 0),
            ti.math.vec2(-2, 0),
        ),
        pytest.param(
            ti.math.vec2(9, 7),
            ti.math.vec2(11, 9),
            ti.math.vec2(10, 0),
            ti.math.vec2(10, 20),
            ti.math.vec2(2, 2),
            ti.math.vec2(-2, 2),
        ),
        pytest.param(
            ti.math.vec2(7, 5),
            ti.math.vec2(5, 7),
            ti.math.vec2(0, 0),
            ti.math.vec2(10, 10),
            ti.math.vec2(-2, 2),
            ti.math.vec2(2, -2),
        ),
        pytest.param(
            ti.math.vec2(2, 2),
            ti.math.vec2(4, 4),
            ti.math.vec2(0, 0),
            ti.math.vec2(10, 10),
            ti.math.vec2(2, 2),
            ti.math.vec2(2, 2),
        ),
        pytest.param(
            ti.math.vec2(2, -6),
            ti.math.vec2(7, -1),
            ti.math.vec2(0, 0),
            ti.math.vec2(10, -10),
            ti.math.vec2(5, 5),
            ti.math.vec2(-5, -5),
        ),
        pytest.param(
            ti.math.vec2(7, 11),
            ti.math.vec2(10, 9),
            ti.math.vec2(0, 10),
            ti.math.vec2(20, 10),
            ti.math.vec2(3, -2),
            ti.math.vec2(3, 2),
        ),
        pytest.param(
            ti.math.vec2(4, 2),
            ti.math.vec2(6, 3),
            ti.math.vec2(5, 0),
            ti.math.vec2(5, 10),
            ti.math.vec2(2, 1),
            ti.math.vec2(-2, 1),
        ),
        pytest.param(
            ti.math.vec2(3, 4),
            ti.math.vec2(3, 4),
            ti.math.vec2(2, 2),
            ti.math.vec2(4, 6),
            ti.math.vec2(0, 0),
            ti.math.vec2(0, 0),
        ),
    ],
)
def test_get_rotated_vector(
    last_pos: ti.math.vec2,
    new_pos: ti.math.vec2,
    border1: ti.math.vec2,
    border2: ti.math.vec2,
    to_rotate: ti.math.vec2,
    expected_vec: ti.math.vec2,
):
    actual_vec = _get_rotated_vector_kernel(
        last_pos, new_pos, border1, border2, to_rotate
    )
    print(actual_vec)
    assert (
        ti.abs(actual_vec.x - expected_vec.x) < EPS
        and ti.abs(actual_vec.y - expected_vec.y) < EPS
    )
