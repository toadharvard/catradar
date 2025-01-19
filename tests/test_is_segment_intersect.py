import pytest
import taichi as ti

from catradar.borders_processor import is_segment_intersect


# Need this wrapper because we can not call `ti.func` out of taichi scope
@ti.kernel
def _is_segment_intersect_kernel(
    seg11: ti.math.vec2,
    seg12: ti.math.vec2,
    seg21: ti.math.vec2,
    seg22: ti.math.vec2,
) -> bool:
    return is_segment_intersect(seg11, seg12, seg21, seg22)


@pytest.mark.parametrize(
    "seg11,seg12,seg21,seg22,expected",
    [
        pytest.param(
            ti.math.vec2(0, 0),
            ti.math.vec2(2, 0),
            ti.math.vec2(1, 1),
            ti.math.vec2(1, -1),
            True,
        ),
        pytest.param(
            ti.math.vec2(0, 0),
            ti.math.vec2(1, 1),
            ti.math.vec2(2, 2),
            ti.math.vec2(3, 3),
            False,
        ),
        pytest.param(
            ti.math.vec2(0, 0),
            ti.math.vec2(4, 4),
            ti.math.vec2(4, 0),
            ti.math.vec2(0, 4),
            True,
        ),
        pytest.param(
            ti.math.vec2(1, 1),
            ti.math.vec2(2, 2),
            ti.math.vec2(2, 2),
            ti.math.vec2(4, 2),
            True,
        ),
        pytest.param(
            ti.math.vec2(1, 1),
            ti.math.vec2(3, 3),
            ti.math.vec2(1, 1),
            ti.math.vec2(3, 3),
            True,
        ),
        pytest.param(
            ti.math.vec2(1, 1),
            ti.math.vec2(4, 1),
            ti.math.vec2(2, 1),
            ti.math.vec2(5, 1),
            True,
        ),
        pytest.param(
            ti.math.vec2(0, 0),
            ti.math.vec2(3, 0),
            ti.math.vec2(0, 1),
            ti.math.vec2(3, 1),
            False,
        ),
        pytest.param(
            ti.math.vec2(0, 0),
            ti.math.vec2(3, 0),
            ti.math.vec2(3, 0),
            ti.math.vec2(6, 0),
            True,
        ),
        pytest.param(
            ti.math.vec2(2, 1),
            ti.math.vec2(2, 4),
            ti.math.vec2(2, 4),
            ti.math.vec2(2, 6),
            True,
        ),
        pytest.param(
            ti.math.vec2(0, 0),
            ti.math.vec2(2, 0),
            ti.math.vec2(3, 0),
            ti.math.vec2(5, 0),
            False,
        ),
        pytest.param(
            ti.math.vec2(1, 2),
            ti.math.vec2(5, 2),
            ti.math.vec2(3, 2),
            ti.math.vec2(7, 2),
            True,
        ),
        pytest.param(
            ti.math.vec2(0, 0),
            ti.math.vec2(2, 0),
            ti.math.vec2(2, 0),
            ti.math.vec2(3, 1),
            True,
        ),
        pytest.param(
            ti.math.vec2(0, 0),
            ti.math.vec2(5, 5),
            ti.math.vec2(3, 3),
            ti.math.vec2(8, 8),
            True,
        ),
        pytest.param(
            ti.math.vec2(1, 1),
            ti.math.vec2(1, 1),
            ti.math.vec2(0, 0),
            ti.math.vec2(2, 2),
            True,
        ),
        pytest.param(
            ti.math.vec2(5, 5),
            ti.math.vec2(5, 5),
            ti.math.vec2(0, 0),
            ti.math.vec2(2, 2),
            False,
        ),
        pytest.param(
            ti.math.vec2(2, 0),
            ti.math.vec2(2, 5),
            ti.math.vec2(2, 2),
            ti.math.vec2(2, 7),
            True,
        ),
        pytest.param(
            ti.math.vec2(0, 0),
            ti.math.vec2(2, 2),
            ti.math.vec2(3, 1),
            ti.math.vec2(5, 2),
            False,
        ),
        pytest.param(
            ti.math.vec2(-1, 3),
            ti.math.vec2(3, -1),
            ti.math.vec2(-1, -1),
            ti.math.vec2(3, 3),
            True,
        ),
        pytest.param(
            ti.math.vec2(0, 0),
            ti.math.vec2(2, 0),
            ti.math.vec2(3, 1),
            ti.math.vec2(3, -1),
            False,
        ),
        pytest.param(
            ti.math.vec2(0, 0),
            ti.math.vec2(2, 0),
            ti.math.vec2(1, 1),
            ti.math.vec2(1, -1),
            True,
        ),
        pytest.param(
            ti.math.vec2(0, 3),
            ti.math.vec2(3, 0),
            ti.math.vec2(1, 0),
            ti.math.vec2(2, 2),
            True,
        ),
        pytest.param(
            ti.math.vec2(10.5, 10.5),
            ti.math.vec2(20.5, 20.5),
            ti.math.vec2(20.0, 15.0),
            ti.math.vec2(15.0, 20.0),
            True,
        ),
        pytest.param(
            ti.math.vec2(14.3, 16.8),
            ti.math.vec2(26.6, 31.2),
            ti.math.vec2(27.7, 33.3),
            ti.math.vec2(50.0, 66.1),
            False,
        ),
        pytest.param(
            ti.math.vec2(10.0, 11.0),
            ti.math.vec2(30.0, 11.0),
            ti.math.vec2(15.5, 9.5),
            ti.math.vec2(15.5, 25.0),
            True,
        ),
        pytest.param(
            ti.math.vec2(12.2, 33.4),
            ti.math.vec2(22.2, 13.4),
            ti.math.vec2(11.0, 11.0),
            ti.math.vec2(33.0, 33.0),
            True,
        ),
        pytest.param(
            ti.math.vec2(10.0, 10.0),
            ti.math.vec2(15.0, 15.0),
            ti.math.vec2(16.0, 17.0),
            ti.math.vec2(22.0, 23.0),
            False,
        ),
        pytest.param(
            ti.math.vec2(20.5, 20.5),
            ti.math.vec2(30.5, 30.5),
            ti.math.vec2(25.5, 25.5),
            ti.math.vec2(48.5, 25.5),
            True,
        ),
        pytest.param(
            ti.math.vec2(23.7, 28.9),
            ti.math.vec2(37.4, 22.1),
            ti.math.vec2(25.5, 20.0),
            ti.math.vec2(26.6, 12.0),
            False,
        ),
        pytest.param(
            ti.math.vec2(33.3, 11.1),
            ti.math.vec2(22.2, 39.9),
            ti.math.vec2(25.5, 25.5),
            ti.math.vec2(30.0, 11.2),
            False,
        ),
        pytest.param(
            ti.math.vec2(11.1, 22.2),
            ti.math.vec2(33.3, 22.2),
            ti.math.vec2(22.2, 22.2),
            ti.math.vec2(44.4, 22.2),
            True,
        ),
        pytest.param(
            ti.math.vec2(18.8, 15.5),
            ti.math.vec2(32.2, 27.7),
            ti.math.vec2(20.0, 12.0),
            ti.math.vec2(24.0, 30.0),
            True,
        ),
    ],
)
def test_is_segment_intersect(
    seg11: ti.math.vec2,
    seg12: ti.math.vec2,
    seg21: ti.math.vec2,
    seg22: ti.math.vec2,
    expected: bool,
):
    assert expected == _is_segment_intersect_kernel(seg11, seg12, seg21, seg22)
