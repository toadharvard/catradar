import taichi as ti

__all__ = ["is_segment_intersect", "get_rotated_vector"]

INF: ti.f32 = 1e9


@ti.func
def _line_intersection(x1, y1, x2, y2, x3, y3, x4, y4) -> ti.math.vec2:
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
def _point_in_rect(px, py, sx1, sy1, sx2, sy2) -> bool:
    return ti.min(sx1, sx2) <= px <= ti.max(sx1, sx2) and ti.min(
        sy1, sy2
    ) <= py <= ti.max(sy1, sy2)


@ti.func
def _calc_angel(a: ti.math.vec2, b: ti.math.vec2) -> ti.f32:
    dot = ti.math.dot(a, b)
    u = ti.math.length(a)
    v = ti.math.length(b)

    res: ti.f32 = 0
    if u == 0 or v == 0:
        res = INF
    else:
        res = ti.math.acos(ti.max(ti.min(dot / (u * v), 1), -1))

    return res


@ti.func
def _rotate_vector(v, alpha):
    cos_a = ti.math.cos(alpha)
    sin_a = ti.math.sin(alpha)

    px_new = v.x * cos_a - v.y * sin_a
    py_new = v.x * sin_a + v.y * cos_a

    return ti.math.vec2(px_new, py_new)


# Returns true, if segments [seg11, seg12] and [seg21, seg22] intersect
@ti.func
def is_segment_intersect(
    seg11: ti.math.vec2,
    seg12: ti.math.vec2,
    seg21: ti.math.vec2,
    seg22: ti.math.vec2,
) -> bool:
    inter = _line_intersection(
        seg11.x, seg11.y, seg12.x, seg12.y, seg21.x, seg21.y, seg22.x, seg22.y
    )
    return (
        (inter.x != INF or inter.y != INF)
        and _point_in_rect(inter.x, inter.y, seg11.x, seg11.y, seg12.x, seg12.y)
        and _point_in_rect(inter.x, inter.y, seg21.x, seg21.y, seg22.x, seg22.y)
    )


# Returns rotated vector for
@ti.func
def get_rotated_vector(
    last_pos: ti.math.vec2,
    new_pos: ti.math.vec2,
    border1: ti.math.vec2,
    border2: ti.math.vec2,
    to_rotate: ti.math.vec2,
) -> ti.math.vec2:
    line = ti.math.vec2(border1.x - border2.x, border1.y - border2.y)
    perp = ti.math.vec2(-line.y, line.x)
    s1_p = ti.math.vec2(border1.x - last_pos.x, border1.y - last_pos.y)
    if ti.math.dot(perp, s1_p) < 0:
        perp = -perp

    p_vec = ti.math.vec2(new_pos.x - last_pos.x, new_pos.y - last_pos.y)
    angel: ti.f32 = _calc_angel(p_vec, perp)
    if p_vec.x * perp.y - p_vec.y * perp.x < 0:
        angel *= -1

    return _rotate_vector(to_rotate * (-1), angel * 2)
