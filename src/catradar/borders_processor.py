import taichi as ti

__all__ = ["is_segment_intersect", "get_rotated_vector"]

INF: ti.f32 = 1e9
EPS: ti.f32 = 1e-8


@ti.func
def _is_same_point(p1, p2):
    return ti.abs(p1.x - p2.x) < EPS and ti.abs(p1.y - p2.y) < EPS


@ti.func
def _point_on_line(p, p1, p2):
    return ti.abs((p2.x - p1.x) * (p.y - p1.y) - (p2.y - p1.y) * (p.x - p1.x)) < EPS


@ti.func
def _line_intersection(p1, p2, p3, p4) -> ti.math.vec2:
    res = ti.math.vec2(INF, INF)

    first_degenerate = _is_same_point(p1, p2)
    second_degenerate = _is_same_point(p3, p4)
    if first_degenerate and second_degenerate:
        if _is_same_point(p1, p3):
            res = p1
    elif first_degenerate:
        if _point_on_line(p1, p3, p4):
            res = p1
    elif second_degenerate:
        if _point_on_line(p3, p1, p2):
            res = p3
    else:
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        x3, y3 = p3.x, p3.y
        x4, y4 = p4.x, p4.y

        d1x, d1y = (x2 - x1), (y2 - y1)
        d2x, d2y = (x4 - x3), (y4 - y3)
        cross_d1_d2 = ti.math.cross(ti.math.vec2(d1x, d1y), ti.math.vec2(d2x, d2y))
        if ti.abs(cross_d1_d2) > EPS:
            cross_p13_d2 = ti.math.cross(
                ti.math.vec2(x3 - x1, y3 - y1), ti.math.vec2(d2x, d2y)
            )
            t = cross_p13_d2 / cross_d1_d2
            ix = x1 + t * d1x
            iy = y1 + t * d1y
            res = ti.math.vec2(ix, iy)
        else:
            cross_p13_d1 = ti.math.cross(
                ti.math.vec2(d1x, d1y), ti.math.vec2(x3 - x1, y3 - y1)
            )
            if ti.abs(cross_p13_d1) < EPS:
                res = ti.math.vec2(x3, y3)
    return res


@ti.func
def _point_in_rect(p, r1, r2) -> bool:
    return (
        ti.min(r1.x, r2.x) - EPS <= p.x <= ti.max(r1.x, r2.x) + EPS
        and ti.min(r1.y, r2.y) - EPS <= p.y <= ti.max(r1.y, r2.y) + EPS
    )


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
    inter = _line_intersection(seg11, seg12, seg21, seg22)
    return (
        (inter.x != INF or inter.y != INF)
        and _point_in_rect(inter, seg11, seg12)
        and _point_in_rect(inter, seg21, seg22)
    )


# Returns rotated vector for vector [to_rotate], when point moves from [last_pos] to [new_pos] through the segment [border1, border2]
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
