import taichi as ti

from catradar.common import (
    STATE_IDLE,
    STATE_INTERACT,
    STATE_INTERSECTION,
)
from catradar.utils import trace

N: ti.i32
R0: ti.f32

positions_to_draw = NotImplemented
colors_to_draw = NotImplemented
border_vertices = ti.Vector.field(3, dtype=ti.f32, shape=4)
border_indices = ti.Vector.field(2, dtype=ti.i32, shape=4)
border_indices[0] = ti.Vector([0, 1])
border_indices[1] = ti.Vector([1, 2])
border_indices[2] = ti.Vector([2, 3])
border_indices[3] = ti.Vector([3, 0])

__all__ = [
    "setup_data_for_scene",
    "draw_borders",
    "draw_circles",
]


def setup_data_for_scene(
    aX: ti.f32, aY: ti.f32, aN: ti.i32, aR0: ti.f32, norm_ratio: ti.f32
):
    global N, R0
    N = aN
    R0 = aR0

    global positions_to_draw, colors_to_draw
    positions_to_draw = ti.Vector.field(3, dtype=ti.f32, shape=N)
    colors_to_draw = ti.Vector.field(3, dtype=ti.f32, shape=N)

    fill_vertices(aX / norm_ratio, aY / norm_ratio, R0 / norm_ratio / 2)


@ti.kernel
def update_colors(
    positions: ti.template(),
    states: ti.template(),
    logged_id: ti.i32,
    render_rate: ti.i32,
    norm_ratio: ti.f32,
):
    for i in range(int(N * render_rate / 100)):
        fixed = positions[i] / norm_ratio
        positions_to_draw[i] = ti.Vector([fixed[0], fixed[1], 0])
        if logged_id == i:
            colors_to_draw[i] = ti.Vector([0.5, 0.5, 0.5])
        else:
            colors_to_draw[i] = ti.Vector([0.0, 0.0, 0.0])

        if states[i] == STATE_IDLE:
            colors_to_draw[i] += ti.Vector([0.0, 0.0, 0.5])
        elif states[i] == STATE_INTERACT:
            colors_to_draw[i] += ti.Vector([0.0, 0.5, 0.0])
        elif states[i] == STATE_INTERSECTION:
            colors_to_draw[i] += ti.Vector([0.5, 0.0, 0.0])


@ti.kernel
def fill_vertices(X: ti.f32, Y: ti.f32, R: ti.f32):
    border_vertices[0] = ti.Vector([-R, -R, 0])
    border_vertices[1] = ti.Vector([X + R, -R, 0])
    border_vertices[2] = ti.Vector([X + R, Y + R, 0])
    border_vertices[3] = ti.Vector([-R, Y + R, 0])


def draw_borders(scene: ti.ui.Scene):
    scene.lines(vertices=border_vertices, indices=border_indices, width=2)


def draw_circles(
    scene: ti.ui.Scene,
    positions: ti.template(),
    states: ti.template(),
    logged_id: ti.i32,
    render_rate: ti.i32,
    norm_ratio,
    window_size: tuple,
):
    if render_rate == 0:  # Do not render at all
        return
    trace(
        lambda: update_colors(positions, states, logged_id, render_rate, norm_ratio),
        "update_colors",
    )

    trace(
        lambda: scene.particles(
            positions_to_draw,
            radius=R0 / 2 / norm_ratio * (window_size[1] / window_size[0]),
            per_vertex_color=colors_to_draw,
            index_offset=0,
            index_count=int(N * render_rate / 100),
        ),
        "draw particles",
    )
