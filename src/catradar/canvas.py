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

__all__ = [
    "setup_data_for_scene",
    "draw_circles",
]


def setup_data_for_scene(aN: ti.i32, aR0: ti.f32):
    global N, R0
    N = aN
    R0 = aR0

    global positions_to_draw, colors_to_draw
    positions_to_draw = ti.Vector.field(3, dtype=ti.f32, shape=N)
    colors_to_draw = ti.Vector.field(3, dtype=ti.f32, shape=N)


@ti.kernel
def update_colors(
    positions: ti.template(),
    states: ti.template(),
    render_rate: ti.i32,
    resol_x: ti.i32,
    resol_y: ti.i32,
):
    for i in range(int(N * render_rate / 100)):
        fixed = positions[i] / ti.Vector([resol_x, resol_y])
        positions_to_draw[i] = ti.Vector([fixed[0], fixed[1], 0])
        if states[i] == STATE_IDLE:
            colors_to_draw[i] = ti.Vector([0.0, 0.0, 1.0])
        elif states[i] == STATE_INTERACT:
            colors_to_draw[i] = ti.Vector([0.0, 1.0, 0.0])
        elif states[i] == STATE_INTERSECTION:
            colors_to_draw[i] = ti.Vector([1.0, 0.0, 0.0])


def draw_circles(
    scene: ti.ui.Scene,
    positions: ti.template(),
    states: ti.template(),
    render_rate: ti.i32,
    resol_x: ti.i32,
    resol_y: ti.i32,
):
    if render_rate == 0:  # Do not render at all
        return
    trace(
        lambda: update_colors(positions, states, render_rate, resol_x, resol_y),
        "update_colors",
    )
    trace(
        lambda: scene.particles(
            positions_to_draw,
            radius=R0 / 2 / resol_x,
            per_vertex_color=colors_to_draw,
            index_offset=0,
            index_count=int(N * render_rate / 100),
        ),
        "draw particles",
    )
