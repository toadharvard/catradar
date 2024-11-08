import taichi as ti
from catradar.common import *

# For interaction with UI
positions_to_draw = ti.Vector.field(2, dtype=ti.f32, shape=N)
states_to_draw = ti.field(dtype=ti.i32, shape=N)
colors_to_draw = ti.Vector.field(3, dtype=ti.f32, shape=N)


# UI module: Draws circles with different colors based on state
@ti.kernel
def update_color_and_positions():
    for i in range(N):
        positions_to_draw[i] = positions[i] / ti.Vector([X, Y])
        if states[i] == STATE_MOVING:
            colors_to_draw[i] = ti.Vector([0.0, 0.0, 1.0])
        elif states[i] == STATE_INTERACT:
            colors_to_draw[i] = ti.Vector([0.0, 1.0, 0.0])
        elif states[i] == STATE_INTERSECTION:
            colors_to_draw[i] = ti.Vector([1.0, 0.0, 0.0])


def draw(canvas: ti.ui.Canvas):
    update_color_and_positions()
    canvas.circles(positions_to_draw, radius=R0 / X, per_vertex_color=colors_to_draw)
