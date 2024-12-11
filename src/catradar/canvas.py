import taichi as ti

from catradar.common import (
    STATE_IDLE,
    STATE_INTERACT,
    STATE_INTERSECTION,
)
from catradar.utils import trace

N: ti.i32
R0: ti.f32
RATIO: ti.i32

positions_to_draw = NotImplemented
colors_to_draw = NotImplemented

# Grid borders (up, bottom, left, right)
grid_borders_vertices = ti.Vector.field(3, dtype=ti.f32, shape=4)
grid_border_indices = ti.Vector.field(2, dtype=ti.i32, shape=4)
grid_border_indices[0] = ti.Vector([0, 1])
grid_border_indices[1] = ti.Vector([1, 2])
grid_border_indices[2] = ti.Vector([2, 3])
grid_border_indices[3] = ti.Vector([3, 0])

bottom_indices = ti.field(dtype=ti.i32, shape=6)
bottom_indices[0] = 0
bottom_indices[1] = 1
bottom_indices[2] = 3
bottom_indices[3] = 3
bottom_indices[4] = 1
bottom_indices[5] = 2

navigate_grid_vertices = NotImplemented


__all__ = ["setup_data_for_scene", "draw_borders", "draw_circles"]


def setup_data_for_scene(
    aX: ti.f32, aY: ti.f32, aN: ti.i32, aR0: ti.f32, aR1: ti.f32, norm_ratio: ti.f32
):
    global N, R0, RATIO
    N = aN
    R0 = aR0
    RATIO = norm_ratio

    global positions_to_draw, colors_to_draw
    positions_to_draw = ti.Vector.field(3, dtype=ti.f32, shape=N)
    colors_to_draw = ti.Vector.field(3, dtype=ti.f32, shape=N)

    fill_vertices(
        aX / norm_ratio, aY / norm_ratio, R0 / norm_ratio / 2, aR1 / norm_ratio
    )


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


# @ti.kernel
def fill_vertices(X: ti.f32, Y: ti.f32, R: ti.f32, R1: ti.f32):
    grid_borders_vertices[0] = ti.Vector([-R, -R, 0])
    grid_borders_vertices[1] = ti.Vector([X + R, -R, 0])
    grid_borders_vertices[2] = ti.Vector([X + R, Y + R, 0])
    grid_borders_vertices[3] = ti.Vector([-R, Y + R, 0])

    global navigate_grid_vertices
    NAVIGATE_GRID_SIZE = R1
    x_lines_count = int(X // NAVIGATE_GRID_SIZE) + 2
    y_lines_count = int(Y // NAVIGATE_GRID_SIZE) + 2
    navigate_grid_vertices = ti.Vector.field(
        3, dtype=ti.f32, shape=(x_lines_count + y_lines_count) * 2
    )
    c = 0

    for i in range(x_lines_count):
        x = ti.min(i * NAVIGATE_GRID_SIZE, X)
        navigate_grid_vertices[c] = ti.Vector([x, 0, 0])
        c += 1
        navigate_grid_vertices[c] = ti.Vector([x, Y, 0])
        c += 1

    for i in range(y_lines_count):
        y = ti.min(i * NAVIGATE_GRID_SIZE, Y)
        navigate_grid_vertices[c] = ti.Vector([0, y, 0])
        c += 1
        navigate_grid_vertices[c] = ti.Vector([X, y, 0])
        c += 1


def draw_borders(scene: ti.ui.Scene, drawn_borders, drawn_borders_count, width):
    scene.lines(
        vertices=grid_borders_vertices, indices=grid_border_indices, width=width
    )
    scene.lines(
        vertices=drawn_borders,
        width=width,
        vertex_count=drawn_borders_count,
        color=(0.4, 0.4, 1),
    )


def draw_bottom(scene: ti.ui.Scene):
    scene.mesh(
        vertices=grid_borders_vertices,
        indices=bottom_indices,
        color=(0.4, 0.4, 0.4),
        two_sided=True,
    )

    scene.lines(vertices=navigate_grid_vertices, color=(1, 1, 1), width=1)


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
