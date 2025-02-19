import time
import taichi as ti
import numpy as np
from catradar.common import STANDARD_MODE
import tkinter as tk

from catradar.utils import trace

from catradar.canvas import draw_circles, setup_data_for_scene, draw_borders

from catradar.positions_updater import (
    initialize_positions,
    update_positions,
    setup_positions_data,
)
from catradar.grid_manager import compute_states, setup_grid_data, update_logs

# Grid parameters
X: ti.f32 = 1000
Y: ti.f32 = 1000
N: ti.i32 = 500  # Count of cats
R0: ti.f32 = 5.0  # The distance at which cats enter the INTERACT state
R1: ti.f32 = 20.0  # The distance at which cats enter the INTERSECTION state

# Limit of cats per cell for grid algorithm.
# User can't change it, this value was obtained by running tests several times with different limits
LIMIT_PER_CELL: ti.i32 = 100

# Other "soft" parameters. "Soft" means that changing these parameters does not require rebuilding the grid
init_opt: ti.i32 = 0  # Option for positions initializer
movement_pattern: ti.i32 = 0
cursor_push_on: ti.i8 = 0  # Flag that indicates that user enable "cursor push"
speed_mult: ti.f32 = 1  # Speed modifier
render_rate: ti.i32 = 100  # Percentage of rendering cats
norm_func: ti.i32 = 0  # Function for calculating distance

# Logging
show_logs = True
print_logs = True
logged_id: ti.i32 = 0
logs = []
current_page = 0
per_page = 50

# Other UI settings
show_borders = True
allow_large_n = False

# The coefficient needed to convert the coordinates of the field to coordinates for the UI
NORM_RATIO = 1000

# Shared between modules memory
positions = NotImplemented  # Positions of cats
states = NotImplemented  # States of cats

# Data structure for first INTERSECTION_NUM intersections
INTERSECTION_NUM = 10
intersections = NotImplemented

settings_buffer = {
    "X": X,
    "Y": Y,
    "N": N,
    "R0": R0,
    "R1": R1,
    "init_opt": init_opt,
}


def draw_ui(gui: ti.ui.Gui):
    """
    Draws the UI controls for parameters like area size, cats count, movement pattern, cats speed and logging.
    This lets users configure and reset the simulation.
    """
    global \
        render_rate, \
        init_opt, \
        movement_pattern, \
        cursor_push_on, \
        speed_mult, \
        norm_func
    global allow_large_n, logged_id, current_page
    LEFT_BORDER = 0.3
    with gui.sub_window("Simulation parameters", 0, 0, LEFT_BORDER, 0.22) as w:
        settings_buffer["X"] = w.slider_float("X", settings_buffer["X"], 1000, 25000)
        settings_buffer["Y"] = w.slider_float("Y", settings_buffer["Y"], 1000, 25000)
        allow_large_n = w.checkbox("Allow large N", allow_large_n)
        if allow_large_n:
            settings_buffer["N"] = w.slider_int(
                "N", settings_buffer["N"], 500, 5_000_000
            )
        else:
            if settings_buffer["N"] >= 50_000:
                settings_buffer["N"] = 50_000
            settings_buffer["N"] = w.slider_int("N", settings_buffer["N"], 500, 50_000)
        settings_buffer["R0"] = w.slider_float("R0", settings_buffer["R0"], 1.0, 10.0)
        settings_buffer["R1"] = w.slider_float("R1", settings_buffer["R1"], 10.0, 50.0)
        settings_buffer["init_opt"] = w.slider_int(
            "Init positions preset", settings_buffer["init_opt"], 0, 1
        )
        if w.button("Reset"):
            reset_grid()
            initialize_positions(positions, init_opt)

    global show_logs, print_logs, logs, show_borders
    with gui.sub_window("Settings", 0, 0.22, LEFT_BORDER, 0.23) as w:
        render_rate = w.slider_int("Render rate", render_rate, 0, 100)
        speed_mult = w.slider_float("Speed", speed_mult, 0.0, 5.0)
        w.text("0 - Free movement, 1 - Carousel, 2 - Colliding")
        movement_pattern = w.slider_int("Movement pattern", movement_pattern, 0, 2)
        w.text("0 - Euclidean, 1 - Manhattan, 2 - Max")
        norm_func = w.slider_int("Distance function preset", norm_func, 0, 2)
        cursor_push_on = w.checkbox("Allow cursor push", cursor_push_on)
        show_borders = w.checkbox("Show borders", show_borders)
        show_logs = w.checkbox("Show logs", show_logs)
        if not show_logs:
            print_logs = True

    if show_logs:
        with gui.sub_window("Logging", 0, 0.45, LEFT_BORDER, 0.55) as w:
            text_button = "Pause" if print_logs else "Continue"
            if w.button(text_button):
                print_logs = not print_logs
            if w.button("Clear"):
                logs = []
            logged_id = w.slider_int("Logged cat index", logged_id, 0, N - 1)
            logs_sz = len(logs)
            current_page = w.slider_int(
                "Page", current_page, 0, max(logs_sz - 1, 0) // per_page
            )
            left = max(0, logs_sz - (current_page + 1) * per_page)
            right = logs_sz - current_page * per_page - 1
            w.text("\n".join(reversed(logs[left : right - 1])))


def setup_all_data():
    """
    Initializes shared between modules data (positions, states, intersections) and calls setup routines for each.
    """
    global positions, states
    positions = ti.Vector.field(2, dtype=ti.f32, shape=N)
    states = ti.field(dtype=ti.i32, shape=N)
    global intersections
    intersections = ti.field(
        dtype=ti.i32,
        shape=(N, INTERSECTION_NUM + 1),
    )

    setup_positions_data(X, Y, N)
    setup_grid_data(X, Y, N, R0, R1, LIMIT_PER_CELL, INTERSECTION_NUM, STANDARD_MODE)
    setup_data_for_scene(X, Y, N, R0, NORM_RATIO)


def reset_grid():
    """
    Resets simulation parameters from settings_buffer and re-initializes all data.
    """
    global X, Y, N, R0, R1, init_opt
    X = settings_buffer["X"]
    Y = settings_buffer["Y"]
    N = settings_buffer["N"]
    R0 = settings_buffer["R0"]
    R1 = settings_buffer["R1"]
    init_opt = settings_buffer["init_opt"]

    setup_all_data()


# Taichi field for storing last position of cursor (it is updated when user clicks on area)
cursor_pos_field = ti.Vector.field(2, dtype=ti.f32, shape=1)


def main():
    # Getting resolution of user screen
    root = tk.Tk()
    root.withdraw()
    init_resol_x = root.winfo_screenwidth()
    init_resol_y = root.winfo_screenheight()
    root.destroy()

    window = ti.ui.Window(
        "Catradar: cat interaction simulation",
        res=(init_resol_x, init_resol_y),
        fps_limit=60,
        vsync=True,
    )
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.make_camera()

    # Initial camera position
    camera_pos = np.array([0.3, 0.5, 1.5])
    # The camera initially "looks" along the Z axis
    camera_dir = np.array([0.0, 0.0, -1.0])
    # The "up" vector
    up_vector = np.array([0.0, 1.0, 0.0])
    right_vector = np.cross(up_vector, camera_dir)
    right_vector = right_vector / np.linalg.norm(right_vector)

    scene.ambient_light((1, 1, 1))

    gui = window.get_gui()

    setup_all_data()
    initialize_positions(positions, init_opt)

    prev_update_time = time.time()

    while window.running:
        speed = 0.01 * camera_pos[2]  # Camera movement speed

        if window.is_pressed("q"):
            # Moving the camera forward
            camera_pos += camera_dir * speed
        if window.is_pressed("e"):
            # Moving the camera back
            camera_pos -= camera_dir * speed

        if window.is_pressed("a"):
            # Moving the camera to the left
            camera_pos += right_vector * speed
        if window.is_pressed("d"):
            # Moving the camera to the right
            camera_pos -= right_vector * speed

        if window.is_pressed("w"):
            # Moving the camera up
            camera_pos += up_vector * speed
        if window.is_pressed("s"):
            # Moving the camera down
            camera_pos -= up_vector * speed
        camera_pos[2] = max(camera_pos[2], 0.2)

        # Setting a new camera position
        camera.position(camera_pos[0], camera_pos[1], camera_pos[2])
        camera.lookat(
            camera_pos[0] + camera_dir[0],
            camera_pos[1] + camera_dir[1],
            camera_pos[2] + camera_dir[2],
        )
        camera.up(up_vector[0], up_vector[1], up_vector[2])
        scene.set_camera(camera)

        cursor_board_pos = ti.math.vec2(-1000, -1000)
        # Cursor info
        if window.is_pressed(ti.GUI.LMB):
            ws = window.get_window_shape()
            cursor_pos = window.get_cursor_pos()
            zoom = 1.2 / camera_pos[2]
            cursor_board_pos[0] = (
                ws[0]
                * (cursor_pos[0] + (camera_pos[0] * zoom * (ws[1] / ws[0]) - 0.5))
                / zoom
            )
            cursor_board_pos[1] = (
                ws[1] * (cursor_pos[1] + camera_pos[1] * zoom - 0.5) / zoom
            )
            cursor_board_pos *= NORM_RATIO / ws[1]

        new_update_time = time.time()
        trace(
            lambda: update_positions(
                positions,
                intersections,
                cursor_board_pos,
                cursor_push_on,
                speed_mult,
                movement_pattern,
                new_update_time - prev_update_time,
            ),
            "update_positions",
        )
        prev_update_time = new_update_time

        trace(
            lambda: compute_states(
                positions,
                states,
                intersections,
                movement_pattern == 2,
                norm_func,
                logged_id if (show_logs and print_logs) else -1,
            ),
            "compute_states",
        )
        if show_logs and print_logs:
            trace(lambda: update_logs(logged_id, logs), "collect_logs")
        if show_borders:
            trace(lambda: draw_borders(scene), "draw_borders")

        draw_circles(
            scene,
            positions,
            states,
            logged_id if show_logs else -1,
            render_rate,
            NORM_RATIO,
            window.get_window_shape(),
        )
        if cursor_push_on and window.is_pressed(ti.GUI.LMB):
            cursor_pos = window.get_cursor_pos()
            cursor_pos_field[0] = ti.Vector([cursor_pos[0], cursor_pos[1]])
            canvas.circles(cursor_pos_field, radius=0.025 * zoom, color=(0.8, 0.7, 0.7))
        trace(lambda: draw_ui(gui), "draw_ui")
        trace(lambda: canvas.scene(scene), "canvas.scene")
        trace(lambda: window.show(), "window.show")


if __name__ == "__main__":
    main()
