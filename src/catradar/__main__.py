from collections import defaultdict
import time
import taichi as ti
from catradar.common import STANDARD_MODE
import tkinter as tk

from catradar.utils import trace

from catradar.canvas import (
    draw_circles,
    setup_data_for_scene,
    draw_borders,
)

from catradar.positions_updater import (
    initialize_positions,
    update_positions,
    setup_positions_data,
)
from catradar.common import state_to_str
from catradar.grid_manager import compute_states, setup_grid_data, update_logs
from catradar.view import camera, default_view, third_person_view

# Grid parameters
X: ti.f32 = 1000
Y: ti.f32 = 1000
N: ti.i32 = 500
R0: ti.f32 = 5.0
R1: ti.f32 = 20.0
LIMIT_PER_CELL: ti.i32 = 100
# Other "soft" parameters
init_opt: ti.i32 = 0
update_opt: ti.i32 = 0
cursor_push_on: ti.i8 = 0
speed_mult: ti.f32 = 1
render_rate: ti.i32 = 100
norm_func: ti.i32 = 0
# Logging
show_logs = True
print_logs = True
logged_id: ti.i32 = 0
logs = defaultdict(list)
show_all = True
current_page = 0
per_page = 50
is_3rd_person_view = False  # for logged cat
# Borders (UI)
show_borders = True

# Get resolution of user screen
root = tk.Tk()
root.withdraw()
init_resol_x = root.winfo_screenwidth()
init_resol_y = root.winfo_screenheight()
root.destroy()
NORM_RATIO = 1000

# Shared between modules memory
positions = NotImplemented  # Positions of circles
states = NotImplemented  # States of circles

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
    "show_all": show_all,
}
allow_large_n = False


def fmt_logs(logs):
    return [
        "{}: {} state: {} -> {}".format(
            time.strftime("%H:%M:%S", time.localtime(timestamp)),
            i,
            state_to_str[prev_state],
            state_to_str[new_state],
        )
        if who_changed_id == -1
        else "{}: {} state: {} -> {} (changed by {})".format(
            time.strftime("%H:%M:%S", time.localtime(timestamp)),
            i,
            state_to_str[prev_state],
            state_to_str[new_state],
            who_changed_id,
        )
        for timestamp, i, prev_state, new_state, who_changed_id in logs
    ]


def draw_ui(gui: ti.ui.Gui):
    global render_rate, init_opt, update_opt, cursor_push_on, speed_mult, norm_func
    global allow_large_n, logged_id, current_page, is_3rd_person_view
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

    global show_logs, print_logs, logs, show_borders, show_all
    with gui.sub_window("Settings", 0, 0.22, LEFT_BORDER, 0.23) as w:
        render_rate = w.slider_int("Render rate", render_rate, 0, 100)
        speed_mult = w.slider_float("Speed", speed_mult, 0.0, 5.0)
        w.text("0 - Free movement, 1 - Carousel, 2 - Colliding")
        update_opt = w.slider_int("Movement pattern", update_opt, 0, 2)
        w.text("0 - Euclidean, 1 - Manhattan, 2 - Max")
        norm_func = w.slider_int("Distance function preset", norm_func, 0, 2)
        if not is_3rd_person_view:
            cursor_push_on = w.checkbox("Allow cursor push", cursor_push_on)
        else:
            cursor_push_on = False

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
                logs = defaultdict(list)

            prev_show_all = settings_buffer["show_all"]
            settings_buffer["show_all"] = w.checkbox(
                "Show All", settings_buffer["show_all"]
            )
            if prev_show_all != settings_buffer["show_all"]:
                current_page = 0

            if not settings_buffer["show_all"]:
                logged_id = w.slider_int("Cat index", logged_id, 0, N - 1)
                is_3rd_person_view = w.checkbox("Track current cat", is_3rd_person_view)
                current_logs = fmt_logs(sorted(logs[logged_id], key=lambda log: log[0]))
            else:
                current_logs = fmt_logs(
                    sorted(
                        (log for i in range(N) for log in logs[i]),
                        key=lambda log: log[0],
                    )
                )

            logs_sz = len(current_logs)
            current_page = w.slider_int(
                "Page", current_page, 0, max(logs_sz - 1, 0) // per_page
            )
            left = max(0, logs_sz - (current_page + 1) * per_page)
            right = logs_sz - current_page * per_page - 1

            w.text("\n".join(reversed(current_logs[left : right - 1])))
    else:
        is_3rd_person_view = False

    global last_click_time, adding_state, borders_count
    if not is_3rd_person_view:
        with gui.sub_window("Adding new border", 0.9, 0, 0.1, 0.2) as w:
            text = "Add border" if adding_state == NO_ADDING_MODE else "Cancel"
            if w.button(text):
                if adding_state == NO_ADDING_MODE:
                    last_click_time = time.time()
                    adding_state = ZERO_POINTS_ADDED
                else:
                    adding_state = NO_ADDING_MODE
            if w.button("Remove last"):
                if borders_count >= 2:
                    borders_count -= 1
                    borders[borders_count] = (0.0, 0.0, 0.0)
                    borders_to_draw[borders_count] = (0.0, 0.0, 0.0)

                    borders_count -= 1
                    borders[borders_count] = (0.0, 0.0, 0.0)
                    borders_to_draw[borders_count] = (0.0, 0.0, 0.0)


def setup_all_data():
    # data shared between all modules
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
    setup_data_for_scene(X, Y, N, R0, R1, NORM_RATIO)


def reset_grid():
    global X, Y, N, R0, R1, init_opt
    X = settings_buffer["X"]
    Y = settings_buffer["Y"]
    N = settings_buffer["N"]
    R0 = settings_buffer["R0"]
    R1 = settings_buffer["R1"]
    init_opt = settings_buffer["init_opt"]

    setup_all_data()


cursor_pos_field = ti.Vector.field(2, dtype=ti.f32, shape=1)

# Borders drawn by user
current_border = ti.Vector.field(3, dtype=ti.f32, shape=2)
borders_count = 0
borders_to_draw = ti.Vector.field(3, dtype=ti.f32, shape=100)
borders = ti.Vector.field(3, dtype=ti.f32, shape=100)

# States of adding borders
NO_ADDING_MODE = 0
ZERO_POINTS_ADDED = 1
ONE_POINT_ADDED = 2
adding_state = NO_ADDING_MODE
# Handling clicks
DELAY = 0.1  # in seconds
last_click_time = 0


def process_click(window, canvas, camera_pos) -> ti.math.vec2:
    cursor_board_pos = ti.math.vec2(-1000, -1000)
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

        if cursor_push_on:
            cursor_pos = window.get_cursor_pos()
            cursor_pos_field[0] = ti.Vector([cursor_pos[0], cursor_pos[1]])
            zoom = 1.2 / camera_pos[2]
            canvas.circles(cursor_pos_field, radius=0.025 * zoom, color=(0.8, 0.7, 0.7))

        global \
            last_click_time, \
            adding_state, \
            current_border, \
            borders_to_draw, \
            borders_count
        if adding_state != NO_ADDING_MODE:
            cur_time = time.time()
            if cur_time - last_click_time < DELAY:
                return cursor_board_pos
            last_click_time = cur_time

            current_point = ti.Vector([cursor_board_pos[0], cursor_board_pos[1], 0])
            if adding_state == ZERO_POINTS_ADDED:
                current_border[0] = current_point
                adding_state = ONE_POINT_ADDED
            else:
                borders_to_draw[borders_count] = current_border[0] / NORM_RATIO
                borders[borders_count] = current_border[0]
                borders_count += 1

                borders_to_draw[borders_count] = current_point / NORM_RATIO
                borders[borders_count] = current_point
                borders_count += 1

                adding_state = NO_ADDING_MODE
    return cursor_board_pos


def main():
    window = ti.ui.Window(
        "Catradar: cat interaction simulation",
        res=(init_resol_x, init_resol_y),
        fps_limit=60,
        vsync=True,
    )
    canvas = window.get_canvas()
    scene = window.get_scene()

    gui = window.get_gui()

    setup_all_data()
    initialize_positions(positions, init_opt)

    prev_update_time = time.time()
    prev_logged_pos = ti.Vector([ti.math.nan, ti.math.nan])

    while window.running:
        if not is_3rd_person_view:
            trace(
                lambda: default_view(scene, window),
                "default_view",
            )
        else:
            trace(
                lambda: third_person_view(
                    scene, NORM_RATIO, prev_logged_pos, positions[logged_id]
                ),
                "third_person_view",
            )

        if logged_id > -1:
            prev_logged_pos[0] = positions[logged_id][0]
            prev_logged_pos[1] = positions[logged_id][1]
        else:
            prev_logged_pos[0] = ti.math.nan
            prev_logged_pos[1] = ti.math.nan

        cursor_board_pos = process_click(window, canvas, camera.curr_position)

        new_update_time = time.time()
        trace(
            lambda: update_positions(
                positions,
                borders,
                borders_count,
                intersections,
                cursor_board_pos,
                cursor_push_on,
                speed_mult,
                update_opt,
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
                update_opt == 2,
                norm_func,
            ),
            "compute_states",
        )
        if show_logs and print_logs:
            trace(lambda: update_logs(logs), "collect_logs")
        if show_borders:
            trace(
                lambda: draw_borders(
                    scene,
                    borders_to_draw,
                    borders_count,
                    width=2 if not is_3rd_person_view else 6,
                ),
                "draw_borders",
            )

        draw_circles(
            scene,
            positions,
            states,
            logged_id if show_logs else -1,
            render_rate,
            NORM_RATIO,
            window.get_window_shape(),
        )
        trace(lambda: draw_ui(gui), "draw_ui")
        trace(lambda: canvas.scene(scene), "canvas.scene")
        trace(lambda: window.show(), "window.show")


if __name__ == "__main__":
    main()
