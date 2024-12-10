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
logs = []
current_page = 0
per_page = 50
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
}
allow_large_n = False


def draw_ui(gui: ti.ui.Gui):
    global render_rate, init_opt, update_opt, cursor_push_on, speed_mult, norm_func
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
        update_opt = w.slider_int("Movement pattern", update_opt, 0, 2)
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

    global last_click_time, ADDING_STATE
    with gui.sub_window("Adding new border", 0.9, 0, 0.1, 0.2) as w:
        text = "Add border" if ADDING_STATE == NO_ADDING_MODE else "Cancel"
        if w.button(text):
            if ADDING_STATE == NO_ADDING_MODE:
                last_click_time = time.time()
                ADDING_STATE = ZERO_POINTS_ADDED
            else:
                ADDING_STATE = NO_ADDING_MODE
        if w.button("Remove last"):
            if len(drawn_borders_lst) >= 2:
                drawn_borders_lst.pop()
                drawn_borders_lst.pop()


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
    setup_data_for_scene(X, Y, N, R0, NORM_RATIO)


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
drawn_borders_lst = []
# States of adding borders
NO_ADDING_MODE = 0
ZERO_POINTS_ADDED = 1
ONE_POINT_ADDED = 2
ADDING_STATE = NO_ADDING_MODE
# Handling clicks
DELAY = 0.1  # in seconds
last_click_time = 0


def process_click(window, camera_pos) -> ti.math.vec2:
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

        # print(cursor_board_pos.x, cursor_board_pos.y)

        global last_click_time, ADDING_STATE, current_border
        if ADDING_STATE != NO_ADDING_MODE:
            cur_time = time.time()
            # Forbid add point in the menu bar
            if cur_time - last_click_time < DELAY:
                return
            last_click_time = cur_time

            current_point = ti.Vector([cursor_board_pos[0], cursor_board_pos[1], 0])
            if ADDING_STATE == ZERO_POINTS_ADDED:
                current_border[0] = current_point
                ADDING_STATE = ONE_POINT_ADDED
            else:
                drawn_borders_lst.append(
                    ti.Vector([current_border[0][0], current_border[0][1], 0])
                )
                drawn_borders_lst.append(current_point)
                ADDING_STATE = NO_ADDING_MODE


def main():
    window = ti.ui.Window(
        "Catradar: cat interaction simulation",
        res=(init_resol_x, init_resol_y),
        fps_limit=60,
        vsync=True,
    )
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.make_camera()

    # Изначальная позиция камеры
    camera_pos = np.array([0.3, 0.5, 1.5])
    # Камера изначально "смотрит" по оси Z
    camera_dir = np.array([0.0, 0.0, -1.0])
    # Вектор "вверх"
    up_vector = np.array([0.0, 1.0, 0.0])
    right_vector = np.cross(up_vector, camera_dir)
    right_vector = right_vector / np.linalg.norm(right_vector)

    scene.ambient_light((1, 1, 1))

    gui = window.get_gui()

    setup_all_data()
    initialize_positions(positions, init_opt)

    prev_update_time = time.time()

    while window.running:
        speed = 0.01 * camera_pos[2]  # Скорость перемещения камеры

        if window.is_pressed("q"):
            # Перемещаем камеру вперед
            camera_pos += camera_dir * speed
        if window.is_pressed("e"):
            # Перемещаем камеру назад
            camera_pos -= camera_dir * speed

        if window.is_pressed("a"):
            # Перемещаем камеру влево
            camera_pos += right_vector * speed
        if window.is_pressed("d"):
            # Перемещаем камеру вправо
            camera_pos -= right_vector * speed

        if window.is_pressed("w"):
            # Перемещаем камеру вверх
            camera_pos += up_vector * speed
        if window.is_pressed("s"):
            # Перемещаем камеру вниз
            camera_pos -= up_vector * speed
        camera_pos[2] = max(camera_pos[2], 0.2)

        # Устанавливаем новую позицию камеры
        camera.position(camera_pos[0], camera_pos[1], camera_pos[2])
        camera.lookat(
            camera_pos[0] + camera_dir[0],
            camera_pos[1] + camera_dir[1],
            camera_pos[2] + camera_dir[2],
        )
        camera.up(up_vector[0], up_vector[1], up_vector[2])
        scene.set_camera(camera)

        cursor_board_pos = process_click(window, camera_pos)

        new_update_time = time.time()
        trace(
            lambda: update_positions(
                positions,
                drawn_borders_lst,
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
                logged_id if (show_logs and print_logs) else -1,
            ),
            "compute_states",
        )
        if show_logs and print_logs:
            trace(lambda: update_logs(logged_id, logs), "collect_logs")
        if show_borders:
            trace(lambda: draw_borders(scene, drawn_borders_lst), "draw_borders")

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
            zoom = 1.2 / camera_pos[2]
            canvas.circles(cursor_pos_field, radius=0.025 * zoom, color=(0.8, 0.7, 0.7))
        trace(lambda: draw_ui(gui), "draw_ui")
        trace(lambda: canvas.scene(scene), "canvas.scene")
        trace(lambda: window.show(), "window.show")


if __name__ == "__main__":
    main()
