import time
import taichi as ti
import numpy as np

from catradar.utils import trace

from catradar.canvas import draw_circles, setup_data_for_scene

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

logged_id: ti.i32 = 0
logs = []
current_page = 0
per_page = 50

window_resol_x = 1000
window_resol_y = 1000

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


def draw_ui(gui: ti.ui.Gui):
    global render_rate, init_opt, update_opt, cursor_push_on, speed_mult, norm_func
    global logged_id, current_page, logs
    with gui.sub_window("Simulation", 0, 0, 0.2, 0.3) as w:
        settings_buffer["X"] = w.slider_float("X", settings_buffer["X"], 1000, 10000)
        settings_buffer["Y"] = w.slider_float("Y", settings_buffer["Y"], 1000, 10000)
        settings_buffer["N"] = w.slider_int("N", settings_buffer["N"], 500, 1_000_000)
        settings_buffer["R0"] = w.slider_float("R0", settings_buffer["R0"], 1.0, 10.0)
        settings_buffer["R1"] = w.slider_float("R1", settings_buffer["R1"], 10.0, 50.0)
        if w.button("Reset"):
            reset_grid()
            initialize_positions(positions, init_opt)

    with gui.sub_window("Settings", 0, 0.3, 0.2, 0.25) as w:
        render_rate = w.slider_int("Render rate", render_rate, 0, 100)
        settings_buffer["init_opt"] = w.slider_int(
            "Position preset", settings_buffer["init_opt"], 0, 1
        )
        if w.button("Set position preset"):
            init_opt = settings_buffer["init_opt"]
            initialize_positions(positions, init_opt)
        w.text("0: Free movement\n1: Carousel\n2: Colliding")
        update_opt = w.slider_int("Movement pattern", update_opt, 0, 2)
        speed_mult = w.slider_float("Speed", speed_mult, 0.0, 5.0)
        cursor_push_on = w.checkbox("Cursor push", cursor_push_on)

        w.text("0: Euclidean\n1: Manhattan\n2: Max")
        norm_func = w.slider_int("Distance preset", norm_func, 0, 2)

    with gui.sub_window("Logs", 0.6, 0.7, 0.4, 0.3) as w:
        if w.button("Clear"):
            logs = []
        logged_id = w.slider_int("Logged circle index", logged_id, 0, N - 1)
        current_page = w.slider_int("Page", current_page, 0, len(logs) // per_page)
        w.text("\n".join(logs[current_page * per_page : (current_page + 1) * per_page]))


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
    setup_grid_data(X, Y, N, R0, R1, LIMIT_PER_CELL, INTERSECTION_NUM)
    setup_data_for_scene(N, R0)


def reset_grid():
    global X, Y, N, R0, R1, LIMIT_PER_CELL
    X = settings_buffer["X"]
    Y = settings_buffer["Y"]
    N = settings_buffer["N"]
    R0 = settings_buffer["R0"]
    R1 = settings_buffer["R1"]

    setup_all_data()


cursor_pos_field = ti.Vector.field(2, dtype=ti.f32, shape=1)


def main():
    window = ti.ui.Window(
        "Catradar: cat interaction simulation",
        res=(window_resol_x, window_resol_y),
        fps_limit=60,
        vsync=True,
    )
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.make_camera()

    # Изначальная позиция камеры
    camera_pos = np.array([0.5, 0.5, 2.0])
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

        cursor_board_pos = ti.math.vec2(-1000, -1000)
        # cursor info
        if window.is_pressed(ti.GUI.LMB):
            ws = window.get_window_shape()
            cursor_pos = window.get_cursor_pos()
            zoom = 1.2 / camera_pos[2]
            # i really don't know how but it work
            cursor_board_pos[0] = (
                ws[0]
                * (cursor_pos[0] + (camera_pos[0] * zoom * (ws[1] / ws[0]) - 0.5))
                / zoom
            )
            cursor_board_pos[1] = (
                ws[1] * (cursor_pos[1] + camera_pos[1] * zoom - 0.5) / zoom
            )
            cursor_board_pos *= window_resol_y / ws[1]
            # gui.text(f"cursor {cursor_board_pos[0]} {cursor_board_pos[1]}")

        new_update_time = time.time()
        trace(
            lambda: update_positions(
                positions,
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
                positions, states, intersections, update_opt == 2, norm_func, logged_id
            ),
            "compute_states",
        )
        trace(lambda: update_logs(logged_id, logs), "collect_logs")
        draw_circles(
            scene,
            positions,
            states,
            logged_id,
            render_rate,
            window_resol_x,
            window_resol_y,
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
