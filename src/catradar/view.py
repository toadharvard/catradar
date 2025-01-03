import taichi as ti
import numpy as np

from catradar.canvas import draw_bottom

__all__ = ["camera", "default_view", "third_person_view"]

camera = ti.ui.make_camera()

forward_vector = np.array([0.0, 0.0, -1.0])
up_vector = np.array([0.0, 1.0, 0.0])
right_vector = np.array([-1.0, 0.0, 0.0])


default_camera_pos = np.array([0.3, 0.5, 1.5])

third_person_prev_angle = 0.0


def default_camera_mover(window: ti.ui.Window, camera_pos: np.ndarray):
    speed = 0.01 * camera_pos[2]  # Скорость перемещения камеры

    if window.is_pressed("q"):
        camera_pos += forward_vector * speed
    if window.is_pressed("e"):
        camera_pos -= forward_vector * speed

    if window.is_pressed("a"):
        camera_pos += right_vector * speed
    if window.is_pressed("d"):
        camera_pos -= right_vector * speed
    if window.is_pressed("w"):
        camera_pos += up_vector * speed
    if window.is_pressed("s"):
        camera_pos -= up_vector * speed
    camera_pos[2] = max(camera_pos[2], 0.2)


def default_camera_set(scene: ti.ui.Scene, camera_pos: np.ndarray):
    # Устанавливаем новую позицию камеры
    camera.position(camera_pos[0], camera_pos[1], camera_pos[2])
    camera.lookat(
        camera_pos[0] + forward_vector[0],
        camera_pos[1] + forward_vector[1],
        camera_pos[2] + forward_vector[2],
    )
    camera.up(up_vector[0], up_vector[1], up_vector[2])
    scene.set_camera(camera)


def cat_camera(scene: ti.ui.Scene, norm_ratio, prev_pos, pos):
    global third_person_prev_angle
    if prev_pos[0] != ti.math.nan:
        dif = prev_pos - pos
        x = pos[0] / norm_ratio
        y = pos[1] / norm_ratio
        angle = ti.math.atan2(dif[0], dif[1])
        if angle == 0:  # if position didn't update, we would want to save old angle
            angle = third_person_prev_angle

        third_person_prev_angle = angle
        cos = ti.math.cos(angle)
        sin = ti.math.sin(angle)

        curr_pos = camera.curr_position
        new_pos = ti.Vector([x + sin * 0.1, y + cos * 0.1, 0.1])
        pos_dif = (new_pos - curr_pos) / 10
        curr_pos = pos_dif + curr_pos

        camera.position(curr_pos[0], curr_pos[1], curr_pos[2])
        camera.lookat(
            x,
            y,
            0,
        )
        camera.up(0, 0, 1)
        scene.set_camera(camera)


def default_view(scene: ti.ui.Scene, window: ti.ui.Window):
    scene.ambient_light((1, 1, 1))

    default_camera_mover(window, default_camera_pos)
    default_camera_set(scene, default_camera_pos)


def third_person_view(scene: ti.ui.Scene, norm_ratio, prev_pos, pos):
    scene.point_light((0, 0, 100), (1, 1, 1))

    cat_camera(scene, norm_ratio, prev_pos, pos)
    draw_bottom(scene)
