import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

# Parameters
# Simulation
X, Y = 1000.0, 1000.0  # Size of the map
RES_X, RES_Y = 1000, 1000
N = 1000  # Number of circles
R0 = 5.0  # Intersection radius threshold
R1 = 20.0  # Interaction radius threshold (R1 > R0)
LIMIT_PER_CELL = 250

# Time interval for updating states
tau = 0.01
# Time step for simulation
dt = 0.01
assert dt <= tau
# Number of substeps before updating states
num_substeps = int(tau / dt)

# Size of each grid cell
grid_cell_size = R1
grid_resolution_x = int(X / grid_cell_size) + 1
grid_resolution_y = int(Y / grid_cell_size) + 1

# States
STATE_IDLE = 0
STATE_INTERACT = 1
STATE_INTERSECTION = 2
state_to_str = {
    STATE_IDLE: "IDLE",
    STATE_INTERACT: "INTERACT",
    STATE_INTERSECTION: "INTERSECTION",
}

# Shared memory - positions of circles
positions = NotImplemented
velocities = NotImplemented
states = NotImplemented

# Grid data structures for collision detection
grid_num_circles = NotImplemented
grid_circles = NotImplemented

# For interaction with UI
positions_to_draw = NotImplemented
states_to_draw = NotImplemented
colors_to_draw = NotImplemented


# Random initialization of positions and velocities
@ti.kernel
def initialize(positions: ti.template(), velocities: ti.template(), opt: ti.types.int8):
    if opt == 0:
        for i in range(N):
            positions[i] = ti.Vector([50 + ti.random() * 10, 50 + ti.random()])
            velocities[i] = ti.Vector([10 + ti.random(), 10 + ti.random()]) * 10
            states[i] = STATE_IDLE
    if opt == 1:
        for i in range(N):
            positions[i] = ti.Vector([ti.random() * X, ti.random() * Y])
            velocities[i] = ti.Vector([ti.random() * 100 - 50, ti.random() * 100 - 50])
            states[i] = STATE_IDLE


# First module: Updates positions and writes to shared memory
@ti.kernel
def update_positions(positions: ti.template(), velocities: ti.template()):
    for i in range(N):
        positions[i] += velocities[i] * dt
        # Boundary conditions
        if positions[i].x < 0:
            positions[i].x = 0
            velocities[i].x *= -1
        if positions[i].x > X:
            positions[i].x = X
            velocities[i].x *= -1
        if positions[i].y < 0:
            positions[i].y = 0
            velocities[i].y *= -1
        if positions[i].y > Y:
            positions[i].y = Y
            velocities[i].y *= -1


logs_prev_state = ti.field(ti.i32, shape=())
logs_new_state = ti.field(ti.i32, shape=())
logs_who_chaged_id = ti.field(ti.i32, shape=())
logged = 0
logs = []

EUCLIDEAN_NORM = 0
MANHATTAN_NORM = 1
MAX_NORM = 2
norm_func = MAX_NORM

norm_result = ti.field(ti.f32, shape=())


@ti.func
def calc_dist(pos_i, pos_j):
    norm_result[None] = 0

    if norm_func == EUCLIDEAN_NORM:
        norm_result[None] = (pos_i - pos_j).norm()

    if norm_func == MANHATTAN_NORM:
        norm_result[None] = ti.abs(pos_i.x - pos_j.x) + ti.abs(pos_i.y - pos_j.y)

    if norm_func == MAX_NORM:
        norm_result[None] = ti.max(ti.abs(pos_i.x - pos_j.x), ti.abs(pos_i.y - pos_j.y))

    return norm_result[None]


# Second module: Reads positions and computes states
@ti.kernel
def compute_states(
    positions: ti.template(),
    states: ti.template(),
    grid_num_circles: ti.template(),
    grid_circles: ti.template(),
):
    grid_num_circles.fill(0)

    # Insert circles into grid
    for idx in range(N):
        pos = positions[idx]
        gx = int(pos.x / grid_cell_size)
        gy = int(pos.y / grid_cell_size)
        num = ti.atomic_add(grid_num_circles[gx, gy], 1)
        if num < LIMIT_PER_CELL:
            grid_circles[gx, gy, num] = idx

    for idx in range(N):
        pos_i = positions[idx]
        gx = int(pos_i.x / grid_cell_size)
        gy = int(pos_i.y / grid_cell_size)

        if logged == idx:
            logs_who_chaged_id[None] = -1  # Изначально нас никто не менял, поэтому -1
            logs_prev_state[None] = states[idx]

        state = STATE_IDLE
        for offset_x in range(-1, 2):
            for offset_y in range(-1, 2):
                ng_x = gx + offset_x
                ng_y = gy + offset_y
                if 0 <= ng_x < grid_resolution_x and 0 <= ng_y < grid_resolution_y:
                    num = grid_num_circles[ng_x, ng_y]
                    for k in range(num):
                        jdx = grid_circles[ng_x, ng_y, k]
                        if jdx != idx:
                            pos_j = positions[jdx]
                            dist = calc_dist(pos_i, pos_j)
                            if dist <= R0:
                                state = STATE_INTERSECTION
                                if logged == idx:
                                    logs_who_chaged_id[None] = jdx
                                break  # Exit early for performance
                            elif dist <= R1:
                                prob = 1.0 / (dist * dist)
                                if ti.random() < prob:
                                    state = STATE_INTERACT
                                    if logged == idx:
                                        logs_who_chaged_id[None] = jdx
                    if state == STATE_INTERSECTION:
                        break  # Exit early if state is determined
            if state == STATE_INTERSECTION:
                break
        if logged == idx:
            logs_new_state[None] = state
        states[idx] = state


@ti.kernel
def update_color_and_positions(
    positions: ti.template(),
    states: ti.template(),
    colors_to_draw: ti.template(),
    positions_to_draw: ti.template(),
):
    for i in range(N):
        fixed = positions[i] / ti.Vector([RES_X, RES_Y])
        positions_to_draw[i] = ti.Vector([fixed[0], fixed[1], 0])
        if states[i] == STATE_IDLE:
            colors_to_draw[i] = ti.Vector([0.0, 0.0, 1.0])
        elif states[i] == STATE_INTERACT:
            colors_to_draw[i] = ti.Vector([0.0, 1.0, 0.0])
        elif states[i] == STATE_INTERSECTION:
            colors_to_draw[i] = ti.Vector([1.0, 0.0, 0.0])


def draw(
    canvas: ti.ui.Scene,
    positions: ti.template(),
    states: ti.template(),
    positions_to_draw: ti.template(),
    colors_to_draw: ti.template(),
):
    update_color_and_positions(positions, states, colors_to_draw, positions_to_draw)
    canvas.particles(
        positions_to_draw, radius=R0 / 2 / RES_X, per_vertex_color=colors_to_draw
    )


initial_opt = 0
reset_settings = {
    "X": X,
    "Y": Y,
    "N": N,
    "R0": R0,
    "R1": R1,
    "LIMIT_PER_CELL": LIMIT_PER_CELL,
    "tau": tau,
}
settings = {
    "initial_opt": initial_opt,
    "norm_func": norm_func,
}
# Сейчас не используется, потому что из taichi scope не получится использовать динамические массивы...
current_page = 0
per_page = 50


def draw_ui(gui: ti.ui.Gui):
    global initial_opt
    global current_page, logged, logs
    global norm_func
    with gui.sub_window("Simulation", 0, 0, 0.2, 0.3) as w:
        reset_settings["X"] = w.slider_float("X", reset_settings["X"], 1000, 5000)
        reset_settings["Y"] = w.slider_float("Y", reset_settings["Y"], 1000, 5000)
        reset_settings["N"] = w.slider_int("N", reset_settings["N"], 500, 500_000)
        reset_settings["R0"] = w.slider_float("R0", reset_settings["R0"], 1.0, 10.0)
        reset_settings["R1"] = w.slider_float("R1", reset_settings["R1"], 10.0, 50.0)
        reset_settings["LIMIT_PER_CELL"] = w.slider_int(
            "LIMIT", reset_settings["LIMIT_PER_CELL"], 100, 2000
        )
        reset_settings["tau"] = w.slider_float("tau", reset_settings["tau"], 0.001, 0.1)
        if w.button("Reset"):
            reset(**{"a" + k: v for k, v in reset_settings.items()})
            initialize(positions, velocities, initial_opt)

    with gui.sub_window("Settings", 0, 0.3, 0.2, 0.2) as w:
        settings["initial_opt"] = w.slider_int(
            "Position preset", settings["initial_opt"], 0, 1
        )
        if w.button("Set"):
            initial_opt = settings["initial_opt"]
            initialize(positions, velocities, initial_opt)

        w.text("0: Euclidean\n1: Manhattan\n2: Max")
        settings["norm_func"] = w.slider_int(
            "Distance preset", settings["norm_func"], 0, 2
        )
        if w.button("Set"):
            norm_func = settings["norm_func"]
    with gui.sub_window("Logs", 0.6, 0.7, 0.4, 0.3) as w:
        if w.button("Clear"):
            logs = []
        logged = w.slider_int("Logged circle index", logged, 0, N - 1)
        current_page = w.slider_int("Page", current_page, 0, len(logs) // per_page)
        w.text("\n".join(logs[current_page * per_page : (current_page + 1) * per_page]))


def reset(
    aX=1000.0,
    aY=1000.0,
    aN=1000,
    aR0=5.0,
    aR1=20.0,
    aLIMIT_PER_CELL=250,
    atau=0.01,
):
    global X, Y, N, R0, R1, LIMIT_PER_CELL, initial_opt
    X, Y = aX, aY
    N = aN
    R0 = aR0
    R1 = aR1
    LIMIT_PER_CELL = aLIMIT_PER_CELL
    global tau, dt, num_substeps
    # Time interval for updating states
    tau = atau
    # Time step for simulation
    dt = atau
    # Number of substeps before updating states
    num_substeps = int(tau / dt)

    global grid_cell_size, grid_resolution_x, grid_resolution_y
    # Size of each grid cell
    grid_cell_size = R1
    grid_resolution_x = int(X / grid_cell_size) + 1
    grid_resolution_y = int(Y / grid_cell_size) + 1

    global positions, velocities, states
    # Shared memory - positions of circles
    positions = ti.Vector.field(2, dtype=ti.f32, shape=N)
    velocities = ti.Vector.field(2, dtype=ti.f32, shape=N)
    states = ti.field(dtype=ti.i32, shape=N)

    global grid_num_circles, grid_circles
    # Grid data structures for collision detection
    grid_num_circles = ti.field(
        dtype=ti.i32, shape=(grid_resolution_x, grid_resolution_y)
    )
    grid_circles = ti.field(
        dtype=ti.i32, shape=(grid_resolution_x, grid_resolution_y, LIMIT_PER_CELL)
    )

    global positions_to_draw, states_to_draw, colors_to_draw
    # For interaction with UI
    positions_to_draw = ti.Vector.field(3, dtype=ti.f32, shape=N)
    states_to_draw = ti.field(dtype=ti.i32, shape=N)
    colors_to_draw = ti.Vector.field(3, dtype=ti.f32, shape=N)


def collect_logs():
    if logs_new_state[None] == logs_prev_state[None]:
        return
    if logs_who_chaged_id[None] == -1:
        logs.append(
            "State of {} id changed: {} -> {}".format(
                logged,
                state_to_str[logs_prev_state[None]],
                state_to_str[logs_new_state[None]],
            )
        )
        return
    logs.append(
        "State of {} id changed: {} -> {} by {} id".format(
            logged,
            state_to_str[logs_prev_state[None]],
            state_to_str[logs_new_state[None]],
            logs_who_chaged_id[None],
        )
    )


def main():
    window = ti.ui.Window("Circles", res=(RES_X, RES_Y), fps_limit=60, vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.make_camera()

    # Изначальная позиция камеры
    camera_pos = np.array([0.5, 0.5, 2.0])
    # Камера изначально "смотрит" по оси Z
    camera_dir = np.array([0.0, 0.0, -1.0])
    # Вектор "вверх"
    up_vector = np.array([0.0, 1.0, 0.0])
    speed = 0.02  # Скорость перемещения камеры

    scene.ambient_light((1, 1, 1))

    gui = window.get_gui()
    accumulated_time = 0.0
    reset()
    initialize(
        positions,
        velocities,
        1,
    )

    while window.running:
        right_vector = np.cross(up_vector, camera_dir)
        right_vector = right_vector / np.linalg.norm(right_vector)

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

        # Устанавливаем новую позицию камеры
        camera.position(camera_pos[0], camera_pos[1], max(camera_pos[2], 0.2))
        camera.lookat(
            camera_pos[0] + camera_dir[0],
            camera_pos[1] + camera_dir[1],
            camera_pos[2] + camera_dir[2],
        )
        camera.up(up_vector[0], up_vector[1], up_vector[2])
        scene.set_camera(camera)

        for _ in range(num_substeps):
            update_positions(positions, velocities)
            accumulated_time += dt
            if accumulated_time >= tau:
                compute_states(
                    positions,
                    states,
                    grid_num_circles,
                    grid_circles,
                )
                collect_logs()
                accumulated_time = 0.0

        draw(scene, positions, states, positions_to_draw, colors_to_draw)
        draw_ui(gui)
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
