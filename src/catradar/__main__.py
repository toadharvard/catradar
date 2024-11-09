import taichi as ti

ti.init(arch=ti.gpu)

# Parameters
# Simulation
X, Y = 1000.0, 1000.0  # Size of the map
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
STATE_MOVING = 0
STATE_INTERACT = 1
STATE_INTERSECTION = 2

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
            states[i] = STATE_MOVING
    if opt == 1:
        for i in range(N):
            positions[i] = ti.Vector([ti.random() * X, ti.random() * Y])
            velocities[i] = ti.Vector([ti.random() * 100 - 50, ti.random() * 100 - 50])
            states[i] = STATE_MOVING


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
        state = STATE_MOVING

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
                            dist = (pos_i - pos_j).norm()
                            if dist <= R0:
                                state = STATE_INTERSECTION
                                break  # Exit early for performance
                            elif dist <= R1:
                                prob = 1.0 / (dist * dist)
                                if ti.random() < prob:
                                    state = STATE_INTERACT
                    if state == STATE_INTERSECTION:
                        break  # Exit early if state is determined
            if state == STATE_INTERSECTION:
                break
        states[idx] = state


@ti.kernel
def update_color_and_positions(
    positions: ti.template(),
    states: ti.template(),
    colors_to_draw: ti.template(),
    positions_to_draw: ti.template(),
):
    for i in range(N):
        positions_to_draw[i] = positions[i] / ti.Vector([X, Y])
        if states[i] == STATE_MOVING:
            colors_to_draw[i] = ti.Vector([0.0, 0.0, 1.0])
        elif states[i] == STATE_INTERACT:
            colors_to_draw[i] = ti.Vector([0.0, 1.0, 0.0])
        elif states[i] == STATE_INTERSECTION:
            colors_to_draw[i] = ti.Vector([1.0, 0.0, 0.0])


def draw(
    canvas: ti.ui.Canvas,
    positions: ti.template(),
    states: ti.template(),
    positions_to_draw: ti.template(),
    colors_to_draw: ti.template(),
):
    update_color_and_positions(positions, states, colors_to_draw, positions_to_draw)
    canvas.circles(positions_to_draw, radius=R0 / X, per_vertex_color=colors_to_draw)


opt = 0
current_settings = {
    "X": X,
    "Y": Y,
    "N": N,
    "R0": R0,
    "R1": R1,
    "LIMIT_PER_CELL": LIMIT_PER_CELL,
    "tau": tau,
    "opt": opt,
}


def draw_ui(gui: ti.ui.Gui):
    with gui.sub_window("Parameters", 0.1, 0.1, 0.3, 0.3) as w:
        current_settings["X"] = w.slider_float("X", current_settings["X"], 1000, 5000)
        current_settings["Y"] = w.slider_float("Y", current_settings["Y"], 1000, 5000)
        current_settings["N"] = w.slider_int("N", current_settings["N"], 500, 500_000)
        current_settings["R0"] = w.slider_float("R0", current_settings["R0"], 1.0, 10.0)
        current_settings["R1"] = w.slider_float(
            "R1", current_settings["R1"], 10.0, 50.0
        )
        current_settings["LIMIT_PER_CELL"] = w.slider_int(
            "LIMIT_PER_CELL", current_settings["LIMIT_PER_CELL"], 100, 500
        )
        current_settings["tau"] = w.slider_float(
            "tau", current_settings["tau"], 0.005, 0.1
        )
        current_settings["opt"] = w.slider_int("opt", current_settings["opt"], 0, 1)
        if w.button("Reset"):
            reset(**{"a" + k: v for k, v in current_settings.items()})
            initialize(positions, velocities, opt)


def reset(
    aX=1000.0,
    aY=1000.0,
    aN=1000,
    aR0=5.0,
    aR1=20.0,
    aLIMIT_PER_CELL=250,
    atau=0.01,
    aopt=0,
):
    global X, Y, N, R0, R1, LIMIT_PER_CELL, opt
    X, Y = aX, aY
    N = aN
    R0 = aR0
    R1 = aR1
    LIMIT_PER_CELL = aLIMIT_PER_CELL
    opt = aopt
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
    positions_to_draw = ti.Vector.field(2, dtype=ti.f32, shape=N)
    states_to_draw = ti.field(dtype=ti.i32, shape=N)
    colors_to_draw = ti.Vector.field(3, dtype=ti.f32, shape=N)


# Main simulation loop
def main():
    window = ti.ui.Window("Circles", res=(1000, 1000), fps_limit=60, vsync=True)
    canvas = window.get_canvas()
    gui = window.get_gui()
    accumulated_time = 0.0
    reset()
    initialize(
        positions,
        velocities,
        0,
    )
    while window.running:
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
                accumulated_time = 0.0
        draw(canvas, positions, states, positions_to_draw, colors_to_draw)
        draw_ui(gui)
        window.show()


if __name__ == "__main__":
    main()
