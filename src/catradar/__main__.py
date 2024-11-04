import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

# Parameters
X, Y = 1000.0, 1000.0  # Size of the map
N = 50000  # Number of circles
R0 = 5.0  # Intersection radius threshold
R1 = 20.0  # Interaction radius threshold (R1 > R0)
tau = 0.01  # Time interval for updating states
dt = 0.001  # Time step for simulation
num_substeps = int(tau / dt)  # Number of substeps before updating states

grid_cell_size = R1  # Size of each grid cell
grid_resolution_x = int(X / grid_cell_size) + 1
grid_resolution_y = int(Y / grid_cell_size) + 1

# States
STATE_MOVING = 0
STATE_INTERACT = 1
STATE_INTERSECTION = 2

# Shared memory - positions of circles
positions = ti.Vector.field(2, dtype=ti.f32, shape=N)
velocities = ti.Vector.field(2, dtype=ti.f32, shape=N)
states = ti.field(dtype=ti.i32, shape=N)

# Grid data structures for collision detection
grid_num_circles = ti.field(dtype=ti.i32, shape=(grid_resolution_x, grid_resolution_y))
grid_circles = ti.field(dtype=ti.i32, shape=(grid_resolution_x, grid_resolution_y, 100))

# For interaction with UI
positions_to_draw = ti.Vector.field(2, dtype=ti.f32, shape=N)
states_to_draw = ti.field(dtype=ti.i32, shape=N)


# Random initialization of positions and velocities
@ti.kernel
def initialize():
    for i in range(N):
        positions[i] = ti.Vector([ti.random() * X, ti.random() * Y])
        angle = ti.random() * 2 * 3.1415926
        speed = ti.random() * 50 + 50  # Random speed between 50 and 100
        velocities[i] = ti.Vector([ti.cos(angle), ti.sin(angle)]) * speed
        states[i] = STATE_MOVING


# First module: Updates positions and writes to shared memory
@ti.kernel
def update_positions():
    for i in range(N):
        velocities[i] += ti.Vector([0.0, 0.0]) * dt  # No external forces
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
def compute_states():
    # Clear grid
    for i, j in grid_num_circles:
        grid_num_circles[i, j] = 0

    # Insert circles into grid
    for idx in range(N):
        pos = positions[idx]
        gx = int(pos.x / grid_cell_size)
        gy = int(pos.y / grid_cell_size)
        num = ti.atomic_add(grid_num_circles[gx, gy], 1)
        if num < 100:
            grid_circles[gx, gy, num] = idx

    # Update states
    for idx in range(N):
        pos_i = positions[idx]
        gx = int(pos_i.x / grid_cell_size)
        gy = int(pos_i.y / grid_cell_size)
        state = STATE_MOVING  # Default state

        # Check neighboring grid cells
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


# Function to copy positions and states to fields used by UI
@ti.kernel
def copy_to_draw():
    for i in range(N):
        positions_to_draw[i] = positions[i]
        states_to_draw[i] = states[i]


# UI module: Draws circles with different colors based on state
def draw(gui):
    positions_np = positions_to_draw.to_numpy()
    positions_norm = positions_np / np.array([X, Y])
    state_np = states_to_draw.to_numpy()
    palette = [0x0000FF, 0x00FF00, 0xFF0000]  # Blue, Green, Red
    palette_indices = np.zeros(N, dtype=np.int32)
    palette_indices[state_np == STATE_MOVING] = 0
    palette_indices[state_np == STATE_INTERACT] = 1
    palette_indices[state_np == STATE_INTERSECTION] = 2
    radius = 2  # Radius in pixels
    gui.circles(
        positions_norm, radius=radius, palette=palette, palette_indices=palette_indices
    )


# Main simulation loop
def main():
    initialize()
    gui = ti.GUI("Circle Simulation", res=(800, 800))
    gui.fps_limit = 200
    accumulated_time = 0.0
    while gui.running:
        for _ in range(num_substeps):
            update_positions()
            accumulated_time += dt
            if accumulated_time >= tau:
                compute_states()
                accumulated_time = 0.0
        copy_to_draw()
        draw(gui)
        gui.show()


if __name__ == "__main__":
    main()
