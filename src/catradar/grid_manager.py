import taichi as ti
from catradar.common import *


# Size of each grid cell
grid_cell_size = R1
grid_resolution_x = int(X / grid_cell_size) + 1
grid_resolution_y = int(Y / grid_cell_size) + 1

# Grid data structures for collision detection
grid_num_circles = ti.field(dtype=ti.i32, shape=(grid_resolution_x, grid_resolution_y))
grid_circles = ti.field(
    dtype=ti.i32, shape=(grid_resolution_x, grid_resolution_y, LIMIT_PER_CELL)
)


# Second module: Reads positions and computes states
@ti.kernel
def compute_states():
    grid_num_circles.fill(0)

    # Insert circles into grid
    for idx in range(N):
        pos = positions[idx]
        gx = int(pos.x / grid_cell_size)
        gy = int(pos.y / grid_cell_size)
        num = ti.atomic_add(grid_num_circles[gx, gy], 1)
        if num < LIMIT_PER_CELL:
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
