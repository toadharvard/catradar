import taichi as ti
from catradar.common import *
from catradar.grid_manager import compute_states
from catradar.gui import draw
from catradar.position_updater import initialize_positions, update_positions


# Random initialization of positions and velocities
@ti.kernel
def initialize():
    initialize_positions()

    for i in range(N):
        states[i] = STATE_MOVING


# Main simulation loop
def main():
    initialize()
    window = ti.ui.Window("Circles", res=(1000, 1000), fps_limit=60, vsync=True)
    canvas = window.get_canvas()
    accumulated_time = 0.0
    while window.running:
        for _ in range(num_substeps):
            update_positions()
            accumulated_time += dt
            if accumulated_time >= tau:
                compute_states()
                accumulated_time = 0.0
        draw(canvas)
        window.show()


if __name__ == "__main__":
    main()
