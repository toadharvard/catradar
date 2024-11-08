from typing import Tuple
import taichi as ti
from catradar.common import (
    N,
    R0,
    R1,
    STATE_MOVING,
    X,
    Y,
    num_substeps,
    dt,
    tau,
)
from catradar.grid_manager import compute_states, initialize_grid
from catradar.gui import draw
from catradar.position_updater import initialize_positions, update_positions


def initialize_states(N: int):
    states = ti.field(dtype=ti.i32, shape=N)

    @ti.kernel
    def set_data():
        for i in range(N):
            states[i] = STATE_MOVING

    set_data()
    return states


# Random initialization of positions and velocities
def initialize(
    argX: float = X,
    argY: float = Y,
    argN: int = N,
    argR0: float = R0,
    argR1: float = R1,
) -> Tuple:
    global X, Y, N, R0, R1, positions, states
    X = argX
    Y = argY
    N = argN
    R0 = argR0
    R1 = argR1

    global positions, states
    positions = initialize_positions(N)
    states = initialize_states(N)
    initialize_grid(R1, X, Y)
    return positions, states


# Main simulation loop
def main():
    positions, states = initialize()
    window = ti.ui.Window("Circles", res=(1000, 1000), fps_limit=60, vsync=True)
    canvas = window.get_canvas()
    accumulated_time = 0.0
    while window.running:
        for _ in range(num_substeps):
            update_positions(positions)
            accumulated_time += dt
            if accumulated_time >= tau:
                compute_states(positions, states)
                accumulated_time = 0.0
        draw(canvas, positions, states)
        window.show()


if __name__ == "__main__":
    main()
