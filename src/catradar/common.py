import taichi as ti

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


# States
STATE_MOVING = 0
STATE_INTERACT = 1
STATE_INTERSECTION = 2

# Shared memory - positions of circles
positions = ti.Vector.field(2, dtype=ti.f32, shape=N)
states = ti.field(dtype=ti.i32, shape=N)
