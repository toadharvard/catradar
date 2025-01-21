# File represents constants used in different modules

# Functions that used to calculate distance between cats
EUCLIDEAN_NORM = 0
MANHATTAN_NORM = 1
MAX_NORM = 2

# Cats states
STATE_IDLE = 0
STATE_INTERACT = 1
STATE_INTERSECTION = 2
state_to_str = {
    STATE_IDLE: "IDLE",
    STATE_INTERACT: "INTERACT",
    STATE_INTERSECTION: "INTERSECTION",
}

# Modes of running program
STANDARD_MODE = 0
TESTING_MODE = 1

# Cats movement patterns
MOVE_PATTERN_FREE = 0
MOVE_PATTERN_CAROUSEL = 1
MOVE_PATTERN_COLLIDING = 2
