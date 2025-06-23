# config.py

"""
Centralized configuration for Q-learning agent parameters and simulation settings.
"""

# Simulation parameters
SIMULATION_CONFIG = {
    'num_rounds': 5000,  # Number of rounds per simulation
    'num_runs': 50,      # Number of runs to average over
}

# Parameters for a standard, non-adaptive Q-learning agent.
# Serves as the stable baseline for comparison.
VANILLA_PARAMS = {
    'lr': 0.1,  # Learning Rate (alpha)
    'df': 0.9,  # Discount Factor (gamma)
    'eps': 0.1,  # Epsilon (fixed exploration rate)
}

# Parameters for the new, truly adaptive agent.
ADAPTIVE_PARAMS = {
    # Initial values that will be adapted during the run
    'initial_lr': 0.3,
    'initial_eps': 0.25,

    # Bounds for the adaptive parameters
    'min_lr': 0.05,
    'max_lr': 0.5,
    'min_eps': 0.01,
    'max_eps': 0.4,

    # Controls how quickly the agent adapts. Higher value = slower change.
    'adaptation_factor': 1.05,

    # How many recent rounds to consider for performance trends
    'reward_window_size': 50,

    # Discount factor remains fixed
    'df': 0.9,
}