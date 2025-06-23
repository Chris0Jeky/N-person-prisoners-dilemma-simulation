# config.py

"""
Centralized configuration for Q-learning agent parameters and simulation settings.
"""

# Simulation parameters
SIMULATION_CONFIG = {
    'num_rounds': 4000,  # Number of rounds per simulation
    'num_runs': 10,      # Number of runs to average over
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
    'initial_lr': 0.15,
    'initial_eps': 0.1,

    # Bounds for the adaptive parameters
    'min_lr': 0.05,
    'max_lr': 0.5,
    'min_eps': 0.01,
    'max_eps': 0.4,

    # Controls how quickly the agent adapts. Higher value = slower change.
    'adaptation_factor': 1.1,

    # How many recent rounds to consider for performance trends
    'reward_window_size': 50,

    # Discount factor remains fixed
    'df': 0.9,
}

# Parameters for Hysteretic Q-learning agent
# Uses different learning rates for positive and negative updates
HYSTERETIC_PARAMS = {
    'lr': 0.1,      # Learning rate for positive updates (good news)
    'beta': 0.01,   # Learning rate for negative updates (bad news)
    'df': 0.9,      # Discount factor
    'eps': 0.1,     # Exploration rate
}