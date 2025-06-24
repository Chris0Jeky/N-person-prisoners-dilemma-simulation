"""
Configuration for Q-learning agent parameters and simulation settings.
"""

# Simulation parameters
SIMULATION_CONFIG = {
    'num_rounds': 20000,  # Number of rounds per simulation
    'num_runs': 200,      # Number of runs to average over
}

# Legacy Q-Learning parameters (with 2-round history tracking)
LEGACY_PARAMS = {
    'lr': 0.15,                  # Learning rate
    'df': 0.95,                  # Discount factor (will be overridden in sensitivity analysis)
    'eps': 0.3,                  # Starting epsilon
    'epsilon_decay': 0.995,      # Epsilon decay rate
    'epsilon_min': 0.05,         # Minimum epsilon
    'optimistic_init': 0.0       # Initial Q-value
}

# Legacy 3-Round Q-Learning parameters (3-round history tracking)
LEGACY_3ROUND_PARAMS = {
    'lr': 0.15,                  # Learning rate
    'df': 0.99,                  # Discount factor (will be overridden in sensitivity analysis)
    'eps': 0.25,                 # Starting epsilon
    'epsilon_decay': 0.998,      # Epsilon decay rate
    'epsilon_min': 0.01,         # Minimum epsilon
    'optimistic_init': -0.3,     # Initial Q-value
    'history_length': 3          # Track 3 rounds instead of 2
}