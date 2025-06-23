# config.py

"""
Centralized configuration for Q-learning agent parameters and simulation settings.
"""

# Simulation parameters
SIMULATION_CONFIG = {
    'num_rounds': 10000,  # Number of rounds per simulation
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
    'initial_lr': 0.15,
    'initial_eps': 0.2,

    # Bounds for the adaptive parameters
    'min_lr': 0.05,
    'max_lr': 0.2,
    'min_eps': 0.01,
    'max_eps': 0.25,

    # Controls how quickly the agent adapts. Higher value = slower change.
    'adaptation_factor': 1.1,

    # How many recent rounds to consider for performance trends
    'reward_window_size': 500,

    # Discount factor
    'df': 0.9,
}

# Parameters for Hysteretic Q-learning agent
# Uses different learning rates for positive and negative updates
HYSTERETIC_PARAMS = {
    'lr': 0.1,      # Learning rate for positive updates (good news)
    'beta': 0.005,   # Learning rate for negative updates (bad news)
    'df': 0.9,      # Discount factor
    'eps': 0.08,     # Exploration rate
}

# Parameters for Modular Q-learning agents
MODULAR_BASE_PARAMS = {
    'lr': 0.1,      # Learning rate
    'df': 0.9,      # Discount factor
    'eps': 0.2,     # Epsilon for epsilon-greedy
}

# Softmax specific parameters
SOFTMAX_PARAMS = {
    'temperature': 2.0,          # Initial temperature
    'min_temperature': 0.01,      # Minimum temperature
    'decay_rate': 0.98,         # Temperature decay rate
    'lr': 0.1,                   # Learning rate
    'df': 0.9,                   # Discount factor
}

# Enhanced Q-Learning parameters (with epsilon decay)
ENHANCED_PARAMS = {
    'learning_rate': 0.1,        # Learning rate
    'discount_factor': 0.9,      # Discount factor  
    'epsilon': 0.2,              # Starting epsilon (higher for initial exploration)
    'epsilon_decay': 0.99,      # Epsilon decay rate per episode
    'epsilon_min': 0.01,         # Minimum epsilon
    'state_type': 'memory_enhanced',       # State representation type
    'memory_length': 50,         # Number of past actions to remember
    'exploration_rate': 0.0,     # Additional random exploration
}