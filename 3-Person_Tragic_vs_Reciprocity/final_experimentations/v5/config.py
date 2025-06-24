# config.py

"""
Centralized configuration for Q-learning agent parameters and simulation settings.
"""

# Simulation parameters
SIMULATION_CONFIG = {
    'num_rounds': 40000,  # Number of rounds per simulation
    'num_runs': 3,      # Number of runs to average over
}

# Parameters for a standard, non-adaptive Q-learning agent.
# Serves as the stable baseline for comparison.
VANILLA_PARAMS = {
    'lr': 0.1,  # Learning Rate (alpha)
    'df': 0.95,  # Discount Factor (gamma)
    'eps': 0.1,  # Epsilon (fixed exploration rate)
}

# Parameters for the new, truly adaptive agent.
ADAPTIVE_PARAMS = {
    # Initial values that will be adapted during the run
    'initial_lr': 0.1,
    'initial_eps': 0.15,

    # Bounds for the adaptive parameters
    'min_lr': 0.03,
    'max_lr': 0.15,
    'min_eps': 0.02,
    'max_eps': 0.15,

    # Controls how quickly the agent adapts. Higher value = slower change.
    'adaptation_factor': 1.08,

    # How many recent rounds to consider for performance trends
    'reward_window_size': 75,

    # Discount factor
    'df': 0.95,
}

# Parameters for Hysteretic Q-learning agent
# Uses different learning rates for positive and negative updates
HYSTERETIC_PARAMS = {
    'lr': 0.12,      # Learning rate for positive updates (good news)
    'beta': 0.002,   # Learning rate for negative updates (bad news)
    'df': 0.95,      # Discount factor
    'eps': 0.05,     # Exploration rate
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
    'epsilon': 0.23,              # Starting epsilon (higher for initial exploration)
    'epsilon_decay': 0.97,      # Epsilon decay rate per episode
    'epsilon_min': 0.005,         # Minimum epsilon
    'state_type': 'memory_enhanced',       # State representation type
    'memory_length': 100,         # Number of past actions to remember
    'exploration_rate': 0.0,     # Additional random exploration
}

# Legacy Q-Learning parameters (with sophisticated state representation)
LEGACY_PARAMS = {
    'lr': 0.12,                  # Learning rate
    'df': 0.98,                  # Discount factor (slightly lower for 2-round)
    'eps': 0.25,                 # Starting epsilon (higher for better exploration)
    'epsilon_decay': 0.998,      # Epsilon decay rate (slower decay)
    'epsilon_min': 0.02,         # Minimum epsilon (higher floor)
    'optimistic_init': 0.0       # Changed from 0.1 to 0.0 (pessimistic)
}

# Legacy 3-Round Q-Learning parameters (3-round history tracking)
LEGACY_3ROUND_PARAMS = {
    'lr': 0.15,                  # Higher learning rate for larger state space
    'df': 0.98,                  # High discount for long-term thinking
    'eps': 0.30,                 # Much more exploration needed
    'epsilon_decay': 0.9995,     # Very slow decay for larger state space
    'epsilon_min': 0.02,         # Higher minimum
    'optimistic_init': -0.1,     # Pessimistic initialization
    'history_length': 3          # Track 3 rounds instead of 2
}