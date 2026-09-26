"""
Configuration for cooperation measurement experiments
Contains only the parameters needed for Legacy and Legacy3Round Q-learners
"""

# Simulation parameters
SIMULATION_CONFIG = {
    'num_rounds': 20000,  # Number of rounds per simulation
    'num_runs': 200,      # Number of runs to average over
}

# Legacy Q-Learning parameters (with sophisticated state representation)
LEGACY_PARAMS = {
    'lr': 0.15,                  # Learning rate
    'df': 0.95,                  # Discount factor (slightly lower for 2-round)
    'eps': 0.3,                  # Starting epsilon (higher for better exploration)
    'epsilon_decay': 0.995,      # Epsilon decay rate (slower decay)
    'epsilon_min': 0.05,         # Minimum epsilon (higher floor)
    'optimistic_init': 0.0       # Initial Q-values
}

# Legacy 3-Round Q-Learning parameters (3-round history tracking)
LEGACY_3ROUND_PARAMS = {
    'lr': 0.15,                  # Higher learning rate for larger state space
    'df': 0.99,                  # High discount for long-term thinking
    'eps': 0.25,                 # Much more exploration needed
    'epsilon_decay': 0.998,      # Very slow decay for larger state space
    'epsilon_min': 0.01,         # Higher minimum
    'optimistic_init': -0.3,     # Pessimistic initialization
    'history_length': 3          # Track 3 rounds instead of 2
}