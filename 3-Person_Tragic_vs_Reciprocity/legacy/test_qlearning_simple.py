#!/usr/bin/env python3
"""
Simple test of Q-learning demo generator with reduced parameters
"""

# Reduce parameters for testing
import sys
sys.path.insert(0, '.')

# Patch the module before importing
import qlearning_demo_generator as ql

# Override parameters for testing
ql.NUM_ROUNDS = 20
ql.NUM_RUNS = 2
ql.TRAINING_ROUNDS = 50

print("Running Q-learning test with reduced parameters...")
print(f"NUM_ROUNDS: {ql.NUM_ROUNDS}")
print(f"NUM_RUNS: {ql.NUM_RUNS}")
print(f"TRAINING_ROUNDS: {ql.TRAINING_ROUNDS}")

# Test basic functionality
print("\nTesting agent creation...")
test_agents = [
    ql.QLearningAgent("QL_1"),
    ql.StaticAgent("AllD_1", "AllD"),
    ql.StaticAgent("TFT_1", "TFT")
]

print("Agents created successfully!")

# Test a simple simulation
print("\nTesting simulation...")
coop_history, score_history = ql.run_pairwise_simulation_extended(test_agents, 10)
print(f"Simulation completed. Got {len(coop_history)} agent histories")

# Test aggregation
print("\nTesting data aggregation...")
test_runs = {
    'QL_1': [[0.5] * 10, [0.6] * 10],
    'AllD_1': [[0.0] * 10, [0.0] * 10],
    'TFT_1': [[0.8] * 10, [0.9] * 10]
}
aggregated = ql.aggregate_agent_data(test_runs)
print(f"Aggregation completed. Got {len(aggregated)} agent results")

print("\nBasic tests passed! The Q-learning generator should work.")