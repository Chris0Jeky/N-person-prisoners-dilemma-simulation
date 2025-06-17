#!/usr/bin/env python3
"""
Run Q-learning demo with customizable parameters
This script runs the Q-learning experiments and generates CSV results
"""

import sys
import os

# Import the Q-learning generator
sys.path.insert(0, '.')
import qlearning_demo_generator as ql

# Set parameters (can be modified)
NUM_ROUNDS = 500
NUM_RUNS = 100
TRAINING_ROUNDS = 0

print("=== Q-Learning Demo Generator ===")
print(f"Configuration:")
print(f"  NUM_ROUNDS: {NUM_ROUNDS}")
print(f"  NUM_RUNS: {NUM_RUNS}")
print(f"  TRAINING_ROUNDS: {TRAINING_ROUNDS}")
print()

# Override module parameters
ql.NUM_ROUNDS = NUM_ROUNDS
ql.NUM_RUNS = NUM_RUNS
ql.TRAINING_ROUNDS = TRAINING_ROUNDS

# Create main results directory
results_dir = "qlearning_results"
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {os.path.abspath(results_dir)}")
print(f"Running {NUM_RUNS} simulations per experiment with {TRAINING_ROUNDS} training rounds...")

# --- 2QL Experiments ---
print("\n=== Running 2QL vs Strategies Experiments ===")
experiments_2ql = ql.setup_2ql_experiments()

# Create 2QL results directory
results_2ql_dir = os.path.join(results_dir, "2QL_experiments")
os.makedirs(results_2ql_dir, exist_ok=True)

# Run pairwise simulations
print("\nRunning 2QL Pairwise simulations...")
pairwise_2ql_coop = {}
pairwise_2ql_scores = {}

for name, agent_list in experiments_2ql.items():
    print(f"  - Running {NUM_RUNS} simulations for: {name}")
    coop_runs, score_runs = ql.run_multiple_simulations_extended(
        ql.run_pairwise_simulation_extended, agent_list, NUM_ROUNDS, NUM_RUNS, TRAINING_ROUNDS)
    
    pairwise_2ql_coop[name] = ql.aggregate_agent_data(coop_runs)
    pairwise_2ql_scores[name] = ql.aggregate_agent_data(score_runs)

# Run N-person simulations
print("\nRunning 2QL Neighbourhood simulations...")
nperson_2ql_coop = {}
nperson_2ql_scores = {}

for name, agent_list in experiments_2ql.items():
    print(f"  - Running {NUM_RUNS} simulations for: {name}")
    coop_runs, score_runs = ql.run_multiple_simulations_extended(
        ql.run_nperson_simulation_extended, agent_list, NUM_ROUNDS, NUM_RUNS, TRAINING_ROUNDS)
    
    nperson_2ql_coop[name] = ql.aggregate_agent_data(coop_runs)
    nperson_2ql_scores[name] = ql.aggregate_agent_data(score_runs)

# Save 2QL data
print("\nSaving 2QL data...")
ql.save_aggregated_data_to_csv(pairwise_2ql_coop, "2QL", "pairwise_cooperation", results_2ql_dir)
ql.save_aggregated_data_to_csv(pairwise_2ql_scores, "2QL", "pairwise_scores", results_2ql_dir)
ql.save_aggregated_data_to_csv(nperson_2ql_coop, "2QL", "nperson_cooperation", results_2ql_dir)
ql.save_aggregated_data_to_csv(nperson_2ql_scores, "2QL", "nperson_scores", results_2ql_dir)

# Create figures directory
figures_2ql_dir = os.path.join(results_2ql_dir, "figures")
os.makedirs(figures_2ql_dir, exist_ok=True)

# Plot 2QL results (if matplotlib available)
print("\nGenerating 2QL plots...")
ql.plot_ql_cooperation(pairwise_2ql_coop, "2QL Pairwise", "2QL", "pairwise",
                      os.path.join(figures_2ql_dir, "2QL_pairwise_cooperation.png"))
ql.plot_ql_scores(pairwise_2ql_scores, "2QL Pairwise", "2QL", "pairwise",
                 os.path.join(figures_2ql_dir, "2QL_pairwise_scores.png"))
ql.plot_ql_cooperation(nperson_2ql_coop, "2QL Neighbourhood", "2QL", "nperson",
                      os.path.join(figures_2ql_dir, "2QL_nperson_cooperation.png"))
ql.plot_ql_scores(nperson_2ql_scores, "2QL Neighbourhood", "2QL", "nperson",
                 os.path.join(figures_2ql_dir, "2QL_nperson_scores.png"))

# --- 1QL Experiments ---
print("\n=== Running 1QL vs All Combinations Experiments ===")
experiments_1ql = ql.setup_1ql_experiments()

# Create 1QL results directory
results_1ql_dir = os.path.join(results_dir, "1QL_experiments")
os.makedirs(results_1ql_dir, exist_ok=True)

# Run pairwise simulations
print("\nRunning 1QL Pairwise simulations...")
pairwise_1ql_coop = {}
pairwise_1ql_scores = {}

for name, agent_list in experiments_1ql.items():
    print(f"  - Running {NUM_RUNS} simulations for: {name}")
    coop_runs, score_runs = ql.run_multiple_simulations_extended(
        ql.run_pairwise_simulation_extended, agent_list, NUM_ROUNDS, NUM_RUNS, TRAINING_ROUNDS)
    
    pairwise_1ql_coop[name] = ql.aggregate_agent_data(coop_runs)
    pairwise_1ql_scores[name] = ql.aggregate_agent_data(score_runs)

# Run N-person simulations
print("\nRunning 1QL Neighbourhood simulations...")
nperson_1ql_coop = {}
nperson_1ql_scores = {}

for name, agent_list in experiments_1ql.items():
    print(f"  - Running {NUM_RUNS} simulations for: {name}")
    coop_runs, score_runs = ql.run_multiple_simulations_extended(
        ql.run_nperson_simulation_extended, agent_list, NUM_ROUNDS, NUM_RUNS, TRAINING_ROUNDS)
    
    nperson_1ql_coop[name] = ql.aggregate_agent_data(coop_runs)
    nperson_1ql_scores[name] = ql.aggregate_agent_data(score_runs)

# Save 1QL data
print("\nSaving 1QL data...")
ql.save_aggregated_data_to_csv(pairwise_1ql_coop, "1QL", "pairwise_cooperation", results_1ql_dir)
ql.save_aggregated_data_to_csv(pairwise_1ql_scores, "1QL", "pairwise_scores", results_1ql_dir)
ql.save_aggregated_data_to_csv(nperson_1ql_coop, "1QL", "nperson_cooperation", results_1ql_dir)
ql.save_aggregated_data_to_csv(nperson_1ql_scores, "1QL", "nperson_scores", results_1ql_dir)

# Create figures directory
figures_1ql_dir = os.path.join(results_1ql_dir, "figures")
os.makedirs(figures_1ql_dir, exist_ok=True)

# Plot 1QL results (if matplotlib available)
print("\nGenerating 1QL plots...")
ql.plot_ql_cooperation(pairwise_1ql_coop, "1QL Pairwise", "1QL", "pairwise",
                      os.path.join(figures_1ql_dir, "1QL_pairwise_cooperation.png"))
ql.plot_ql_scores(pairwise_1ql_scores, "1QL Pairwise", "1QL", "pairwise",
                 os.path.join(figures_1ql_dir, "1QL_pairwise_scores.png"))
ql.plot_ql_cooperation(nperson_1ql_coop, "1QL Neighbourhood", "1QL", "nperson",
                      os.path.join(figures_1ql_dir, "1QL_nperson_cooperation.png"))
ql.plot_ql_scores(nperson_1ql_scores, "1QL Neighbourhood", "1QL", "nperson",
                 os.path.join(figures_1ql_dir, "1QL_nperson_scores.png"))

print(f"\nDone! All Q-learning results saved to '{results_dir}' directory.")
print("\nFolder structure created:")
print(f"  {results_dir}/")
print(f"    2QL_experiments/")
print(f"      csv/          - CSV files with detailed data")
print(f"      figures/      - Plots (if matplotlib available)")
print(f"    1QL_experiments/")
print(f"      csv/          - CSV files with detailed data")
print(f"      figures/      - Plots (if matplotlib available)")