#!/usr/bin/env python3
"""
Recovery script to continue the multi-agent scaling experiments from where they crashed.
This script looks for existing results and continues from the next uncompleted scenario.
"""

import os
import glob
import re
from multi_agent_scaling_demo import *
from final_agents import LegacyQLearner
from config import LEGACY_PARAMS

def find_completed_scenarios(output_dir):
    """Find which scenarios have already been completed by looking at saved plots"""
    completed = set()
    
    # Check for individual scenario plots
    plot_dir = os.path.join(output_dir, "scenario_plots")
    if os.path.exists(plot_dir):
        for plot_file in glob.glob(os.path.join(plot_dir, "*.png")):
            # Extract scenario name from filename
            filename = os.path.basename(plot_file)
            scenario_name = filename.replace('.png', '')
            completed.add(scenario_name)
    
    # Also check the main directory for older format
    for plot_file in glob.glob(os.path.join(output_dir, "*agents_*.png")):
        filename = os.path.basename(plot_file)
        scenario_name = filename.replace('.png', '')
        if scenario_name != 'scaling_comparison':  # Skip the summary plot
            completed.add(scenario_name)
    
    return completed


def extract_results_from_completed(output_dir, scenario_name):
    """Try to extract results from CSV files if available"""
    # This is a placeholder - in practice, we'd need to save raw results
    # For now, we'll just skip completed scenarios
    return None


if __name__ == "__main__":
    NUM_ROUNDS = SIMULATION_CONFIG['num_rounds']
    NUM_RUNS = SIMULATION_CONFIG['num_runs']
    OUTPUT_DIR = "scaling_experiment_results"
    
    # Find what's already been completed
    completed_scenarios = find_completed_scenarios(OUTPUT_DIR)
    print(f"Found {len(completed_scenarios)} completed scenarios:")
    for scenario in sorted(completed_scenarios):
        print(f"  - {scenario}")
    
    # Group sizes to test
    GROUP_SIZES = [3, 5, 7, 10, 15, 20, 25]
    
    # Opponent types from v6
    OPPONENT_TYPES = {
        "AllC": {"strategy": "AllC", "error_rate": 0.0},
        "AllD": {"strategy": "AllD", "error_rate": 0.0},
        "Random": {"strategy": "Random", "error_rate": 0.0},
        "TFT": {"strategy": "TFT", "error_rate": 0.0},
    }
    
    USE_PARALLEL = True
    n_cores = cpu_count()
    n_processes = max(1, n_cores - 1) if USE_PARALLEL else 1
    
    print(f"\nResuming scaling experiments...")
    print(f"Using {NUM_ROUNDS} rounds and {NUM_RUNS} runs per scenario")
    if USE_PARALLEL:
        print(f"Using {n_processes} processes on {n_cores} available CPU cores")
    
    # Track all results
    all_scenario_results = {}
    scaling_results = {}
    
    total_start_time = time.time()
    scenarios_run = 0
    
    # --- Resume from Test 2: Mixed QL types ---
    print("\n=== Resuming mixed QL types with varying group sizes ===")
    
    for group_size in GROUP_SIZES:
        if group_size < 4:
            continue
        
        scenario_name = f"{group_size}agents_MixedQL"
        
        if scenario_name in completed_scenarios:
            print(f"\nScenario: {scenario_name} (already completed, skipping)")
            continue
        
        print(f"\nScenario: {scenario_name}")
        scenarios_run += 1
        
        # Create mixed QL agents
        agents = []
        n_legacy3 = group_size // 2
        n_nodecay = group_size - n_legacy3
        
        # Add Legacy3Round agents
        for i in range(n_legacy3):
            agents.append(Legacy3RoundQLearner(
                agent_id=f"Legacy3Round_QL_{i+1}",
                params=LEGACY_3ROUND_PARAMS
            ))
        
        # Add QLNoDecay agents
        for i in range(n_nodecay):
            agents.append(QLNoDecay(agent_id=f"QLNoDecay_{i+1}"))
        
        # Run experiment
        start_time = time.time()
        p_data, n_data = run_experiment_set(agents, NUM_ROUNDS, NUM_RUNS, USE_PARALLEL)
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.1f}s")
        
        all_scenario_results[scenario_name] = (p_data, n_data)
        plot_scenario_results((p_data, n_data), scenario_name, OUTPUT_DIR)
    
    # --- Test 3: All same QL type ---
    print("\n=== Resuming all same QL type with varying group sizes ===")
    
    for group_size in GROUP_SIZES:
        # All Legacy3Round
        scenario_name = f"{group_size}agents_AllLegacy3Round"
        
        if scenario_name not in completed_scenarios:
            print(f"\nScenario: {scenario_name}")
            scenarios_run += 1
            
            agents = []
            for i in range(group_size):
                agents.append(Legacy3RoundQLearner(
                    agent_id=f"Legacy3Round_QL_{i+1}",
                    params=LEGACY_3ROUND_PARAMS
                ))
            
            start_time = time.time()
            p_data, n_data = run_experiment_set(agents, NUM_ROUNDS, NUM_RUNS, USE_PARALLEL)
            elapsed = time.time() - start_time
            print(f"  Completed in {elapsed:.1f}s")
            
            all_scenario_results[scenario_name] = (p_data, n_data)
            plot_scenario_results((p_data, n_data), scenario_name, OUTPUT_DIR)
        else:
            print(f"\nScenario: {scenario_name} (already completed, skipping)")
        
        # All QLNoDecay
        scenario_name = f"{group_size}agents_AllQLNoDecay"
        
        if scenario_name not in completed_scenarios:
            print(f"\nScenario: {scenario_name}")
            scenarios_run += 1
            
            agents = []
            for i in range(group_size):
                agents.append(QLNoDecay(agent_id=f"QLNoDecay_{i+1}"))
            
            start_time = time.time()
            p_data, n_data = run_experiment_set(agents, NUM_ROUNDS, NUM_RUNS, USE_PARALLEL)
            elapsed = time.time() - start_time
            print(f"  Completed in {elapsed:.1f}s")
            
            all_scenario_results[scenario_name] = (p_data, n_data)
            plot_scenario_results((p_data, n_data), scenario_name, OUTPUT_DIR)
        else:
            print(f"\nScenario: {scenario_name} (already completed, skipping)")
    
    # Calculate total time
    total_elapsed = time.time() - total_start_time
    print(f"\nRecovery complete! Ran {scenarios_run} additional scenarios")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    
    if scenarios_run > 0:
        print("\nNote: To generate the final scaling comparison plot and CSV,")
        print("you'll need to run the fixed script from the beginning with all data.")