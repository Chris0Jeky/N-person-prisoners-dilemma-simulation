#!/usr/bin/env python3
"""
Enhanced Q-Learning Demo Generator using Unified Agent API
This version uses the simplified unified API to eliminate all agent type checking bugs.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from datetime import datetime
from itertools import combinations

# Import unified agents and simulation runner
from unified_agents import StaticAgent, EnhancedQLearningAgent
from unified_simulation_runner import (
    run_pairwise_simulation, 
    run_nperson_simulation,
    run_multiple_simulations
)

# --- Experiment Setup ---
def setup_2ql_experiments():
    """Setup 2 Enhanced QL vs various strategies experiments."""
    strategies = ["AllD", "AllC", "TFT", "TFT-E", "Random"]
    experiments = {}
    
    for strategy in strategies:
        exp_name = f"2 EQL + 1 {strategy}"
        agents = [
            EnhancedQLearningAgent(agent_id="EQL_1"),
            EnhancedQLearningAgent(agent_id="EQL_2"),
            StaticAgent(agent_id=f"{strategy}_1", strategy_name=strategy, 
                       exploration_rate=0.1 if strategy == "TFT-E" else 0.0)
        ]
        experiments[exp_name] = agents
    
    return experiments


def setup_1ql_experiments():
    """Setup 1 Enhanced QL vs all possible 2-agent combinations."""
    strategies = ["AllD", "AllC", "TFT", "TFT-E", "Random"]
    experiments = {}
    
    # 1QL vs homogeneous pairs
    for strategy in strategies:
        exp_name = f"1 EQL + 2 {strategy}"
        agents = [
            EnhancedQLearningAgent(agent_id="EQL_1"),
            StaticAgent(agent_id=f"{strategy}_1", strategy_name=strategy,
                       exploration_rate=0.1 if strategy == "TFT-E" else 0.0),
            StaticAgent(agent_id=f"{strategy}_2", strategy_name=strategy,
                       exploration_rate=0.1 if strategy == "TFT-E" else 0.0)
        ]
        experiments[exp_name] = agents
    
    # 1QL vs heterogeneous pairs
    for combo in combinations(strategies, 2):
        exp_name = f"1 EQL + 1 {combo[0]} + 1 {combo[1]}"
        agents = [
            EnhancedQLearningAgent(agent_id="EQL_1"),
            StaticAgent(agent_id=f"{combo[0]}_1", strategy_name=combo[0],
                       exploration_rate=0.1 if combo[0] == "TFT-E" else 0.0),
            StaticAgent(agent_id=f"{combo[1]}_1", strategy_name=combo[1],
                       exploration_rate=0.1 if combo[1] == "TFT-E" else 0.0)
        ]
        experiments[exp_name] = agents
    
    return experiments


def aggregate_agent_data(agent_runs):
    """Aggregate per-agent data from multiple runs."""
    aggregated = {}
    for agent_id, runs in agent_runs.items():
        runs_array = np.array(runs)
        mean_values = np.mean(runs_array, axis=0)
        std_values = np.std(runs_array, axis=0)
        
        n_runs = len(runs)
        sem = std_values / np.sqrt(n_runs)
        ci_95 = 1.96 * sem
        
        aggregated[agent_id] = {
            'mean': mean_values,
            'std': std_values,
            'lower_95': mean_values - ci_95,
            'upper_95': mean_values + ci_95,
            'all_runs': runs
        }
    
    return aggregated


# --- Plotting Functions ---
def save_aggregated_data_to_csv(data, exp_type, game_mode, results_dir):
    """Saves aggregated data to CSV files."""
    csv_dir = os.path.join(results_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    
    # Save individual agent data for each experiment
    for exp_name, exp_data in data.items():
        # Create DataFrame with all agent data
        dfs = []
        for agent_id, stats in exp_data.items():
            # Build all columns at once to avoid fragmentation
            columns = {
                'Round': range(1, len(stats['mean']) + 1),
                f'{agent_id}_mean': stats['mean'],
                f'{agent_id}_std': stats['std'],
                f'{agent_id}_lower_95': stats['lower_95'],
                f'{agent_id}_upper_95': stats['upper_95']
            }
            
            # Add individual runs to the columns dict
            for i, run in enumerate(stats['all_runs']):
                columns[f'{agent_id}_run_{i+1}'] = run
            
            # Create DataFrame with all columns at once
            df = pd.DataFrame(columns)
            dfs.append(df)
        
        # Merge all agent data
        combined_df = dfs[0]
        for df in dfs[1:]:
            combined_df = pd.merge(combined_df, df, on='Round')
        
        # Clean filename
        clean_name = exp_name.replace(' ', '_').replace('+', 'plus')
        filename = f"{exp_type}_{game_mode}_{clean_name}.csv"
        filepath = os.path.join(csv_dir, filename)
        combined_df.to_csv(filepath, index=False)
        print(f"  - Saved: {filename}")
    
    # Also save summary file
    summary_data = []
    for exp_name, exp_data in data.items():
        for agent_id, stats in exp_data.items():
            avg_coop = np.mean(stats['mean'])
            final_score = stats['mean'][-1] if 'score' in agent_id else avg_coop
            summary_data.append({
                'Experiment': exp_name,
                'Agent': agent_id,
                'Avg_Cooperation': avg_coop,
                'Final_Score': final_score
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_filename = f"{exp_type}_{game_mode}_summary.csv"
    summary_filepath = os.path.join(csv_dir, summary_filename)
    summary_df.to_csv(summary_filepath, index=False)
    print(f"  - Saved summary: {summary_filename}")


def plot_ql_cooperation(coop_data, title, exp_type, game_mode, save_path=None):
    """Plot cooperation rates for Q-learning experiments."""
    sns.set_style("whitegrid")
    
    # Determine subplot layout
    n_experiments = len(coop_data)
    n_cols = 3
    n_rows = (n_experiments + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows), constrained_layout=True)
    fig.suptitle(title + " - Cooperation Rates (15 run average)", fontsize=16, weight='bold')
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    for i, (exp_name, exp_data) in enumerate(coop_data.items()):
        ax = axes_flat[i]
        
        # Group by agent type
        agent_type_data = {}
        for agent_id, data in exp_data.items():
            agent_type = agent_id.split('_')[0]
            if agent_type not in agent_type_data:
                agent_type_data[agent_type] = []
            agent_type_data[agent_type].append(data)
        
        # Plot each agent type
        colors = {'EQL': 'darkblue', 'TFT': 'green', 'TFT-E': 'lightgreen', 
                 'AllC': 'orange', 'AllD': 'red', 'Random': 'purple'}
        
        for agent_type, data_list in agent_type_data.items():
            # Average across agents of same type
            all_means = [d['mean'] for d in data_list]
            avg_mean = np.mean(all_means, axis=0)
            rounds = range(1, len(avg_mean) + 1)
            
            # Calculate confidence interval
            if len(data_list) > 1:
                std_mean = np.std(all_means, axis=0)
                n_agents = len(data_list)
                sem = std_mean / np.sqrt(n_agents)
                ci_95 = 1.96 * sem
                ax.fill_between(rounds, avg_mean - ci_95, avg_mean + ci_95,
                              alpha=0.2, color=colors.get(agent_type, 'black'))
            
            ax.plot(rounds, avg_mean, color=colors.get(agent_type, 'black'),
                   linewidth=2.5, label=agent_type, marker='o', markersize=2)
            
            # Add smoothed line using rolling average
            smoothing_window = 20
            smoothed_mean = pd.Series(avg_mean).rolling(
                window=smoothing_window, min_periods=1, center=True).mean()
            ax.plot(rounds, smoothed_mean, color=colors.get(agent_type, 'black'),
                   linewidth=2, linestyle='--', alpha=0.7)
        
        ax.set_title(exp_name, fontsize=12)
        ax.set_xlabel("Round")
        ax.set_ylabel("Cooperation Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(coop_data), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved figure: {os.path.basename(save_path)}")
    
    plt.show()


def plot_ql_scores(score_data, title, exp_type, game_mode, save_path=None):
    """Plot cumulative scores for Q-learning experiments."""
    sns.set_style("whitegrid")
    
    # Determine subplot layout
    n_experiments = len(score_data)
    n_cols = 3
    n_rows = (n_experiments + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows), constrained_layout=True)
    fig.suptitle(title + " - Cumulative Scores (15 run average)", fontsize=16, weight='bold')
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    for i, (exp_name, exp_data) in enumerate(score_data.items()):
        ax = axes_flat[i]
        
        # Group by agent type
        agent_type_data = {}
        for agent_id, data in exp_data.items():
            agent_type = agent_id.split('_')[0]
            if agent_type not in agent_type_data:
                agent_type_data[agent_type] = []
            agent_type_data[agent_type].append(data)
        
        # Plot each agent type
        colors = {'EQL': 'darkblue', 'TFT': 'green', 'TFT-E': 'lightgreen', 
                 'AllC': 'orange', 'AllD': 'red', 'Random': 'purple'}
        
        for agent_type, data_list in agent_type_data.items():
            # Average across agents of same type
            all_means = [d['mean'] for d in data_list]
            avg_mean = np.mean(all_means, axis=0)
            rounds = range(1, len(avg_mean) + 1)
            
            ax.plot(rounds, avg_mean, color=colors.get(agent_type, 'black'),
                   linewidth=2.5, label=agent_type, marker='o', markersize=2)
            
            # Add smoothed line using rolling average
            smoothing_window = 20
            smoothed_mean = pd.Series(avg_mean).rolling(
                window=smoothing_window, min_periods=1, center=True).mean()
            ax.plot(rounds, smoothed_mean, color=colors.get(agent_type, 'black'),
                   linewidth=2, linestyle='--', alpha=0.7)
        
        ax.set_title(exp_name, fontsize=12)
        ax.set_xlabel("Round")
        ax.set_ylabel("Cumulative Score")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(score_data), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved figure: {os.path.basename(save_path)}")
    
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    NUM_ROUNDS = 1000
    NUM_RUNS = 500
    TRAINING_ROUNDS = 0
    
    # Create main results directory
    results_dir = "enhanced_qlearning_results_unified"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {os.path.abspath(results_dir)}")
    print(f"Running {NUM_RUNS} simulations per experiment with {TRAINING_ROUNDS} training rounds...")
    print("Using Enhanced Q-Learning with unified agent API")
    
    # --- 2 Enhanced QL Experiments ---
    print("\n=== Running 2 Enhanced QL vs Strategies Experiments ===")
    experiments_2ql = setup_2ql_experiments()
    
    # Create 2EQL results directory
    results_2ql_dir = os.path.join(results_dir, "2EQL_experiments")
    os.makedirs(results_2ql_dir, exist_ok=True)
    
    # Run pairwise simulations
    print("\nRunning 2 Enhanced QL Pairwise simulations...")
    pairwise_2ql_coop = {}
    pairwise_2ql_scores = {}
    
    for name, agent_list in experiments_2ql.items():
        print(f"  - Running {NUM_RUNS} simulations for: {name}")
        coop_runs, score_runs = run_multiple_simulations(
            run_pairwise_simulation, agent_list, NUM_ROUNDS, NUM_RUNS, TRAINING_ROUNDS)
        
        pairwise_2ql_coop[name] = aggregate_agent_data(coop_runs)
        pairwise_2ql_scores[name] = aggregate_agent_data(score_runs)
    
    # Run N-person simulations
    print("\nRunning 2 Enhanced QL Neighbourhood simulations...")
    nperson_2ql_coop = {}
    nperson_2ql_scores = {}
    
    for name, agent_list in experiments_2ql.items():
        print(f"  - Running {NUM_RUNS} simulations for: {name}")
        coop_runs, score_runs = run_multiple_simulations(
            run_nperson_simulation, agent_list, NUM_ROUNDS, NUM_RUNS, TRAINING_ROUNDS)
        
        nperson_2ql_coop[name] = aggregate_agent_data(coop_runs)
        nperson_2ql_scores[name] = aggregate_agent_data(score_runs)
    
    # Save 2EQL data
    print("\nSaving 2 Enhanced QL data...")
    save_aggregated_data_to_csv(pairwise_2ql_coop, "2EQL", "pairwise_cooperation", results_2ql_dir)
    save_aggregated_data_to_csv(pairwise_2ql_scores, "2EQL", "pairwise_scores", results_2ql_dir)
    save_aggregated_data_to_csv(nperson_2ql_coop, "2EQL", "nperson_cooperation", results_2ql_dir)
    save_aggregated_data_to_csv(nperson_2ql_scores, "2EQL", "nperson_scores", results_2ql_dir)
    
    # Create figures directory
    figures_2ql_dir = os.path.join(results_2ql_dir, "figures")
    os.makedirs(figures_2ql_dir, exist_ok=True)
    
    # Plot 2EQL results
    print("\nGenerating 2 Enhanced QL plots...")
    plot_ql_cooperation(pairwise_2ql_coop, "2 Enhanced QL Pairwise", "2EQL", "pairwise",
                       os.path.join(figures_2ql_dir, "2EQL_pairwise_cooperation.png"))
    plot_ql_scores(pairwise_2ql_scores, "2 Enhanced QL Pairwise", "2EQL", "pairwise",
                  os.path.join(figures_2ql_dir, "2EQL_pairwise_scores.png"))
    plot_ql_cooperation(nperson_2ql_coop, "2 Enhanced QL Neighbourhood", "2EQL", "nperson",
                       os.path.join(figures_2ql_dir, "2EQL_nperson_cooperation.png"))
    plot_ql_scores(nperson_2ql_scores, "2 Enhanced QL Neighbourhood", "2EQL", "nperson",
                  os.path.join(figures_2ql_dir, "2EQL_nperson_scores.png"))
    
    # --- 1 Enhanced QL Experiments ---
    print("\n=== Running 1 Enhanced QL vs All Combinations Experiments ===")
    experiments_1ql = setup_1ql_experiments()
    
    # Create 1EQL results directory
    results_1ql_dir = os.path.join(results_dir, "1EQL_experiments")
    os.makedirs(results_1ql_dir, exist_ok=True)
    
    # Run pairwise simulations
    print("\nRunning 1 Enhanced QL Pairwise simulations...")
    pairwise_1ql_coop = {}
    pairwise_1ql_scores = {}
    
    for name, agent_list in experiments_1ql.items():
        print(f"  - Running {NUM_RUNS} simulations for: {name}")
        coop_runs, score_runs = run_multiple_simulations(
            run_pairwise_simulation, agent_list, NUM_ROUNDS, NUM_RUNS, TRAINING_ROUNDS)
        
        pairwise_1ql_coop[name] = aggregate_agent_data(coop_runs)
        pairwise_1ql_scores[name] = aggregate_agent_data(score_runs)
    
    # Run N-person simulations
    print("\nRunning 1 Enhanced QL Neighbourhood simulations...")
    nperson_1ql_coop = {}
    nperson_1ql_scores = {}
    
    for name, agent_list in experiments_1ql.items():
        print(f"  - Running {NUM_RUNS} simulations for: {name}")
        coop_runs, score_runs = run_multiple_simulations(
            run_nperson_simulation, agent_list, NUM_ROUNDS, NUM_RUNS, TRAINING_ROUNDS)
        
        nperson_1ql_coop[name] = aggregate_agent_data(coop_runs)
        nperson_1ql_scores[name] = aggregate_agent_data(score_runs)
    
    # Save 1EQL data
    print("\nSaving 1 Enhanced QL data...")
    save_aggregated_data_to_csv(pairwise_1ql_coop, "1EQL", "pairwise_cooperation", results_1ql_dir)
    save_aggregated_data_to_csv(pairwise_1ql_scores, "1EQL", "pairwise_scores", results_1ql_dir)
    save_aggregated_data_to_csv(nperson_1ql_coop, "1EQL", "nperson_cooperation", results_1ql_dir)
    save_aggregated_data_to_csv(nperson_1ql_scores, "1EQL", "nperson_scores", results_1ql_dir)
    
    # Create figures directory
    figures_1ql_dir = os.path.join(results_1ql_dir, "figures")
    os.makedirs(figures_1ql_dir, exist_ok=True)
    
    # Plot 1EQL results
    print("\nGenerating 1 Enhanced QL plots...")
    plot_ql_cooperation(pairwise_1ql_coop, "1 Enhanced QL Pairwise", "1EQL", "pairwise",
                      os.path.join(figures_1ql_dir, "1EQL_pairwise_cooperation.png"))
    plot_ql_scores(pairwise_1ql_scores, "1 Enhanced QL Pairwise", "1EQL", "pairwise",
                 os.path.join(figures_1ql_dir, "1EQL_pairwise_scores.png"))
    plot_ql_cooperation(nperson_1ql_coop, "1 Enhanced QL Neighbourhood", "1EQL", "nperson",
                      os.path.join(figures_1ql_dir, "1EQL_nperson_cooperation.png"))
    plot_ql_scores(nperson_1ql_scores, "1 Enhanced QL Neighbourhood", "1EQL", "nperson",
                 os.path.join(figures_1ql_dir, "1EQL_nperson_scores.png"))
    
    print(f"\nDone! All Enhanced Q-learning results saved to '{results_dir}' directory.")
    print("\nFolder structure created:")
    print(f"  {results_dir}/")
    print(f"    2EQL_experiments/")
    print(f"      csv/          - CSV files with detailed data")
    print(f"      figures/      - Plots")
    print(f"    1EQL_experiments/")
    print(f"      csv/          - CSV files with detailed data")
    print(f"      figures/      - Plots")