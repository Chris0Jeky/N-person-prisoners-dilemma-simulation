#!/usr/bin/env python3
"""
Final Q-Learning Demo Generator using the robust Unified API
Generates comprehensive results that can be compared with the comparison scripts
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from datetime import datetime

# Import the clean, unified agents and simulation runner
from final_agents import StaticAgent, SimpleQLearningAgent, EnhancedQLearningAgent
from final_simulation_runner import run_pairwise_simulation, run_nperson_simulation

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def setup_2ql_experiments(agent_type='enhanced'):
    """Setup 2 QL vs various strategies experiments."""
    strategies = ["AllD", "AllC", "TFT", "TFT-E", "Random"]
    experiments = {}
    
    ql_class = EnhancedQLearningAgent if agent_type == 'enhanced' else SimpleQLearningAgent
    ql_name = "EQL" if agent_type == 'enhanced' else "QL"
    
    for strategy in strategies:
        exp_name = f"2 {ql_name} + 1 {strategy}"
        agents = [
            ql_class(agent_id=f"{ql_name}_1"),
            ql_class(agent_id=f"{ql_name}_2"),
            StaticAgent(agent_id=f"{strategy}_1", strategy_name=strategy, 
                       exploration_rate=0.1 if strategy == "TFT-E" else 0.0)
        ]
        experiments[exp_name] = agents
    
    return experiments


def setup_1ql_experiments(agent_type='enhanced'):
    """Setup 1 QL vs all possible 2-agent combinations."""
    strategies = ["AllD", "AllC", "TFT", "TFT-E", "Random"]
    experiments = {}
    
    ql_class = EnhancedQLearningAgent if agent_type == 'enhanced' else SimpleQLearningAgent
    ql_name = "EQL" if agent_type == 'enhanced' else "QL"
    
    # 1QL vs homogeneous pairs
    for strategy in strategies:
        exp_name = f"1 {ql_name} + 2 {strategy}"
        agents = [
            ql_class(agent_id=f"{ql_name}_1"),
            StaticAgent(agent_id=f"{strategy}_1", strategy_name=strategy,
                       exploration_rate=0.1 if strategy == "TFT-E" else 0.0),
            StaticAgent(agent_id=f"{strategy}_2", strategy_name=strategy,
                       exploration_rate=0.1 if strategy == "TFT-E" else 0.0)
        ]
        experiments[exp_name] = agents
    
    # 1QL vs heterogeneous pairs
    for combo in combinations(strategies, 2):
        exp_name = f"1 {ql_name} + 1 {combo[0]} + 1 {combo[1]}"
        agents = [
            ql_class(agent_id=f"{ql_name}_1"),
            StaticAgent(agent_id=f"{combo[0]}_1", strategy_name=combo[0],
                       exploration_rate=0.1 if combo[0] == "TFT-E" else 0.0),
            StaticAgent(agent_id=f"{combo[1]}_1", strategy_name=combo[1],
                       exploration_rate=0.1 if combo[1] == "TFT-E" else 0.0)
        ]
        experiments[exp_name] = agents
    
    return experiments


def run_multiple_simulations(simulation_func, agent_templates, num_rounds, num_runs=100):
    """Run multiple simulations and aggregate results."""
    all_coop_runs = {agent.agent_id: [] for agent in agent_templates}
    all_score_runs = {agent.agent_id: [] for agent in agent_templates}
    
    for run in range(num_runs):
        # Create fresh agents from templates
        fresh_agents = []
        for agent in agent_templates:
            if isinstance(agent, EnhancedQLearningAgent):
                fresh_agents.append(EnhancedQLearningAgent(
                    agent_id=agent.agent_id,
                    lr=agent.lr,
                    df=agent.df,
                    eps=agent.initial_epsilon,
                    eps_decay=agent.epsilon_decay,
                    eps_min=agent.epsilon_min,
                    state_type=agent.state_type
                ))
            elif isinstance(agent, SimpleQLearningAgent):
                fresh_agents.append(SimpleQLearningAgent(
                    agent_id=agent.agent_id,
                    lr=agent.lr,
                    df=agent.df,
                    eps=agent.epsilon
                ))
            else:  # StaticAgent
                fresh_agents.append(StaticAgent(
                    agent_id=agent.agent_id,
                    strategy_name=agent.strategy_name,
                    exploration_rate=agent.exploration_rate
                ))
        
        coop_history, score_history = simulation_func(fresh_agents, num_rounds)
        
        for agent_id in coop_history:
            all_coop_runs[agent_id].append(coop_history[agent_id])
            all_score_runs[agent_id].append(score_history[agent_id])
    
    return all_coop_runs, all_score_runs


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


def save_aggregated_data_to_csv(data, exp_type, game_mode, results_dir):
    """Save aggregated data to CSV files."""
    csv_dir = os.path.join(results_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    
    # Save individual agent data for each experiment
    for exp_name, exp_data in data.items():
        # Create DataFrame with all agent data
        dfs = []
        for agent_id, stats in exp_data.items():
            # Build all columns at once
            columns = {
                'Round': range(1, len(stats['mean']) + 1),
                f'{agent_id}_mean': stats['mean'],
                f'{agent_id}_std': stats['std'],
                f'{agent_id}_lower_95': stats['lower_95'],
                f'{agent_id}_upper_95': stats['upper_95']
            }
            
            # Add individual runs
            for i, run in enumerate(stats['all_runs']):
                columns[f'{agent_id}_run_{i+1}'] = run
            
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
    
    # Save summary file
    summary_data = []
    for exp_name, exp_data in data.items():
        for agent_id, stats in exp_data.items():
            avg_coop = np.mean(stats['mean'])
            final_score = stats['mean'][-1] if 'score' in game_mode else avg_coop
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


def plot_cooperation_rates(coop_data, title, exp_type, game_mode, save_path=None):
    """Plot cooperation rates for experiments."""
    # Determine subplot layout
    n_experiments = len(coop_data)
    n_cols = 3
    n_rows = (n_experiments + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows), constrained_layout=True)
    fig.suptitle(title + " - Cooperation Rates", fontsize=16, weight='bold')
    
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
        colors = {'QL': 'blue', 'EQL': 'darkblue', 'TFT': 'green', 'TFT-E': 'lightgreen', 
                 'AllC': 'orange', 'AllD': 'red', 'Random': 'purple'}
        
        for agent_type, data_list in agent_type_data.items():
            # Average across agents of same type
            all_means = [d['mean'] for d in data_list]
            avg_mean = np.mean(all_means, axis=0)
            rounds = range(1, len(avg_mean) + 1)
            
            ax.plot(rounds, avg_mean, color=colors.get(agent_type, 'black'),
                   linewidth=2, label=agent_type)
        
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
    
    plt.close()


def plot_scores(score_data, title, exp_type, game_mode, save_path=None):
    """Plot cumulative scores for experiments."""
    # Similar layout to cooperation plots
    n_experiments = len(score_data)
    n_cols = 3
    n_rows = (n_experiments + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows), constrained_layout=True)
    fig.suptitle(title + " - Cumulative Scores", fontsize=16, weight='bold')
    
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
        colors = {'QL': 'blue', 'EQL': 'darkblue', 'TFT': 'green', 'TFT-E': 'lightgreen', 
                 'AllC': 'orange', 'AllD': 'red', 'Random': 'purple'}
        
        for agent_type, data_list in agent_type_data.items():
            # Average across agents of same type
            all_means = [d['mean'] for d in data_list]
            avg_mean = np.mean(all_means, axis=0)
            rounds = range(1, len(avg_mean) + 1)
            
            ax.plot(rounds, avg_mean, color=colors.get(agent_type, 'black'),
                   linewidth=2, label=agent_type)
        
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
    
    plt.close()


def main():
    """Main function to run all experiments."""
    NUM_ROUNDS = 1000
    NUM_RUNS = 500
    
    # Determine which agent type to use
    use_enhanced = True  # Set to False to run basic QL experiments
    agent_type = 'enhanced' if use_enhanced else 'simple'
    results_base_dir = 'final_enhanced_qlearning_results' if use_enhanced else 'final_qlearning_results'
    
    # Create main results directory
    os.makedirs(results_base_dir, exist_ok=True)
    print(f"Results will be saved to: {os.path.abspath(results_base_dir)}")
    print(f"Running {NUM_RUNS} simulations per experiment with {NUM_ROUNDS} rounds each...")
    print(f"Using {'Enhanced' if use_enhanced else 'Basic'} Q-Learning agents")
    
    # --- 2 QL Experiments ---
    print("\n=== Running 2 QL vs Strategies Experiments ===")
    experiments_2ql = setup_2ql_experiments(agent_type)
    
    # Create results directory
    ql_type = '2EQL' if use_enhanced else '2QL'
    results_2ql_dir = os.path.join(results_base_dir, f"{ql_type}_experiments")
    os.makedirs(results_2ql_dir, exist_ok=True)
    
    # Run pairwise simulations
    print(f"\nRunning {ql_type} Pairwise simulations...")
    pairwise_2ql_coop = {}
    pairwise_2ql_scores = {}
    
    for name, agent_list in experiments_2ql.items():
        print(f"  - Running {NUM_RUNS} simulations for: {name}")
        coop_runs, score_runs = run_multiple_simulations(
            run_pairwise_simulation, agent_list, NUM_ROUNDS, NUM_RUNS)
        
        pairwise_2ql_coop[name] = aggregate_agent_data(coop_runs)
        pairwise_2ql_scores[name] = aggregate_agent_data(score_runs)
    
    # Run N-person simulations
    print(f"\nRunning {ql_type} Neighbourhood simulations...")
    nperson_2ql_coop = {}
    nperson_2ql_scores = {}
    
    for name, agent_list in experiments_2ql.items():
        print(f"  - Running {NUM_RUNS} simulations for: {name}")
        coop_runs, score_runs = run_multiple_simulations(
            run_nperson_simulation, agent_list, NUM_ROUNDS, NUM_RUNS)
        
        nperson_2ql_coop[name] = aggregate_agent_data(coop_runs)
        nperson_2ql_scores[name] = aggregate_agent_data(score_runs)
    
    # Save data
    print(f"\nSaving {ql_type} data...")
    save_aggregated_data_to_csv(pairwise_2ql_coop, ql_type, "pairwise_cooperation", results_2ql_dir)
    save_aggregated_data_to_csv(pairwise_2ql_scores, ql_type, "pairwise_scores", results_2ql_dir)
    save_aggregated_data_to_csv(nperson_2ql_coop, ql_type, "nperson_cooperation", results_2ql_dir)
    save_aggregated_data_to_csv(nperson_2ql_scores, ql_type, "nperson_scores", results_2ql_dir)
    
    # Create figures directory
    figures_2ql_dir = os.path.join(results_2ql_dir, "figures")
    os.makedirs(figures_2ql_dir, exist_ok=True)
    
    # Plot results
    print(f"\nGenerating {ql_type} plots...")
    plot_cooperation_rates(pairwise_2ql_coop, f"{ql_type} Pairwise", ql_type, "pairwise",
                          os.path.join(figures_2ql_dir, f"{ql_type}_pairwise_cooperation.png"))
    plot_scores(pairwise_2ql_scores, f"{ql_type} Pairwise", ql_type, "pairwise",
               os.path.join(figures_2ql_dir, f"{ql_type}_pairwise_scores.png"))
    plot_cooperation_rates(nperson_2ql_coop, f"{ql_type} Neighbourhood", ql_type, "nperson",
                          os.path.join(figures_2ql_dir, f"{ql_type}_nperson_cooperation.png"))
    plot_scores(nperson_2ql_scores, f"{ql_type} Neighbourhood", ql_type, "nperson",
               os.path.join(figures_2ql_dir, f"{ql_type}_nperson_scores.png"))
    
    # --- 1 QL Experiments ---
    print("\n=== Running 1 QL vs All Combinations Experiments ===")
    experiments_1ql = setup_1ql_experiments(agent_type)
    
    # Create results directory
    ql_type = '1EQL' if use_enhanced else '1QL'
    results_1ql_dir = os.path.join(results_base_dir, f"{ql_type}_experiments")
    os.makedirs(results_1ql_dir, exist_ok=True)
    
    # Run pairwise simulations
    print(f"\nRunning {ql_type} Pairwise simulations...")
    pairwise_1ql_coop = {}
    pairwise_1ql_scores = {}
    
    for name, agent_list in experiments_1ql.items():
        print(f"  - Running {NUM_RUNS} simulations for: {name}")
        coop_runs, score_runs = run_multiple_simulations(
            run_pairwise_simulation, agent_list, NUM_ROUNDS, NUM_RUNS)
        
        pairwise_1ql_coop[name] = aggregate_agent_data(coop_runs)
        pairwise_1ql_scores[name] = aggregate_agent_data(score_runs)
    
    # Run N-person simulations
    print(f"\nRunning {ql_type} Neighbourhood simulations...")
    nperson_1ql_coop = {}
    nperson_1ql_scores = {}
    
    for name, agent_list in experiments_1ql.items():
        print(f"  - Running {NUM_RUNS} simulations for: {name}")
        coop_runs, score_runs = run_multiple_simulations(
            run_nperson_simulation, agent_list, NUM_ROUNDS, NUM_RUNS)
        
        nperson_1ql_coop[name] = aggregate_agent_data(coop_runs)
        nperson_1ql_scores[name] = aggregate_agent_data(score_runs)
    
    # Save data
    print(f"\nSaving {ql_type} data...")
    save_aggregated_data_to_csv(pairwise_1ql_coop, ql_type, "pairwise_cooperation", results_1ql_dir)
    save_aggregated_data_to_csv(pairwise_1ql_scores, ql_type, "pairwise_scores", results_1ql_dir)
    save_aggregated_data_to_csv(nperson_1ql_coop, ql_type, "nperson_cooperation", results_1ql_dir)
    save_aggregated_data_to_csv(nperson_1ql_scores, ql_type, "nperson_scores", results_1ql_dir)
    
    # Create figures directory
    figures_1ql_dir = os.path.join(results_1ql_dir, "figures")
    os.makedirs(figures_1ql_dir, exist_ok=True)
    
    # Plot results
    print(f"\nGenerating {ql_type} plots...")
    plot_cooperation_rates(pairwise_1ql_coop, f"{ql_type} Pairwise", ql_type, "pairwise",
                          os.path.join(figures_1ql_dir, f"{ql_type}_pairwise_cooperation.png"))
    plot_scores(pairwise_1ql_scores, f"{ql_type} Pairwise", ql_type, "pairwise",
               os.path.join(figures_1ql_dir, f"{ql_type}_pairwise_scores.png"))
    plot_cooperation_rates(nperson_1ql_coop, f"{ql_type} Neighbourhood", ql_type, "nperson",
                          os.path.join(figures_1ql_dir, f"{ql_type}_nperson_cooperation.png"))
    plot_scores(nperson_1ql_scores, f"{ql_type} Neighbourhood", ql_type, "nperson",
               os.path.join(figures_1ql_dir, f"{ql_type}_nperson_scores.png"))
    
    print(f"\nDone! All results saved to '{results_base_dir}' directory.")
    print("\nFolder structure created:")
    print(f"  {results_base_dir}/")
    print(f"    {ql_type}_experiments/")
    print(f"      csv/          - CSV files with detailed data")
    print(f"      figures/      - Plots")
    
    # Generate comparison info
    if use_enhanced:
        print("\nTo compare these results with basic Q-learning:")
        print("1. Run this script again with use_enhanced = False")
        print("2. Then run the comparison script")


if __name__ == "__main__":
    main()