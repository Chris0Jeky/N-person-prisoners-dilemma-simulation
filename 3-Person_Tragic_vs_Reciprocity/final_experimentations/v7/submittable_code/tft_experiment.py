#!/usr/bin/env python3
"""
Comprehensive Q-Learning vs TFT Experiment
Tests Vanilla and Adaptive Q-learners against TFT agents with different discount factors

Scenarios:
- 1 QL vs 2 TFT
- 1 QL vs 2 TFT-E (10% error)
- 2 QL vs 1 TFT
- 2 QL vs 1 TFT-E (10% error)

Each scenario is tested with:
- Vanilla Q-learning (df=0.6 and df=0.95)
- Adaptive Q-learning (df=0.6 and df=0.95)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
from collections import defaultdict
import time
import json

# Add parent directory to path to import modules
sys.path.append('..')

from final_agents import StaticAgent, PairwiseAdaptiveQLearner
from final_simulation import run_pairwise_tournament, run_nperson_simulation

# Constants
PAYOFFS_2P = {(0, 0): (3, 3), (0, 1): (0, 5), (1, 0): (5, 0), (1, 1): (1, 1)}
T, R, P, S = 5, 3, 1, 0

# Simulation parameters
NUM_ROUNDS = 10000
NUM_RUNS = 25
OUTPUT_DIR = "tft_experiment_results"

# Q-learner configurations with different discount factors
VANILLA_DF_06 = {
    'lr': 0.08,
    'df': 0.6,
    'eps': 0.1,
}

VANILLA_DF_095 = {
    'lr': 0.08,
    'df': 0.95,
    'eps': 0.1,
}

ADAPTIVE_DF_06 = {
    'initial_lr': 0.1,
    'initial_eps': 0.15,
    'min_lr': 0.03,
    'max_lr': 0.15,
    'min_eps': 0.02,
    'max_eps': 0.15,
    'adaptation_factor': 1.08,
    'reward_window_size': 75,
    'df': 0.6,
}

ADAPTIVE_DF_095 = {
    'initial_lr': 0.1,
    'initial_eps': 0.15,
    'min_lr': 0.03,
    'max_lr': 0.15,
    'min_eps': 0.02,
    'max_eps': 0.15,
    'adaptation_factor': 1.08,
    'reward_window_size': 75,
    'df': 0.95,
}

def run_experiment_set(agents, num_rounds, num_runs):
    """Runs both pairwise and neighborhood simulations for a given agent configuration."""
    p_runs = []
    n_runs = []
    
    for run in range(num_runs):
        # Create fresh copies of agents for each run
        fresh_agents = []
        for agent in agents:
            if isinstance(agent, PairwiseAdaptiveQLearner):
                fresh_agents.append(PairwiseAdaptiveQLearner(agent.agent_id, agent.params))
            else:
                fresh_agents.append(StaticAgent(agent.agent_id, agent.strategy_name, agent.error_rate))
        
        p_runs.append(run_pairwise_tournament(fresh_agents, num_rounds))
        n_runs.append(run_nperson_simulation(fresh_agents, num_rounds))
    
    # Aggregate results
    p_agg = {aid: {m: np.mean([r[aid][m] for r in p_runs if aid in r], axis=0) for m in ['coop_rate', 'score']} 
             for aid in p_runs[0]}
    n_agg = {aid: {m: np.mean([r[aid][m] for r in n_runs if aid in r], axis=0) for m in ['coop_rate', 'score']} 
             for aid in n_runs[0]}
    
    return p_agg, n_agg

def smooth_data(data, window_size=50):
    """Apply rolling average to smooth data"""
    if len(data) < window_size:
        window_size = max(1, len(data) // 10)
    
    # Manual rolling average
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        smoothed.append(np.mean(data[start:end]))
    
    return np.array(smoothed)

def save_results_csv(results, scenario_name):
    """Save results to CSV files"""
    csv_dir = os.path.join(OUTPUT_DIR, "csv_files")
    os.makedirs(csv_dir, exist_ok=True)
    
    # Prepare data for CSV
    data_rows = []
    
    for agent_id, metrics in results.items():
        # Average cooperation rate over all rounds
        avg_coop = np.mean(metrics['coop_rate'])
        final_score = metrics['score'][-1] if len(metrics['score']) > 0 else 0
        
        data_rows.append({
            'Agent': agent_id,
            'Average_Cooperation_Rate': avg_coop,
            'Final_Score': final_score,
            'Agent_Type': agent_id.split('_')[0],
            'Scenario': scenario_name
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(data_rows)
    csv_path = os.path.join(csv_dir, f"{scenario_name}.csv")
    df.to_csv(csv_path, index=False)
    
    return df

def plot_scenario(p_data, n_data, scenario_name, save_path):
    """Create a 4-panel plot for a scenario"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Scenario: {scenario_name}", fontsize=16)
    
    # Define colors for different agent types
    colors = {
        'Vanilla': '#1f77b4',
        'Adaptive': '#ff7f0e',
        'TFT': '#2ca02c',
        'TFT-E': '#d62728',
    }
    
    # Plot each agent's performance
    for agent_id in p_data.keys():
        # Determine agent type and color
        if 'Vanilla' in agent_id:
            color = colors['Vanilla']
            label = agent_id
        elif 'Adaptive' in agent_id:
            color = colors['Adaptive']
            label = agent_id
        elif 'TFT-E' in agent_id:
            color = colors['TFT-E']
            label = agent_id
        elif 'TFT' in agent_id:
            color = colors['TFT']
            label = agent_id
        else:
            color = 'gray'
            label = agent_id
        
        # Smooth cooperation rates
        p_coop_smooth = smooth_data(p_data[agent_id]['coop_rate'], 100)
        n_coop_smooth = smooth_data(n_data[agent_id]['coop_rate'], 100)
        
        # Plot
        axes[0, 0].plot(p_coop_smooth, label=label, color=color, linewidth=2)
        axes[0, 1].plot(p_data[agent_id]['score'], label=label, color=color, linewidth=2)
        axes[1, 0].plot(n_coop_smooth, label=label, color=color, linewidth=2)
        axes[1, 1].plot(n_data[agent_id]['score'], label=label, color=color, linewidth=2)
    
    # Configure axes
    axes[0, 0].set_title('Pairwise Cooperation Rate')
    axes[0, 0].set_ylabel('Cooperation Rate')
    axes[0, 0].set_ylim(-0.05, 1.05)
    
    axes[0, 1].set_title('Pairwise Cumulative Score')
    axes[0, 1].set_ylabel('Cumulative Score')
    
    axes[1, 0].set_title('Neighborhood Cooperation Rate')
    axes[1, 0].set_ylabel('Cooperation Rate')
    axes[1, 0].set_ylim(-0.05, 1.05)
    
    axes[1, 1].set_title('Neighborhood Cumulative Score')
    axes[1, 1].set_ylabel('Cumulative Score')
    
    for ax in axes.flat:
        ax.set_xlabel('Round')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_comparison_heatmap(all_results, save_path):
    """Create a heatmap comparing all scenarios"""
    scenarios = []
    agent_types = []
    avg_coop_pairwise = []
    avg_coop_neighbor = []
    final_score_pairwise = []
    final_score_neighbor = []
    
    for scenario_name, (p_data, n_data) in all_results.items():
        for agent_id in p_data.keys():
            if 'QL' in agent_id:  # Only include Q-learners
                scenarios.append(scenario_name)
                agent_types.append(agent_id)
                avg_coop_pairwise.append(np.mean(p_data[agent_id]['coop_rate']))
                avg_coop_neighbor.append(np.mean(n_data[agent_id]['coop_rate']))
                final_score_pairwise.append(p_data[agent_id]['score'][-1])
                final_score_neighbor.append(n_data[agent_id]['score'][-1])
    
    # Create DataFrame
    df = pd.DataFrame({
        'Scenario': scenarios,
        'Agent': agent_types,
        'Pairwise_Coop': avg_coop_pairwise,
        'Neighbor_Coop': avg_coop_neighbor,
        'Pairwise_Score': final_score_pairwise,
        'Neighbor_Score': final_score_neighbor
    })
    
    # Create pivot table for heatmap
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Comparison Across All Scenarios', fontsize=16)
    
    # Cooperation rates
    pivot_coop_p = df.pivot_table(values='Pairwise_Coop', index='Agent', columns='Scenario')
    pivot_coop_n = df.pivot_table(values='Neighbor_Coop', index='Agent', columns='Scenario')
    
    sns.heatmap(pivot_coop_p, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[0, 0], vmin=0, vmax=1)
    axes[0, 0].set_title('Pairwise Cooperation Rates')
    
    sns.heatmap(pivot_coop_n, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[0, 1], vmin=0, vmax=1)
    axes[0, 1].set_title('Neighborhood Cooperation Rates')
    
    # Scores (normalized by max possible score)
    max_score = R * NUM_ROUNDS * 2  # Maximum possible score in 3-agent scenario
    pivot_score_p = df.pivot_table(values='Pairwise_Score', index='Agent', columns='Scenario') / max_score
    pivot_score_n = df.pivot_table(values='Neighbor_Score', index='Agent', columns='Scenario') / max_score
    
    sns.heatmap(pivot_score_p, annot=True, fmt='.3f', cmap='RdYlBu', ax=axes[1, 0])
    axes[1, 0].set_title('Pairwise Normalized Scores')
    
    sns.heatmap(pivot_score_n, annot=True, fmt='.3f', cmap='RdYlBu', ax=axes[1, 1])
    axes[1, 1].set_title('Neighborhood Normalized Scores')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_summary_stats(all_results):
    """Save summary statistics to a JSON file"""
    summary = {}
    
    for scenario_name, (p_data, n_data) in all_results.items():
        scenario_summary = {}
        
        for agent_id in p_data.keys():
            agent_summary = {
                'pairwise': {
                    'avg_cooperation': float(np.mean(p_data[agent_id]['coop_rate'])),
                    'final_score': float(p_data[agent_id]['score'][-1]),
                    'score_per_round': float(p_data[agent_id]['score'][-1] / NUM_ROUNDS)
                },
                'neighborhood': {
                    'avg_cooperation': float(np.mean(n_data[agent_id]['coop_rate'])),
                    'final_score': float(n_data[agent_id]['score'][-1]),
                    'score_per_round': float(n_data[agent_id]['score'][-1] / NUM_ROUNDS)
                }
            }
            scenario_summary[agent_id] = agent_summary
        
        summary[scenario_name] = scenario_summary
    
    with open(os.path.join(OUTPUT_DIR, 'summary_statistics.json'), 'w') as f:
        json.dump(summary, f, indent=2)

def main():
    """Run all experiments"""
    print("=" * 80)
    print("Q-LEARNING VS TFT COMPREHENSIVE EXPERIMENT")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    figures_dir = os.path.join(OUTPUT_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Define all scenarios
    scenarios = {
        # 1 QL vs 2 TFT scenarios
        "1_Vanilla_DF06_vs_2_TFT": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Vanilla_QL_DF06", VANILLA_DF_06),
                StaticAgent("TFT_1", "TFT", 0.0),
                StaticAgent("TFT_2", "TFT", 0.0)
            ]
        },
        "1_Vanilla_DF095_vs_2_TFT": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Vanilla_QL_DF095", VANILLA_DF_095),
                StaticAgent("TFT_1", "TFT", 0.0),
                StaticAgent("TFT_2", "TFT", 0.0)
            ]
        },
        "1_Adaptive_DF06_vs_2_TFT": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Adaptive_QL_DF06", ADAPTIVE_DF_06),
                StaticAgent("TFT_1", "TFT", 0.0),
                StaticAgent("TFT_2", "TFT", 0.0)
            ]
        },
        "1_Adaptive_DF095_vs_2_TFT": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Adaptive_QL_DF095", ADAPTIVE_DF_095),
                StaticAgent("TFT_1", "TFT", 0.0),
                StaticAgent("TFT_2", "TFT", 0.0)
            ]
        },
        
        # 1 QL vs 2 TFT-E scenarios
        "1_Vanilla_DF06_vs_2_TFT-E": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Vanilla_QL_DF06", VANILLA_DF_06),
                StaticAgent("TFT-E_1", "TFT-E", 0.1),
                StaticAgent("TFT-E_2", "TFT-E", 0.1)
            ]
        },
        "1_Vanilla_DF095_vs_2_TFT-E": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Vanilla_QL_DF095", VANILLA_DF_095),
                StaticAgent("TFT-E_1", "TFT-E", 0.1),
                StaticAgent("TFT-E_2", "TFT-E", 0.1)
            ]
        },
        "1_Adaptive_DF06_vs_2_TFT-E": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Adaptive_QL_DF06", ADAPTIVE_DF_06),
                StaticAgent("TFT-E_1", "TFT-E", 0.1),
                StaticAgent("TFT-E_2", "TFT-E", 0.1)
            ]
        },
        "1_Adaptive_DF095_vs_2_TFT-E": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Adaptive_QL_DF095", ADAPTIVE_DF_095),
                StaticAgent("TFT-E_1", "TFT-E", 0.1),
                StaticAgent("TFT-E_2", "TFT-E", 0.1)
            ]
        },
        
        # 2 QL vs 1 TFT scenarios
        "2_Vanilla_DF06_vs_1_TFT": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Vanilla_QL1_DF06", VANILLA_DF_06),
                PairwiseAdaptiveQLearner("Vanilla_QL2_DF06", VANILLA_DF_06),
                StaticAgent("TFT", "TFT", 0.0)
            ]
        },
        "2_Vanilla_DF095_vs_1_TFT": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Vanilla_QL1_DF095", VANILLA_DF_095),
                PairwiseAdaptiveQLearner("Vanilla_QL2_DF095", VANILLA_DF_095),
                StaticAgent("TFT", "TFT", 0.0)
            ]
        },
        "2_Adaptive_DF06_vs_1_TFT": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Adaptive_QL1_DF06", ADAPTIVE_DF_06),
                PairwiseAdaptiveQLearner("Adaptive_QL2_DF06", ADAPTIVE_DF_06),
                StaticAgent("TFT", "TFT", 0.0)
            ]
        },
        "2_Adaptive_DF095_vs_1_TFT": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Adaptive_QL1_DF095", ADAPTIVE_DF_095),
                PairwiseAdaptiveQLearner("Adaptive_QL2_DF095", ADAPTIVE_DF_095),
                StaticAgent("TFT", "TFT", 0.0)
            ]
        },
        
        # 2 QL vs 1 TFT-E scenarios
        "2_Vanilla_DF06_vs_1_TFT-E": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Vanilla_QL1_DF06", VANILLA_DF_06),
                PairwiseAdaptiveQLearner("Vanilla_QL2_DF06", VANILLA_DF_06),
                StaticAgent("TFT-E", "TFT-E", 0.1)
            ]
        },
        "2_Vanilla_DF095_vs_1_TFT-E": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Vanilla_QL1_DF095", VANILLA_DF_095),
                PairwiseAdaptiveQLearner("Vanilla_QL2_DF095", VANILLA_DF_095),
                StaticAgent("TFT-E", "TFT-E", 0.1)
            ]
        },
        "2_Adaptive_DF06_vs_1_TFT-E": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Adaptive_QL1_DF06", ADAPTIVE_DF_06),
                PairwiseAdaptiveQLearner("Adaptive_QL2_DF06", ADAPTIVE_DF_06),
                StaticAgent("TFT-E", "TFT-E", 0.1)
            ]
        },
        "2_Adaptive_DF095_vs_1_TFT-E": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Adaptive_QL1_DF095", ADAPTIVE_DF_095),
                PairwiseAdaptiveQLearner("Adaptive_QL2_DF095", ADAPTIVE_DF_095),
                StaticAgent("TFT-E", "TFT-E", 0.1)
            ]
        }
    }
    
    all_results = {}
    total_scenarios = len(scenarios)
    
    # Run all scenarios
    for i, (scenario_name, scenario_config) in enumerate(scenarios.items(), 1):
        print(f"\n[{i}/{total_scenarios}] Running scenario: {scenario_name}")
        start_time = time.time()
        
        # Create agents
        agents = scenario_config["agents"]()
        
        # Run experiment
        p_data, n_data = run_experiment_set(agents, NUM_ROUNDS, NUM_RUNS)
        all_results[scenario_name] = (p_data, n_data)
        
        # Save CSV
        save_results_csv(p_data, f"{scenario_name}_pairwise")
        save_results_csv(n_data, f"{scenario_name}_neighborhood")
        
        # Create plot
        plot_path = os.path.join(figures_dir, f"{scenario_name}.png")
        plot_scenario(p_data, n_data, scenario_name, plot_path)
        
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.1f}s")
    
    # Create comparison heatmap
    print("\nCreating comparison heatmap...")
    heatmap_path = os.path.join(OUTPUT_DIR, "comparison_heatmap.png")
    create_comparison_heatmap(all_results, heatmap_path)
    
    # Save summary statistics
    print("Saving summary statistics...")
    save_summary_stats(all_results)
    
    # Create discount factor comparison plots
    print("Creating discount factor comparison plots...")
    create_df_comparison_plots(all_results)
    
    print(f"\nAll experiments complete! Results saved to '{OUTPUT_DIR}'")
    print("Files created:")
    print(f"  - {len(scenarios)} scenario plots in figures/")
    print(f"  - {len(scenarios) * 2} CSV files in csv_files/")
    print("  - comparison_heatmap.png")
    print("  - summary_statistics.json")
    print("  - discount_factor_comparison.png")

def create_df_comparison_plots(all_results):
    """Create plots specifically comparing discount factor effects"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Effect of Discount Factor on Q-Learning Performance', fontsize=16)
    
    # Vanilla QL comparison
    scenarios_vanilla = ["1_Vanilla_DF06_vs_2_TFT", "1_Vanilla_DF095_vs_2_TFT",
                        "1_Vanilla_DF06_vs_2_TFT-E", "1_Vanilla_DF095_vs_2_TFT-E"]
    
    # Adaptive QL comparison
    scenarios_adaptive = ["1_Adaptive_DF06_vs_2_TFT", "1_Adaptive_DF095_vs_2_TFT",
                         "1_Adaptive_DF06_vs_2_TFT-E", "1_Adaptive_DF095_vs_2_TFT-E"]
    
    # Plot comparisons
    for ax_idx, (scenarios, title_prefix) in enumerate([(scenarios_vanilla, "Vanilla"), 
                                                        (scenarios_adaptive, "Adaptive")]):
        for scenario in scenarios:
            if scenario in all_results:
                p_data, n_data = all_results[scenario]
                
                # Find QL agent
                ql_agent = [aid for aid in p_data.keys() if 'QL' in aid][0]
                
                # Determine line style and label
                if 'DF06' in scenario:
                    ls = '--'
                    df_label = 'DF=0.6'
                else:
                    ls = '-'
                    df_label = 'DF=0.95'
                
                if 'TFT-E' in scenario:
                    color = '#d62728'
                    opponent = 'vs TFT-E'
                else:
                    color = '#2ca02c'
                    opponent = 'vs TFT'
                
                label = f"{df_label} {opponent}"
                
                # Smooth data
                p_coop_smooth = smooth_data(p_data[ql_agent]['coop_rate'], 100)
                n_coop_smooth = smooth_data(n_data[ql_agent]['coop_rate'], 100)
                
                # Plot
                axes[0, ax_idx].plot(p_coop_smooth, ls=ls, color=color, label=label, linewidth=2)
                axes[1, ax_idx].plot(n_coop_smooth, ls=ls, color=color, label=label, linewidth=2)
    
    # Configure axes
    axes[0, 0].set_title('Vanilla QL - Pairwise')
    axes[0, 1].set_title('Adaptive QL - Pairwise')
    axes[1, 0].set_title('Vanilla QL - Neighborhood')
    axes[1, 1].set_title('Adaptive QL - Neighborhood')
    
    for ax in axes.flat:
        ax.set_xlabel('Round')
        ax.set_ylabel('Cooperation Rate')
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'discount_factor_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()