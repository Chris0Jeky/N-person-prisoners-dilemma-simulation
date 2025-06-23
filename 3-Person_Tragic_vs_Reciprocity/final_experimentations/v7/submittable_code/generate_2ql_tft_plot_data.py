#!/usr/bin/env python3
"""
Generate cooperation and score data for 2 QL vs 1 TFT scenario
with DF=0.4 for pairwise and DF=0.95 for neighborhood
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from collections import defaultdict

# Add parent directory to path to import modules
sys.path.append('.')

from final_agents_v2 import VanillaQLearner, StaticAgent
from final_simulation import run_pairwise_tournament, run_nperson_simulation

# Constants
PAYOFFS_2P = {(0, 0): (3, 3), (0, 1): (0, 5), (1, 0): (5, 0), (1, 1): (1, 1)}
T, R, P, S = 5, 3, 1, 0

# Simulation parameters
NUM_ROUNDS = 10000
NUM_RUNS = 50  # More runs for smoother data

# Q-learner configurations
QL_CONFIG_PAIRWISE = {
    'lr': 0.08,
    'df': 0.4,  # DF=0.4 for pairwise
    'eps': 0.1,
}

QL_CONFIG_NEIGHBORHOOD = {
    'lr': 0.08,
    'df': 0.95,  # DF=0.95 for neighborhood
    'eps': 0.1,
}

def run_mixed_df_experiment(num_rounds, num_runs):
    """Run experiment with different DFs for pairwise and neighborhood"""
    p_runs = []
    n_runs = []
    
    for run in range(num_runs):
        # Create agents for pairwise (DF=0.4)
        agents_pairwise = [
            VanillaQLearner("QL1", QL_CONFIG_PAIRWISE),
            VanillaQLearner("QL2", QL_CONFIG_PAIRWISE),
            StaticAgent("TFT", "TFT", 0.0)
        ]
        
        # Create agents for neighborhood (DF=0.95)
        agents_neighborhood = [
            VanillaQLearner("QL1", QL_CONFIG_NEIGHBORHOOD),
            VanillaQLearner("QL2", QL_CONFIG_NEIGHBORHOOD),
            StaticAgent("TFT", "TFT", 0.0)
        ]
        
        # Run simulations
        p_runs.append(run_pairwise_tournament(agents_pairwise, num_rounds))
        n_runs.append(run_nperson_simulation(agents_neighborhood, num_rounds))
    
    # Aggregate results
    p_agg = {aid: {m: np.mean([r[aid][m] for r in p_runs if aid in r], axis=0) for m in ['coop_rate', 'score']} 
             for aid in p_runs[0]}
    n_agg = {aid: {m: np.mean([r[aid][m] for r in n_runs if aid in r], axis=0) for m in ['coop_rate', 'score']} 
             for aid in n_runs[0]}
    
    return p_agg, n_agg

def smooth_data(data, window_size=100):
    """Apply rolling average to smooth data"""
    if len(data) < window_size:
        window_size = max(1, len(data) // 10)
    
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        smoothed.append(np.mean(data[start:end]))
    
    return np.array(smoothed)

def save_plot_data(p_data, n_data):
    """Save data in CSV format for plotting"""
    
    # Create DataFrames for cooperation rates
    rounds = np.arange(NUM_ROUNDS)
    
    # Smooth the data for better visualization
    coop_data = {
        'Round': rounds,
        'QL1_Pairwise_Coop': smooth_data(p_data['QL1']['coop_rate']),
        'QL2_Pairwise_Coop': smooth_data(p_data['QL2']['coop_rate']),
        'TFT_Pairwise_Coop': smooth_data(p_data['TFT']['coop_rate']),
        'QL1_Neighborhood_Coop': smooth_data(n_data['QL1']['coop_rate']),
        'QL2_Neighborhood_Coop': smooth_data(n_data['QL2']['coop_rate']),
        'TFT_Neighborhood_Coop': smooth_data(n_data['TFT']['coop_rate'])
    }
    
    # Score data (cumulative, no smoothing needed)
    score_data = {
        'Round': rounds,
        'QL1_Pairwise_Score': p_data['QL1']['score'],
        'QL2_Pairwise_Score': p_data['QL2']['score'],
        'TFT_Pairwise_Score': p_data['TFT']['score'],
        'QL1_Neighborhood_Score': n_data['QL1']['score'],
        'QL2_Neighborhood_Score': n_data['QL2']['score'],
        'TFT_Neighborhood_Score': n_data['TFT']['score']
    }
    
    # Save to CSV
    df_coop = pd.DataFrame(coop_data)
    df_score = pd.DataFrame(score_data)
    
    df_coop.to_csv('2QL_vs_TFT_cooperation_rates.csv', index=False)
    df_score.to_csv('2QL_vs_TFT_scores.csv', index=False)
    
    # Also save summary statistics
    summary = {
        'Agent': ['QL1', 'QL2', 'TFT'],
        'Pairwise_Avg_Coop': [
            np.mean(p_data['QL1']['coop_rate']),
            np.mean(p_data['QL2']['coop_rate']),
            np.mean(p_data['TFT']['coop_rate'])
        ],
        'Pairwise_Final_Score': [
            p_data['QL1']['score'][-1],
            p_data['QL2']['score'][-1],
            p_data['TFT']['score'][-1]
        ],
        'Neighborhood_Avg_Coop': [
            np.mean(n_data['QL1']['coop_rate']),
            np.mean(n_data['QL2']['coop_rate']),
            np.mean(n_data['TFT']['coop_rate'])
        ],
        'Neighborhood_Final_Score': [
            n_data['QL1']['score'][-1],
            n_data['QL2']['score'][-1],
            n_data['TFT']['score'][-1]
        ]
    }
    
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv('2QL_vs_TFT_summary.csv', index=False)
    
    print("\nSummary Statistics:")
    print(df_summary.to_string(index=False))
    
    return df_coop, df_score, df_summary

def create_visualization(p_data, n_data):
    """Create a 4-panel plot showing the results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('2 QL vs 1 TFT: DF=0.4 (Pairwise) vs DF=0.95 (Neighborhood)', fontsize=16)
    
    colors = {'QL1': '#1f77b4', 'QL2': '#ff7f0e', 'TFT': '#2ca02c'}
    
    # Plot cooperation rates
    for agent_id, color in colors.items():
        # Smooth cooperation rates
        p_coop_smooth = smooth_data(p_data[agent_id]['coop_rate'])
        n_coop_smooth = smooth_data(n_data[agent_id]['coop_rate'])
        
        axes[0, 0].plot(p_coop_smooth, label=agent_id, color=color, linewidth=2)
        axes[0, 1].plot(n_coop_smooth, label=agent_id, color=color, linewidth=2)
        axes[1, 0].plot(p_data[agent_id]['score'], label=agent_id, color=color, linewidth=2)
        axes[1, 1].plot(n_data[agent_id]['score'], label=agent_id, color=color, linewidth=2)
    
    # Configure axes
    axes[0, 0].set_title('Pairwise Cooperation Rate (DF=0.4)')
    axes[0, 0].set_ylabel('Cooperation Rate')
    axes[0, 0].set_ylim(-0.05, 1.05)
    
    axes[0, 1].set_title('Neighborhood Cooperation Rate (DF=0.95)')
    axes[0, 1].set_ylabel('Cooperation Rate')
    axes[0, 1].set_ylim(-0.05, 1.05)
    
    axes[1, 0].set_title('Pairwise Cumulative Score (DF=0.4)')
    axes[1, 0].set_ylabel('Cumulative Score')
    
    axes[1, 1].set_title('Neighborhood Cumulative Score (DF=0.95)')
    axes[1, 1].set_ylabel('Cumulative Score')
    
    for ax in axes.flat:
        ax.set_xlabel('Round')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('2QL_vs_TFT_mixed_DF.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("=" * 80)
    print("2 QL vs 1 TFT EXPERIMENT")
    print("Pairwise: DF=0.4 (short-term focus)")
    print("Neighborhood: DF=0.95 (long-term focus)")
    print("=" * 80)
    
    print(f"\nRunning {NUM_RUNS} simulations of {NUM_ROUNDS} rounds each...")
    
    # Run experiment
    p_data, n_data = run_mixed_df_experiment(NUM_ROUNDS, NUM_RUNS)
    
    # Save data for plotting
    df_coop, df_score, df_summary = save_plot_data(p_data, n_data)
    
    # Create visualization
    create_visualization(p_data, n_data)
    
    print("\nFiles generated:")
    print("- 2QL_vs_TFT_cooperation_rates.csv")
    print("- 2QL_vs_TFT_scores.csv")
    print("- 2QL_vs_TFT_summary.csv")
    print("- 2QL_vs_TFT_mixed_DF.png")
    
    print("\nExperiment complete!")

if __name__ == "__main__":
    main()