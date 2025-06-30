#!/usr/bin/env python3
"""
Single Run for 3 TFT-E Scenario Only
Runs one simulation for both pairwise and neighbourhood strategies
Imports from the existing static_figure_generator.py module
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from datetime import datetime

# Import from the local static_figure_generator module
from static_figure_generator import (
    StaticAgent, 
    run_pairwise_simulation_extended,
    run_nperson_simulation_extended,
    COOPERATE, DEFECT
)


def plot_3tfte_comparison(pairwise_coop, pairwise_scores, nperson_coop, nperson_scores, save_dir):
    """Creates comprehensive plots comparing pairwise and neighbourhood for 3 TFT-E."""
    sns.set_style("whitegrid")
    
    # Create figure with 2 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    fig.suptitle("3 TFT-E Agents: Pairwise vs Neighbourhood (Single Run)", fontsize=18, weight='bold')
    
    # Plot 1: Pairwise Cooperation
    ax = axes[0, 0]
    for agent_id, coop_history in pairwise_coop.items():
        rounds = range(1, len(coop_history) + 1)
        ax.plot(rounds, coop_history, linewidth=2.5, label=agent_id, alpha=0.8)
    
    # Add average cooperation line
    avg_coop = np.mean([coop_history for coop_history in pairwise_coop.values()], axis=0)
    ax.plot(rounds, avg_coop, 'k--', linewidth=3, label='Average', alpha=0.7)
    
    ax.set_title("Pairwise: Cooperation Rates", fontsize=14, weight='bold')
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Cooperation Rate", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Neighbourhood Cooperation
    ax = axes[0, 1]
    for agent_id, coop_history in nperson_coop.items():
        rounds = range(1, len(coop_history) + 1)
        ax.plot(rounds, coop_history, linewidth=2.5, label=agent_id, alpha=0.8)
    
    # Add average cooperation line
    avg_coop = np.mean([coop_history for coop_history in nperson_coop.values()], axis=0)
    ax.plot(rounds, avg_coop, 'k--', linewidth=3, label='Average', alpha=0.7)
    
    ax.set_title("Neighbourhood: Cooperation Rates", fontsize=14, weight='bold')
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Cooperation Rate", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Pairwise Scores
    ax = axes[1, 0]
    for agent_id, score_history in pairwise_scores.items():
        rounds = range(1, len(score_history) + 1)
        ax.plot(rounds, score_history, linewidth=2.5, label=agent_id, alpha=0.8)
    
    ax.set_title("Pairwise: Cumulative Scores", fontsize=14, weight='bold')
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Cumulative Score", fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Neighbourhood Scores
    ax = axes[1, 1]
    for agent_id, score_history in nperson_scores.items():
        rounds = range(1, len(score_history) + 1)
        ax.plot(rounds, score_history, linewidth=2.5, label=agent_id, alpha=0.8)
    
    ax.set_title("Neighbourhood: Cumulative Scores", fontsize=14, weight='bold')
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Cumulative Score", fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, "3_TFT-E_comparison.png"), dpi=300, bbox_inches='tight')
    print(f"  - Saved: 3_TFT-E_comparison.png")
    plt.close()
    
    # Create a summary comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    fig.suptitle("3 TFT-E: Average Cooperation Comparison", fontsize=16, weight='bold')
    
    # Left plot: Average cooperation over time
    ax = axes[0]
    rounds = range(1, NUM_ROUNDS + 1)
    
    pairwise_avg = np.mean([coop_history for coop_history in pairwise_coop.values()], axis=0)
    nperson_avg = np.mean([coop_history for coop_history in nperson_coop.values()], axis=0)
    
    ax.plot(rounds, pairwise_avg, 'b-', linewidth=3, label='Pairwise', alpha=0.8)
    ax.plot(rounds, nperson_avg, 'r-', linewidth=3, label='Neighbourhood', alpha=0.8)
    
    # Add smoothed lines
    window = 20
    pairwise_smooth = pd.Series(pairwise_avg).rolling(window=window, min_periods=1, center=True).mean()
    nperson_smooth = pd.Series(nperson_avg).rolling(window=window, min_periods=1, center=True).mean()
    
    ax.plot(rounds, pairwise_smooth, 'b--', linewidth=2, label=f'Pairwise (smoothed, w={window})', alpha=0.6)
    ax.plot(rounds, nperson_smooth, 'r--', linewidth=2, label=f'Neighbourhood (smoothed, w={window})', alpha=0.6)
    
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Average Cooperation Rate", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Right plot: Final scores comparison
    ax = axes[1]
    
    final_scores_pairwise = [scores[-1] for scores in pairwise_scores.values()]
    final_scores_nperson = [scores[-1] for scores in nperson_scores.values()]
    
    x = np.arange(3)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, final_scores_pairwise, width, label='Pairwise', alpha=0.8)
    bars2 = ax.bar(x + width/2, final_scores_nperson, width, label='Neighbourhood', alpha=0.8)
    
    ax.set_ylabel('Final Score', fontsize=12)
    ax.set_xlabel('Agent', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['TFT-E_1', 'TFT-E_2', 'TFT-E_3'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    plt.savefig(os.path.join(save_dir, "3_TFT-E_summary.png"), dpi=300, bbox_inches='tight')
    print(f"  - Saved: 3_TFT-E_summary.png")
    plt.close()


def save_3tfte_data_to_csv(pairwise_coop, pairwise_scores, nperson_coop, nperson_scores, save_dir):
    """Saves the 3 TFT-E data to CSV files."""
    # Pairwise data
    pairwise_df_data = {'Round': range(1, len(next(iter(pairwise_coop.values()))) + 1)}
    for agent_id, coop_history in pairwise_coop.items():
        pairwise_df_data[f'{agent_id}_cooperation'] = coop_history
    for agent_id, score_history in pairwise_scores.items():
        pairwise_df_data[f'{agent_id}_score'] = score_history
    
    # Add average cooperation
    avg_coop = np.mean([coop_history for coop_history in pairwise_coop.values()], axis=0)
    pairwise_df_data['average_cooperation'] = avg_coop
    
    pairwise_df = pd.DataFrame(pairwise_df_data)
    pairwise_df.to_csv(os.path.join(save_dir, "3_TFT-E_pairwise_data.csv"), index=False)
    print(f"  - Saved: 3_TFT-E_pairwise_data.csv")
    
    # Neighbourhood data
    nperson_df_data = {'Round': range(1, len(next(iter(nperson_coop.values()))) + 1)}
    for agent_id, coop_history in nperson_coop.items():
        nperson_df_data[f'{agent_id}_cooperation'] = coop_history
    for agent_id, score_history in nperson_scores.items():
        nperson_df_data[f'{agent_id}_score'] = score_history
    
    # Add average cooperation
    avg_coop = np.mean([coop_history for coop_history in nperson_coop.values()], axis=0)
    nperson_df_data['average_cooperation'] = avg_coop
    
    nperson_df = pd.DataFrame(nperson_df_data)
    nperson_df.to_csv(os.path.join(save_dir, "3_TFT-E_neighbourhood_data.csv"), index=False)
    print(f"  - Saved: 3_TFT-E_neighbourhood_data.csv")
    
    # Summary statistics
    summary_data = {
        'Metric': ['Mean Cooperation (Pairwise)', 'Mean Cooperation (Neighbourhood)',
                   'Final Avg Score (Pairwise)', 'Final Avg Score (Neighbourhood)',
                   'Cooperation Std Dev (Pairwise)', 'Cooperation Std Dev (Neighbourhood)'],
        'Value': [
            np.mean([np.mean(coop) for coop in pairwise_coop.values()]),
            np.mean([np.mean(coop) for coop in nperson_coop.values()]),
            np.mean([scores[-1] for scores in pairwise_scores.values()]),
            np.mean([scores[-1] for scores in nperson_scores.values()]),
            np.mean([np.std(coop) for coop in pairwise_coop.values()]),
            np.mean([np.std(coop) for coop in nperson_coop.values()])
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(save_dir, "3_TFT-E_summary_stats.csv"), index=False)
    print(f"  - Saved: 3_TFT-E_summary_stats.csv")


# --- Main Execution ---

if __name__ == "__main__":
    NUM_ROUNDS = 500
    
    # Setup only 3 TFT-E agents
    agents = [
        StaticAgent(agent_id="TFT-E_1", strategy_name="TFT-E", exploration_rate=0.3, exploration_decay=0.07),
        StaticAgent(agent_id="TFT-E_2", strategy_name="TFT-E", exploration_rate=0.3, exploration_decay=0.07),
        StaticAgent(agent_id="TFT-E_3", strategy_name="TFT-E", exploration_rate=0.3, exploration_decay=0.07),
    ]

    # --- Create results directory ---
    results_dir = "results_3tfte_single"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {os.path.abspath(results_dir)}")
    print(f"Running single simulation for 3 TFT-E agents...")

    # --- Run Single Pairwise Simulation ---
    print(f"\nRunning Pairwise simulation...")
    tft_coop_pw, pairwise_coop, pairwise_scores = run_pairwise_simulation_extended(agents, NUM_ROUNDS)
    
    # Print some statistics
    avg_coop_pairwise = np.mean([np.mean(coop) for coop in pairwise_coop.values()])
    print(f"  - Average cooperation rate: {avg_coop_pairwise:.3f}")
    print(f"  - Final scores: {[scores[-1] for scores in pairwise_scores.values()]}")

    # --- Run Single Neighbourhood Simulation ---
    print(f"\nRunning Neighbourhood simulation...")
    tft_coop_np, nperson_coop, nperson_scores = run_nperson_simulation_extended(agents, NUM_ROUNDS)
    
    # Print some statistics
    avg_coop_nperson = np.mean([np.mean(coop) for coop in nperson_coop.values()])
    print(f"  - Average cooperation rate: {avg_coop_nperson:.3f}")
    print(f"  - Final scores: {[scores[-1] for scores in nperson_scores.values()]}")

    # --- Save Data to CSV ---
    print("\nSaving data to CSV files...")
    save_3tfte_data_to_csv(pairwise_coop, pairwise_scores, nperson_coop, nperson_scores, results_dir)

    # --- Generate Plots ---
    print("\nGenerating plots...")
    plot_3tfte_comparison(pairwise_coop, pairwise_scores, nperson_coop, nperson_scores, results_dir)

    print(f"\nDone! All results saved to the '{results_dir}' directory.")