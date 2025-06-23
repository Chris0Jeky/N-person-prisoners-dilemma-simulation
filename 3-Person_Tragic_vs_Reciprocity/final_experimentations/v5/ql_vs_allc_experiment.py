"""
Q-Learning vs Always Cooperate Control Experiment
This script runs the missing control experiments for Table 2:
- Vanilla Q-Learning vs Always Cooperate in Pairwise voting
- Vanilla Q-Learning vs Always Cooperate in Neighbour voting
Modified to use 1 QL vs 2 AllC configuration
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from final_agents import StaticAgent, PairwiseAdaptiveQLearner, NeighborhoodAdaptiveQLearner
from final_simulation import run_pairwise_tournament, run_nperson_simulation
from config import VANILLA_PARAMS


def create_vanilla_qlearning_agent(agent_id, game_type):
    """Create a vanilla Q-learning agent with standard parameters"""
    
    if game_type == "pairwise":
        # For pairwise, use the adaptive learner with vanilla params
        agent = PairwiseAdaptiveQLearner(agent_id, VANILLA_PARAMS)
    else:  # neighborhood
        agent = NeighborhoodAdaptiveQLearner(agent_id, VANILLA_PARAMS)
    
    return agent


def run_ql_vs_allc_experiment(voting_model="pairwise", num_rounds=10000, num_runs=25):
    """
    Run experiment with 1 Q-Learning agent vs 2 Always Cooperate agents
    
    Parameters:
    - voting_model: "pairwise" or "neighborhood"
    - num_rounds: Number of rounds per run
    - num_runs: Number of independent runs
    """
    
    print(f"\nRunning Q-Learning vs Always Cooperate - {voting_model} voting")
    print(f"Configuration: {num_runs} runs, {num_rounds:,} rounds each")
    print("Setup: 1 Q-Learning agent vs 2 Always Cooperate agents")
    
    # Storage for results across all runs
    all_cooperation_rates = {
        'QL': [],
        'AllC1': [],
        'AllC2': []
    }
    all_scores = {
        'QL': [],
        'AllC1': [],
        'AllC2': []
    }
    
    # Run multiple independent simulations
    for run in range(num_runs):
        if run % 5 == 0:
            print(f"  Run {run+1}/{num_runs}...")
        
        # Create fresh agents for each run - 1 QL vs 2 AllC
        agents = [
            create_vanilla_qlearning_agent("QL", voting_model),
            StaticAgent("AllC1", "AllC"),
            StaticAgent("AllC2", "AllC")
        ]
        
        # Run tournament
        if voting_model == "pairwise":
            history = run_pairwise_tournament(agents, num_rounds)
        else:  # neighborhood
            history = run_nperson_simulation(agents, num_rounds)
        
        # Store results
        for agent_id in ['QL', 'AllC1', 'AllC2']:
            all_cooperation_rates[agent_id].append(history[agent_id]['coop_rate'])
            all_scores[agent_id].append(history[agent_id]['score'])
    
    # Convert to numpy arrays for easier manipulation
    for agent_id in ['QL', 'AllC1', 'AllC2']:
        all_cooperation_rates[agent_id] = np.array(all_cooperation_rates[agent_id])
        all_scores[agent_id] = np.array(all_scores[agent_id])
    
    return all_cooperation_rates, all_scores


def save_results_to_csv(cooperation_rates, scores, voting_model, output_dir):
    """Save experiment results to CSV files"""
    
    # Calculate mean and std across runs - subsample for CSV to avoid huge files
    # Take every 100th round for the detailed CSV
    subsample_rate = 100
    results_data = []
    
    for round_idx in range(0, cooperation_rates['QL'].shape[1], subsample_rate):
        row = {
            'round': round_idx + 1,
            'QL_coop_mean': np.mean(cooperation_rates['QL'][:, round_idx]),
            'QL_coop_std': np.std(cooperation_rates['QL'][:, round_idx]),
            'AllC1_coop_mean': np.mean(cooperation_rates['AllC1'][:, round_idx]),
            'AllC1_coop_std': np.std(cooperation_rates['AllC1'][:, round_idx]),
            'AllC2_coop_mean': np.mean(cooperation_rates['AllC2'][:, round_idx]),
            'AllC2_coop_std': np.std(cooperation_rates['AllC2'][:, round_idx]),
            'QL_score_mean': np.mean(scores['QL'][:, round_idx]),
            'QL_score_std': np.std(scores['QL'][:, round_idx]),
            'AllC1_score_mean': np.mean(scores['AllC1'][:, round_idx]),
            'AllC1_score_std': np.std(scores['AllC1'][:, round_idx]),
            'AllC2_score_mean': np.mean(scores['AllC2'][:, round_idx]),
            'AllC2_score_std': np.std(scores['AllC2'][:, round_idx])
        }
        results_data.append(row)
    
    # Save to CSV
    df = pd.DataFrame(results_data)
    csv_path = output_dir / f"ql_vs_allc_{voting_model}_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")
    
    # Also save summary statistics
    # Calculate final cooperation rates (last 10% of rounds)
    final_rounds = cooperation_rates['QL'].shape[1] // 10
    summary_data = {
        'voting_model': voting_model,
        'num_runs': cooperation_rates['QL'].shape[0],
        'num_rounds': cooperation_rates['QL'].shape[1],
        'QL_final_coop': np.mean(cooperation_rates['QL'][:, -final_rounds:]),
        'AllC1_final_coop': np.mean(cooperation_rates['AllC1'][:, -final_rounds:]),
        'AllC2_final_coop': np.mean(cooperation_rates['AllC2'][:, -final_rounds:]),
        'QL_final_score': np.mean(scores['QL'][:, -1]),
        'AllC1_final_score': np.mean(scores['AllC1'][:, -1]),
        'AllC2_final_score': np.mean(scores['AllC2'][:, -1]),
        'QL_avg_score_per_round': np.mean(scores['QL'][:, -1]) / cooperation_rates['QL'].shape[1],
        'AllC_avg_score_per_round': np.mean(scores['AllC1'][:, -1]) / cooperation_rates['AllC1'].shape[1]
    }
    
    summary_df = pd.DataFrame([summary_data])
    summary_path = output_dir / f"ql_vs_allc_{voting_model}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved summary: {summary_path}")


def plot_results(cooperation_rates, scores, voting_model, output_dir):
    """Generate plots for the experiment results"""
    
    # Calculate means and confidence intervals
    mean_coop = {}
    ci_coop = {}
    mean_scores = {}
    
    for agent_id in ['QL', 'AllC1', 'AllC2']:
        mean_coop[agent_id] = np.mean(cooperation_rates[agent_id], axis=0)
        std_coop = np.std(cooperation_rates[agent_id], axis=0)
        ci_coop[agent_id] = 1.96 * std_coop / np.sqrt(cooperation_rates[agent_id].shape[0])
        mean_scores[agent_id] = np.mean(scores[agent_id], axis=0)
    
    # Apply smoothing
    window_size = 50
    for agent_id in ['QL', 'AllC1', 'AllC2']:
        mean_coop[agent_id] = pd.Series(mean_coop[agent_id]).rolling(
            window=window_size, min_periods=1, center=True
        ).mean().values
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot cooperation rates
    rounds = np.arange(1, len(mean_coop['QL']) + 1)
    
    # Subsample for plotting if too many points
    if len(rounds) > 5000:
        plot_indices = np.linspace(0, len(rounds)-1, 5000, dtype=int)
        rounds_plot = rounds[plot_indices]
        mean_coop_plot = {k: v[plot_indices] for k, v in mean_coop.items()}
        ci_coop_plot = {k: v[plot_indices] for k, v in ci_coop.items()}
        mean_scores_plot = {k: v[plot_indices] for k, v in mean_scores.items()}
    else:
        rounds_plot = rounds
        mean_coop_plot = mean_coop
        ci_coop_plot = ci_coop
        mean_scores_plot = mean_scores
    
    # Colors for agents
    colors = {'QL': 'blue', 'AllC1': 'red', 'AllC2': 'orange'}
    
    # Plot cooperation rates
    for agent_id in ['QL', 'AllC1', 'AllC2']:
        label = 'Q-Learning' if agent_id == 'QL' else f'Always Cooperate {agent_id[-1]}'
        ax1.plot(rounds_plot, mean_coop_plot[agent_id], label=label, 
                color=colors[agent_id], linewidth=2)
        ax1.fill_between(rounds_plot, 
                        mean_coop_plot[agent_id] - ci_coop_plot[agent_id],
                        mean_coop_plot[agent_id] + ci_coop_plot[agent_id],
                        alpha=0.2, color=colors[agent_id])
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Cooperation Rate')
    ax1.set_title(f'1 Q-Learning vs 2 Always Cooperate - {voting_model.capitalize()} Voting\n'
                  f'Cooperation Rates ({cooperation_rates["QL"].shape[0]} runs, '
                  f'{cooperation_rates["QL"].shape[1]:,} rounds)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Add horizontal line at 0 to highlight defection
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Plot cumulative scores
    for agent_id in ['QL', 'AllC1', 'AllC2']:
        label = 'Q-Learning' if agent_id == 'QL' else f'Always Cooperate {agent_id[-1]}'
        ax2.plot(rounds_plot, mean_scores_plot[agent_id], label=label, 
                color=colors[agent_id], linewidth=2)
    
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Cumulative Score')
    ax2.set_title('Cumulative Scores')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = output_dir / f"ql_vs_allc_{voting_model}_plot.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved figure: {fig_path}")
    
    # Create a zoomed-in plot for the first 5000 rounds to see early learning
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    early_rounds = min(5000, len(mean_coop['QL']))
    for agent_id in ['QL', 'AllC1', 'AllC2']:
        label = 'Q-Learning' if agent_id == 'QL' else f'Always Cooperate {agent_id[-1]}'
        ax.plot(range(early_rounds), mean_coop[agent_id][:early_rounds], 
               label=label, color=colors[agent_id], linewidth=2)
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Cooperation Rate')
    ax.set_title(f'Early Learning Phase - First {early_rounds} Rounds\n'
                f'{voting_model.capitalize()} Voting')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    zoom_path = output_dir / f"ql_vs_allc_{voting_model}_early_learning.png"
    plt.savefig(zoom_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved early learning plot: {zoom_path}")


def main():
    """Main function to run all experiments"""
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results_ql_vs_allc_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    # Save experiment configuration
    config = {
        "experiment": "Q-Learning vs Always Cooperate Control",
        "timestamp": timestamp,
        "num_rounds": 10000,
        "num_runs": 25,
        "voting_models": ["pairwise", "neighborhood"],
        "agents": ["QL (Vanilla Q-Learning)", "AllC1 (Always Cooperate)", "AllC2 (Always Cooperate)"],
        "setup": "1 Q-Learning agent vs 2 Always Cooperate agents",
        "q_learning_params": VANILLA_PARAMS
    }
    
    with open(output_dir / "experiment_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Starting Q-Learning vs Always Cooperate Control Experiments")
    print("Setup: 1 Q-Learning agent vs 2 Always Cooperate agents")
    print("=" * 60)
    
    # Run experiments for both voting models
    for voting_model in ["pairwise", "neighborhood"]:
        cooperation_rates, scores = run_ql_vs_allc_experiment(
            voting_model=voting_model,
            num_rounds=10000,
            num_runs=25
        )
        
        # Save results
        save_results_to_csv(cooperation_rates, scores, voting_model, output_dir)
        plot_results(cooperation_rates, scores, voting_model, output_dir)
    
    print("\n" + "=" * 60)
    print(f"All experiments completed! Results saved in: {output_dir}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob("*")):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()