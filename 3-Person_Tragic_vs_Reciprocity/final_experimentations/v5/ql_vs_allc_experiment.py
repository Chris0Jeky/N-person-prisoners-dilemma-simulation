"""
Q-Learning vs Always Cooperate Control Experiment
This script runs the missing control experiments for Table 2:
- Vanilla Q-Learning vs Always Cooperate in Pairwise voting
- Vanilla Q-Learning vs Always Cooperate in Neighbour voting
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


def run_ql_vs_allc_experiment(voting_model="pairwise", num_rounds=1000, num_runs=100):
    """
    Run experiment with 2 Q-Learning agents vs 1 Always Cooperate agent
    
    Parameters:
    - voting_model: "pairwise" or "neighborhood"
    - num_rounds: Number of rounds per run
    - num_runs: Number of independent runs
    """
    
    print(f"\nRunning Q-Learning vs Always Cooperate - {voting_model} voting")
    print(f"Configuration: {num_runs} runs, {num_rounds} rounds each")
    
    # Storage for results across all runs
    all_cooperation_rates = {
        'QL1': [],
        'QL2': [],
        'AllC': []
    }
    all_scores = {
        'QL1': [],
        'QL2': [],
        'AllC': []
    }
    
    # Run multiple independent simulations
    for run in range(num_runs):
        if run % 10 == 0:
            print(f"  Run {run+1}/{num_runs}...")
        
        # Create fresh agents for each run
        agents = [
            create_vanilla_qlearning_agent("QL1", voting_model),
            create_vanilla_qlearning_agent("QL2", voting_model),
            StaticAgent("AllC", "AllC")
        ]
        
        # Run tournament
        if voting_model == "pairwise":
            history = run_pairwise_tournament(agents, num_rounds)
        else:  # neighborhood
            history = run_nperson_simulation(agents, num_rounds)
        
        # Store results
        for agent_id in ['QL1', 'QL2', 'AllC']:
            all_cooperation_rates[agent_id].append(history[agent_id]['coop_rate'])
            all_scores[agent_id].append(history[agent_id]['score'])
    
    # Convert to numpy arrays for easier manipulation
    for agent_id in ['QL1', 'QL2', 'AllC']:
        all_cooperation_rates[agent_id] = np.array(all_cooperation_rates[agent_id])
        all_scores[agent_id] = np.array(all_scores[agent_id])
    
    return all_cooperation_rates, all_scores


def save_results_to_csv(cooperation_rates, scores, voting_model, output_dir):
    """Save experiment results to CSV files"""
    
    # Calculate mean and std across runs
    results_data = []
    
    for round_idx in range(cooperation_rates['QL1'].shape[1]):
        row = {
            'round': round_idx + 1,
            'QL1_coop_mean': np.mean(cooperation_rates['QL1'][:, round_idx]),
            'QL1_coop_std': np.std(cooperation_rates['QL1'][:, round_idx]),
            'QL2_coop_mean': np.mean(cooperation_rates['QL2'][:, round_idx]),
            'QL2_coop_std': np.std(cooperation_rates['QL2'][:, round_idx]),
            'AllC_coop_mean': np.mean(cooperation_rates['AllC'][:, round_idx]),
            'AllC_coop_std': np.std(cooperation_rates['AllC'][:, round_idx]),
            'QL1_score_mean': np.mean(scores['QL1'][:, round_idx]),
            'QL1_score_std': np.std(scores['QL1'][:, round_idx]),
            'QL2_score_mean': np.mean(scores['QL2'][:, round_idx]),
            'QL2_score_std': np.std(scores['QL2'][:, round_idx]),
            'AllC_score_mean': np.mean(scores['AllC'][:, round_idx]),
            'AllC_score_std': np.std(scores['AllC'][:, round_idx])
        }
        results_data.append(row)
    
    # Save to CSV
    df = pd.DataFrame(results_data)
    csv_path = output_dir / f"ql_vs_allc_{voting_model}_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")
    
    # Also save summary statistics
    summary_data = {
        'voting_model': voting_model,
        'num_runs': cooperation_rates['QL1'].shape[0],
        'num_rounds': cooperation_rates['QL1'].shape[1],
        'QL1_final_coop': np.mean(cooperation_rates['QL1'][:, -100:]),
        'QL2_final_coop': np.mean(cooperation_rates['QL2'][:, -100:]),
        'AllC_final_coop': np.mean(cooperation_rates['AllC'][:, -100:]),
        'QL1_final_score': np.mean(scores['QL1'][:, -1]),
        'QL2_final_score': np.mean(scores['QL2'][:, -1]),
        'AllC_final_score': np.mean(scores['AllC'][:, -1])
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
    
    for agent_id in ['QL1', 'QL2', 'AllC']:
        mean_coop[agent_id] = np.mean(cooperation_rates[agent_id], axis=0)
        std_coop = np.std(cooperation_rates[agent_id], axis=0)
        ci_coop[agent_id] = 1.96 * std_coop / np.sqrt(cooperation_rates[agent_id].shape[0])
        mean_scores[agent_id] = np.mean(scores[agent_id], axis=0)
    
    # Apply smoothing
    window_size = 50
    for agent_id in ['QL1', 'QL2', 'AllC']:
        mean_coop[agent_id] = pd.Series(mean_coop[agent_id]).rolling(
            window=window_size, min_periods=1, center=True
        ).mean().values
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot cooperation rates
    rounds = np.arange(1, len(mean_coop['QL1']) + 1)
    
    # Colors for agents
    colors = {'QL1': 'blue', 'QL2': 'green', 'AllC': 'red'}
    
    for agent_id in ['QL1', 'QL2', 'AllC']:
        ax1.plot(rounds, mean_coop[agent_id], label=agent_id, color=colors[agent_id], linewidth=2)
        ax1.fill_between(rounds, 
                        mean_coop[agent_id] - ci_coop[agent_id],
                        mean_coop[agent_id] + ci_coop[agent_id],
                        alpha=0.2, color=colors[agent_id])
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Cooperation Rate')
    ax1.set_title(f'Q-Learning vs Always Cooperate - {voting_model.capitalize()} Voting\nCooperation Rates (100 runs)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Plot cumulative scores
    for agent_id in ['QL1', 'QL2', 'AllC']:
        ax2.plot(rounds, mean_scores[agent_id], label=agent_id, color=colors[agent_id], linewidth=2)
    
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
        "num_rounds": 1000,
        "num_runs": 100,
        "voting_models": ["pairwise", "neighborhood"],
        "agents": ["QL1 (Vanilla Q-Learning)", "QL2 (Vanilla Q-Learning)", "AllC (Always Cooperate)"]
    }
    
    with open(output_dir / "experiment_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Starting Q-Learning vs Always Cooperate Control Experiments")
    print("=" * 60)
    
    # Run experiments for both voting models
    for voting_model in ["pairwise", "neighborhood"]:
        cooperation_rates, scores = run_ql_vs_allc_experiment(
            voting_model=voting_model,
            num_rounds=1000,
            num_runs=100
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