"""
Q-Learning vs Tit-for-Tat (TFT) Control Experiment
This script runs experiments for:
- Vanilla Q-Learning vs 2 TFT agents in Pairwise voting
- Vanilla Q-Learning vs 2 TFT agents in Neighbourhood voting
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


def run_ql_vs_tft_experiment(voting_model="pairwise", num_rounds=5000, num_runs=25, tft_error_rate=0.0):
    """
    Run experiment with 1 Q-Learning agent vs 2 TFT agents
    
    Parameters:
    - voting_model: "pairwise" or "neighborhood"
    - num_rounds: Number of rounds per run
    - num_runs: Number of independent runs
    - tft_error_rate: Error rate for TFT agents (0.0 for perfect TFT, 0.1 for 10% error)
    """
    
    strategy_name = "TFT-E" if tft_error_rate > 0 else "TFT"
    print(f"\nRunning Q-Learning vs {strategy_name} - {voting_model} voting")
    print(f"Configuration: {num_runs} runs, {num_rounds:,} rounds each")
    print(f"Setup: 1 Q-Learning agent vs 2 {strategy_name} agents")
    if tft_error_rate > 0:
        print(f"TFT Error Rate: {tft_error_rate:.1%}")
    
    # Storage for results across all runs
    all_cooperation_rates = {
        'QL': [],
        'TFT1': [],
        'TFT2': []
    }
    all_scores = {
        'QL': [],
        'TFT1': [],
        'TFT2': []
    }
    
    # Run multiple independent simulations
    for run in range(num_runs):
        if run % 5 == 0:
            print(f"  Run {run+1}/{num_runs}...")
        
        # Create fresh agents for each run - 1 QL vs 2 TFT
        agents = [
            create_vanilla_qlearning_agent("QL", voting_model),
            StaticAgent("TFT1", strategy_name, error_rate=tft_error_rate),
            StaticAgent("TFT2", strategy_name, error_rate=tft_error_rate)
        ]
        
        # Run tournament
        if voting_model == "pairwise":
            history = run_pairwise_tournament(agents, num_rounds)
        else:  # neighborhood
            history = run_nperson_simulation(agents, num_rounds)
        
        # Store results
        for agent_id in ['QL', 'TFT1', 'TFT2']:
            all_cooperation_rates[agent_id].append(history[agent_id]['coop_rate'])
            all_scores[agent_id].append(history[agent_id]['score'])
    
    # Convert to numpy arrays for easier manipulation
    for agent_id in ['QL', 'TFT1', 'TFT2']:
        all_cooperation_rates[agent_id] = np.array(all_cooperation_rates[agent_id])
        all_scores[agent_id] = np.array(all_scores[agent_id])
    
    return all_cooperation_rates, all_scores


def save_results_to_csv(cooperation_rates, scores, voting_model, output_dir, tft_error_rate=0.0):
    """Save experiment results to CSV files"""
    
    strategy_suffix = f"_e{int(tft_error_rate*100)}" if tft_error_rate > 0 else ""
    
    # Calculate mean and std across runs - subsample for CSV to avoid huge files
    # Take every 100th round for the detailed CSV
    subsample_rate = 100
    results_data = []
    
    for round_idx in range(0, cooperation_rates['QL'].shape[1], subsample_rate):
        row = {
            'round': round_idx + 1,
            'QL_coop_mean': np.mean(cooperation_rates['QL'][:, round_idx]),
            'QL_coop_std': np.std(cooperation_rates['QL'][:, round_idx]),
            'TFT1_coop_mean': np.mean(cooperation_rates['TFT1'][:, round_idx]),
            'TFT1_coop_std': np.std(cooperation_rates['TFT1'][:, round_idx]),
            'TFT2_coop_mean': np.mean(cooperation_rates['TFT2'][:, round_idx]),
            'TFT2_coop_std': np.std(cooperation_rates['TFT2'][:, round_idx]),
            'QL_score_mean': np.mean(scores['QL'][:, round_idx]),
            'QL_score_std': np.std(scores['QL'][:, round_idx]),
            'TFT1_score_mean': np.mean(scores['TFT1'][:, round_idx]),
            'TFT1_score_std': np.std(scores['TFT1'][:, round_idx]),
            'TFT2_score_mean': np.mean(scores['TFT2'][:, round_idx]),
            'TFT2_score_std': np.std(scores['TFT2'][:, round_idx])
        }
        results_data.append(row)
    
    # Save to CSV
    df = pd.DataFrame(results_data)
    csv_path = output_dir / f"ql_vs_tft{strategy_suffix}_{voting_model}_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")
    
    # Also save summary statistics
    # Calculate final cooperation rates (last 10% of rounds)
    final_rounds = cooperation_rates['QL'].shape[1] // 10
    summary_data = {
        'voting_model': voting_model,
        'num_runs': cooperation_rates['QL'].shape[0],
        'num_rounds': cooperation_rates['QL'].shape[1],
        'tft_error_rate': tft_error_rate,
        'QL_final_coop': np.mean(cooperation_rates['QL'][:, -final_rounds:]),
        'TFT1_final_coop': np.mean(cooperation_rates['TFT1'][:, -final_rounds:]),
        'TFT2_final_coop': np.mean(cooperation_rates['TFT2'][:, -final_rounds:]),
        'QL_final_score': np.mean(scores['QL'][:, -1]),
        'TFT1_final_score': np.mean(scores['TFT1'][:, -1]),
        'TFT2_final_score': np.mean(scores['TFT2'][:, -1]),
        'QL_avg_score_per_round': np.mean(scores['QL'][:, -1]) / cooperation_rates['QL'].shape[1],
        'TFT_avg_score_per_round': np.mean(scores['TFT1'][:, -1]) / cooperation_rates['TFT1'].shape[1]
    }
    
    summary_df = pd.DataFrame([summary_data])
    summary_path = output_dir / f"ql_vs_tft{strategy_suffix}_{voting_model}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved summary: {summary_path}")


def plot_results(cooperation_rates, scores, voting_model, output_dir, tft_error_rate=0.0):
    """Generate plots for the experiment results"""
    
    strategy_name = "TFT-E" if tft_error_rate > 0 else "TFT"
    strategy_suffix = f"_e{int(tft_error_rate*100)}" if tft_error_rate > 0 else ""
    
    # Calculate means and confidence intervals
    mean_coop = {}
    ci_coop = {}
    mean_scores = {}
    
    for agent_id in ['QL', 'TFT1', 'TFT2']:
        mean_coop[agent_id] = np.mean(cooperation_rates[agent_id], axis=0)
        std_coop = np.std(cooperation_rates[agent_id], axis=0)
        ci_coop[agent_id] = 1.96 * std_coop / np.sqrt(cooperation_rates[agent_id].shape[0])
        mean_scores[agent_id] = np.mean(scores[agent_id], axis=0)
    
    # Apply smoothing
    window_size = 50
    for agent_id in ['QL', 'TFT1', 'TFT2']:
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
    colors = {'QL': 'blue', 'TFT1': 'green', 'TFT2': 'lightgreen'}
    
    # Plot cooperation rates
    for agent_id in ['QL', 'TFT1', 'TFT2']:
        label = 'Q-Learning' if agent_id == 'QL' else f'{strategy_name} {agent_id[-1]}'
        ax1.plot(rounds_plot, mean_coop_plot[agent_id], label=label, 
                color=colors[agent_id], linewidth=2)
        ax1.fill_between(rounds_plot, 
                        mean_coop_plot[agent_id] - ci_coop_plot[agent_id],
                        mean_coop_plot[agent_id] + ci_coop_plot[agent_id],
                        alpha=0.2, color=colors[agent_id])
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Cooperation Rate')
    title = f'1 Q-Learning vs 2 {strategy_name} - {voting_model.capitalize()} Voting\n'
    title += f'Cooperation Rates ({cooperation_rates["QL"].shape[0]} runs, '
    title += f'{cooperation_rates["QL"].shape[1]:,} rounds)'
    if tft_error_rate > 0:
        title += f'\nTFT Error Rate: {tft_error_rate:.1%}'
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Add horizontal line at 0 to highlight defection
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Plot cumulative scores
    for agent_id in ['QL', 'TFT1', 'TFT2']:
        label = 'Q-Learning' if agent_id == 'QL' else f'{strategy_name} {agent_id[-1]}'
        ax2.plot(rounds_plot, mean_scores_plot[agent_id], label=label, 
                color=colors[agent_id], linewidth=2)
    
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Cumulative Score')
    ax2.set_title('Cumulative Scores')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = output_dir / f"ql_vs_tft{strategy_suffix}_{voting_model}_plot.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved figure: {fig_path}")
    
    # Create a zoomed-in plot for the first 5000 rounds to see early learning
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    early_rounds = min(5000, len(mean_coop['QL']))
    for agent_id in ['QL', 'TFT1', 'TFT2']:
        label = 'Q-Learning' if agent_id == 'QL' else f'{strategy_name} {agent_id[-1]}'
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
    zoom_path = output_dir / f"ql_vs_tft{strategy_suffix}_{voting_model}_early_learning.png"
    plt.savefig(zoom_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved early learning plot: {zoom_path}")


def main():
    """Main function to run all experiments"""
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results_ql_vs_tft_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    # Configuration
    config = {
        "experiment": "Q-Learning vs Tit-for-Tat (TFT) Control",
        "timestamp": timestamp,
        "num_rounds": 5000,
        "num_runs": 25,
        "voting_models": ["pairwise", "neighborhood"],
        "agents": ["QL (Vanilla Q-Learning)", "TFT1 (Tit-for-Tat)", "TFT2 (Tit-for-Tat)"],
        "setup": "1 Q-Learning agent vs 2 TFT agents",
        "q_learning_params": VANILLA_PARAMS,
        "tft_configurations": [
            {"error_rate": 0.0, "description": "Perfect TFT"},
            {"error_rate": 0.1, "description": "TFT with 10% error rate"}
        ]
    }
    
    with open(output_dir / "experiment_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Starting Q-Learning vs Tit-for-Tat Control Experiments")
    print("Setup: 1 Q-Learning agent vs 2 TFT agents")
    print("=" * 60)
    
    # Run experiments for both voting models and both TFT configurations
    for tft_config in config["tft_configurations"]:
        error_rate = tft_config["error_rate"]
        print(f"\n{tft_config['description']}:")
        
        for voting_model in ["pairwise", "neighborhood"]:
            cooperation_rates, scores = run_ql_vs_tft_experiment(
                voting_model=voting_model,
                num_rounds=5000,
                num_runs=25,
                tft_error_rate=error_rate
            )
            
            # Save results
            save_results_to_csv(cooperation_rates, scores, voting_model, output_dir, error_rate)
            plot_results(cooperation_rates, scores, voting_model, output_dir, error_rate)
    
    print("\n" + "=" * 60)
    print(f"All experiments completed! Results saved in: {output_dir}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob("*")):
        print(f"  - {file.name}")
    
    print("\nNote: The experiment tests Q-Learning against both perfect TFT and TFT with errors.")


if __name__ == "__main__":
    main()