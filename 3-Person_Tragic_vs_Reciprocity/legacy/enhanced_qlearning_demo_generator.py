#!/usr/bin/env python3
"""
Enhanced Q-Learning Demo Generator for N-Person Prisoner's Dilemma
Experiments with advanced Q-learning agents featuring optimistic initialization
and sophisticated state representations
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from datetime import datetime
from itertools import combinations
from collections import deque

# --- Constants and Payoffs ---
COOPERATE = 0
DEFECT = 1
PAYOFFS_2P = {  # 2-Player Payoffs
    (COOPERATE, COOPERATE): (3, 3), (COOPERATE, DEFECT): (0, 5),
    (DEFECT, COOPERATE): (5, 0), (DEFECT, DEFECT): (1, 1),
}
T, R, P, S = 5, 3, 1, 0  # N-Person Payoff Constants


def nperson_payoff(my_move, num_other_cooperators, total_agents):
    """Calculates N-Person payoff based on the linear formula."""
    if total_agents <= 1: return R if my_move == COOPERATE else P
    if my_move == COOPERATE:
        return S + (R - S) * (num_other_cooperators / (total_agents - 1))
    else:  # Defect
        return P + (T - P) * (num_other_cooperators / (total_agents - 1))


# --- Agent Classes ---
class StaticAgent:
    """Static strategy agent (AllC, AllD, TFT, TFT-E, Random)"""
    def __init__(self, agent_id, strategy_name, exploration_rate=0.0):
        self.agent_id = agent_id
        self.strategy_name = strategy_name
        self.exploration_rate = exploration_rate
        self.opponent_last_moves = {}

    def choose_pairwise_action(self, opponent_id):
        """Choose action for a 2-player game."""
        intended_move = None
        if self.strategy_name in ["TFT", "TFT-E"]:
            intended_move = self.opponent_last_moves.get(opponent_id, COOPERATE)
        elif self.strategy_name == "AllC":
            intended_move = COOPERATE
        elif self.strategy_name == "AllD":
            intended_move = DEFECT
        elif self.strategy_name == "Random":
            intended_move = random.choice([COOPERATE, DEFECT])

        # Apply exploration for TFT-E
        if self.strategy_name == "TFT-E" and random.random() < self.exploration_rate:
            return 1 - intended_move
        return intended_move

    def choose_nperson_action(self, prev_round_group_coop_ratio):
        """Choose action for an N-player group game."""
        intended_move = None
        if self.strategy_name in ["TFT", "TFT-E"]:
            if prev_round_group_coop_ratio is None:  # First round
                intended_move = COOPERATE
            else:
                intended_move = COOPERATE if random.random() < prev_round_group_coop_ratio else DEFECT
        elif self.strategy_name == "AllC":
            intended_move = COOPERATE
        elif self.strategy_name == "AllD":
            intended_move = DEFECT
        elif self.strategy_name == "Random":
            intended_move = random.choice([COOPERATE, DEFECT])

        # Apply exploration for TFT-E
        if self.strategy_name == "TFT-E" and random.random() < self.exploration_rate:
            return 1 - intended_move
        return intended_move

    def reset(self):
        self.opponent_last_moves = {}


# Import the corrected Enhanced Q-Learning agent from the enhanced_qlearning_agents module
from enhanced_qlearning_agents import EnhancedQLearningAgent


# --- Simulation Functions ---
def run_pairwise_simulation_extended(agents, num_rounds):
    """Runs a single pairwise tournament with enhanced Q-learning support."""
    for agent in agents: agent.reset()

    # Initialize tracking
    all_coop_history = {agent.agent_id: [] for agent in agents}
    score_history = {agent.agent_id: [] for agent in agents}
    cumulative_scores = {agent.agent_id: 0 for agent in agents}

    for round_num in range(num_rounds):
        round_payoffs = {agent.agent_id: 0 for agent in agents}
        round_moves = {agent.agent_id: {} for agent in agents}

        # All pairs play one round
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                if hasattr(agent1, 'choose_action_pairwise'):
                    move1, _ = agent1.choose_action_pairwise(agent2.agent_id, round_num)
                else:
                    move1 = agent1.choose_pairwise_action(agent2.agent_id)
                    
                if hasattr(agent2, 'choose_action_pairwise'):
                    move2, _ = agent2.choose_action_pairwise(agent1.agent_id, round_num)
                else:
                    move2 = agent2.choose_pairwise_action(agent1.agent_id)

                # Record moves
                round_moves[agent1.agent_id][agent2.agent_id] = move1
                round_moves[agent2.agent_id][agent1.agent_id] = move2

                # Update opponent history for static agents
                if hasattr(agent1, 'opponent_last_moves'):
                    agent1.opponent_last_moves[agent2.agent_id] = move2
                if hasattr(agent2, 'opponent_last_moves'):
                    agent2.opponent_last_moves[agent1.agent_id] = move1

                # Calculate payoffs
                payoff1, payoff2 = PAYOFFS_2P[(move1, move2)]
                round_payoffs[agent1.agent_id] += payoff1
                round_payoffs[agent2.agent_id] += payoff2

                # Update Q-values for Enhanced QL agents
                if hasattr(agent1, 'strategy_name') and agent1.strategy_name == "EnhancedQLearning":
                    agent1.record_interaction(agent2.agent_id, move2, payoff1, move1, move1, round_num)
                elif hasattr(agent1, 'record_interaction'):
                    agent1.record_interaction(agent2.agent_id, move2, payoff1, move1, move1, round_num)
                    
                if hasattr(agent2, 'strategy_name') and agent2.strategy_name == "EnhancedQLearning":
                    agent2.record_interaction(agent1.agent_id, move1, payoff2, move2, move2, round_num)
                elif hasattr(agent2, 'record_interaction'):
                    agent2.record_interaction(agent1.agent_id, move1, payoff2, move2, move2, round_num)

        # Update scores and track history
        for agent in agents:
            cumulative_scores[agent.agent_id] += round_payoffs[agent.agent_id]
            score_history[agent.agent_id].append(cumulative_scores[agent.agent_id])
            
            # Calculate cooperation rate
            if len(agents) > 1:
                coop_count = sum(1 for move in round_moves[agent.agent_id].values() if move == COOPERATE)
                coop_rate = coop_count / (len(agents) - 1)
                all_coop_history[agent.agent_id].append(coop_rate)

    return all_coop_history, score_history


def run_nperson_simulation_extended(agents, num_rounds):
    """Runs a single N-Person simulation with enhanced Q-learning support."""
    all_coop_history = {agent.agent_id: [] for agent in agents}
    score_history = {agent.agent_id: [] for agent in agents}
    cumulative_scores = {agent.agent_id: 0 for agent in agents}

    prev_round_coop_ratio = None
    num_total_agents = len(agents)

    for round_num in range(num_rounds):
        moves = {}
        for agent in agents:
            if hasattr(agent, 'choose_action'):
                intended_move, actual_move = agent.choose_action(prev_round_coop_ratio, round_num)
                moves[agent.agent_id] = actual_move
            else:
                moves[agent.agent_id] = agent.choose_nperson_action(prev_round_coop_ratio)

        num_cooperators = list(moves.values()).count(COOPERATE)
        current_coop_ratio = num_cooperators / num_total_agents if num_total_agents > 0 else 0

        # Calculate payoffs for each agent
        for agent in agents:
            my_move = moves[agent.agent_id]
            num_other_cooperators = num_cooperators - (1 if my_move == COOPERATE else 0)
            payoff = nperson_payoff(my_move, num_other_cooperators, num_total_agents)
            cumulative_scores[agent.agent_id] += payoff
            score_history[agent.agent_id].append(cumulative_scores[agent.agent_id])
            
            # Track individual cooperation
            all_coop_history[agent.agent_id].append(1 if my_move == COOPERATE else 0)
            
            # Update Q-values for Enhanced QL agents
            if hasattr(agent, 'strategy_name') and agent.strategy_name == "EnhancedQLearning":
                agent.record_round_outcome(my_move, payoff, current_coop_ratio)
            elif hasattr(agent, 'record_round_outcome'):
                agent.record_round_outcome(my_move, payoff)

        # Update state for next round
        prev_round_coop_ratio = current_coop_ratio

    return all_coop_history, score_history


# --- Multiple Run Support ---
def run_multiple_simulations_extended(simulation_func, agents, num_rounds, num_runs=100,
                                    training_rounds=1000):
    """Run multiple simulations with enhanced Q-learning training phase."""
    all_coop_runs = {agent.agent_id: [] for agent in agents}
    all_score_runs = {agent.agent_id: [] for agent in agents}
    
    for run in range(num_runs):
        # Create fresh agents for each run
        fresh_agents = []
        for agent in agents:
            if agent.strategy_name == "EnhancedQLearning":
                fresh_agents.append(EnhancedQLearningAgent(
                    agent_id=agent.agent_id,
                    learning_rate=0.1,
                    discount_factor=0.9,
                    epsilon=0.3,  # Start with higher exploration
                    epsilon_decay=0.999,
                    epsilon_min=0.01,
                    state_type="basic",  # Use basic state by default to match SimpleQLearning
                    exploration_rate=0.0  # No additional exploration noise
                ))
            else:
                fresh_agents.append(StaticAgent(
                    agent_id=agent.agent_id,
                    strategy_name=agent.strategy_name,
                    exploration_rate=getattr(agent, 'exploration_rate', 0.0)
                ))
        
        # Training phase for Q-learning agents
        if any(agent.strategy_name == "EnhancedQLearning" for agent in fresh_agents):
            for _ in range(training_rounds // num_rounds):
                simulation_func(fresh_agents, num_rounds)
                for agent in fresh_agents:
                    if agent.strategy_name == "EnhancedQLearning":
                        agent.decay_epsilon()
        
        # Main simulation run
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
            
            # Add Savitzky-Golay filter smoothing if scipy is available
            try:
                from scipy.signal import savgol_filter
                # Use window size that's odd and less than data length
                savgol_window = min(smoothing_window * 2 + 1, len(avg_mean) - 1)
                if savgol_window % 2 == 0:
                    savgol_window -= 1
                if savgol_window >= 5:  # Minimum window size for savgol
                    smoothed_savgol = savgol_filter(avg_mean, savgol_window, 3)
                    ax.plot(rounds, smoothed_savgol, color=colors.get(agent_type, 'black'),
                           linewidth=2, linestyle='-.', alpha=0.8)
            except ImportError:
                pass
        
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
            
            # Add Savitzky-Golay filter smoothing if scipy is available
            try:
                from scipy.signal import savgol_filter
                # Use window size that's odd and less than data length
                savgol_window = min(smoothing_window * 2 + 1, len(avg_mean) - 1)
                if savgol_window % 2 == 0:
                    savgol_window -= 1
                if savgol_window >= 5:  # Minimum window size for savgol
                    smoothed_savgol = savgol_filter(avg_mean, savgol_window, 3)
                    ax.plot(rounds, smoothed_savgol, color=colors.get(agent_type, 'black'),
                           linewidth=2, linestyle='-.', alpha=0.8)
            except ImportError:
                pass
        
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
    results_dir = "enhanced_qlearning_results"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {os.path.abspath(results_dir)}")
    print(f"Running {NUM_RUNS} simulations per experiment with {TRAINING_ROUNDS} training rounds...")
    print("Using Enhanced Q-Learning with optimistic initialization and proportion_discretized states")
    
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
        coop_runs, score_runs = run_multiple_simulations_extended(
            run_pairwise_simulation_extended, agent_list, NUM_ROUNDS, NUM_RUNS, TRAINING_ROUNDS)
        
        pairwise_2ql_coop[name] = aggregate_agent_data(coop_runs)
        pairwise_2ql_scores[name] = aggregate_agent_data(score_runs)
    
    # Run N-person simulations
    print("\nRunning 2 Enhanced QL Neighbourhood simulations...")
    nperson_2ql_coop = {}
    nperson_2ql_scores = {}
    
    for name, agent_list in experiments_2ql.items():
        print(f"  - Running {NUM_RUNS} simulations for: {name}")
        coop_runs, score_runs = run_multiple_simulations_extended(
            run_nperson_simulation_extended, agent_list, NUM_ROUNDS, NUM_RUNS, TRAINING_ROUNDS)
        
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
        coop_runs, score_runs = run_multiple_simulations_extended(
            run_pairwise_simulation_extended, agent_list, NUM_ROUNDS, NUM_RUNS, TRAINING_ROUNDS)
        
        pairwise_1ql_coop[name] = aggregate_agent_data(coop_runs)
        pairwise_1ql_scores[name] = aggregate_agent_data(score_runs)
    
    # Run N-person simulations
    print("\nRunning 1 Enhanced QL Neighbourhood simulations...")
    nperson_1ql_coop = {}
    nperson_1ql_scores = {}
    
    for name, agent_list in experiments_1ql.items():
        print(f"  - Running {NUM_RUNS} simulations for: {name}")
        coop_runs, score_runs = run_multiple_simulations_extended(
            run_nperson_simulation_extended, agent_list, NUM_ROUNDS, NUM_RUNS, TRAINING_ROUNDS)
        
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