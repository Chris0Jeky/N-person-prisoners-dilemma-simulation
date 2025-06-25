import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from datetime import datetime

# Static Figure Generator for N-Person Prisoner's Dilemma
# Modified version: Only generates 2TFT-E with AllC and 2TFT-E with AllD figures

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


# --- Agent Class ---
class StaticAgent:
    def __init__(self, agent_id, strategy_name, exploration_rate=0.0):
        self.agent_id = agent_id
        self.strategy_name = strategy_name
        self.exploration_rate = exploration_rate
        # For Pairwise TFT
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

        # Apply exploration for TFT-E
        if self.strategy_name == "TFT-E" and random.random() < self.exploration_rate:
            return 1 - intended_move
        return intended_move

    def choose_nperson_action(self, prev_round_group_coop_ratio):
        """Choose action for an N-player group game."""
        intended_move = None
        if self.strategy_name in ["TFT", "TFT-E"]:
            # This is the pTFT logic
            if prev_round_group_coop_ratio is None:  # First round
                intended_move = COOPERATE
            else:
                intended_move = COOPERATE if random.random() < prev_round_group_coop_ratio else DEFECT
        elif self.strategy_name == "AllC":
            intended_move = COOPERATE
        elif self.strategy_name == "AllD":
            intended_move = DEFECT

        # Apply exploration for TFT-E
        if self.strategy_name == "TFT-E" and random.random() < self.exploration_rate:
            return 1 - intended_move
        return intended_move

    def reset(self):
        self.opponent_last_moves = {}


# --- Simulation Functions ---

def run_pairwise_simulation_extended(agents, num_rounds):
    """Runs a single pairwise tournament and logs cooperation and scores for all agents."""
    for agent in agents: agent.reset()

    # Initialize tracking
    tft_agents = [agent for agent in agents if "TFT" in agent.strategy_name]
    tft_coop_history = []
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
                move1 = agent1.choose_pairwise_action(agent2.agent_id)
                move2 = agent2.choose_pairwise_action(agent1.agent_id)

                # Record moves for cooperation tracking
                round_moves[agent1.agent_id][agent2.agent_id] = move1
                round_moves[agent2.agent_id][agent1.agent_id] = move2

                # Record opponent's move for next round's TFT decision
                agent1.opponent_last_moves[agent2.agent_id] = move2
                agent2.opponent_last_moves[agent1.agent_id] = move1

                payoff1, payoff2 = PAYOFFS_2P[(move1, move2)]
                round_payoffs[agent1.agent_id] += payoff1
                round_payoffs[agent2.agent_id] += payoff2

        # Update cumulative scores and track history
        for agent in agents:
            cumulative_scores[agent.agent_id] += round_payoffs[agent.agent_id]
            score_history[agent.agent_id].append(cumulative_scores[agent.agent_id])
            
            # Calculate cooperation rate for this agent
            if len(agents) > 1:
                coop_count = sum(1 for move in round_moves[agent.agent_id].values() if move == COOPERATE)
                coop_rate = coop_count / (len(agents) - 1)
                all_coop_history[agent.agent_id].append(coop_rate)

        # Log the cooperation rate of TFT agents in this round
        if tft_agents:
            tft_coop_count = 0
            for tft_agent in tft_agents:
                for opponent in agents:
                    if tft_agent.agent_id != opponent.agent_id:
                        if round_moves[tft_agent.agent_id][opponent.agent_id] == COOPERATE:
                            tft_coop_count += 1
            avg_coop_rate = tft_coop_count / (len(tft_agents) * (len(agents) - 1))
            tft_coop_history.append(avg_coop_rate)

    return tft_coop_history, all_coop_history, score_history


def run_pairwise_simulation(agents, num_rounds):
    """Backward compatible wrapper that returns only TFT cooperation."""
    tft_coop_history, _, _ = run_pairwise_simulation_extended(agents, num_rounds)
    return tft_coop_history


def run_nperson_simulation_extended(agents, num_rounds):
    """Runs a single N-Person simulation and logs cooperation and scores for all agents."""
    tft_agents = [agent for agent in agents if "TFT" in agent.strategy_name]
    tft_coop_history = []
    all_coop_history = {agent.agent_id: [] for agent in agents}
    score_history = {agent.agent_id: [] for agent in agents}
    cumulative_scores = {agent.agent_id: 0 for agent in agents}

    prev_round_coop_ratio = None
    num_total_agents = len(agents)

    for round_num in range(num_rounds):
        moves = {agent.agent_id: agent.choose_nperson_action(prev_round_coop_ratio) for agent in agents}

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

        # Log the cooperation rate of TFT agents in this round
        if tft_agents:
            tft_coop_count = sum(1 for agent in tft_agents if moves[agent.agent_id] == COOPERATE)
            avg_coop_rate = tft_coop_count / len(tft_agents) if tft_agents else 0
            tft_coop_history.append(avg_coop_rate)

        # Update state for next round
        prev_round_coop_ratio = current_coop_ratio

    return tft_coop_history, all_coop_history, score_history


def run_nperson_simulation(agents, num_rounds):
    """Backward compatible wrapper that returns only TFT cooperation."""
    tft_coop_history, _, _ = run_nperson_simulation_extended(agents, num_rounds)
    return tft_coop_history


# --- Multiple Run Support ---

def run_multiple_simulations(simulation_func, agents, num_rounds, num_runs=15):
    """Run multiple simulations and collect all results."""
    all_runs = []
    for run in range(num_runs):
        # Create fresh agents for each run to avoid state carryover
        fresh_agents = []
        for agent in agents:
            fresh_agents.append(StaticAgent(
                agent_id=agent.agent_id,
                strategy_name=agent.strategy_name,
                exploration_rate=agent.exploration_rate
            ))
        
        run_history = simulation_func(fresh_agents, num_rounds)
        all_runs.append(run_history)
    
    return all_runs


def run_multiple_simulations_extended(simulation_func, agents, num_rounds, num_runs=15):
    """Run multiple simulations and collect extended results (TFT coop, all coop, scores)."""
    all_tft_runs = []
    all_coop_runs = {agent.agent_id: [] for agent in agents}
    all_score_runs = {agent.agent_id: [] for agent in agents}
    
    for run in range(num_runs):
        # Create fresh agents for each run
        fresh_agents = []
        for agent in agents:
            fresh_agents.append(StaticAgent(
                agent_id=agent.agent_id,
                strategy_name=agent.strategy_name,
                exploration_rate=agent.exploration_rate
            ))
        
        tft_history, coop_history, score_history = simulation_func(fresh_agents, num_rounds)
        all_tft_runs.append(tft_history)
        
        for agent_id in coop_history:
            all_coop_runs[agent_id].append(coop_history[agent_id])
            all_score_runs[agent_id].append(score_history[agent_id])
    
    return all_tft_runs, all_coop_runs, all_score_runs


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


def aggregate_results(all_runs):
    """Aggregate results from multiple runs into statistics."""
    # Convert to numpy array for easier computation
    runs_array = np.array(all_runs)
    
    # Calculate statistics for each round
    mean_values = np.mean(runs_array, axis=0)
    std_values = np.std(runs_array, axis=0)
    
    # Calculate 95% confidence intervals
    n_runs = len(all_runs)
    sem = std_values / np.sqrt(n_runs)  # Standard error of the mean
    ci_95 = 1.96 * sem  # 95% confidence interval
    
    lower_bound = mean_values - ci_95
    upper_bound = mean_values + ci_95
    
    return {
        'mean': mean_values,
        'std': std_values,
        'lower_95': lower_bound,
        'upper_95': upper_bound,
        'all_runs': all_runs
    }


# --- Part 2: Experiment Setup and Plotting ---

def setup_experiments():
    """Defines only the two agent compositions: 2 TFT-E + 1 AllC and 2 TFT-E + 1 AllD."""
    return {
        "2 TFT-E + 1 AllC": [
            StaticAgent(agent_id="TFT-E_1", strategy_name="TFT-E", exploration_rate=0.1),
            StaticAgent(agent_id="TFT-E_2", strategy_name="TFT-E", exploration_rate=0.1),
            StaticAgent(agent_id="AllC_1", strategy_name="AllC"),
        ],
        "2 TFT-E + 1 AllD": [
            StaticAgent(agent_id="TFT-E_1", strategy_name="TFT-E", exploration_rate=0.1),
            StaticAgent(agent_id="TFT-E_2", strategy_name="TFT-E", exploration_rate=0.1),
            StaticAgent(agent_id="AllD_1", strategy_name="AllD"),
        ],
    }


def save_aggregated_data_to_csv(data, filename_prefix, results_dir):
    """Saves the aggregated simulation data to CSV files with statistics."""
    # Save individual experiment files with statistics
    for exp_name, stats in data.items():
        # Create a DataFrame with statistics
        # Build all columns at once to avoid fragmentation
        columns = {
            'Round': range(1, len(stats['mean']) + 1),
            'Mean_Cooperation_Rate': stats['mean'],
            'Std_Dev': stats['std'],
            'Lower_95_CI': stats['lower_95'],
            'Upper_95_CI': stats['upper_95']
        }
        
        # Add individual run columns to the dict
        for i, run in enumerate(stats['all_runs']):
            columns[f'Run_{i+1}'] = run
        
        # Create DataFrame with all columns at once
        df = pd.DataFrame(columns)

        # Clean experiment name for filename
        clean_name = exp_name.replace(' ', '_').replace('+', '_plus_')
        filename = f"{filename_prefix}_{clean_name}_aggregated.csv"
        filepath = os.path.join(results_dir, filename)

        df.to_csv(filepath, index=False)
        print(f"  - Saved: {filename}")

    # Also save a combined summary file with just means
    num_rounds = len(next(iter(data.values()))['mean'])
    # Build all columns at once to avoid fragmentation
    combined_columns = {'Round': range(1, num_rounds + 1)}
    
    for exp_name, stats in data.items():
        clean_name = exp_name.replace(' ', '_').replace('+', '_plus_')
        combined_columns[f'{clean_name}_mean'] = stats['mean']
        combined_columns[f'{clean_name}_std'] = stats['std']
    
    # Create DataFrame with all columns at once
    combined_df = pd.DataFrame(combined_columns)

    combined_filename = f"{filename_prefix}_all_experiments_summary.csv"
    combined_filepath = os.path.join(results_dir, combined_filename)
    combined_df.to_csv(combined_filepath, index=False)
    print(f"  - Saved combined summary: {combined_filename}")


def plot_aggregated_results(data, title, smoothing_window=15, save_path=None):
    """Creates a 1x2 grid of plots showing mean and confidence intervals for only 2 experiments."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle(title + " (15 runs with 95% CI)", fontsize=16, weight='bold')

    axes_flat = axes if isinstance(axes, np.ndarray) else [axes]
    rounds = None
    
    for i, (exp_name, stats) in enumerate(data.items()):
        ax = axes_flat[i]
        
        if rounds is None:
            rounds = range(1, len(stats['mean']) + 1)
        
        # Plot confidence interval as shaded region
        ax.fill_between(rounds, stats['lower_95'], stats['upper_95'], 
                       alpha=0.3, color='blue', label='95% CI')
        
        # Plot mean
        ax.plot(rounds, stats['mean'], color='blue', linewidth=2.5, 
               label='Mean', marker='o', markersize=2)
        
        # Plot smoothed mean using rolling average
        smoothed_mean = pd.Series(stats['mean']).rolling(
            window=smoothing_window, min_periods=1, center=True).mean()
        ax.plot(rounds, smoothed_mean, color='darkblue', linewidth=2, 
               linestyle='--', label=f'Smoothed (window={smoothing_window})')
        
        # Add Savitzky-Golay filter smoothing if scipy is available
        try:
            from scipy.signal import savgol_filter
            # Use window size that's odd and less than data length
            savgol_window = min(smoothing_window * 2 + 1, len(stats['mean']) - 1)
            if savgol_window % 2 == 0:
                savgol_window -= 1
            if savgol_window >= 5:  # Minimum window size for savgol
                smoothed_savgol = savgol_filter(stats['mean'], savgol_window, 3)
                ax.plot(rounds, smoothed_savgol, color='darkgreen', linewidth=2, 
                       linestyle='-.', label=f'Savitzky-Golay (window={savgol_window})')
        except ImportError:
            pass
        
        # Add individual run traces (faint)
        for run in stats['all_runs'][:5]:  # Show first 5 runs as examples
            ax.plot(rounds, run, alpha=0.1, color='gray', linewidth=0.5)

        ax.set_title(exp_name, fontsize=12)
        ax.set_xlabel("Round")
        ax.set_ylabel("Avg. TFT/TFT-E Cooperation Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved figure: {os.path.basename(save_path)}")

    plt.show()


def plot_agent_scores(score_data, title, save_path=None):
    """Creates plots showing cumulative scores for each agent type over time."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle(title + " - Agent Scores (15 run average)", fontsize=16, weight='bold')
    
    axes_flat = axes if isinstance(axes, np.ndarray) else [axes]
    
    for i, (exp_name, exp_data) in enumerate(score_data.items()):
        ax = axes_flat[i]
        
        # Group scores by agent type
        agent_type_scores = {}
        for agent_id, scores in exp_data.items():
            # Extract agent type from ID (e.g., "TFT_1" -> "TFT")
            agent_type = agent_id.split('_')[0]
            if agent_type not in agent_type_scores:
                agent_type_scores[agent_type] = []
            agent_type_scores[agent_type].append(scores['mean'])
        
        # Plot each agent type
        colors = {'TFT': 'blue', 'TFT-E': 'green', 'AllC': 'orange', 'AllD': 'red'}
        for agent_type, scores_list in agent_type_scores.items():
            # Average across agents of the same type
            avg_scores = np.mean(scores_list, axis=0)
            rounds = range(1, len(avg_scores) + 1)
            
            ax.plot(rounds, avg_scores, color=colors.get(agent_type, 'black'), 
                   linewidth=2.5, label=agent_type, marker='o', markersize=2)
        
        ax.set_title(exp_name, fontsize=12)
        ax.set_xlabel("Round")
        ax.set_ylabel("Cumulative Score")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved figure: {os.path.basename(save_path)}")
    
    plt.show()


def plot_all_agent_cooperation(coop_data, title, save_path=None):
    """Creates plots showing cooperation rates for all agent types."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle(title + " - All Agent Cooperation Rates (15 run average)", fontsize=16, weight='bold')
    
    axes_flat = axes if isinstance(axes, np.ndarray) else [axes]
    
    for i, (exp_name, exp_data) in enumerate(coop_data.items()):
        ax = axes_flat[i]
        
        # Group cooperation by agent type
        agent_type_coop = {}
        for agent_id, coop in exp_data.items():
            # Extract agent type from ID
            agent_type = agent_id.split('_')[0]
            if agent_type not in agent_type_coop:
                agent_type_coop[agent_type] = []
            agent_type_coop[agent_type].append(coop['mean'])
        
        # Plot each agent type
        colors = {'TFT': 'blue', 'TFT-E': 'green', 'AllC': 'orange', 'AllD': 'red'}
        for agent_type, coop_list in agent_type_coop.items():
            # Average across agents of the same type
            avg_coop = np.mean(coop_list, axis=0)
            rounds = range(1, len(avg_coop) + 1)
            
            # Also calculate and show confidence intervals
            std_coop = np.std(coop_list, axis=0)
            n_agents = len(coop_list)
            sem = std_coop / np.sqrt(n_agents) if n_agents > 0 else 0
            ci_95 = 1.96 * sem
            
            color = colors.get(agent_type, 'black')
            ax.fill_between(rounds, avg_coop - ci_95, avg_coop + ci_95, 
                           alpha=0.2, color=color)
            ax.plot(rounds, avg_coop, color=color, linewidth=2.5, 
                   label=agent_type, marker='o', markersize=2)
        
        ax.set_title(exp_name, fontsize=12)
        ax.set_xlabel("Round")
        ax.set_ylabel("Cooperation Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved figure: {os.path.basename(save_path)}")
    
    plt.show()


# --- Part 3: Main Execution ---

if __name__ == "__main__":
    NUM_ROUNDS = 200
    NUM_RUNS = 50
    experiments = setup_experiments()

    # --- Create results directory ---
    results_dir = "results_2tfte_only"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {os.path.abspath(results_dir)}")
    print(f"Running {NUM_RUNS} simulations per experiment for statistical aggregation...")

    # --- Run Extended Pairwise Simulations ---
    print(f"\nRunning {NUM_RUNS} Extended Pairwise simulations for each experiment...")
    pairwise_tft_aggregated = {}
    pairwise_all_coop_aggregated = {}
    pairwise_scores_aggregated = {}
    
    for name, agent_list in experiments.items():
        print(f"  - Running {NUM_RUNS} simulations for: {name}")
        tft_runs, coop_runs, score_runs = run_multiple_simulations_extended(
            run_pairwise_simulation_extended, agent_list, NUM_ROUNDS, NUM_RUNS)
        
        pairwise_tft_aggregated[name] = aggregate_results(tft_runs)
        pairwise_all_coop_aggregated[name] = aggregate_agent_data(coop_runs)
        pairwise_scores_aggregated[name] = aggregate_agent_data(score_runs)
        
        print(f"    Mean TFT cooperation rate: {np.mean(pairwise_tft_aggregated[name]['mean']):.3f} ± {np.mean(pairwise_tft_aggregated[name]['std']):.3f}")

    # --- Run Extended Neighbourhood Simulations ---
    print(f"\nRunning {NUM_RUNS} Extended Neighbourhood simulations for each experiment...")
    neighbourhood_tft_aggregated = {}
    neighbourhood_all_coop_aggregated = {}
    neighbourhood_scores_aggregated = {}
    
    for name, agent_list in experiments.items():
        print(f"  - Running {NUM_RUNS} simulations for: {name}")
        tft_runs, coop_runs, score_runs = run_multiple_simulations_extended(
            run_nperson_simulation_extended, agent_list, NUM_ROUNDS, NUM_RUNS)
        
        neighbourhood_tft_aggregated[name] = aggregate_results(tft_runs)
        neighbourhood_all_coop_aggregated[name] = aggregate_agent_data(coop_runs)
        neighbourhood_scores_aggregated[name] = aggregate_agent_data(score_runs)
        
        print(f"    Mean TFT cooperation rate: {np.mean(neighbourhood_tft_aggregated[name]['mean']):.3f} ± {np.mean(neighbourhood_tft_aggregated[name]['std']):.3f}")

    # --- Save Aggregated Data to CSV ---
    print("\nSaving aggregated data to CSV files...")
    save_aggregated_data_to_csv(pairwise_tft_aggregated, "pairwise_tft_cooperation", results_dir)
    save_aggregated_data_to_csv(neighbourhood_tft_aggregated, "neighbourhood_tft_cooperation", results_dir)

    # --- Generate All Plots ---
    print("\nGenerating and saving all plots...")
    
    # Original TFT cooperation plots
    plot_aggregated_results(pairwise_tft_aggregated,
                           title="Figure 1: TFT Cooperation Dynamics with Pairwise Voting",
                           save_path=os.path.join(results_dir, "figure1_pairwise_tft_cooperation.png"))
    plot_aggregated_results(neighbourhood_tft_aggregated,
                           title="Figure 2: TFT Cooperation Dynamics with Neighbourhood Voting",
                           save_path=os.path.join(results_dir, "figure2_neighbourhood_tft_cooperation.png"))
    
    # Agent scores plots
    plot_agent_scores(pairwise_scores_aggregated,
                     title="Figure 3: Pairwise Agent Scores",
                     save_path=os.path.join(results_dir, "figure3_pairwise_agent_scores.png"))
    plot_agent_scores(neighbourhood_scores_aggregated,
                     title="Figure 4: Neighbourhood Agent Scores",
                     save_path=os.path.join(results_dir, "figure4_neighbourhood_agent_scores.png"))
    
    # All agent cooperation plots
    plot_all_agent_cooperation(pairwise_all_coop_aggregated,
                              title="Figure 5: Pairwise All Agents",
                              save_path=os.path.join(results_dir, "figure5_pairwise_all_cooperation.png"))
    plot_all_agent_cooperation(neighbourhood_all_coop_aggregated,
                              title="Figure 6: Neighbourhood All Agents",
                              save_path=os.path.join(results_dir, "figure6_neighbourhood_all_cooperation.png"))

    print(f"\nDone! All results from {NUM_RUNS} runs saved to the '{results_dir}' directory.")