import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from datetime import datetime

# Import Q-learning agents and other necessary components
from qlearning_agents import SimpleQLearningAgent, NPDLQLearningAgent
from enhanced_qlearning_agents import EnhancedQLearningAgent
from extended_agents import QLearningNPersonWrapper, QLearningPairwiseWrapper

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


# --- Agent Classes for Static Strategies ---
class StaticAgent:
    def __init__(self, agent_id, strategy_name):
        self.agent_id = agent_id
        self.strategy_name = strategy_name
        self.opponent_last_moves = {}

    def choose_pairwise_action(self, opponent_id):
        """Choose action for a 2-player game."""
        if self.strategy_name == "TFT":
            return self.opponent_last_moves.get(opponent_id, COOPERATE)
        elif self.strategy_name == "AllC":
            return COOPERATE
        elif self.strategy_name == "AllD":
            return DEFECT

    def choose_nperson_action(self, prev_round_group_coop_ratio):
        """Choose action for an N-player group game."""
        if self.strategy_name == "TFT":
            # Probabilistic TFT (pTFT)
            if prev_round_group_coop_ratio is None:  # First round
                return COOPERATE
            else:
                return COOPERATE if random.random() < prev_round_group_coop_ratio else DEFECT
        elif self.strategy_name == "AllC":
            return COOPERATE
        elif self.strategy_name == "AllD":
            return DEFECT

    def reset(self):
        self.opponent_last_moves = {}

    def update_pairwise(self, opponent_id, my_move, opponent_move, payoff):
        """Update state after a pairwise interaction."""
        self.opponent_last_moves[opponent_id] = opponent_move

    def update_nperson(self, round_data):
        """Update state after an N-person round."""
        pass  # Static agents don't need to update


# --- Simulation Functions ---
def run_pairwise_simulation_extended(agents, num_rounds, training_rounds=100):
    """Runs a pairwise tournament with Q-learning agents and tracks all metrics."""
    # Reset all agents
    for agent in agents:
        if hasattr(agent, 'reset'):
            agent.reset()
    
    # Initialize tracking
    ql_agents = [agent for agent in agents if 'QL' in agent.agent_id or 'Q-Learning' in str(type(agent))]
    ql_coop_history = []
    all_coop_history = {agent.agent_id: [] for agent in agents}
    score_history = {agent.agent_id: [] for agent in agents}
    cumulative_scores = {agent.agent_id: 0 for agent in agents}
    
    total_rounds = training_rounds + num_rounds
    
    for round_num in range(total_rounds):
        round_payoffs = {agent.agent_id: 0 for agent in agents}
        round_moves = {agent.agent_id: {} for agent in agents}
        
        # All pairs play one round
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                
                # Get actions
                if hasattr(agent1, 'choose_pairwise_action'):
                    move1 = agent1.choose_pairwise_action(agent2.agent_id)
                else:
                    # For wrapped Q-learning agents
                    move1 = agent1.get_action(agent2.agent_id)
                    
                if hasattr(agent2, 'choose_pairwise_action'):
                    move2 = agent2.choose_pairwise_action(agent1.agent_id)
                else:
                    move2 = agent2.get_action(agent1.agent_id)
                
                # Record moves
                round_moves[agent1.agent_id][agent2.agent_id] = move1
                round_moves[agent2.agent_id][agent1.agent_id] = move2
                
                # Calculate payoffs
                payoff1, payoff2 = PAYOFFS_2P[(move1, move2)]
                round_payoffs[agent1.agent_id] += payoff1
                round_payoffs[agent2.agent_id] += payoff2
                
                # Update agents
                if hasattr(agent1, 'update_pairwise'):
                    agent1.update_pairwise(agent2.agent_id, move1, move2, payoff1)
                elif hasattr(agent1, 'update'):
                    agent1.update(agent2.agent_id, move1, move2, payoff1)
                    
                if hasattr(agent2, 'update_pairwise'):
                    agent2.update_pairwise(agent1.agent_id, move2, move1, payoff2)
                elif hasattr(agent2, 'update'):
                    agent2.update(agent1.agent_id, move2, move1, payoff2)
        
        # Update cumulative scores and track history (only after training)
        if round_num >= training_rounds:
            for agent in agents:
                cumulative_scores[agent.agent_id] += round_payoffs[agent.agent_id]
                score_history[agent.agent_id].append(cumulative_scores[agent.agent_id])
                
                # Calculate cooperation rate
                if len(agents) > 1:
                    coop_count = sum(1 for move in round_moves[agent.agent_id].values() if move == COOPERATE)
                    coop_rate = coop_count / (len(agents) - 1)
                    all_coop_history[agent.agent_id].append(coop_rate)
            
            # Track Q-learning agent cooperation
            if ql_agents:
                ql_coop_count = 0
                ql_interactions = 0
                for ql_agent in ql_agents:
                    for move in round_moves[ql_agent.agent_id].values():
                        if move == COOPERATE:
                            ql_coop_count += 1
                        ql_interactions += 1
                avg_coop_rate = ql_coop_count / ql_interactions if ql_interactions > 0 else 0
                ql_coop_history.append(avg_coop_rate)
    
    return ql_coop_history, all_coop_history, score_history


def run_nperson_simulation_extended(agents, num_rounds, training_rounds=100):
    """Runs an N-person simulation with Q-learning agents and tracks all metrics."""
    # Reset all agents
    for agent in agents:
        if hasattr(agent, 'reset'):
            agent.reset()
    
    # Initialize tracking
    ql_agents = [agent for agent in agents if 'QL' in agent.agent_id or 'Q-Learning' in str(type(agent))]
    ql_coop_history = []
    all_coop_history = {agent.agent_id: [] for agent in agents}
    score_history = {agent.agent_id: [] for agent in agents}
    cumulative_scores = {agent.agent_id: 0 for agent in agents}
    
    prev_round_coop_ratio = None
    num_total_agents = len(agents)
    total_rounds = training_rounds + num_rounds
    
    for round_num in range(total_rounds):
        # Get moves from all agents
        moves = {}
        for agent in agents:
            if hasattr(agent, 'choose_nperson_action'):
                moves[agent.agent_id] = agent.choose_nperson_action(prev_round_coop_ratio)
            else:
                # For wrapped Q-learning agents
                moves[agent.agent_id] = agent.get_action('group', prev_round_coop_ratio)
        
        num_cooperators = list(moves.values()).count(COOPERATE)
        current_coop_ratio = num_cooperators / num_total_agents if num_total_agents > 0 else 0
        
        # Calculate payoffs
        round_payoffs = {}
        for agent in agents:
            my_move = moves[agent.agent_id]
            num_other_cooperators = num_cooperators - (1 if my_move == COOPERATE else 0)
            payoff = nperson_payoff(my_move, num_other_cooperators, num_total_agents)
            round_payoffs[agent.agent_id] = payoff
            
            # Update agent
            if hasattr(agent, 'update_nperson'):
                round_data = {
                    'my_move': my_move,
                    'group_coop_ratio': current_coop_ratio,
                    'payoff': payoff
                }
                agent.update_nperson(round_data)
            elif hasattr(agent, 'update'):
                agent.update('group', my_move, current_coop_ratio, payoff)
        
        # Track metrics (only after training)
        if round_num >= training_rounds:
            for agent in agents:
                cumulative_scores[agent.agent_id] += round_payoffs[agent.agent_id]
                score_history[agent.agent_id].append(cumulative_scores[agent.agent_id])
                
                # Track individual cooperation
                all_coop_history[agent.agent_id].append(1 if moves[agent.agent_id] == COOPERATE else 0)
            
            # Track Q-learning cooperation
            if ql_agents:
                ql_coop_count = sum(1 for agent in ql_agents if moves[agent.agent_id] == COOPERATE)
                avg_coop_rate = ql_coop_count / len(ql_agents) if ql_agents else 0
                ql_coop_history.append(avg_coop_rate)
        
        # Update state for next round
        prev_round_coop_ratio = current_coop_ratio
    
    return ql_coop_history, all_coop_history, score_history


# --- Multiple Run Support ---
def run_multiple_simulations_extended(simulation_func, agents, num_rounds, num_runs=20, training_rounds=100):
    """Run multiple simulations and collect extended results."""
    all_ql_runs = []
    all_coop_runs = {agent.agent_id: [] for agent in agents}
    all_score_runs = {agent.agent_id: [] for agent in agents}
    
    for run in range(num_runs):
        print(f"    Run {run + 1}/{num_runs}...", end='\r')
        
        # Create fresh agents for each run
        fresh_agents = []
        for agent in agents:
            if isinstance(agent, StaticAgent):
                fresh_agents.append(StaticAgent(agent.agent_id, agent.strategy_name))
            elif 'SimpleQLearning' in str(type(agent)):
                # Create new simple Q-learning agent
                fresh_agents.append(SimpleQLearningAgent(agent.agent_id))
            elif 'EnhancedQLearning' in str(type(agent)):
                # Create new enhanced Q-learning agent with best config
                fresh_agents.append(EnhancedQLearningAgent(
                    agent.agent_id,
                    epsilon_decay=0.995,
                    exclude_self=True,
                    epsilon_min=0.01
                ))
            else:
                # For other agent types, try to create a fresh copy
                fresh_agents.append(agent.__class__(agent.agent_id))
        
        ql_history, coop_history, score_history = simulation_func(fresh_agents, num_rounds, training_rounds)
        all_ql_runs.append(ql_history)
        
        for agent_id in coop_history:
            all_coop_runs[agent_id].append(coop_history[agent_id])
            all_score_runs[agent_id].append(score_history[agent_id])
    
    print(f"    Completed {num_runs} runs.          ")
    return all_ql_runs, all_coop_runs, all_score_runs


def aggregate_results(all_runs):
    """Aggregate results from multiple runs into statistics."""
    runs_array = np.array(all_runs)
    mean_values = np.mean(runs_array, axis=0)
    std_values = np.std(runs_array, axis=0)
    
    n_runs = len(all_runs)
    sem = std_values / np.sqrt(n_runs)
    ci_95 = 1.96 * sem
    
    return {
        'mean': mean_values,
        'std': std_values,
        'lower_95': mean_values - ci_95,
        'upper_95': mean_values + ci_95,
        'all_runs': all_runs
    }


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
def setup_qlearning_experiments():
    """Define Q-learning experiments with different opponents."""
    return {
        "2 Simple QL + 1 TFT": [
            SimpleQLearningAgent("SimpleQL_1"),
            SimpleQLearningAgent("SimpleQL_2"),
            StaticAgent("TFT_1", "TFT"),
        ],
        "2 Simple QL + 1 AllD": [
            SimpleQLearningAgent("SimpleQL_1"),
            SimpleQLearningAgent("SimpleQL_2"),
            StaticAgent("AllD_1", "AllD"),
        ],
        "2 Simple QL + 1 AllC": [
            SimpleQLearningAgent("SimpleQL_1"),
            SimpleQLearningAgent("SimpleQL_2"),
            StaticAgent("AllC_1", "AllC"),
        ],
        "2 Enhanced QL + 1 TFT": [
            EnhancedQLearningAgent("EnhQL_1", epsilon_decay=0.995, exclude_self=True, epsilon_min=0.01),
            EnhancedQLearningAgent("EnhQL_2", epsilon_decay=0.995, exclude_self=True, epsilon_min=0.01),
            StaticAgent("TFT_1", "TFT"),
        ],
        "2 Enhanced QL + 1 AllD": [
            EnhancedQLearningAgent("EnhQL_1", epsilon_decay=0.995, exclude_self=True, epsilon_min=0.01),
            EnhancedQLearningAgent("EnhQL_2", epsilon_decay=0.995, exclude_self=True, epsilon_min=0.01),
            StaticAgent("AllD_1", "AllD"),
        ],
        "2 Enhanced QL + 1 AllC": [
            EnhancedQLearningAgent("EnhQL_1", epsilon_decay=0.995, exclude_self=True, epsilon_min=0.01),
            EnhancedQLearningAgent("EnhQL_2", epsilon_decay=0.995, exclude_self=True, epsilon_min=0.01),
            StaticAgent("AllC_1", "AllC"),
        ],
    }


# --- Plotting Functions ---
def save_aggregated_data_to_csv(data, filename_prefix, results_dir):
    """Saves the aggregated simulation data to CSV files with statistics."""
    for exp_name, stats in data.items():
        df = pd.DataFrame({
            'Round': range(1, len(stats['mean']) + 1),
            'Mean_Value': stats['mean'],
            'Std_Dev': stats['std'],
            'Lower_95_CI': stats['lower_95'],
            'Upper_95_CI': stats['upper_95']
        })
        
        # Add individual run columns
        for i, run in enumerate(stats['all_runs']):
            df[f'Run_{i+1}'] = run

        # Clean experiment name for filename
        clean_name = exp_name.replace(' ', '_').replace('+', '_plus_')
        filename = f"{filename_prefix}_{clean_name}_aggregated.csv"
        filepath = os.path.join(results_dir, filename)

        df.to_csv(filepath, index=False)
        print(f"  - Saved: {filename}")

    # Combined summary file
    num_rounds = len(next(iter(data.values()))['mean'])
    combined_df = pd.DataFrame({'Round': range(1, num_rounds + 1)})
    
    for exp_name, stats in data.items():
        clean_name = exp_name.replace(' ', '_').replace('+', '_plus_')
        combined_df[f'{clean_name}_mean'] = stats['mean']
        combined_df[f'{clean_name}_std'] = stats['std']

    combined_filename = f"{filename_prefix}_all_experiments_summary.csv"
    combined_filepath = os.path.join(results_dir, combined_filename)
    combined_df.to_csv(combined_filepath, index=False)
    print(f"  - Saved combined summary: {combined_filename}")


def plot_qlearning_cooperation(data, title, save_path=None):
    """Creates plots showing Q-learning cooperation rates with confidence intervals."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(title + " (20 runs with 95% CI)", fontsize=16, weight='bold')
    
    axes_flat = axes.flatten()
    
    for i, (exp_name, stats) in enumerate(data.items()):
        ax = axes_flat[i]
        rounds = range(1, len(stats['mean']) + 1)
        
        # Plot confidence interval
        ax.fill_between(rounds, stats['lower_95'], stats['upper_95'], 
                       alpha=0.3, color='blue', label='95% CI')
        
        # Plot mean
        ax.plot(rounds, stats['mean'], color='blue', linewidth=2.5, 
               label='Mean', marker='o', markersize=2)
        
        # Plot smoothed mean
        smoothed = pd.Series(stats['mean']).rolling(window=10, min_periods=1, center=True).mean()
        ax.plot(rounds, smoothed, color='darkblue', linewidth=2, 
               linestyle='--', label='Smoothed')
        
        ax.set_title(exp_name, fontsize=12)
        ax.set_xlabel("Round (after training)")
        ax.set_ylabel("Q-Learning Cooperation Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved figure: {os.path.basename(save_path)}")
    
    plt.show()


def plot_agent_scores(score_data, title, save_path=None):
    """Creates plots showing cumulative scores for each agent type."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(title + " - Agent Scores (20 run average)", fontsize=16, weight='bold')
    
    axes_flat = axes.flatten()
    
    for i, (exp_name, exp_data) in enumerate(score_data.items()):
        ax = axes_flat[i]
        
        # Group scores by agent type
        agent_type_scores = {}
        for agent_id, scores in exp_data.items():
            # Determine agent type
            if 'SimpleQL' in agent_id:
                agent_type = 'Simple QL'
            elif 'EnhQL' in agent_id:
                agent_type = 'Enhanced QL'
            else:
                agent_type = agent_id.split('_')[0]  # TFT, AllC, AllD
            
            if agent_type not in agent_type_scores:
                agent_type_scores[agent_type] = []
            agent_type_scores[agent_type].append(scores['mean'])
        
        # Plot each agent type
        colors = {'Simple QL': 'green', 'Enhanced QL': 'darkgreen', 
                 'TFT': 'blue', 'AllC': 'orange', 'AllD': 'red'}
        
        for agent_type, scores_list in agent_type_scores.items():
            avg_scores = np.mean(scores_list, axis=0)
            rounds = range(1, len(avg_scores) + 1)
            
            ax.plot(rounds, avg_scores, color=colors.get(agent_type, 'black'), 
                   linewidth=2.5, label=agent_type, marker='o', markersize=2)
        
        ax.set_title(exp_name, fontsize=12)
        ax.set_xlabel("Round (after training)")
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
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(title + " - All Agent Cooperation Rates (20 run average)", fontsize=16, weight='bold')
    
    axes_flat = axes.flatten()
    
    for i, (exp_name, exp_data) in enumerate(coop_data.items()):
        ax = axes_flat[i]
        
        # Group cooperation by agent type
        agent_type_coop = {}
        for agent_id, coop in exp_data.items():
            # Determine agent type
            if 'SimpleQL' in agent_id:
                agent_type = 'Simple QL'
            elif 'EnhQL' in agent_id:
                agent_type = 'Enhanced QL'
            else:
                agent_type = agent_id.split('_')[0]
            
            if agent_type not in agent_type_coop:
                agent_type_coop[agent_type] = []
            agent_type_coop[agent_type].append(coop['mean'])
        
        # Plot each agent type
        colors = {'Simple QL': 'green', 'Enhanced QL': 'darkgreen',
                 'TFT': 'blue', 'AllC': 'orange', 'AllD': 'red'}
        
        for agent_type, coop_list in agent_type_coop.items():
            avg_coop = np.mean(coop_list, axis=0)
            rounds = range(1, len(avg_coop) + 1)
            
            # Calculate confidence intervals
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
        ax.set_xlabel("Round (after training)")
        ax.set_ylabel("Cooperation Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved figure: {os.path.basename(save_path)}")
    
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    NUM_ROUNDS = 200      # Measurement rounds
    TRAINING_ROUNDS = 100 # Training rounds for Q-learning
    NUM_RUNS = 20         # Number of simulation runs
    
    experiments = setup_qlearning_experiments()
    
    # Create results directory
    results_dir = "qlearning_results"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Q-Learning results will be saved to: {os.path.abspath(results_dir)}")
    print(f"Running {NUM_RUNS} simulations per experiment ({TRAINING_ROUNDS} training + {NUM_ROUNDS} measurement rounds)...")
    
    # --- Run Extended Pairwise Simulations ---
    print(f"\nRunning Extended Pairwise Q-Learning simulations...")
    pairwise_ql_aggregated = {}
    pairwise_all_coop_aggregated = {}
    pairwise_scores_aggregated = {}
    
    for name, agent_list in experiments.items():
        print(f"  - Running {NUM_RUNS} simulations for: {name}")
        ql_runs, coop_runs, score_runs = run_multiple_simulations_extended(
            run_pairwise_simulation_extended, agent_list, NUM_ROUNDS, NUM_RUNS, TRAINING_ROUNDS)
        
        pairwise_ql_aggregated[name] = aggregate_results(ql_runs)
        pairwise_all_coop_aggregated[name] = aggregate_agent_data(coop_runs)
        pairwise_scores_aggregated[name] = aggregate_agent_data(score_runs)
        
        print(f"    Mean Q-Learning cooperation: {np.mean(pairwise_ql_aggregated[name]['mean']):.3f} ± {np.mean(pairwise_ql_aggregated[name]['std']):.3f}")
    
    # --- Run Extended Neighbourhood Simulations ---
    print(f"\nRunning Extended Neighbourhood Q-Learning simulations...")
    neighbourhood_ql_aggregated = {}
    neighbourhood_all_coop_aggregated = {}
    neighbourhood_scores_aggregated = {}
    
    for name, agent_list in experiments.items():
        print(f"  - Running {NUM_RUNS} simulations for: {name}")
        ql_runs, coop_runs, score_runs = run_multiple_simulations_extended(
            run_nperson_simulation_extended, agent_list, NUM_ROUNDS, NUM_RUNS, TRAINING_ROUNDS)
        
        neighbourhood_ql_aggregated[name] = aggregate_results(ql_runs)
        neighbourhood_all_coop_aggregated[name] = aggregate_agent_data(coop_runs)
        neighbourhood_scores_aggregated[name] = aggregate_agent_data(score_runs)
        
        print(f"    Mean Q-Learning cooperation: {np.mean(neighbourhood_ql_aggregated[name]['mean']):.3f} ± {np.mean(neighbourhood_ql_aggregated[name]['std']):.3f}")
    
    # --- Save Aggregated Data to CSV ---
    print("\nSaving aggregated data to CSV files...")
    save_aggregated_data_to_csv(pairwise_ql_aggregated, "pairwise_ql_cooperation", results_dir)
    save_aggregated_data_to_csv(neighbourhood_ql_aggregated, "neighbourhood_ql_cooperation", results_dir)
    
    # --- Generate All Plots ---
    print("\nGenerating and saving all plots...")
    
    # Q-Learning cooperation plots
    plot_qlearning_cooperation(pairwise_ql_aggregated,
                              title="Figure 1: Q-Learning Cooperation with Pairwise Voting",
                              save_path=os.path.join(results_dir, "figure1_pairwise_ql_cooperation.png"))
    plot_qlearning_cooperation(neighbourhood_ql_aggregated,
                              title="Figure 2: Q-Learning Cooperation with Neighbourhood Voting",
                              save_path=os.path.join(results_dir, "figure2_neighbourhood_ql_cooperation.png"))
    
    # Agent scores plots
    plot_agent_scores(pairwise_scores_aggregated,
                     title="Figure 3: Pairwise Q-Learning Agent Scores",
                     save_path=os.path.join(results_dir, "figure3_pairwise_ql_scores.png"))
    plot_agent_scores(neighbourhood_scores_aggregated,
                     title="Figure 4: Neighbourhood Q-Learning Agent Scores",
                     save_path=os.path.join(results_dir, "figure4_neighbourhood_ql_scores.png"))
    
    # All agent cooperation plots
    plot_all_agent_cooperation(pairwise_all_coop_aggregated,
                              title="Figure 5: Pairwise All Agent Cooperation",
                              save_path=os.path.join(results_dir, "figure5_pairwise_all_cooperation.png"))
    plot_all_agent_cooperation(neighbourhood_all_coop_aggregated,
                              title="Figure 6: Neighbourhood All Agent Cooperation",
                              save_path=os.path.join(results_dir, "figure6_neighbourhood_all_cooperation.png"))
    
    print(f"\nDone! All Q-Learning results from {NUM_RUNS} runs saved to the '{results_dir}' directory.")