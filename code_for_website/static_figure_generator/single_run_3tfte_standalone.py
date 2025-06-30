#!/usr/bin/env python3
"""
Standalone Single Run for 3 TFT-E Scenario
Runs one simulation for both pairwise and neighbourhood strategies
No external dependencies except standard libraries
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from datetime import datetime

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
    def __init__(self, agent_id, strategy_name, exploration_rate=0.0, exploration_decay=0.0):
        self.agent_id = agent_id
        self.strategy_name = strategy_name
        self.exploration_rate = exploration_rate
        self.initial_exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.round_count = 0
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
        if self.strategy_name == "TFT-E" and random.random() < self._get_current_exploration_rate():
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
        if self.strategy_name == "TFT-E" and random.random() < self._get_current_exploration_rate():
            return 1 - intended_move
        return intended_move

    def _get_current_exploration_rate(self):
        """Calculate the current exploration rate based on decay."""
        if self.exploration_decay == 0:
            return self.exploration_rate
        # Exponential decay: rate = initial_rate * (1 - decay_rate)^round_count
        return self.initial_exploration_rate * ((1 - self.exploration_decay) ** self.round_count)
    
    def increment_round(self):
        """Increment the round counter for exploration decay."""
        self.round_count += 1
    
    def reset(self):
        self.opponent_last_moves = {}
        self.round_count = 0
        self.exploration_rate = self.initial_exploration_rate


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
        
        # Increment round counter for all agents (for exploration decay)
        for agent in agents:
            agent.increment_round()

    return tft_coop_history, all_coop_history, score_history


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
        
        # Increment round counter for all agents (for exploration decay)
        for agent in agents:
            agent.increment_round()

    return tft_coop_history, all_coop_history, score_history


def plot_3tfte_comparison(pairwise_coop, pairwise_scores, nperson_coop, nperson_scores, save_dir, num_rounds):
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
    rounds = range(1, num_rounds + 1)
    
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
    plot_3tfte_comparison(pairwise_coop, pairwise_scores, nperson_coop, nperson_scores, results_dir, NUM_ROUNDS)

    print(f"\nDone! All results saved to the '{results_dir}' directory.")