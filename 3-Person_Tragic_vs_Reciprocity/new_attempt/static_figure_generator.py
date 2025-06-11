import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from datetime import datetime

# --- Part 1: Agent and Simulation Logic (Simplified for Static Policies) ---

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
            # This is the pTFT logic described in the paper
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

def run_pairwise_simulation(agents, num_rounds):
    """Runs a single pairwise tournament and logs TFT cooperation."""
    for agent in agents: agent.reset()

    # Identify TFT agents to log their performance
    tft_agents = [agent for agent in agents if "TFT" in agent.strategy_name]
    tft_coop_history = []

    for _ in range(num_rounds):
        round_payoffs = {agent.agent_id: 0 for agent in agents}

        # All pairs play one round
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                move1 = agent1.choose_pairwise_action(agent2.agent_id)
                move2 = agent2.choose_pairwise_action(agent1.agent_id)

                # Record opponent's move for next round's TFT decision
                agent1.opponent_last_moves[agent2.agent_id] = move2
                agent2.opponent_last_moves[agent1.agent_id] = move1

                payoff1, payoff2 = PAYOFFS_2P[(move1, move2)]
                round_payoffs[agent1.agent_id] += payoff1
                round_payoffs[agent2.agent_id] += payoff2

        # Log the cooperation rate of TFT agents in this round
        if tft_agents:
            tft_coop_count = 0
            # To get a fair per-round rate, we check the intended moves for all pairings
            for tft_agent in tft_agents:
                for opponent in agents:
                    if tft_agent.agent_id != opponent.agent_id:
                        if tft_agent.choose_pairwise_action(opponent.agent_id) == COOPERATE:
                            tft_coop_count += 1
            avg_coop_rate = tft_coop_count / (len(tft_agents) * (len(agents) - 1))
            tft_coop_history.append(avg_coop_rate)

    return tft_coop_history


def run_nperson_simulation(agents, num_rounds):
    """Runs a single N-Person simulation and logs TFT cooperation."""
    tft_agents = [agent for agent in agents if "TFT" in agent.strategy_name]
    tft_coop_history = []

    prev_round_coop_ratio = None
    num_total_agents = len(agents)

    for _ in range(num_rounds):
        moves = {agent.agent_id: agent.choose_nperson_action(prev_round_coop_ratio) for agent in agents}

        num_cooperators = list(moves.values()).count(COOPERATE)
        current_coop_ratio = num_cooperators / num_total_agents if num_total_agents > 0 else 0

        # Log the cooperation rate of TFT agents in this round
        if tft_agents:
            tft_coop_count = sum(1 for agent in tft_agents if moves[agent.agent_id] == COOPERATE)
            avg_coop_rate = tft_coop_count / len(tft_agents) if tft_agents else 0
            tft_coop_history.append(avg_coop_rate)

        # Update state for next round
        prev_round_coop_ratio = current_coop_ratio

    return tft_coop_history


# --- Part 2: Experiment Setup and Plotting ---

def setup_experiments():
    """Defines the four agent compositions for the static policy experiments."""
    return {
        "3 TFT": [
            StaticAgent(agent_id="TFT_1", strategy_name="TFT"),
            StaticAgent(agent_id="TFT_2", strategy_name="TFT"),
            StaticAgent(agent_id="TFT_3", strategy_name="TFT"),
        ],
        "2 TFT-E + 1 AllD": [
            StaticAgent(agent_id="TFT-E_1", strategy_name="TFT-E", exploration_rate=0.1),
            StaticAgent(agent_id="TFT-E_2", strategy_name="TFT-E", exploration_rate=0.1),
            StaticAgent(agent_id="AllD_1", strategy_name="AllD"),
        ],
        "2 TFT + 1 AllD": [
            StaticAgent(agent_id="TFT_1", strategy_name="TFT"),
            StaticAgent(agent_id="TFT_2", strategy_name="TFT"),
            StaticAgent(agent_id="AllD_1", strategy_name="AllD"),
        ],
        "2 TFT-E + 1 AllC": [
            StaticAgent(agent_id="TFT-E_1", strategy_name="TFT-E", exploration_rate=0.1),
            StaticAgent(agent_id="TFT-E_2", strategy_name="TFT-E", exploration_rate=0.1),
            StaticAgent(agent_id="AllC_1", strategy_name="AllC"),
        ],
    }


def save_data_to_csv(data, filename_prefix, results_dir):
    """Saves the simulation data to CSV files."""
    # Save individual experiment files
    for exp_name, history in data.items():
        # Create a DataFrame with round numbers and cooperation rates
        df = pd.DataFrame({
            'Round': range(1, len(history) + 1),
            'TFT_Cooperation_Rate': history
        })
        
        # Clean experiment name for filename
        clean_name = exp_name.replace(' ', '_').replace('+', '_plus_')
        filename = f"{filename_prefix}_{clean_name}.csv"
        filepath = os.path.join(results_dir, filename)
        
        df.to_csv(filepath, index=False)
        print(f"  - Saved: {filename}")
    
    # Also save a combined file with all experiments
    # Get number of rounds from the first history
    num_rounds = len(next(iter(data.values())))
    combined_df = pd.DataFrame({'Round': range(1, num_rounds + 1)})
    for exp_name, history in data.items():
        clean_name = exp_name.replace(' ', '_').replace('+', '_plus_')
        combined_df[clean_name] = history
    
    combined_filename = f"{filename_prefix}_all_experiments.csv"
    combined_filepath = os.path.join(results_dir, combined_filename)
    combined_df.to_csv(combined_filepath, index=False)
    print(f"  - Saved combined: {combined_filename}")


def plot_results(data, title, smoothing_window=5, save_path=None):
    """Creates a 2x2 grid of plots for the given simulation data."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(title, fontsize=16, weight='bold')

    axes_flat = axes.flatten()
    for i, (exp_name, history) in enumerate(data.items()):
        ax = axes_flat[i]

        # Raw data
        ax.plot(history, color='skyblue', alpha=0.6, label='Raw Cooperation')

        # Smoothed data
        smoothed_history = pd.Series(history).rolling(window=smoothing_window, min_periods=1, center=True).mean()
        ax.plot(smoothed_history, color='blue', linewidth=2.5, label=f'Smoothed (window={smoothing_window})')

        ax.set_title(exp_name, fontsize=12)
        ax.set_xlabel("Round")
        ax.set_ylabel("Avg. TFT/TFT-E Cooperation Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()

    # Hide any unused subplots
    for i in range(len(data), len(axes_flat)):
        axes_flat[i].set_visible(False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved figure: {os.path.basename(save_path)}")
    
    plt.show()


# --- Part 3: Main Execution ---

if __name__ == "__main__":
    NUM_ROUNDS = 50  # As per the paper draft
    experiments = setup_experiments()
    
    # --- Create results directory ---
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {os.path.abspath(results_dir)}")

    # --- Run Simulations for Figure 1 (Pairwise) ---
    print("\nRunning Pairwise simulations for Figure 1...")
    pairwise_data = {}
    for name, agent_list in experiments.items():
        print(f"  - Simulating: {name}")
        pairwise_data[name] = run_pairwise_simulation(agent_list, NUM_ROUNDS)

    # --- Run Simulations for Figure 2 (Neighbourhood) ---
    print("\nRunning Neighbourhood simulations for Figure 2...")
    neighbourhood_data = {}
    for name, agent_list in experiments.items():
        print(f"  - Simulating: {name}")
        neighbourhood_data[name] = run_nperson_simulation(agent_list, NUM_ROUNDS)

    # --- Save Data to CSV ---
    print("\nSaving data to CSV files...")
    save_data_to_csv(pairwise_data, "pairwise_cooperation", results_dir)
    save_data_to_csv(neighbourhood_data, "neighbourhood_cooperation", results_dir)
    
    # --- Generate and Save Plots ---
    print("\nGenerating and saving plots...")
    plot_results(pairwise_data, 
                title="Figure 1: TFT Cooperation Dynamics with Pairwise Voting",
                save_path=os.path.join(results_dir, "figure1_pairwise_cooperation.png"))
    plot_results(neighbourhood_data, 
                title="Figure 2: TFT Cooperation Dynamics with Neighbourhood Voting",
                save_path=os.path.join(results_dir, "figure2_neighbourhood_cooperation.png"))
    
    print("\nDone! All results saved to the 'results' directory.")