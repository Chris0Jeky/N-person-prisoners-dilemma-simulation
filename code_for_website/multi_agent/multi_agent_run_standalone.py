#!/usr/bin/env python3
"""
Standalone Q-Learning Swarm Experiment

This script runs simulations to observe the behavior of a single Q-Learning agent
within a "swarm" of Tit-for-Tat (TFT) agents. It is designed to be a single,
self-contained file for easy distribution and testing.

Three scenarios are tested, varying the total number of agents:
1.  5 Agents:  1 Q-Learner vs. 4 TFT agents
2.  7 Agents:  1 Q-Learner vs. 6 TFT agents
3.  25 Agents: 1 Q-Learner vs. 24 TFT agents

For each scenario, the script runs both pairwise and N-person (neighbourhood)
simulations, generating plots and raw data CSV files as output.
"""

import random
import os
from datetime import datetime
from collections import defaultdict

# --- Optional Library Imports and Fallbacks ---
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not found. Using slower, built-in math functions.")

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not found. CSV output will be basic and plots will lack smoothing.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not found. Plot generation will be skipped.")

# --- Fallback Math Functions (if numpy is unavailable) ---
if not HAS_NUMPY:
    def np_mean(data, axis=None):
        if axis is None:
            flat_list = [item for sublist in data for item in sublist] if isinstance(data[0], list) else data
            return sum(flat_list) / len(flat_list) if flat_list else 0
        if axis == 0:
            if not data or not data[0]: return []
            return [sum(col) / len(col) for col in zip(*data)]
        return data


    def np_std(data, axis=None):
        if axis is None:
            mean = np_mean(data)
            flat_list = [item for sublist in data for item in sublist] if isinstance(data[0], list) else data
            if not flat_list: return 0
            return (sum((x - mean) ** 2 for x in flat_list) / len(flat_list)) ** 0.5
        if axis == 0:
            if not data or not data[0]: return []
            means = np_mean(data, axis=0)
            return [(sum((row[i] - means[i]) ** 2 for row in data) / len(data)) ** 0.5 for i in range(len(data[0]))]
        return data


    def np_sqrt(x):
        return x ** 0.5


    class MockNumpy:
        mean = staticmethod(np_mean)
        std = staticmethod(np_std)
        sqrt = staticmethod(np_sqrt)

        def array(self, data): return list(data)


    np = MockNumpy()

# --- Constants and Payoffs ---
COOPERATE = 0
DEFECT = 1

# 2-Player Payoffs
PAYOFFS_2P = {
    (COOPERATE, COOPERATE): (3, 3), (COOPERATE, DEFECT): (0, 5),
    (DEFECT, COOPERATE): (5, 0), (DEFECT, DEFECT): (1, 1),
}
# N-Person Payoff Constants
T, R, P, S = 5, 3, 1, 0


def nperson_payoff(my_move, num_other_cooperators, total_agents):
    """Calculates N-Person payoff based on a linear formula."""
    if total_agents <= 1:
        return R if my_move == COOPERATE else P
    # Avoid division by zero if agent is alone
    n_minus_1 = total_agents - 1
    if n_minus_1 == 0:
        return R if my_move == COOPERATE else P

    if my_move == COOPERATE:
        return S + (R - S) * (num_other_cooperators / n_minus_1)
    else:  # Defect
        return P + (T - P) * (num_other_cooperators / n_minus_1)


# --- Agent Classes ---

class StaticTFTAgent:
    """A standard Tit-for-Tat agent."""

    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.strategy_name = "TFT"
        self.opponent_last_moves = {}

    def choose_pairwise_action(self, opponent_id):
        """In pairwise, copy the opponent's last move (or cooperate first)."""
        return self.opponent_last_moves.get(opponent_id, COOPERATE)

    def choose_nperson_action(self, prev_round_group_coop_ratio):
        """In N-person, probabilistically copy the group's cooperation."""
        if prev_round_group_coop_ratio is None:  # First round
            return COOPERATE
        else:
            return COOPERATE if random.random() < prev_round_group_coop_ratio else DEFECT

    def reset(self):
        """Resets agent state for a new simulation run."""
        self.opponent_last_moves = {}


class LegacyQLearner:
    """
    Q-Learning agent with a sophisticated state representation based on
    2-round history and cooperation trends.
    """

    def __init__(self, agent_id, lr, df, eps, epsilon_decay, epsilon_min, **kwargs):
        self.agent_id = agent_id
        self.strategy_name = "QLearner"
        self.learning_rate = lr
        self.discount_factor = df
        self.epsilon = eps
        self.initial_epsilon = eps
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table_pairwise = {}
        self.q_table_nperson = {}
        self.reset()

    def _get_state_pairwise(self, opponent_id):
        """Get state based on last 2 rounds of interaction history."""
        my_hist = self.my_history_pairwise.get(opponent_id, [])
        opp_hist = self.opp_history_pairwise.get(opponent_id, [])

        if len(my_hist) < 2: return 'initial'

        m, o = self._move_to_char, self._move_to_char
        return f"M{m(my_hist[0])}{m(my_hist[1])}_O{o(opp_hist[0])}{o(opp_hist[1])}"

    def _get_state_nperson(self):
        """Get state based on cooperation trends and my recent behavior."""
        if len(self.coop_ratio_history) < 2: return 'initial'

        r_t2, r_t1 = self.coop_ratio_history[0], self.coop_ratio_history[1]
        trend = 'up' if r_t1 > r_t2 + 0.1 else 'down' if r_t1 < r_t2 - 0.1 else 'stable'

        my_hist = self.my_history_nperson
        my_recent = ""
        if len(my_hist) >= 2:
            my_recent = f"_M{self._move_to_char(my_hist[0])}{self._move_to_char(my_hist[1])}"

        return f"{self._ratio_to_category(r_t1)}_{trend}{my_recent}"

    def _move_to_char(self, move):
        return 'C' if move == COOPERATE else 'D'

    def _ratio_to_category(self, r):
        return 'low' if r <= 0.33 else 'medium' if r <= 0.67 else 'high'

    def _ensure_state_exists(self, state, q_table):
        if state not in q_table:
            q_table[state] = {COOPERATE: 0.5, DEFECT: 0.3}  # Optimistic initialization

    def _choose_action(self, state, q_table):
        """Choose action using an epsilon-greedy policy."""
        self._ensure_state_exists(state, q_table)
        if random.random() < self.epsilon:
            return random.choice([COOPERATE, DEFECT])
        q_values = q_table[state]
        return COOPERATE if q_values[COOPERATE] >= q_values[DEFECT] else DEFECT

    def choose_pairwise_action(self, opponent_id):
        state = self._get_state_pairwise(opponent_id)
        action = self._choose_action(state, self.q_table_pairwise)
        self.last_state_pairwise[opponent_id] = state
        self.last_action_pairwise[opponent_id] = action
        return action

    def choose_nperson_action(self, prev_round_group_coop_ratio):
        if prev_round_group_coop_ratio is not None:
            self.coop_ratio_history.append(prev_round_group_coop_ratio)
            if len(self.coop_ratio_history) > 2: self.coop_ratio_history.pop(0)

        state = self._get_state_nperson()
        action = self._choose_action(state, self.q_table_nperson)
        self.last_state_nperson = state
        self.last_action_nperson = action
        return action

    def update_pairwise_q_value(self, opponent_id, my_move, opp_move, payoff):
        # Update history
        self.my_history_pairwise[opponent_id].append(my_move)
        self.opp_history_pairwise[opponent_id].append(opp_move)
        if len(self.my_history_pairwise[opponent_id]) > 2:
            self.my_history_pairwise[opponent_id].pop(0)
            self.opp_history_pairwise[opponent_id].pop(0)

        # Update Q-table
        last_state = self.last_state_pairwise.get(opponent_id)
        if last_state:
            state, action = last_state, self.last_action_pairwise[opponent_id]
            next_state = self._get_state_pairwise(opponent_id)
            self._update_q_table(self.q_table_pairwise, state, action, payoff, next_state)

    def update_nperson_q_value(self, my_move, payoff):
        self.my_history_nperson.append(my_move)
        if len(self.my_history_nperson) > 2: self.my_history_nperson.pop(0)

        if self.last_state_nperson:
            state, action = self.last_state_nperson, self.last_action_nperson
            next_state = self._get_state_nperson()
            self._update_q_table(self.q_table_nperson, state, action, payoff, next_state)

    def _update_q_table(self, q_table, state, action, reward, next_state):
        self._ensure_state_exists(state, q_table)
        self._ensure_state_exists(next_state, q_table)

        current_q = q_table[state][action]
        max_next_q = max(q_table[next_state].values())
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        q_table[state][action] = new_q

    def reset(self):
        """Resets agent state, decaying epsilon for the new run."""
        if hasattr(self, 'my_history_pairwise'):  # Decay epsilon after the first run
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.my_history_pairwise = defaultdict(list)
        self.opp_history_pairwise = defaultdict(list)
        self.my_history_nperson = []
        self.coop_ratio_history = []
        self.last_state_pairwise = {}
        self.last_action_pairwise = {}
        self.last_state_nperson = None
        self.last_action_nperson = None


# --- Simulation Engines ---

def run_pairwise_simulation(agents, num_rounds):
    """Runs a single pairwise tournament for a given set of agents."""
    for agent in agents: agent.reset()

    coop_history = {agent.agent_id: [] for agent in agents}
    score_history = {agent.agent_id: [] for agent in agents}
    cumulative_scores = {agent.agent_id: 0 for agent in agents}

    for _ in range(num_rounds):
        round_payoffs = {agent.agent_id: 0 for agent in agents}
        round_moves = defaultdict(dict)
        agent_pairs = [(agents[i], agents[j]) for i in range(len(agents)) for j in range(i + 1, len(agents))]

        for agent1, agent2 in agent_pairs:
            move1 = agent1.choose_pairwise_action(agent2.agent_id)
            move2 = agent2.choose_pairwise_action(agent1.agent_id)

            round_moves[agent1.agent_id][agent2.agent_id] = move1
            round_moves[agent2.agent_id][agent1.agent_id] = move2

            payoff1, payoff2 = PAYOFFS_2P[(move1, move2)]
            round_payoffs[agent1.agent_id] += payoff1
            round_payoffs[agent2.agent_id] += payoff2

            # Update agents with opponent's move
            if isinstance(agent1, StaticTFTAgent): agent1.opponent_last_moves[agent2.agent_id] = move2
            if isinstance(agent2, StaticTFTAgent): agent2.opponent_last_moves[agent1.agent_id] = move1
            if isinstance(agent1, LegacyQLearner): agent1.update_pairwise_q_value(agent2.agent_id, move1, move2,
                                                                                  payoff1)
            if isinstance(agent2, LegacyQLearner): agent2.update_pairwise_q_value(agent1.agent_id, move2, move1,
                                                                                  payoff2)

        for agent in agents:
            cumulative_scores[agent.agent_id] += round_payoffs[agent.agent_id]
            score_history[agent.agent_id].append(cumulative_scores[agent.agent_id])

            coop_count = sum(1 for move in round_moves[agent.agent_id].values() if move == COOPERATE)
            coop_rate = coop_count / (len(agents) - 1) if len(agents) > 1 else 0
            coop_history[agent.agent_id].append(coop_rate)

    return coop_history, score_history


def run_nperson_simulation(agents, num_rounds):
    """Runs a single N-person (neighbourhood) simulation."""
    for agent in agents: agent.reset()

    coop_history = {agent.agent_id: [] for agent in agents}
    score_history = {agent.agent_id: [] for agent in agents}
    cumulative_scores = {agent.agent_id: 0 for agent in agents}

    prev_round_coop_ratio = None
    num_total_agents = len(agents)

    for _ in range(num_rounds):
        moves = {agent.agent_id: agent.choose_nperson_action(prev_round_coop_ratio) for agent in agents}

        num_cooperators = list(moves.values()).count(COOPERATE)
        current_coop_ratio = num_cooperators / num_total_agents if num_total_agents > 0 else 0

        for agent in agents:
            my_move = moves[agent.agent_id]
            num_other_cooperators = num_cooperators - (1 if my_move == COOPERATE else 0)
            payoff = nperson_payoff(my_move, num_other_cooperators, num_total_agents)

            cumulative_scores[agent.agent_id] += payoff
            score_history[agent.agent_id].append(cumulative_scores[agent.agent_id])
            coop_history[agent.agent_id].append(1 if my_move == COOPERATE else 0)

            if isinstance(agent, LegacyQLearner):
                agent.update_nperson_q_value(my_move, payoff)

        prev_round_coop_ratio = current_coop_ratio

    return coop_history, score_history


# --- Experiment Orchestration ---

def run_experiment(simulation_func, create_agents_func, num_rounds, num_runs):
    """Runs multiple simulations for an experiment and aggregates results."""
    all_coop_runs = defaultdict(list)
    all_score_runs = defaultdict(list)

    for i in range(num_runs):
        print(f"\r  - Running simulation {i + 1}/{num_runs}...", end="")
        agents = create_agents_func()
        coop_h, score_h = simulation_func(agents, num_rounds)
        for agent_id, history in coop_h.items():
            all_coop_runs[agent_id].append(history)
        for agent_id, history in score_h.items():
            all_score_runs[agent_id].append(history)
    print(" Done.")
    return all_coop_runs, all_score_runs


def aggregate_results(all_runs):
    """Aggregates raw run data into mean, std, and CI for QL and TFT groups."""
    ql_runs_coop, tft_runs_coop = [], []
    ql_runs_score, tft_runs_score = [], []

    for agent_id, runs in all_runs[0].items():  # Coop runs
        if "QLearner" in agent_id:
            ql_runs_coop = runs
        else:
            tft_runs_coop.append(runs)

    for agent_id, runs in all_runs[1].items():  # Score runs
        if "QLearner" in agent_id:
            ql_runs_score = runs
        else:
            tft_runs_score.append(runs)

    # Average the TFT agent data across all TFTs for each run
    avg_tft_coop = np.mean(tft_runs_coop, axis=0) if tft_runs_coop else []
    avg_tft_score = np.mean(tft_runs_score, axis=0) if tft_runs_score else []

    def get_stats(run_data):
        if len(run_data) == 0: return {}
        mean = np.mean(run_data, axis=0)
        std = np.std(run_data, axis=0)
        sem = std / np.sqrt(len(run_data))
        ci_95 = 1.96 * sem
        return {'mean': mean, 'lower_95': mean - ci_95, 'upper_95': mean + ci_95, 'raw': run_data}

    return {
        'QLearner_Coop': get_stats(ql_runs_coop),
        'TFT_Coop': get_stats(avg_tft_coop),
        'QLearner_Score': get_stats(ql_runs_score),
        'TFT_Score': get_stats(avg_tft_score),
    }


# --- Plotting and Saving ---

def plot_and_save_results(agg_data, scenario_name, num_rounds, save_dir):
    """Generates and saves a 2x2 plot comparing QL and TFT performance."""
    if not HAS_PLOTTING: return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    fig.suptitle(
        f"Performance in {scenario_name}: Q-Learner vs. TFT Swarm (Avg of {len(agg_data['QLearner_Coop']['raw'])} runs)",
        fontsize=16)
    rounds = range(1, num_rounds + 1)

    # Plot 1: Pairwise Cooperation
    ax = axes[0, 0]
    for key, color, label in [('QLearner_Coop', 'blue', 'QLearner'), ('TFT_Coop', 'red', 'TFT (Avg)')]:
        data = agg_data['pairwise'].get(key)
        if data:
            ax.plot(rounds, data['mean'], label=label, color=color, lw=2)
            ax.fill_between(rounds, data['lower_95'], data['upper_95'], color=color, alpha=0.2)
    ax.set_title('Pairwise Cooperation Rate')
    ax.set_ylabel('Cooperation Rate');
    ax.set_xlabel('Round');
    ax.set_ylim(-0.05, 1.05);
    ax.legend();
    ax.grid(True)

    # Plot 2: N-Person Cooperation
    ax = axes[0, 1]
    for key, color, label in [('QLearner_Coop', 'blue', 'QLearner'), ('TFT_Coop', 'red', 'TFT (Avg)')]:
        data = agg_data['nperson'].get(key)
        if data:
            ax.plot(rounds, data['mean'], label=label, color=color, lw=2)
            ax.fill_between(rounds, data['lower_95'], data['upper_95'], color=color, alpha=0.2)
    ax.set_title('N-Person Cooperation Rate')
    ax.set_ylabel('Cooperation Rate');
    ax.set_xlabel('Round');
    ax.set_ylim(-0.05, 1.05);
    ax.legend();
    ax.grid(True)

    # Plot 3: Pairwise Score
    ax = axes[1, 0]
    for key, color, label in [('QLearner_Score', 'blue', 'QLearner'), ('TFT_Score', 'red', 'TFT (Avg)')]:
        data = agg_data['pairwise'].get(key)
        if data:
            ax.plot(rounds, data['mean'], label=label, color=color, lw=2)
    ax.set_title('Pairwise Cumulative Score')
    ax.set_ylabel('Score');
    ax.set_xlabel('Round');
    ax.legend();
    ax.grid(True)

    # Plot 4: N-Person Score
    ax = axes[1, 1]
    for key, color, label in [('QLearner_Score', 'blue', 'QLearner'), ('TFT_Score', 'red', 'TFT (Avg)')]:
        data = agg_data['nperson'].get(key)
        if data:
            ax.plot(rounds, data['mean'], label=label, color=color, lw=2)
    ax.set_title('N-Person Cumulative Score')
    ax.set_ylabel('Score');
    ax.set_xlabel('Round');
    ax.legend();
    ax.grid(True)

    filepath = os.path.join(save_dir, f"{scenario_name.replace(' ', '_')}_summary_plot.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"  - Saved plot to: {filepath}")


def save_data_to_csv(agg_data, scenario_name, num_rounds, save_dir):
    """Saves the aggregated data to a CSV file."""
    if not HAS_PANDAS:
        print("  - Skipping CSV export (pandas not available).")
        return

    dfs = []
    for game_mode in ['pairwise', 'nperson']:
        for metric_type in ['Coop', 'Score']:
            for agent_type in ['QLearner', 'TFT']:
                key = f"{agent_type}_{metric_type}"
                if key in agg_data[game_mode]:
                    data = agg_data[game_mode][key]['mean']
                    df = pd.DataFrame(data, columns=[f'{game_mode}_{key}_mean'])
                    dfs.append(df)

    if dfs:
        full_df = pd.concat(dfs, axis=1)
        full_df.insert(0, 'Round', range(1, num_rounds + 1))
        filepath = os.path.join(save_dir, f"{scenario_name.replace(' ', '_')}_aggregated_data.csv")
        full_df.to_csv(filepath, index=False)
        print(f"  - Saved CSV data to: {filepath}")


# --- Main Execution ---

if __name__ == "__main__":
    # --- Experiment Parameters ---
    NUM_ROUNDS = 500
    NUM_RUNS = 50
    SCENARIOS = [5, 7, 25]  # Total number of agents in each scenario

    # Q-Learning parameters for the LegacyQLearner
    LEGACY_PARAMS = {
        'lr': 0.15,  # Learning rate
        'df': 0.95,  # Discount factor
        'eps': 0.3,  # Starting epsilon
        'epsilon_decay': 0.995,  # Epsilon decay rate
        'epsilon_min': 0.05,  # Minimum epsilon
    }

    # --- Setup ---
    main_results_dir = "results_ql_vs_tft_swarm"
    os.makedirs(main_results_dir, exist_ok=True)
    print(f"Starting QL vs. TFT Swarm Experiment")
    print(f"Parameters: {NUM_ROUNDS} rounds/sim, {NUM_RUNS} sims/experiment.")
    print(f"Output will be saved in: {os.path.abspath(main_results_dir)}\n")

    # --- Run Experiments for each scenario ---
    for n_agents in SCENARIOS:
        scenario_name = f"{n_agents}_agents"
        scenario_dir = os.path.join(main_results_dir, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)
        print(f"--- Running Scenario: {scenario_name} ({1} QL vs. {n_agents - 1} TFT) ---")

        # Use a closure to create fresh agents for each run, including a Q-Learner
        # that correctly decays its epsilon across runs.
        ql_agent = LegacyQLearner(agent_id="QLearner_1", **LEGACY_PARAMS)


        def create_agents_for_run():
            tft_agents = [StaticTFTAgent(f"TFT_{i + 1}") for i in range(n_agents - 1)]
            # QL agent resets internally, which also handles epsilon decay for a new run
            ql_agent.reset()
            return [ql_agent] + tft_agents


        # Run Pairwise Simulations
        print("Running Pairwise simulations...")
        pw_coop, pw_score = run_experiment(
            run_pairwise_simulation, create_agents_for_run, NUM_ROUNDS, NUM_RUNS
        )

        # Run N-Person Simulations
        print("Running N-Person simulations...")
        np_coop, np_score = run_experiment(
            run_nperson_simulation, create_agents_for_run, NUM_ROUNDS, NUM_RUNS
        )

        # Aggregate and save results
        print("Aggregating and saving results...")
        aggregated_data = {
            'pairwise': aggregate_results((pw_coop, pw_score)),
            'nperson': aggregate_results((np_coop, np_score))
        }

        plot_and_save_results(aggregated_data, scenario_name, NUM_ROUNDS, scenario_dir)
        save_data_to_csv(aggregated_data, scenario_name, NUM_ROUNDS, scenario_dir)
        print("-" * 50 + "\n")

    print("All experiments complete.")