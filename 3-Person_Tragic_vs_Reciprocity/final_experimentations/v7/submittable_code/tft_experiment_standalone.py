#!/usr/bin/env python3
"""
Comprehensive Q-Learning vs TFT Experiment (Standalone Version)
Tests Vanilla and Adaptive Q-learners against TFT agents with different discount factors

This is a self-contained version that doesn't require external imports.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import random
from collections import defaultdict, deque
import time
import json

# Constants
COOPERATE, DEFECT = 0, 1
PAYOFFS_2P = {(0, 0): (3, 3), (0, 1): (0, 5), (1, 0): (5, 0), (1, 1): (1, 1)}
T, R, P, S = 5, 3, 1, 0

# Simulation parameters
NUM_ROUNDS = 10000
NUM_RUNS = 25
OUTPUT_DIR = "tft_experiment_results"

# Q-learner configurations with different discount factors
VANILLA_DF_04 = {
    'lr': 0.08,
    'df': 0.4,
    'eps': 0.1,
}

VANILLA_DF_095 = {
    'lr': 0.08,
    'df': 0.95,
    'eps': 0.1,
}

ADAPTIVE_DF_04 = {
    'initial_lr': 0.1,
    'initial_eps': 0.15,
    'min_lr': 0.03,
    'max_lr': 0.15,
    'min_eps': 0.02,
    'max_eps': 0.15,
    'adaptation_factor': 1.08,
    'reward_window_size': 75,
    'df': 0.4,
}

ADAPTIVE_DF_095 = {
    'initial_lr': 0.1,
    'initial_eps': 0.15,
    'min_lr': 0.03,
    'max_lr': 0.15,
    'min_eps': 0.02,
    'max_eps': 0.15,
    'adaptation_factor': 1.08,
    'reward_window_size': 75,
    'df': 0.95,
}


# --- Agent Implementations ---
class BaseAgent:
    def __init__(self, agent_id, strategy_name):
        self.agent_id, self.strategy_name = agent_id, strategy_name
        self.total_score = 0

    def reset(self): 
        self.total_score = 0


class StaticAgent(BaseAgent):
    def __init__(self, agent_id, strategy_name="TFT", error_rate=0.0, **kwargs):
        super().__init__(agent_id, strategy_name)
        self.strategy_name = strategy_name
        self.error_rate = error_rate
        self.opponent_last_moves = {}
        self.last_neighborhood_move = COOPERATE

    def _apply_error(self, intended_move):
        """Apply error rate to the intended move"""
        if random.random() < self.error_rate:
            return random.choice([COOPERATE, DEFECT])
        return intended_move

    def choose_pairwise_action(self, opponent_id):
        if self.strategy_name == "AllC":
            intended = COOPERATE
        elif self.strategy_name == "AllD":
            intended = DEFECT
        elif self.strategy_name == "Random":
            intended = random.choice([COOPERATE, DEFECT])
        elif self.strategy_name == "TFT" or self.strategy_name == "TFT-E":
            intended = self.opponent_last_moves.get(opponent_id, COOPERATE)
        else:  # Default TFT
            intended = self.opponent_last_moves.get(opponent_id, COOPERATE)
        
        return self._apply_error(intended)

    def choose_neighborhood_action(self, coop_ratio):
        if self.strategy_name == "AllC":
            intended = COOPERATE
        elif self.strategy_name == "AllD":
            intended = DEFECT
        elif self.strategy_name == "Random":
            intended = random.choice([COOPERATE, DEFECT])
        elif self.strategy_name == "TFT" or self.strategy_name == "TFT-E":
            if coop_ratio is None:
                intended = COOPERATE
            else:
                intended = COOPERATE if random.random() < coop_ratio else DEFECT
        else:
            if coop_ratio is None:
                intended = COOPERATE
            else:
                intended = COOPERATE if random.random() < coop_ratio else DEFECT
        
        return self._apply_error(intended)

    def record_pairwise_outcome(self, opponent_id, my_move, opponent_move, reward):
        self.total_score += reward
        self.opponent_last_moves[opponent_id] = opponent_move

    def record_neighborhood_outcome(self, coop_ratio, reward):
        self.total_score += reward
        self.last_neighborhood_move = COOPERATE if coop_ratio and coop_ratio >= 0.5 else DEFECT

    def reset(self):
        super().reset()
        self.opponent_last_moves.clear()
        self.last_neighborhood_move = COOPERATE


class PairwiseAdaptiveQLearner(BaseAgent):
    def __init__(self, agent_id, params, **kwargs):
        super().__init__(agent_id, "PairwiseAdaptive")
        self.params = params
        self.reset()

    def _get_state(self, opponent_id):
        if opponent_id not in self.histories:
            self.histories[opponent_id] = self._make_history_deque()
        history = self.histories[opponent_id]
        if len(history) < 2: return "start"
        return str(tuple(history))

    def choose_pairwise_action(self, opponent_id):
        state = self._get_state(opponent_id)
        if opponent_id not in self.q_tables:
            self.q_tables[opponent_id] = defaultdict(lambda: defaultdict(float))
        
        if random.random() < self._get_epsilon():
            return random.choice([COOPERATE, DEFECT])
        else:
            q_values = self.q_tables[opponent_id][state]
            if q_values[COOPERATE] == q_values[DEFECT]:
                return random.choice([COOPERATE, DEFECT])
            return max(q_values, key=q_values.get)

    def record_pairwise_outcome(self, opponent_id, my_move, opponent_move, reward):
        self.total_score += reward
        state = self._get_state(opponent_id)
        
        if opponent_id not in self.histories:
            self.histories[opponent_id] = self._make_history_deque()
        self.histories[opponent_id].append((my_move, opponent_move))
        
        next_state = self._get_state(opponent_id)
        self._update_q_value(opponent_id, state, my_move, reward, next_state)
        self._adapt_parameters(reward)

    def choose_neighborhood_action(self, coop_ratio):
        # First, we need to get the current state (before this round's ratio is added)
        state = self._get_neighborhood_state()
        
        if random.random() < self._get_epsilon():
            return random.choice([COOPERATE, DEFECT])
        else:
            q_values = self.n_q_table[state]
            if q_values[COOPERATE] == q_values[DEFECT]:
                return random.choice([COOPERATE, DEFECT])
            return max(q_values, key=q_values.get)

    def record_neighborhood_outcome(self, coop_ratio, reward):
        self.total_score += reward
        
        # Get current state before updating history
        state = self._get_neighborhood_state()
        
        # Update cooperation ratio history with category
        if coop_ratio is not None:
            category = self._categorize_coop_ratio(coop_ratio)
            self.coop_ratio_history.append(category)
        
        # Get next state after updating history
        next_state = self._get_neighborhood_state()
        
        if self.last_coop_ratio is None:
            my_last_move = COOPERATE
        else:
            my_last_move = COOPERATE if coop_ratio > self.last_coop_ratio else DEFECT
        
        self._update_neighborhood_q_value(state, my_last_move, reward, next_state)
        self._adapt_parameters(reward)
        self.last_coop_ratio = coop_ratio

    def _categorize_coop_ratio(self, coop_ratio):
        """Convert cooperation ratio to category"""
        if coop_ratio < 0.33:
            return "low"
        elif coop_ratio < 0.67:
            return "medium"
        else:
            return "high"

    def _get_neighborhood_state(self):
        """Get state as tuple of last 4 cooperation ratio categories"""
        if len(self.coop_ratio_history) == 0:
            return "start"
        elif len(self.coop_ratio_history) < 4:
            # Pad with 'start' if we don't have 4 rounds yet
            padding = ['start'] * (4 - len(self.coop_ratio_history))
            return str(tuple(padding + list(self.coop_ratio_history)))
        else:
            # Return tuple of last 4 categories
            return str(tuple(self.coop_ratio_history))

    def _update_q_value(self, opponent_id, state, action, reward, next_state):
        lr = self._get_learning_rate()
        df = self.params['df']
        
        current_q = self.q_tables[opponent_id][state][action]
        max_next_q = max(self.q_tables[opponent_id][next_state].values()) if self.q_tables[opponent_id][next_state] else 0
        new_q = current_q + lr * (reward + df * max_next_q - current_q)
        self.q_tables[opponent_id][state][action] = new_q

    def _update_neighborhood_q_value(self, state, action, reward, next_state):
        lr = self._get_learning_rate()
        df = self.params['df']
        
        current_q = self.n_q_table[state][action]
        max_next_q = max(self.n_q_table[next_state].values()) if self.n_q_table[next_state] else 0
        new_q = current_q + lr * (reward + df * max_next_q - current_q)
        self.n_q_table[state][action] = new_q

    def _get_learning_rate(self):
        if 'lr' in self.params:
            return self.params['lr']
        else:
            return self.current_lr

    def _get_epsilon(self):
        if 'eps' in self.params:
            return self.params['eps']
        else:
            return self.current_eps

    def _adapt_parameters(self, reward):
        if 'initial_lr' not in self.params:
            return
            
        self.reward_history.append(reward)
        
        window_size = self.params.get('reward_window_size', 75)
        if len(self.reward_history) < window_size:
            return
        
        recent_rewards = list(self.reward_history)[-window_size:]
        avg_reward = np.mean(recent_rewards)
        
        mid_point = window_size // 2
        first_half_avg = np.mean(recent_rewards[:mid_point])
        second_half_avg = np.mean(recent_rewards[mid_point:])
        
        factor = self.params.get('adaptation_factor', 1.08)
        
        if second_half_avg > first_half_avg:
            self.current_lr = max(self.params['min_lr'], self.current_lr / factor)
        else:
            self.current_lr = min(self.params['max_lr'], self.current_lr * factor)
        
        if avg_reward > 2.5:
            self.current_eps = max(self.params['min_eps'], self.current_eps / factor)
        else:
            self.current_eps = min(self.params['max_eps'], self.current_eps * factor)

    def _make_history_deque(self):
        return deque(maxlen=2)

    def reset(self):
        super().reset()
        self.q_tables = {}
        self.n_q_table = defaultdict(lambda: defaultdict(float))
        self.histories = {}
        self.last_coop_ratio = None
        self.coop_ratio_history = deque(maxlen=4)  # Store last 4 cooperation ratio categories
        self.reward_history = deque(maxlen=1000)
        
        if 'initial_lr' in self.params:
            self.current_lr = self.params['initial_lr']
            self.current_eps = self.params['initial_eps']
        else:
            self.current_lr = self.params.get('lr', 0.1)
            self.current_eps = self.params.get('eps', 0.1)


# --- Simulation Functions ---
def nperson_payoff(my_move, num_cooperators, total_agents):
    others_coop = num_cooperators - (1 - my_move)
    if my_move == 0:  # cooperate
        return S + (R - S) * (others_coop / (total_agents - 1))
    else:  # defect
        return P + (T - P) * (others_coop / (total_agents - 1))


def run_pairwise_tournament(agents, num_rounds):
    for agent in agents: agent.reset()
    history = {a.agent_id: {'coop_rate': [], 'score': []} for a in agents}
    agent_map = {a.agent_id: a for a in agents}
    
    for _ in range(num_rounds):
        moves = {}
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                moves[(agent1.agent_id, agent2.agent_id)] = (
                    agent1.choose_pairwise_action(agent2.agent_id),
                    agent2.choose_pairwise_action(agent1.agent_id)
                )
        round_moves_by_agent = defaultdict(list)
        for (id1, id2), (move1, move2) in moves.items():
            payoff1, payoff2 = PAYOFFS_2P[(move1, move2)]
            agent_map[id1].record_pairwise_outcome(id2, move1, move2, payoff1)
            agent_map[id2].record_pairwise_outcome(id1, move2, move1, payoff2)
            round_moves_by_agent[id1].append(move1)
            round_moves_by_agent[id2].append(move2)
        
        for agent in agents:
            agent_moves = round_moves_by_agent[agent.agent_id]
            if not agent_moves: continue
            history[agent.agent_id]['coop_rate'].append(agent_moves.count(0) / len(agent_moves))
            history[agent.agent_id]['score'].append(agent.total_score)
    return history


def run_nperson_simulation(agents, num_rounds):
    for agent in agents: agent.reset()
    history = {a.agent_id: {'coop_rate': [], 'score': []} for a in agents}
    coop_ratio = None
    
    for _ in range(num_rounds):
        moves = {a.agent_id: a.choose_neighborhood_action(coop_ratio) for a in agents}
        num_cooperators = list(moves.values()).count(0)
        current_coop_ratio = num_cooperators / len(agents)
        for agent in agents:
            my_move = moves[agent.agent_id]
            payoff = nperson_payoff(my_move, num_cooperators, len(agents))
            agent.record_neighborhood_outcome(current_coop_ratio, payoff)
            history[agent.agent_id]['coop_rate'].append(1 - my_move)
            history[agent.agent_id]['score'].append(agent.total_score)
        coop_ratio = current_coop_ratio
    return history


# --- Analysis Functions ---
def run_experiment_set(agents, num_rounds, num_runs):
    """Runs both pairwise and neighborhood simulations for a given agent configuration."""
    p_runs = []
    n_runs = []
    
    for run in range(num_runs):
        fresh_agents = []
        for agent in agents:
            if isinstance(agent, PairwiseAdaptiveQLearner):
                fresh_agents.append(PairwiseAdaptiveQLearner(agent.agent_id, agent.params))
            else:
                fresh_agents.append(StaticAgent(agent.agent_id, agent.strategy_name, agent.error_rate))
        
        p_runs.append(run_pairwise_tournament(fresh_agents, num_rounds))
        n_runs.append(run_nperson_simulation(fresh_agents, num_rounds))
    
    p_agg = {aid: {m: np.mean([r[aid][m] for r in p_runs if aid in r], axis=0) for m in ['coop_rate', 'score']} 
             for aid in p_runs[0]}
    n_agg = {aid: {m: np.mean([r[aid][m] for r in n_runs if aid in r], axis=0) for m in ['coop_rate', 'score']} 
             for aid in n_runs[0]}
    
    return p_agg, n_agg


def smooth_data(data, window_size=50):
    """Apply rolling average to smooth data"""
    if len(data) < window_size:
        window_size = max(1, len(data) // 10)
    
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        smoothed.append(np.mean(data[start:end]))
    
    return np.array(smoothed)


def save_results_csv(results, scenario_name):
    """Save results to CSV files"""
    csv_dir = os.path.join(OUTPUT_DIR, "csv_files")
    os.makedirs(csv_dir, exist_ok=True)
    
    data_rows = []
    
    for agent_id, metrics in results.items():
        avg_coop = np.mean(metrics['coop_rate'])
        final_score = metrics['score'][-1] if len(metrics['score']) > 0 else 0
        
        data_rows.append({
            'Agent': agent_id,
            'Average_Cooperation_Rate': avg_coop,
            'Final_Score': final_score,
            'Agent_Type': agent_id.split('_')[0],
            'Scenario': scenario_name
        })
    
    df = pd.DataFrame(data_rows)
    csv_path = os.path.join(csv_dir, f"{scenario_name}.csv")
    df.to_csv(csv_path, index=False)
    
    return df


def plot_scenario(p_data, n_data, scenario_name, save_path):
    """Create a 4-panel plot for a scenario"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Scenario: {scenario_name}", fontsize=16)
    
    colors = {
        'Vanilla': '#1f77b4',
        'Adaptive': '#ff7f0e',
        'TFT': '#2ca02c',
        'TFT-E': '#d62728',
    }
    
    for agent_id in p_data.keys():
        if 'Vanilla' in agent_id:
            color = colors['Vanilla']
            label = agent_id
        elif 'Adaptive' in agent_id:
            color = colors['Adaptive']
            label = agent_id
        elif 'TFT-E' in agent_id:
            color = colors['TFT-E']
            label = agent_id
        elif 'TFT' in agent_id:
            color = colors['TFT']
            label = agent_id
        else:
            color = 'gray'
            label = agent_id
        
        p_coop_smooth = smooth_data(p_data[agent_id]['coop_rate'], 100)
        n_coop_smooth = smooth_data(n_data[agent_id]['coop_rate'], 100)
        
        axes[0, 0].plot(p_coop_smooth, label=label, color=color, linewidth=2)
        axes[0, 1].plot(p_data[agent_id]['score'], label=label, color=color, linewidth=2)
        axes[1, 0].plot(n_coop_smooth, label=label, color=color, linewidth=2)
        axes[1, 1].plot(n_data[agent_id]['score'], label=label, color=color, linewidth=2)
    
    axes[0, 0].set_title('Pairwise Cooperation Rate')
    axes[0, 0].set_ylabel('Cooperation Rate')
    axes[0, 0].set_ylim(-0.05, 1.05)
    
    axes[0, 1].set_title('Pairwise Cumulative Score')
    axes[0, 1].set_ylabel('Cumulative Score')
    
    axes[1, 0].set_title('Neighborhood Cooperation Rate')
    axes[1, 0].set_ylabel('Cooperation Rate')
    axes[1, 0].set_ylim(-0.05, 1.05)
    
    axes[1, 1].set_title('Neighborhood Cumulative Score')
    axes[1, 1].set_ylabel('Cumulative Score')
    
    for ax in axes.flat:
        ax.set_xlabel('Round')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_heatmap(all_results, save_path):
    """Create a heatmap comparing all scenarios"""
    scenarios = []
    agent_types = []
    avg_coop_pairwise = []
    avg_coop_neighbor = []
    final_score_pairwise = []
    final_score_neighbor = []
    
    for scenario_name, (p_data, n_data) in all_results.items():
        for agent_id in p_data.keys():
            if 'QL' in agent_id:
                scenarios.append(scenario_name)
                agent_types.append(agent_id)
                avg_coop_pairwise.append(np.mean(p_data[agent_id]['coop_rate']))
                avg_coop_neighbor.append(np.mean(n_data[agent_id]['coop_rate']))
                final_score_pairwise.append(p_data[agent_id]['score'][-1])
                final_score_neighbor.append(n_data[agent_id]['score'][-1])
    
    df = pd.DataFrame({
        'Scenario': scenarios,
        'Agent': agent_types,
        'Pairwise_Coop': avg_coop_pairwise,
        'Neighbor_Coop': avg_coop_neighbor,
        'Pairwise_Score': final_score_pairwise,
        'Neighbor_Score': final_score_neighbor
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Comparison Across All Scenarios', fontsize=16)
    
    pivot_coop_p = df.pivot_table(values='Pairwise_Coop', index='Agent', columns='Scenario')
    pivot_coop_n = df.pivot_table(values='Neighbor_Coop', index='Agent', columns='Scenario')
    
    sns.heatmap(pivot_coop_p, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[0, 0], vmin=0, vmax=1)
    axes[0, 0].set_title('Pairwise Cooperation Rates')
    
    sns.heatmap(pivot_coop_n, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[0, 1], vmin=0, vmax=1)
    axes[0, 1].set_title('Neighborhood Cooperation Rates')
    
    max_score = R * NUM_ROUNDS * 2
    pivot_score_p = df.pivot_table(values='Pairwise_Score', index='Agent', columns='Scenario') / max_score
    pivot_score_n = df.pivot_table(values='Neighbor_Score', index='Agent', columns='Scenario') / max_score
    
    sns.heatmap(pivot_score_p, annot=True, fmt='.3f', cmap='RdYlBu', ax=axes[1, 0])
    axes[1, 0].set_title('Pairwise Normalized Scores')
    
    sns.heatmap(pivot_score_n, annot=True, fmt='.3f', cmap='RdYlBu', ax=axes[1, 1])
    axes[1, 1].set_title('Neighborhood Normalized Scores')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_summary_stats(all_results):
    """Save summary statistics to a JSON file"""
    summary = {}
    
    for scenario_name, (p_data, n_data) in all_results.items():
        scenario_summary = {}
        
        for agent_id in p_data.keys():
            agent_summary = {
                'pairwise': {
                    'avg_cooperation': float(np.mean(p_data[agent_id]['coop_rate'])),
                    'final_score': float(p_data[agent_id]['score'][-1]),
                    'score_per_round': float(p_data[agent_id]['score'][-1] / NUM_ROUNDS)
                },
                'neighborhood': {
                    'avg_cooperation': float(np.mean(n_data[agent_id]['coop_rate'])),
                    'final_score': float(n_data[agent_id]['score'][-1]),
                    'score_per_round': float(n_data[agent_id]['score'][-1] / NUM_ROUNDS)
                }
            }
            scenario_summary[agent_id] = agent_summary
        
        summary[scenario_name] = scenario_summary
    
    with open(os.path.join(OUTPUT_DIR, 'summary_statistics.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def create_df_comparison_plots(all_results):
    """Create plots specifically comparing discount factor effects"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Effect of Discount Factor on Q-Learning Performance', fontsize=16)
    
    # Define distinct colors for each combination
    color_scheme = {
        ('DF04', 'TFT'): '#1f77b4',     # Blue
        ('DF04', 'TFT-E'): '#ff7f0e',   # Orange  
        ('DF095', 'TFT'): '#2ca02c',    # Green
        ('DF095', 'TFT-E'): '#d62728',  # Red
    }
    
    # Define line styles for better distinction
    line_styles = {
        'DF04': ':',      # Dotted for short-term
        'DF095': '-'      # Solid for long-term
    }
    
    scenarios_vanilla = ["1_Vanilla_DF04_vs_2_TFT", "1_Vanilla_DF095_vs_2_TFT",
                        "1_Vanilla_DF04_vs_2_TFT-E", "1_Vanilla_DF095_vs_2_TFT-E"]
    
    scenarios_adaptive = ["1_Adaptive_DF04_vs_2_TFT", "1_Adaptive_DF095_vs_2_TFT",
                         "1_Adaptive_DF04_vs_2_TFT-E", "1_Adaptive_DF095_vs_2_TFT-E"]
    
    for ax_idx, (scenarios, title_prefix) in enumerate([(scenarios_vanilla, "Vanilla"), 
                                                        (scenarios_adaptive, "Adaptive")]):
        for scenario in scenarios:
            if scenario in all_results:
                p_data, n_data = all_results[scenario]
                
                ql_agent = [aid for aid in p_data.keys() if 'QL' in aid][0]
                
                # Determine discount factor and opponent
                if 'DF04' in scenario:
                    df_key = 'DF04'
                    df_label = 'DF=0.4'
                else:
                    df_key = 'DF095'
                    df_label = 'DF=0.95'
                
                if 'TFT-E' in scenario:
                    opponent_key = 'TFT-E'
                    opponent = 'vs TFT-E (10% error)'
                else:
                    opponent_key = 'TFT'
                    opponent = 'vs TFT'
                
                # Get color and line style
                color = color_scheme[(df_key, opponent_key)]
                ls = line_styles[df_key]
                label = f"{df_label} {opponent}"
                
                p_coop_smooth = smooth_data(p_data[ql_agent]['coop_rate'], 100)
                n_coop_smooth = smooth_data(n_data[ql_agent]['coop_rate'], 100)
                
                axes[0, ax_idx].plot(p_coop_smooth, ls=ls, color=color, label=label, linewidth=2.5)
                axes[1, ax_idx].plot(n_coop_smooth, ls=ls, color=color, label=label, linewidth=2.5)
    
    axes[0, 0].set_title('Vanilla QL - Pairwise')
    axes[0, 1].set_title('Adaptive QL - Pairwise')
    axes[1, 0].set_title('Vanilla QL - Neighborhood')
    axes[1, 1].set_title('Adaptive QL - Neighborhood')
    
    for ax in axes.flat:
        ax.set_xlabel('Round')
        ax.set_ylabel('Cooperation Rate')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'discount_factor_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Updated discount factor comparison plot saved!")


def main():
    """Run all experiments"""
    print("=" * 80)
    print("Q-LEARNING VS TFT COMPREHENSIVE EXPERIMENT")
    print("=" * 80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    figures_dir = os.path.join(OUTPUT_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    scenarios = {
        "1_Vanilla_DF04_vs_2_TFT": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Vanilla_QL_DF04", VANILLA_DF_04),
                StaticAgent("TFT_1", "TFT", 0.0),
                StaticAgent("TFT_2", "TFT", 0.0)
            ]
        },
        "1_Vanilla_DF095_vs_2_TFT": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Vanilla_QL_DF095", VANILLA_DF_095),
                StaticAgent("TFT_1", "TFT", 0.0),
                StaticAgent("TFT_2", "TFT", 0.0)
            ]
        },
        "1_Adaptive_DF04_vs_2_TFT": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Adaptive_QL_DF04", ADAPTIVE_DF_04),
                StaticAgent("TFT_1", "TFT", 0.0),
                StaticAgent("TFT_2", "TFT", 0.0)
            ]
        },
        "1_Adaptive_DF095_vs_2_TFT": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Adaptive_QL_DF095", ADAPTIVE_DF_095),
                StaticAgent("TFT_1", "TFT", 0.0),
                StaticAgent("TFT_2", "TFT", 0.0)
            ]
        },
        "1_Vanilla_DF04_vs_2_TFT-E": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Vanilla_QL_DF04", VANILLA_DF_04),
                StaticAgent("TFT-E_1", "TFT-E", 0.1),
                StaticAgent("TFT-E_2", "TFT-E", 0.1)
            ]
        },
        "1_Vanilla_DF095_vs_2_TFT-E": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Vanilla_QL_DF095", VANILLA_DF_095),
                StaticAgent("TFT-E_1", "TFT-E", 0.1),
                StaticAgent("TFT-E_2", "TFT-E", 0.1)
            ]
        },
        "1_Adaptive_DF04_vs_2_TFT-E": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Adaptive_QL_DF04", ADAPTIVE_DF_04),
                StaticAgent("TFT-E_1", "TFT-E", 0.1),
                StaticAgent("TFT-E_2", "TFT-E", 0.1)
            ]
        },
        "1_Adaptive_DF095_vs_2_TFT-E": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Adaptive_QL_DF095", ADAPTIVE_DF_095),
                StaticAgent("TFT-E_1", "TFT-E", 0.1),
                StaticAgent("TFT-E_2", "TFT-E", 0.1)
            ]
        },
        "2_Vanilla_DF04_vs_1_TFT": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Vanilla_QL1_DF04", VANILLA_DF_04),
                PairwiseAdaptiveQLearner("Vanilla_QL2_DF04", VANILLA_DF_04),
                StaticAgent("TFT", "TFT", 0.0)
            ]
        },
        "2_Vanilla_DF095_vs_1_TFT": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Vanilla_QL1_DF095", VANILLA_DF_095),
                PairwiseAdaptiveQLearner("Vanilla_QL2_DF095", VANILLA_DF_095),
                StaticAgent("TFT", "TFT", 0.0)
            ]
        },
        "2_Adaptive_DF04_vs_1_TFT": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Adaptive_QL1_DF04", ADAPTIVE_DF_04),
                PairwiseAdaptiveQLearner("Adaptive_QL2_DF04", ADAPTIVE_DF_04),
                StaticAgent("TFT", "TFT", 0.0)
            ]
        },
        "2_Adaptive_DF095_vs_1_TFT": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Adaptive_QL1_DF095", ADAPTIVE_DF_095),
                PairwiseAdaptiveQLearner("Adaptive_QL2_DF095", ADAPTIVE_DF_095),
                StaticAgent("TFT", "TFT", 0.0)
            ]
        },
        "2_Vanilla_DF04_vs_1_TFT-E": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Vanilla_QL1_DF04", VANILLA_DF_04),
                PairwiseAdaptiveQLearner("Vanilla_QL2_DF04", VANILLA_DF_04),
                StaticAgent("TFT-E", "TFT-E", 0.1)
            ]
        },
        "2_Vanilla_DF095_vs_1_TFT-E": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Vanilla_QL1_DF095", VANILLA_DF_095),
                PairwiseAdaptiveQLearner("Vanilla_QL2_DF095", VANILLA_DF_095),
                StaticAgent("TFT-E", "TFT-E", 0.1)
            ]
        },
        "2_Adaptive_DF04_vs_1_TFT-E": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Adaptive_QL1_DF04", ADAPTIVE_DF_04),
                PairwiseAdaptiveQLearner("Adaptive_QL2_DF04", ADAPTIVE_DF_04),
                StaticAgent("TFT-E", "TFT-E", 0.1)
            ]
        },
        "2_Adaptive_DF095_vs_1_TFT-E": {
            "agents": lambda: [
                PairwiseAdaptiveQLearner("Adaptive_QL1_DF095", ADAPTIVE_DF_095),
                PairwiseAdaptiveQLearner("Adaptive_QL2_DF095", ADAPTIVE_DF_095),
                StaticAgent("TFT-E", "TFT-E", 0.1)
            ]
        }
    }
    
    all_results = {}
    total_scenarios = len(scenarios)
    
    for i, (scenario_name, scenario_config) in enumerate(scenarios.items(), 1):
        print(f"\n[{i}/{total_scenarios}] Running scenario: {scenario_name}")
        start_time = time.time()
        
        agents = scenario_config["agents"]()
        
        p_data, n_data = run_experiment_set(agents, NUM_ROUNDS, NUM_RUNS)
        all_results[scenario_name] = (p_data, n_data)
        
        save_results_csv(p_data, f"{scenario_name}_pairwise")
        save_results_csv(n_data, f"{scenario_name}_neighborhood")
        
        plot_path = os.path.join(figures_dir, f"{scenario_name}.png")
        plot_scenario(p_data, n_data, scenario_name, plot_path)
        
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.1f}s")
    
    print("\nCreating comparison heatmap...")
    heatmap_path = os.path.join(OUTPUT_DIR, "comparison_heatmap.png")
    create_comparison_heatmap(all_results, heatmap_path)
    
    print("Saving summary statistics...")
    save_summary_stats(all_results)
    
    print("Creating discount factor comparison plots...")
    create_df_comparison_plots(all_results)
    
    print(f"\nAll experiments complete! Results saved to '{OUTPUT_DIR}'")
    print("Files created:")
    print(f"  - {len(scenarios)} scenario plots in figures/")
    print(f"  - {len(scenarios) * 2} CSV files in csv_files/")
    print("  - comparison_heatmap.png")
    print("  - summary_statistics.json")
    print("  - discount_factor_comparison.png")


if __name__ == "__main__":
    main()