#!/usr/bin/env python3
"""
Multi-agent scaling experiments to understand the effect of group size on Q-learning agents.
V2: Includes 1QL, 2QL, and 3QL variations with SimpleQL and advanced QL learners.

Tests:
- Group sizes: 3, 5, 7, 10, 15, 20, 25 total agents
- QL counts: 1, 2, or 3 Q-learners per scenario
- Uses Legacy3RoundQLearner, LegacyQLearner, and SimpleQLearner
- Tests the combinations similar to v6/multi_agent_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import csv
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import time
from datetime import datetime
import pickle
import random

from final_agents import StaticAgent, Legacy3RoundQLearner, LegacyQLearner, BaseAgent
from config import LEGACY_3ROUND_PARAMS, LEGACY_PARAMS, SIMULATION_CONFIG
from save_config import save_detailed_config

# --- Payoff Logic ---
PAYOFFS_2P = {(0, 0): (3, 3), (0, 1): (0, 5), (1, 0): (5, 0), (1, 1): (1, 1)}
T, R, P, S = 5, 3, 1, 0


def nperson_payoff(my_move, num_cooperators, total_agents):
    """Calculate N-person payoff"""
    others_coop = num_cooperators - (1 - my_move)
    if my_move == 0:  # cooperate
        return S + (R - S) * (others_coop / (total_agents - 1))
    else:  # defect
        return P + (T - P) * (others_coop / (total_agents - 1))


# --- Simple QL Agent ---
class SimpleQLearner(BaseAgent):
    """Simple Q-Learning agent with fixed parameters and no decay"""
    def __init__(self, agent_id, params=None):
        super().__init__(agent_id, "SimpleQL")
        # Simple fixed parameters
        self.lr = 0.1
        self.df = 0.95
        self.eps = 0.1
        
        # Q-tables
        self.q_tables = {}  # opponent_id -> Q-table for pairwise
        self.n_q_table = {}  # Q-table for neighborhood
        
        # State tracking
        self.opponent_last_moves = {}
        self.last_coop_ratio = None
    
    def reset(self):
        super().reset()
        self.q_tables = {}
        self.n_q_table = {}
        self.opponent_last_moves = {}
        self.last_coop_ratio = None
    
    def _get_pairwise_state(self, opponent_id):
        """Simple state: opponent's last move"""
        if opponent_id not in self.opponent_last_moves:
            return 'initial'
        return 'C' if self.opponent_last_moves[opponent_id] == 0 else 'D'
    
    def _get_neighborhood_state(self):
        """Simple state: cooperation ratio category"""
        if self.last_coop_ratio is None:
            return 'initial'
        if self.last_coop_ratio <= 0.33:
            return 'low'
        elif self.last_coop_ratio <= 0.67:
            return 'medium'
        else:
            return 'high'
    
    def _ensure_state_exists(self, state, q_table):
        """Initialize Q-values for new states"""
        if state not in q_table:
            q_table[state] = {0: 0.0, 1: 0.0}  # C: 0, D: 1
    
    def choose_pairwise_action(self, opponent_id):
        """Choose action for pairwise interaction"""
        # Initialize Q-table for new opponent
        if opponent_id not in self.q_tables:
            self.q_tables[opponent_id] = {}
        
        state = self._get_pairwise_state(opponent_id)
        self._ensure_state_exists(state, self.q_tables[opponent_id])
        
        # Epsilon-greedy action selection
        if random.random() < self.eps:
            return random.choice([0, 1])
        else:
            q_values = self.q_tables[opponent_id][state]
            if q_values[0] >= q_values[1]:
                return 0  # Cooperate
            else:
                return 1  # Defect
    
    def choose_neighborhood_action(self, coop_ratio):
        """Choose action for neighborhood interaction"""
        self.last_coop_ratio = coop_ratio
        state = self._get_neighborhood_state()
        self._ensure_state_exists(state, self.n_q_table)
        
        # Epsilon-greedy action selection
        if random.random() < self.eps:
            action = random.choice([0, 1])
        else:
            q_values = self.n_q_table[state]
            if q_values[0] >= q_values[1]:
                action = 0
            else:
                action = 1
        
        self.last_n_action = action
        self.last_n_state = state
        return action
    
    def record_pairwise_outcome(self, opponent_id, my_move, opponent_move, reward):
        """Record outcome and update Q-values"""
        self.total_score += reward
        
        # Update opponent's last move
        self.opponent_last_moves[opponent_id] = opponent_move
        
        # Q-learning update
        if opponent_id in self.q_tables:
            state = self._get_pairwise_state(opponent_id)
            self._ensure_state_exists(state, self.q_tables[opponent_id])
            
            # Get next state value
            next_state = 'C' if opponent_move == 0 else 'D'
            self._ensure_state_exists(next_state, self.q_tables[opponent_id])
            max_next_q = max(self.q_tables[opponent_id][next_state].values())
            
            # Update Q-value
            current_q = self.q_tables[opponent_id][state][my_move]
            self.q_tables[opponent_id][state][my_move] = current_q + self.lr * (reward + self.df * max_next_q - current_q)
    
    def record_neighborhood_outcome(self, coop_ratio, reward):
        """Record outcome and update Q-values"""
        self.total_score += reward
        
        # Q-learning update
        if hasattr(self, 'last_n_state') and hasattr(self, 'last_n_action'):
            # Get next state
            self.last_coop_ratio = coop_ratio
            next_state = self._get_neighborhood_state()
            self._ensure_state_exists(next_state, self.n_q_table)
            max_next_q = max(self.n_q_table[next_state].values())
            
            # Update Q-value
            current_q = self.n_q_table[self.last_n_state][self.last_n_action]
            self.n_q_table[self.last_n_state][self.last_n_action] = current_q + self.lr * (reward + self.df * max_next_q - current_q)


# --- Helper function to recreate agents safely ---
def recreate_agent(agent):
    """Safely recreate an agent for parallel processing"""
    agent_dict = agent.__dict__.copy()
    
    # Remove attributes that shouldn't be passed to __init__
    attributes_to_remove = ['strategy_name', 'agent_id', 'total_score', 'pairwise_history', 
                           'pairwise_q_tables', 'neighborhood_q_table', 'neighborhood_history',
                           'last_neighborhood_action', 'epsilon', 'interactions_count',
                           'q_tables', 'n_q_table', 'opponent_last_moves', 'last_coop_ratio',
                           'last_n_state', 'last_n_action']
    
    for attr in attributes_to_remove:
        agent_dict.pop(attr, None)
    
    # Get the agent class and ID
    agent_class = type(agent)
    agent_id = agent.agent_id
    
    # Special handling for our custom classes
    if isinstance(agent, Legacy3RoundQLearner):
        return Legacy3RoundQLearner(agent_id, agent.params)
    elif isinstance(agent, LegacyQLearner):
        return LegacyQLearner(agent_id, agent.params)
    elif isinstance(agent, SimpleQLearner):
        return SimpleQLearner(agent_id)
    elif isinstance(agent, StaticAgent):
        return StaticAgent(agent_id, strategy_name=agent.strategy_name, error_rate=agent.error_rate)
    else:
        # Generic recreation
        return agent_class(agent_id, **agent_dict)


# --- Simulation Runners ---
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


def run_single_pairwise(args):
    """Helper function for parallel pairwise simulation"""
    agents, num_rounds = args
    return run_pairwise_tournament([recreate_agent(a) for a in agents], num_rounds)


def run_single_nperson(args):
    """Helper function for parallel n-person simulation"""
    agents, num_rounds = args
    return run_nperson_simulation([recreate_agent(a) for a in agents], num_rounds)


def run_experiment_set(agents, num_rounds, num_runs, use_parallel=True):
    """Runs both pairwise and neighborhood simulations for a given agent configuration."""
    if use_parallel:
        n_processes = max(1, cpu_count() - 1)
        
        with Pool(processes=n_processes) as pool:
            p_args = [(agents, num_rounds) for _ in range(num_runs)]
            p_runs = pool.map(run_single_pairwise, p_args)
            
            n_args = [(agents, num_rounds) for _ in range(num_runs)]
            n_runs = pool.map(run_single_nperson, n_args)
    else:
        p_runs = [run_pairwise_tournament([recreate_agent(a) for a in agents], num_rounds) for _ in range(num_runs)]
        n_runs = [run_nperson_simulation([recreate_agent(a) for a in agents], num_rounds) for _ in range(num_runs)]
    
    # Aggregate results
    p_agg = {aid: {m: np.mean([r[aid][m] for r in p_runs if aid in r], axis=0) for m in ['coop_rate', 'score']} 
             for aid in p_runs[0]}
    n_agg = {aid: {m: np.mean([r[aid][m] for r in n_runs if aid in r], axis=0) for m in ['coop_rate', 'score']} 
             for aid in n_runs[0]}
    
    return p_agg, n_agg


def smooth_data(data, window_size=50):
    """Apply rolling average to smooth data"""
    if len(data) < window_size:
        window_size = max(1, len(data) // 10)
    
    series = pd.Series(data)
    smoothed = series.rolling(window=window_size, center=True, min_periods=1).mean()
    return smoothed.values


def plot_scenario_comparison(results, scenario_name, output_dir):
    """Create comparison plots for different QL types in a scenario"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{scenario_name}', fontsize=16)
    
    p_data_by_type = defaultdict(list)
    n_data_by_type = defaultdict(list)
    
    # Group data by QL type
    for ql_type, (p_data, n_data) in results.items():
        # Find QL agents
        ql_agents = [aid for aid in p_data.keys() if 'QL' in aid]
        
        if ql_agents:
            # Average across QL agents of this type
            avg_p_coop = np.mean([p_data[aid]['coop_rate'] for aid in ql_agents], axis=0)
            avg_p_score = np.mean([p_data[aid]['score'] for aid in ql_agents], axis=0)
            avg_n_coop = np.mean([n_data[aid]['coop_rate'] for aid in ql_agents], axis=0)
            avg_n_score = np.mean([n_data[aid]['score'] for aid in ql_agents], axis=0)
            
            p_data_by_type[ql_type] = {'coop': avg_p_coop, 'score': avg_p_score}
            n_data_by_type[ql_type] = {'coop': avg_n_coop, 'score': avg_n_score}
    
    # Colors for different QL types
    colors = {
        'SimpleQL': '#1f77b4',
        'LegacyQL': '#ff7f0e', 
        'Legacy3Round': '#2ca02c'
    }
    
    # Plot results
    for ql_type in ['SimpleQL', 'LegacyQL', 'Legacy3Round']:
        if ql_type in p_data_by_type:
            color = colors.get(ql_type, '#808080')
            
            # Apply smoothing to cooperation rates
            p_coop_smooth = smooth_data(p_data_by_type[ql_type]['coop'], 100)
            n_coop_smooth = smooth_data(n_data_by_type[ql_type]['coop'], 100)
            
            # Plot with raw data in background (low alpha) and smoothed in foreground
            axes[0, 0].plot(p_data_by_type[ql_type]['coop'], color=color, alpha=0.2, linewidth=0.5)
            axes[0, 0].plot(p_coop_smooth, label=ql_type, color=color, linewidth=2.5)
            
            axes[1, 0].plot(n_data_by_type[ql_type]['coop'], color=color, alpha=0.2, linewidth=0.5)
            axes[1, 0].plot(n_coop_smooth, label=ql_type, color=color, linewidth=2.5)
            
            axes[0, 1].plot(p_data_by_type[ql_type]['score'], label=ql_type, color=color, linewidth=2)
            axes[1, 1].plot(n_data_by_type[ql_type]['score'], label=ql_type, color=color, linewidth=2)
    
    axes[0, 0].set_title('Pairwise Cooperation Rate')
    axes[0, 0].set_ylabel('Cooperation Rate')
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Pairwise Cumulative Score')
    axes[0, 1].set_ylabel('Cumulative Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Neighborhood Cooperation Rate')
    axes[1, 0].set_ylabel('Cooperation Rate')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Neighborhood Cumulative Score')
    axes[1, 1].set_ylabel('Cumulative Score')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save in subdirectory
    plot_dir = os.path.join(output_dir, "scenario_plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f"{scenario_name}.png"), dpi=150)
    plt.close()


def save_summary_results_csv(all_results, output_dir):
    """Save summary results to CSV files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Summary data
    summary_data = []
    
    for scenario_name, scenario_results in all_results.items():
        for ql_type, (p_data, n_data) in scenario_results.items():
            # Find QL agents
            ql_agents = [aid for aid in p_data.keys() if 'QL' in aid]
            
            if ql_agents:
                # Calculate metrics
                p_avg_coop = np.mean([np.mean(p_data[aid]['coop_rate']) for aid in ql_agents])
                p_final_coop = np.mean([np.mean(p_data[aid]['coop_rate'][-1000:]) for aid in ql_agents])
                p_final_score = np.mean([p_data[aid]['score'][-1] for aid in ql_agents])
                
                n_avg_coop = np.mean([np.mean(n_data[aid]['coop_rate']) for aid in ql_agents])
                n_final_coop = np.mean([np.mean(n_data[aid]['coop_rate'][-1000:]) for aid in ql_agents])
                n_final_score = np.mean([n_data[aid]['score'][-1] for aid in ql_agents])
                
                summary_data.append({
                    'scenario': scenario_name,
                    'ql_type': ql_type,
                    'pairwise_avg_coop': p_avg_coop,
                    'pairwise_final_coop': p_final_coop,
                    'pairwise_final_score': p_final_score,
                    'neighborhood_avg_coop': n_avg_coop,
                    'neighborhood_final_coop': n_final_coop,
                    'neighborhood_final_score': n_final_score
                })
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, f"summary_results_{timestamp}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary results saved to: {summary_path}")
    
    # Create detailed time series for select scenarios
    detailed_scenarios = [s for s in all_results.keys() if '2QL' in s and 'AllC' in s]
    
    for scenario_name in detailed_scenarios:
        if scenario_name not in all_results:
            continue
            
        scenario_data = []
        for ql_type, (p_data, n_data) in all_results[scenario_name].items():
            ql_agents = [aid for aid in p_data.keys() if 'QL' in aid]
            
            if ql_agents:
                # Average across QL agents
                p_coop = np.mean([p_data[aid]['coop_rate'] for aid in ql_agents], axis=0)
                p_score = np.mean([p_data[aid]['score'] for aid in ql_agents], axis=0)
                n_coop = np.mean([n_data[aid]['coop_rate'] for aid in ql_agents], axis=0)
                n_score = np.mean([n_data[aid]['score'] for aid in ql_agents], axis=0)
                
                for round_num in range(len(p_coop)):
                    scenario_data.append({
                        'round': round_num + 1,
                        'ql_type': ql_type,
                        'pairwise_coop': p_coop[round_num],
                        'pairwise_score': p_score[round_num],
                        'neighborhood_coop': n_coop[round_num],
                        'neighborhood_score': n_score[round_num]
                    })
        
        if scenario_data:
            detailed_df = pd.DataFrame(scenario_data)
            detailed_path = os.path.join(output_dir, f"{scenario_name}_timeseries_{timestamp}.csv")
            detailed_df.to_csv(detailed_path, index=False)


if __name__ == "__main__":
    NUM_ROUNDS = SIMULATION_CONFIG['num_rounds']
    NUM_RUNS = SIMULATION_CONFIG['num_runs']
    OUTPUT_DIR = "scaling_experiment_results_v2"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Group sizes to test
    GROUP_SIZES = [3, 5, 7, 10, 15, 20, 25]
    
    # QL counts to test
    QL_COUNTS = [1, 2, 3]
    
    # Opponent types
    OPPONENT_TYPES = {
        "AllC": {"strategy": "AllC", "error_rate": 0.0},
        "AllD": {"strategy": "AllD", "error_rate": 0.0},
        "Random": {"strategy": "Random", "error_rate": 0.0},
        "TFT": {"strategy": "TFT", "error_rate": 0.0},
    }
    
    # Q-learner configurations
    QL_CONFIGS = {
        "SimpleQL": {"class": SimpleQLearner, "params": None},
        "LegacyQL": {"class": LegacyQLearner, "params": LEGACY_PARAMS},
        "Legacy3Round": {"class": Legacy3RoundQLearner, "params": LEGACY_3ROUND_PARAMS},
    }
    
    USE_PARALLEL = True
    n_cores = cpu_count()
    n_processes = max(1, n_cores - 1) if USE_PARALLEL else 1
    
    print(f"Running scaling experiments with group sizes: {GROUP_SIZES}")
    print(f"QL counts per scenario: {QL_COUNTS}")
    print(f"Using {NUM_ROUNDS} rounds and {NUM_RUNS} runs per scenario")
    if USE_PARALLEL:
        print(f"Using {n_processes} processes on {n_cores} available CPU cores")
    
    all_results = {}
    total_start_time = time.time()
    
    # --- Test 1: N QL agents vs varying numbers of opponents ---
    print("\n=== Testing N QL agents vs opponents ===")
    
    for group_size in GROUP_SIZES:
        for n_ql in QL_COUNTS:
            if n_ql >= group_size:
                continue  # Skip if not enough room for opponents
            
            n_opponents = group_size - n_ql
            
            for opp_name, opp_config in OPPONENT_TYPES.items():
                scenario_name = f"{group_size}agents_{n_ql}QL_vs_{n_opponents}{opp_name}"
                print(f"\nScenario: {scenario_name}")
                
                scenario_results = {}
                
                for ql_name, ql_config in QL_CONFIGS.items():
                    print(f"  Testing {ql_name}...", end='', flush=True)
                    agents = []
                    
                    # Add QL agents
                    for i in range(n_ql):
                        if ql_config["params"]:
                            agents.append(ql_config["class"](
                                agent_id=f"{ql_name}_{i+1}",
                                params=ql_config["params"]
                            ))
                        else:
                            agents.append(ql_config["class"](
                                agent_id=f"{ql_name}_{i+1}"
                            ))
                    
                    # Add opponents
                    for i in range(n_opponents):
                        agents.append(StaticAgent(
                            agent_id=f"{opp_name}_{i+1}",
                            strategy_name=opp_config["strategy"],
                            error_rate=opp_config["error_rate"]
                        ))
                    
                    # Run experiment
                    start_time = time.time()
                    p_data, n_data = run_experiment_set(agents, NUM_ROUNDS, NUM_RUNS, USE_PARALLEL)
                    elapsed = time.time() - start_time
                    print(f" done in {elapsed:.1f}s")
                    
                    scenario_results[ql_name] = (p_data, n_data)
                
                all_results[scenario_name] = scenario_results
                plot_scenario_comparison(scenario_results, scenario_name, OUTPUT_DIR)
    
    # --- Test 2: All QL scenarios (different types) ---
    print("\n=== Testing all-QL scenarios ===")
    
    for group_size in GROUP_SIZES:
        # Test each QL type separately
        for ql_name, ql_config in QL_CONFIGS.items():
            scenario_name = f"{group_size}agents_All{ql_name}"
            print(f"\nScenario: {scenario_name}")
            
            agents = []
            for i in range(group_size):
                if ql_config["params"]:
                    agents.append(ql_config["class"](
                        agent_id=f"{ql_name}_{i+1}",
                        params=ql_config["params"]
                    ))
                else:
                    agents.append(ql_config["class"](
                        agent_id=f"{ql_name}_{i+1}"
                    ))
            
            start_time = time.time()
            p_data, n_data = run_experiment_set(agents, NUM_ROUNDS, NUM_RUNS, USE_PARALLEL)
            elapsed = time.time() - start_time
            print(f"  Completed in {elapsed:.1f}s")
            
            # Store as single result type for all-QL scenario
            all_results[scenario_name] = {ql_name: (p_data, n_data)}
        
        # Mixed QL types scenario
        if group_size >= 3:  # Need at least 3 for meaningful mix
            scenario_name = f"{group_size}agents_MixedQL"
            print(f"\nScenario: {scenario_name}")
            
            agents = []
            ql_types = list(QL_CONFIGS.keys())
            
            # Distribute agents roughly evenly among QL types
            for i in range(group_size):
                ql_type = ql_types[i % len(ql_types)]
                ql_config = QL_CONFIGS[ql_type]
                
                if ql_config["params"]:
                    agents.append(ql_config["class"](
                        agent_id=f"{ql_type}_{(i // len(ql_types)) + 1}",
                        params=ql_config["params"]
                    ))
                else:
                    agents.append(ql_config["class"](
                        agent_id=f"{ql_type}_{(i // len(ql_types)) + 1}"
                    ))
            
            start_time = time.time()
            p_data, n_data = run_experiment_set(agents, NUM_ROUNDS, NUM_RUNS, USE_PARALLEL)
            elapsed = time.time() - start_time
            print(f"  Completed in {elapsed:.1f}s")
            
            # Store mixed results
            all_results[scenario_name] = {"Mixed": (p_data, n_data)}
    
    # --- Save Results ---
    print("\n=== Saving results ===")
    save_summary_results_csv(all_results, OUTPUT_DIR)
    
    # --- Create summary ---
    summary_path = os.path.join(OUTPUT_DIR, "experiment_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MULTI-AGENT SCALING EXPERIMENT SUMMARY V2\n")
        f.write("="*80 + "\n\n")
        f.write(f"Group sizes tested: {GROUP_SIZES}\n")
        f.write(f"QL counts tested: {QL_COUNTS}\n")
        f.write(f"Number of rounds: {NUM_ROUNDS}\n")
        f.write(f"Number of runs per scenario: {NUM_RUNS}\n")
        f.write(f"Total scenarios tested: {len(all_results)}\n")
        f.write("\nQL Agent Types:\n")
        f.write("- SimpleQL: Basic Q-learning with fixed parameters (lr=0.1, df=0.95, eps=0.1)\n")
        f.write("- LegacyQL: 2-round history with sophisticated state representation\n")
        f.write("- Legacy3Round: 3-round history with sophisticated state representation\n")
        f.write("\nExperiment Types:\n")
        f.write("1. N QL agents (1, 2, or 3) vs varying numbers of opponents\n")
        f.write("2. All same QL type scenarios\n")
        f.write("3. Mixed QL type scenarios\n")
    
    # Calculate total time
    total_elapsed = time.time() - total_start_time
    print(f"\nAll experiments complete! Results saved to '{OUTPUT_DIR}'")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")