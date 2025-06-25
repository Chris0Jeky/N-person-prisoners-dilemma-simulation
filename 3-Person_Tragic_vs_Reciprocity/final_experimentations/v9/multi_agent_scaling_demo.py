#!/usr/bin/env python3
"""
Multi-agent scaling experiments to understand the effect of group size on Q-learning agents.
FIXED VERSION: Handles the strategy_name issue in parallel processing.

Tests:
- Group sizes: 3, 5, 7, 10, 15, 20, 25 total agents
- Uses Legacy3RoundQLearner with LEGACY_3ROUND_PARAMS
- Uses a modified QL without epsilon decay as comparison
- Tests the combinations from v6/multi_agent_demo.py
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

from final_agents import StaticAgent, Legacy3RoundQLearner
from config import LEGACY_3ROUND_PARAMS, SIMULATION_CONFIG
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


# --- Custom QL Agent without epsilon decay ---
class QLNoDecay(Legacy3RoundQLearner):
    """Q-learner without epsilon decay - fixed exploration rate"""
    def __init__(self, agent_id, params=None, **kwargs):
        # Create modified params without decay
        modified_params = {
            'lr': 0.1,
            'df': 0.95,
            'eps': 0.1,
            'epsilon_decay': 1.0,  # No decay
            'epsilon_min': 0.1,    # Same as initial
            'optimistic_init': 1.0,  # Cooperative initialization
            'history_length': 3
        }
        # Don't pass strategy_name to parent
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'strategy_name'}
        super().__init__(agent_id, modified_params, **filtered_kwargs)
        self.strategy_name = "QLNoDecay"


# --- Helper function to recreate agents safely ---
def recreate_agent(agent):
    """Safely recreate an agent for parallel processing"""
    agent_dict = agent.__dict__.copy()
    
    # Remove attributes that shouldn't be passed to __init__
    attributes_to_remove = ['strategy_name', 'agent_id', 'total_score', 'pairwise_history', 
                           'pairwise_q_tables', 'neighborhood_q_table', 'neighborhood_history',
                           'last_neighborhood_action', 'epsilon', 'interactions_count']
    
    for attr in attributes_to_remove:
        agent_dict.pop(attr, None)
    
    # Get the agent class and ID
    agent_class = type(agent)
    agent_id = agent.agent_id
    
    # Special handling for our custom classes
    if isinstance(agent, QLNoDecay):
        return QLNoDecay(agent_id)
    elif isinstance(agent, Legacy3RoundQLearner):
        return Legacy3RoundQLearner(agent_id, agent.params)
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


def save_checkpoint(data, filename):
    """Save checkpoint data to file"""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(filename):
    """Load checkpoint data from file"""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Checkpoint loaded from {filename}")
        return data
    return None


def plot_scaling_results(scaling_results, output_dir):
    """Create plots showing how cooperation scales with group size"""
    
    # Extract data for plotting
    group_sizes = sorted(scaling_results.keys())
    
    # Data structures for different metrics
    legacy3_pairwise_coop = []
    legacy3_neighborhood_coop = []
    nodecay_pairwise_coop = []
    nodecay_neighborhood_coop = []
    
    legacy3_pairwise_score = []
    legacy3_neighborhood_score = []
    nodecay_pairwise_score = []
    nodecay_neighborhood_score = []
    
    for size in group_sizes:
        if size not in scaling_results:
            continue
            
        p_data, n_data = scaling_results[size]
        
        # Find QL agents
        legacy3_agents = [aid for aid in p_data.keys() if 'Legacy3Round' in aid]
        nodecay_agents = [aid for aid in p_data.keys() if 'QLNoDecay' in aid]
        
        if legacy3_agents:
            # Average final cooperation rates (last 1000 rounds)
            legacy3_pairwise_coop.append(np.mean([np.mean(p_data[aid]['coop_rate'][-1000:]) for aid in legacy3_agents]))
            legacy3_neighborhood_coop.append(np.mean([np.mean(n_data[aid]['coop_rate'][-1000:]) for aid in legacy3_agents]))
            legacy3_pairwise_score.append(np.mean([p_data[aid]['score'][-1] for aid in legacy3_agents]))
            legacy3_neighborhood_score.append(np.mean([n_data[aid]['score'][-1] for aid in legacy3_agents]))
        
        if nodecay_agents:
            nodecay_pairwise_coop.append(np.mean([np.mean(p_data[aid]['coop_rate'][-1000:]) for aid in nodecay_agents]))
            nodecay_neighborhood_coop.append(np.mean([np.mean(n_data[aid]['coop_rate'][-1000:]) for aid in nodecay_agents]))
            nodecay_pairwise_score.append(np.mean([p_data[aid]['score'][-1] for aid in nodecay_agents]))
            nodecay_neighborhood_score.append(np.mean([n_data[aid]['score'][-1] for aid in nodecay_agents]))
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Q-Learning Performance Scaling with Group Size', fontsize=16)
    
    # Plot cooperation rates
    axes[0, 0].plot(group_sizes[:len(legacy3_pairwise_coop)], legacy3_pairwise_coop, 'o-', label='Legacy3Round', linewidth=2, markersize=8)
    axes[0, 0].plot(group_sizes[:len(nodecay_pairwise_coop)], nodecay_pairwise_coop, 's-', label='QL No Decay', linewidth=2, markersize=8)
    axes[0, 0].set_title('Pairwise Final Cooperation Rate')
    axes[0, 0].set_xlabel('Group Size')
    axes[0, 0].set_ylabel('Cooperation Rate')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1.05)
    
    axes[1, 0].plot(group_sizes[:len(legacy3_neighborhood_coop)], legacy3_neighborhood_coop, 'o-', label='Legacy3Round', linewidth=2, markersize=8)
    axes[1, 0].plot(group_sizes[:len(nodecay_neighborhood_coop)], nodecay_neighborhood_coop, 's-', label='QL No Decay', linewidth=2, markersize=8)
    axes[1, 0].set_title('Neighborhood Final Cooperation Rate')
    axes[1, 0].set_xlabel('Group Size')
    axes[1, 0].set_ylabel('Cooperation Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1.05)
    
    # Plot final scores
    axes[0, 1].plot(group_sizes[:len(legacy3_pairwise_score)], legacy3_pairwise_score, 'o-', label='Legacy3Round', linewidth=2, markersize=8)
    axes[0, 1].plot(group_sizes[:len(nodecay_pairwise_score)], nodecay_pairwise_score, 's-', label='QL No Decay', linewidth=2, markersize=8)
    axes[0, 1].set_title('Pairwise Final Score')
    axes[0, 1].set_xlabel('Group Size')
    axes[0, 1].set_ylabel('Total Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 1].plot(group_sizes[:len(legacy3_neighborhood_score)], legacy3_neighborhood_score, 'o-', label='Legacy3Round', linewidth=2, markersize=8)
    axes[1, 1].plot(group_sizes[:len(nodecay_neighborhood_score)], nodecay_neighborhood_score, 's-', label='QL No Decay', linewidth=2, markersize=8)
    axes[1, 1].set_title('Neighborhood Final Score')
    axes[1, 1].set_xlabel('Group Size')
    axes[1, 1].set_ylabel('Total Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scaling_comparison.png'), dpi=150)
    plt.close()


def plot_scenario_results(scenario_results, scenario_name, output_dir):
    """Create detailed plots for a specific scenario"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{scenario_name}', fontsize=16)
    
    p_data, n_data = scenario_results
    
    # Separate QL types
    legacy3_agents = sorted([aid for aid in p_data.keys() if 'Legacy3Round' in aid])
    nodecay_agents = sorted([aid for aid in p_data.keys() if 'QLNoDecay' in aid])
    other_agents = sorted([aid for aid in p_data.keys() if aid not in legacy3_agents + nodecay_agents])
    
    # Define colors
    colors_legacy3 = plt.cm.Blues(np.linspace(0.5, 0.9, len(legacy3_agents)))
    colors_nodecay = plt.cm.Reds(np.linspace(0.5, 0.9, len(nodecay_agents)))
    colors_other = plt.cm.Greys(np.linspace(0.3, 0.7, len(other_agents)))
    
    # Plot cooperation rates and scores
    for i, (agents, colors, label_prefix) in enumerate([
        (legacy3_agents, colors_legacy3, 'L3R'),
        (nodecay_agents, colors_nodecay, 'ND'),
        (other_agents, colors_other, '')
    ]):
        for j, aid in enumerate(agents):
            label = f"{label_prefix} {aid}" if label_prefix else aid
            linewidth = 2 if 'QL' in aid else 1
            alpha = 1.0 if 'QL' in aid else 0.5
            
            # Smooth cooperation rates
            p_coop_smooth = smooth_data(p_data[aid]['coop_rate'], 100)
            n_coop_smooth = smooth_data(n_data[aid]['coop_rate'], 100)
            
            axes[0, 0].plot(p_coop_smooth, label=label, color=colors[j], linewidth=linewidth, alpha=alpha)
            axes[1, 0].plot(n_coop_smooth, label=label, color=colors[j], linewidth=linewidth, alpha=alpha)
            axes[0, 1].plot(p_data[aid]['score'], label=label, color=colors[j], linewidth=linewidth, alpha=alpha)
            axes[1, 1].plot(n_data[aid]['score'], label=label, color=colors[j], linewidth=linewidth, alpha=alpha)
    
    axes[0, 0].set_title('Pairwise Cooperation Rate')
    axes[0, 0].set_ylabel('Cooperation Rate')
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].legend(fontsize=8, loc='best')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Pairwise Cumulative Score')
    axes[0, 1].set_ylabel('Cumulative Score')
    axes[0, 1].legend(fontsize=8, loc='best')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Neighborhood Cooperation Rate')
    axes[1, 0].set_ylabel('Cooperation Rate')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].legend(fontsize=8, loc='best')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Neighborhood Cumulative Score')
    axes[1, 1].set_ylabel('Cumulative Score')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].legend(fontsize=8, loc='best')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save in subdirectory
    plot_dir = os.path.join(output_dir, "scenario_plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f"{scenario_name}.png"), dpi=150)
    plt.close()


def save_scaling_results_csv(scaling_results, output_dir):
    """Save scaling results to CSV"""
    csv_path = os.path.join(output_dir, "scaling_results.csv")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'group_size', 'agent_type', 'mode',
            'avg_coop_rate', 'final_coop_rate', 'final_score', 'score_per_round'
        ])
        
        for size, (p_data, n_data) in scaling_results.items():
            # Legacy3Round agents
            legacy3_agents = [aid for aid in p_data.keys() if 'Legacy3Round' in aid]
            if legacy3_agents:
                # Pairwise
                avg_coop = np.mean([np.mean(p_data[aid]['coop_rate']) for aid in legacy3_agents])
                final_coop = np.mean([np.mean(p_data[aid]['coop_rate'][-1000:]) for aid in legacy3_agents])
                final_score = np.mean([p_data[aid]['score'][-1] for aid in legacy3_agents])
                score_per_round = final_score / len(p_data[legacy3_agents[0]]['score'])
                writer.writerow([size, 'Legacy3Round', 'pairwise', avg_coop, final_coop, final_score, score_per_round])
                
                # Neighborhood
                avg_coop = np.mean([np.mean(n_data[aid]['coop_rate']) for aid in legacy3_agents])
                final_coop = np.mean([np.mean(n_data[aid]['coop_rate'][-1000:]) for aid in legacy3_agents])
                final_score = np.mean([n_data[aid]['score'][-1] for aid in legacy3_agents])
                score_per_round = final_score / len(n_data[legacy3_agents[0]]['score'])
                writer.writerow([size, 'Legacy3Round', 'neighborhood', avg_coop, final_coop, final_score, score_per_round])
            
            # QLNoDecay agents
            nodecay_agents = [aid for aid in p_data.keys() if 'QLNoDecay' in aid]
            if nodecay_agents:
                # Pairwise
                avg_coop = np.mean([np.mean(p_data[aid]['coop_rate']) for aid in nodecay_agents])
                final_coop = np.mean([np.mean(p_data[aid]['coop_rate'][-1000:]) for aid in nodecay_agents])
                final_score = np.mean([p_data[aid]['score'][-1] for aid in nodecay_agents])
                score_per_round = final_score / len(p_data[nodecay_agents[0]]['score'])
                writer.writerow([size, 'QLNoDecay', 'pairwise', avg_coop, final_coop, final_score, score_per_round])
                
                # Neighborhood
                avg_coop = np.mean([np.mean(n_data[aid]['coop_rate']) for aid in nodecay_agents])
                final_coop = np.mean([np.mean(n_data[aid]['coop_rate'][-1000:]) for aid in nodecay_agents])
                final_score = np.mean([n_data[aid]['score'][-1] for aid in nodecay_agents])
                score_per_round = final_score / len(n_data[nodecay_agents[0]]['score'])
                writer.writerow([size, 'QLNoDecay', 'neighborhood', avg_coop, final_coop, final_score, score_per_round])
    
    print(f"Scaling results saved to: {csv_path}")


if __name__ == "__main__":
    NUM_ROUNDS = SIMULATION_CONFIG['num_rounds']
    NUM_RUNS = SIMULATION_CONFIG['num_runs']
    OUTPUT_DIR = "scaling_experiment_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Group sizes to test
    GROUP_SIZES = [3, 5, 7, 10, 15, 20, 25]
    
    # Opponent types from v6
    OPPONENT_TYPES = {
        "AllC": {"strategy": "AllC", "error_rate": 0.0},
        "AllD": {"strategy": "AllD", "error_rate": 0.0},
        "Random": {"strategy": "Random", "error_rate": 0.0},
        "TFT": {"strategy": "TFT", "error_rate": 0.0},
    }
    
    USE_PARALLEL = True
    n_cores = cpu_count()
    n_processes = max(1, n_cores - 1) if USE_PARALLEL else 1
    
    print(f"Running scaling experiments with group sizes: {GROUP_SIZES}")
    print(f"Using {NUM_ROUNDS} rounds and {NUM_RUNS} runs per scenario")
    if USE_PARALLEL:
        print(f"Using {n_processes} processes on {n_cores} available CPU cores")
    
    # Track all results
    all_scenario_results = {}
    scaling_results = {}  # For plotting scaling behavior
    
    # Check for checkpoint
    checkpoint_file = os.path.join(OUTPUT_DIR, "checkpoint.pkl")
    checkpoint_data = load_checkpoint(checkpoint_file)
    
    if checkpoint_data:
        all_scenario_results = checkpoint_data.get('all_scenario_results', {})
        scaling_results = checkpoint_data.get('scaling_results', {})
        print(f"Resuming from checkpoint with {len(all_scenario_results)} completed scenarios")
    
    total_start_time = time.time()
    
    # --- Test 1: 2 QL agents vs varying numbers of opponents ---
    print("\n=== Testing 2 QL agents with varying group sizes ===")
    
    for group_size in GROUP_SIZES:
        if group_size < 3:
            continue  # Need at least 3 agents
        
        n_opponents = group_size - 2  # 2 QL agents, rest are opponents
        
        for opp_name, opp_config in OPPONENT_TYPES.items():
            scenario_name = f"{group_size}agents_2QL_vs_{n_opponents}{opp_name}"
            
            # Skip if already completed
            if scenario_name in all_scenario_results:
                print(f"\nScenario: {scenario_name} (already completed, skipping)")
                continue
                
            print(f"\nScenario: {scenario_name}")
            
            # Create agents
            agents = []
            
            # Add 2 Legacy3Round QL agents
            agents.append(Legacy3RoundQLearner(agent_id="Legacy3Round_QL_1", params=LEGACY_3ROUND_PARAMS))
            agents.append(Legacy3RoundQLearner(agent_id="Legacy3Round_QL_2", params=LEGACY_3ROUND_PARAMS))
            
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
            print(f"  Completed in {elapsed:.1f}s")
            
            all_scenario_results[scenario_name] = (p_data, n_data)
            
            # Save for scaling analysis (only for AllC opponent)
            if opp_name == "AllC":
                scaling_results[group_size] = (p_data, n_data)
            
            # Plot individual scenario
            plot_scenario_results((p_data, n_data), scenario_name, OUTPUT_DIR)
            
            # Save checkpoint after each scenario
            save_checkpoint({
                'all_scenario_results': all_scenario_results,
                'scaling_results': scaling_results
            }, checkpoint_file)
    
    # --- Test 2: Mixed QL types ---
    print("\n=== Testing mixed QL types with varying group sizes ===")
    
    for group_size in GROUP_SIZES:
        if group_size < 4:
            continue  # Need at least 4 agents for meaningful comparison
        
        scenario_name = f"{group_size}agents_MixedQL"
        
        # Skip if already completed
        if scenario_name in all_scenario_results:
            print(f"\nScenario: {scenario_name} (already completed, skipping)")
            continue
            
        print(f"\nScenario: {scenario_name}")
        
        # Create mixed QL agents
        agents = []
        n_legacy3 = group_size // 2
        n_nodecay = group_size - n_legacy3
        
        # Add Legacy3Round agents
        for i in range(n_legacy3):
            agents.append(Legacy3RoundQLearner(
                agent_id=f"Legacy3Round_QL_{i+1}",
                params=LEGACY_3ROUND_PARAMS
            ))
        
        # Add QLNoDecay agents
        for i in range(n_nodecay):
            agents.append(QLNoDecay(agent_id=f"QLNoDecay_{i+1}"))
        
        # Run experiment
        start_time = time.time()
        p_data, n_data = run_experiment_set(agents, NUM_ROUNDS, NUM_RUNS, USE_PARALLEL)
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.1f}s")
        
        all_scenario_results[scenario_name] = (p_data, n_data)
        plot_scenario_results((p_data, n_data), scenario_name, OUTPUT_DIR)
        
        # Save checkpoint
        save_checkpoint({
            'all_scenario_results': all_scenario_results,
            'scaling_results': scaling_results
        }, checkpoint_file)
    
    # --- Test 3: All same QL type ---
    print("\n=== Testing all same QL type with varying group sizes ===")
    
    for group_size in GROUP_SIZES:
        # All Legacy3Round
        scenario_name = f"{group_size}agents_AllLegacy3Round"
        
        if scenario_name not in all_scenario_results:
            print(f"\nScenario: {scenario_name}")
            
            agents = []
            for i in range(group_size):
                agents.append(Legacy3RoundQLearner(
                    agent_id=f"Legacy3Round_QL_{i+1}",
                    params=LEGACY_3ROUND_PARAMS
                ))
            
            start_time = time.time()
            p_data, n_data = run_experiment_set(agents, NUM_ROUNDS, NUM_RUNS, USE_PARALLEL)
            elapsed = time.time() - start_time
            print(f"  Completed in {elapsed:.1f}s")
            
            all_scenario_results[scenario_name] = (p_data, n_data)
            plot_scenario_results((p_data, n_data), scenario_name, OUTPUT_DIR)
            
            # Save checkpoint
            save_checkpoint({
                'all_scenario_results': all_scenario_results,
                'scaling_results': scaling_results
            }, checkpoint_file)
        else:
            print(f"\nScenario: {scenario_name} (already completed, skipping)")
        
        # All QLNoDecay
        scenario_name = f"{group_size}agents_AllQLNoDecay"
        
        if scenario_name not in all_scenario_results:
            print(f"\nScenario: {scenario_name}")
            
            agents = []
            for i in range(group_size):
                agents.append(QLNoDecay(agent_id=f"QLNoDecay_{i+1}"))
            
            start_time = time.time()
            p_data, n_data = run_experiment_set(agents, NUM_ROUNDS, NUM_RUNS, USE_PARALLEL)
            elapsed = time.time() - start_time
            print(f"  Completed in {elapsed:.1f}s")
            
            all_scenario_results[scenario_name] = (p_data, n_data)
            plot_scenario_results((p_data, n_data), scenario_name, OUTPUT_DIR)
            
            # Save checkpoint
            save_checkpoint({
                'all_scenario_results': all_scenario_results,
                'scaling_results': scaling_results
            }, checkpoint_file)
        else:
            print(f"\nScenario: {scenario_name} (already completed, skipping)")
    
    # --- Create scaling comparison plot ---
    print("\n=== Creating scaling analysis plots ===")
    plot_scaling_results(scaling_results, OUTPUT_DIR)
    
    # --- Save results to CSV ---
    print("\n=== Saving results to CSV ===")
    save_scaling_results_csv(scaling_results, OUTPUT_DIR)
    
    # --- Create summary ---
    summary_path = os.path.join(OUTPUT_DIR, "experiment_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MULTI-AGENT SCALING EXPERIMENT SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Group sizes tested: {GROUP_SIZES}\n")
        f.write(f"Number of rounds: {NUM_ROUNDS}\n")
        f.write(f"Number of runs per scenario: {NUM_RUNS}\n")
        f.write(f"Total scenarios tested: {len(all_scenario_results)}\n")
        f.write("\nQL Agent Types:\n")
        f.write("- Legacy3Round: Uses LEGACY_3ROUND_PARAMS with 3-round history\n")
        f.write("- QLNoDecay: No epsilon decay, fixed exploration rate of 0.1\n")
        f.write("\nExperiment Types:\n")
        f.write("1. 2 QL agents vs varying numbers of opponents (AllC, AllD, Random, TFT)\n")
        f.write("2. Mixed QL types (half Legacy3Round, half QLNoDecay)\n")
        f.write("3. All same QL type scenarios\n")
    
    # Calculate total time
    total_elapsed = time.time() - total_start_time
    print(f"\nAll experiments complete! Results saved to '{OUTPUT_DIR}'")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    
    # Clean up checkpoint file
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("Checkpoint file removed.")