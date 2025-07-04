#!/usr/bin/env python3
"""
Multi-agent experiments testing the effect of group size on Q-learning agents.

Tests:
- 5, 7, 19, and 25 total agents
- 1 or 3 Q-learning agents per scenario
- Vanilla QL and Hysteretic QL only
- Special scenarios: all same QL type, and mixed QL types
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

from final_agents import StaticAgent, PairwiseAdaptiveQLearner, HystereticQLearner
from config import VANILLA_PARAMS, HYSTERETIC_PARAMS, SIMULATION_CONFIG
from save_config import save_detailed_config

# --- Payoff Logic ---
PAYOFFS_2P = {(0, 0): (3, 3), (0, 1): (0, 5), (1, 0): (5, 0), (1, 1): (1, 1)}
T, R, P, S = 5, 3, 1, 0


def nperson_payoff(my_move, num_cooperators, total_agents):
    # num_cooperators includes all agents who cooperated (including self if applicable)
    # Calculate the number of OTHER agents who cooperated
    others_coop = num_cooperators - (1 - my_move)  # subtract 1 if I cooperated
    if my_move == 0:  # cooperate
        return S + (R - S) * (others_coop / (total_agents - 1))
    else:  # defect
        return P + (T - P) * (others_coop / (total_agents - 1))


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
    return run_pairwise_tournament([type(a)(**a.__dict__) for a in agents], num_rounds)

def run_single_nperson(args):
    """Helper function for parallel n-person simulation"""
    agents, num_rounds = args
    return run_nperson_simulation([type(a)(**a.__dict__) for a in agents], num_rounds)

def run_experiment_set(agents, num_rounds, num_runs, use_parallel=True):
    """Runs both pairwise and neighborhood simulations for a given agent configuration."""
    if use_parallel:
        # Determine number of processes (leave 1 CPU free for system)
        n_processes = max(1, cpu_count() - 1)
        
        with Pool(processes=n_processes) as pool:
            # Pairwise simulations
            p_args = [(agents, num_rounds) for _ in range(num_runs)]
            p_runs = pool.map(run_single_pairwise, p_args)
            
            # Neighborhood simulations
            n_args = [(agents, num_rounds) for _ in range(num_runs)]
            n_runs = pool.map(run_single_nperson, n_args)
    else:
        # Original sequential version
        p_runs = [run_pairwise_tournament([type(a)(**a.__dict__) for a in agents], num_rounds) for _ in range(num_runs)]
        n_runs = [run_nperson_simulation([type(a)(**a.__dict__) for a in agents], num_rounds) for _ in range(num_runs)]
    
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
    
    import pandas as pd
    series = pd.Series(data)
    smoothed = series.rolling(window=window_size, center=True, min_periods=1).mean()
    return smoothed.values


def save_scenario_csv(scenario_name, p_data, n_data, output_dir, subsample_rate=100):
    """Save detailed results for a scenario to CSV file"""
    csv_dir = os.path.join(output_dir, "csv_results")
    os.makedirs(csv_dir, exist_ok=True)
    
    # Create CSV for this scenario
    csv_path = os.path.join(csv_dir, f"{scenario_name}.csv")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['round', 'voting_model']
        agent_ids = sorted(p_data.keys())
        for aid in agent_ids:
            header.extend([f'{aid}_coop_rate', f'{aid}_score'])
        writer.writerow(header)
        
        # Get number of rounds
        num_rounds = len(p_data[agent_ids[0]]['coop_rate'])
        
        # Write pairwise data
        for round_idx in range(0, num_rounds, subsample_rate):
            row = [round_idx + 1, 'pairwise']
            for aid in agent_ids:
                row.extend([
                    p_data[aid]['coop_rate'][round_idx],
                    p_data[aid]['score'][round_idx]
                ])
            writer.writerow(row)
        
        # Write neighborhood data
        for round_idx in range(0, num_rounds, subsample_rate):
            row = [round_idx + 1, 'neighborhood']
            for aid in agent_ids:
                row.extend([
                    n_data[aid]['coop_rate'][round_idx],
                    n_data[aid]['score'][round_idx]
                ])
            writer.writerow(row)
    
    # Also create a summary CSV
    summary_csv_path = os.path.join(csv_dir, f"{scenario_name}_summary.csv")
    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['agent_id', 'agent_type', 'voting_model', 'avg_coop_rate', 'final_coop_rate', 'final_score', 'avg_score_per_round'])
        
        # Write summary data
        for aid in agent_ids:
            agent_type = 'QL' if 'QL' in aid else aid.split('_')[0]
            
            # Pairwise
            p_avg_coop = np.mean(p_data[aid]['coop_rate'])
            p_final_coop = np.mean(p_data[aid]['coop_rate'][-1000:])  # Last 1000 rounds
            p_final_score = p_data[aid]['score'][-1]
            p_avg_score_per_round = p_final_score / num_rounds
            
            writer.writerow([aid, agent_type, 'pairwise', p_avg_coop, p_final_coop, p_final_score, p_avg_score_per_round])
            
            # Neighborhood
            n_avg_coop = np.mean(n_data[aid]['coop_rate'])
            n_final_coop = np.mean(n_data[aid]['coop_rate'][-1000:])
            n_final_score = n_data[aid]['score'][-1]
            n_avg_score_per_round = n_final_score / num_rounds
            
            writer.writerow([aid, agent_type, 'neighborhood', n_avg_coop, n_final_coop, n_final_score, n_avg_score_per_round])


def create_heatmap(all_results, output_dir):
    """Create heatmaps showing final cooperation rates across scenarios"""
    # Prepare data for heatmap
    scenarios = []
    pairwise_coop = []
    neighborhood_coop = []
    
    for scenario_name, (p_data, n_data) in sorted(all_results.items()):
        # Find QL agents
        ql_agents = [aid for aid in p_data.keys() if 'QL' in aid]
        
        if ql_agents:
            # Calculate average final cooperation rate for QL agents
            p_final_coop = np.mean([np.mean(p_data[aid]['coop_rate'][-1000:]) for aid in ql_agents])
            n_final_coop = np.mean([np.mean(n_data[aid]['coop_rate'][-1000:]) for aid in ql_agents])
            
            scenarios.append(scenario_name)
            pairwise_coop.append(p_final_coop)
            neighborhood_coop.append(n_final_coop)
    
    # Create figure with two heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Reshape data for heatmap (organize by agent count and opponent type)
    agent_counts = sorted(set(int(s.split('_')[0]) for s in scenarios if not ('AllQL' in s or 'MixedQL' in s)))
    opponent_types = sorted(set(s.split('_')[-1] for s in scenarios if not ('AllQL' in s or 'MixedQL' in s)))
    
    # Create matrices for standard scenarios
    p_matrix = np.zeros((len(agent_counts), len(opponent_types)))
    n_matrix = np.zeros((len(agent_counts), len(opponent_types)))
    
    for i, count in enumerate(agent_counts):
        for j, opp in enumerate(opponent_types):
            # Find matching scenarios
            for k, scenario in enumerate(scenarios):
                if str(count) in scenario and opp in scenario and not ('AllQL' in scenario or 'MixedQL' in scenario):
                    p_matrix[i, j] = pairwise_coop[k]
                    n_matrix[i, j] = neighborhood_coop[k]
                    break
    
    # Plot heatmaps
    sns.heatmap(p_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
                xticklabels=opponent_types, yticklabels=agent_counts,
                ax=ax1, vmin=0, vmax=1, cbar_kws={'label': 'Cooperation Rate'})
    ax1.set_title('Pairwise Final Cooperation Rates')
    ax1.set_xlabel('Opponent Type')
    ax1.set_ylabel('Total Agent Count')
    
    sns.heatmap(n_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=opponent_types, yticklabels=agent_counts,
                ax=ax2, vmin=0, vmax=1, cbar_kws={'label': 'Cooperation Rate'})
    ax2.set_title('Neighborhood Final Cooperation Rates')
    ax2.set_xlabel('Opponent Type')
    ax2.set_ylabel('Total Agent Count')
    
    plt.suptitle('Q-Learning Agent Performance Across Scenarios', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cooperation_heatmap.png'), dpi=150)
    plt.close(fig)
    
    # Create separate heatmap for special scenarios
    special_scenarios = [s for s in scenarios if 'AllQL' in s or 'MixedQL' in s]
    if special_scenarios:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        special_p_coop = [pairwise_coop[scenarios.index(s)] for s in special_scenarios]
        special_n_coop = [neighborhood_coop[scenarios.index(s)] for s in special_scenarios]
        
        data = np.array([special_p_coop, special_n_coop])
        
        sns.heatmap(data, annot=True, fmt='.2f', cmap='RdYlGn',
                   xticklabels=[s.replace('_', ' ') for s in special_scenarios],
                   yticklabels=['Pairwise', 'Neighborhood'],
                   ax=ax, vmin=0, vmax=1, cbar_kws={'label': 'Cooperation Rate'})
        ax.set_title('Special Scenarios - Final Cooperation Rates')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'special_scenarios_heatmap.png'), dpi=150)
        plt.close(fig)


def plot_individual_scenario(scenario_name, p_data, n_data, output_dir):
    """Create detailed plot for individual scenario"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Scenario: {scenario_name}', fontsize=16)
    
    # Separate QL and non-QL agents
    ql_agents = sorted([aid for aid in p_data.keys() if 'QL' in aid])
    other_agents = sorted([aid for aid in p_data.keys() if 'QL' not in aid])
    
    # Define colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(p_data)))
    color_map = {aid: colors[i] for i, aid in enumerate(sorted(p_data.keys()))}
    
    # Pairwise cooperation rates
    for aid in ql_agents:
        smoothed = smooth_data(p_data[aid]['coop_rate'], window_size=50)
        axes[0, 0].plot(smoothed, label=aid, color=color_map[aid], linewidth=2)
    for aid in other_agents:
        smoothed = smooth_data(p_data[aid]['coop_rate'], window_size=50)
        axes[0, 0].plot(smoothed, label=aid, color=color_map[aid], linewidth=1, alpha=0.7)
    
    axes[0, 0].set_title('Pairwise Cooperation Rates')
    axes[0, 0].set_ylabel('Cooperation Rate')
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Pairwise scores
    for aid in ql_agents:
        axes[0, 1].plot(p_data[aid]['score'], label=aid, color=color_map[aid], linewidth=2)
    for aid in other_agents:
        axes[0, 1].plot(p_data[aid]['score'], label=aid, color=color_map[aid], linewidth=1, alpha=0.7)
    
    axes[0, 1].set_title('Pairwise Cumulative Scores')
    axes[0, 1].set_ylabel('Cumulative Score')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Neighborhood cooperation rates
    for aid in ql_agents:
        smoothed = smooth_data(n_data[aid]['coop_rate'], window_size=50)
        axes[1, 0].plot(smoothed, label=aid, color=color_map[aid], linewidth=2)
    for aid in other_agents:
        smoothed = smooth_data(n_data[aid]['coop_rate'], window_size=50)
        axes[1, 0].plot(smoothed, label=aid, color=color_map[aid], linewidth=1, alpha=0.7)
    
    axes[1, 0].set_title('Neighborhood Cooperation Rates')
    axes[1, 0].set_ylabel('Cooperation Rate')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Neighborhood scores
    for aid in ql_agents:
        axes[1, 1].plot(n_data[aid]['score'], label=aid, color=color_map[aid], linewidth=2)
    for aid in other_agents:
        axes[1, 1].plot(n_data[aid]['score'], label=aid, color=color_map[aid], linewidth=1, alpha=0.7)
    
    axes[1, 1].set_title('Neighborhood Cumulative Scores')
    axes[1, 1].set_ylabel('Cumulative Score')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save in a subdirectory
    plot_dir = os.path.join(output_dir, "individual_plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f"{scenario_name}.png"), dpi=150)
    plt.close(fig)


def plot_multi_agent_comparison(results, title, save_path, num_rounds=None):
    """Plot results for multi-agent experiments"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle(title, fontsize=20)
    
    # Extract agent counts and QL counts from results
    scenarios = sorted(results.keys())
    
    for idx, (scenario_name, (p_data, n_data)) in enumerate(results.items()):
        # Parse scenario name to get details
        parts = scenario_name.split('_')
        total_agents = int(parts[0])
        num_ql = int(parts[1].replace('QL', ''))
        
        # Find QL agents in results
        ql_agents = [aid for aid in p_data.keys() if 'QL' in aid]
        
        # Calculate average performance for QL agents
        if ql_agents:
            avg_p_coop = np.mean([p_data[aid]['coop_rate'] for aid in ql_agents], axis=0)
            avg_p_score = np.mean([p_data[aid]['score'] for aid in ql_agents], axis=0)
            avg_n_coop = np.mean([n_data[aid]['coop_rate'] for aid in ql_agents], axis=0)
            avg_n_score = np.mean([n_data[aid]['score'] for aid in ql_agents], axis=0)
            
            # Apply smoothing
            smooth_window = max(50, (num_rounds or SIMULATION_CONFIG['num_rounds']) // 100)
            avg_p_coop_smooth = smooth_data(avg_p_coop, smooth_window)
            avg_n_coop_smooth = smooth_data(avg_n_coop, smooth_window)
            
            # Plot
            label = f"{total_agents} agents, {num_ql} QL"
            
            axes[0, 0].plot(avg_p_coop_smooth, label=label, linewidth=2)
            axes[0, 1].plot(avg_p_score, label=label, linewidth=2)
            axes[1, 0].plot(avg_n_coop_smooth, label=label, linewidth=2)
            axes[1, 1].plot(avg_n_score, label=label, linewidth=2)
    
    axes[0, 0].set_title('Pairwise Cooperation Rate'); axes[0, 0].set_ylabel('Cooperation Rate')
    axes[0, 1].set_title('Pairwise Cumulative Score'); axes[0, 1].set_ylabel('Cumulative Score')
    axes[1, 0].set_title('Neighborhood Cooperation Rate'); axes[1, 0].set_ylabel('Cooperation Rate')
    axes[1, 1].set_title('Neighborhood Cumulative Score'); axes[1, 1].set_ylabel('Cumulative Score')
    
    for ax in axes.flat:
        ax.set_xlabel('Round')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def create_master_summary_csv(all_results, output_dir):
    """Create a master CSV summarizing all scenarios"""
    csv_path = os.path.join(output_dir, "master_summary.csv")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'scenario', 'total_agents', 'num_ql', 'ql_type', 'opponent_type',
            'pairwise_avg_coop', 'pairwise_final_coop', 'pairwise_final_score',
            'neighborhood_avg_coop', 'neighborhood_final_coop', 'neighborhood_final_score'
        ])
        
        # Process each scenario
        for scenario_name, (p_data, n_data) in sorted(all_results.items()):
            # Parse scenario details
            parts = scenario_name.split('_')
            total_agents = int(parts[0])
            
            # Determine scenario type and details
            if 'AllQL' in scenario_name:
                num_ql = total_agents
                ql_type = parts[-1]  # Last part should be QL type
                opponent_type = 'None'
            elif 'MixedQL' in scenario_name:
                # Extract numbers from mixed scenario name
                mixed_parts = scenario_name.split('_')[-1]  # e.g., "2v3h"
                num_ql = total_agents
                ql_type = 'Mixed'
                opponent_type = 'Mixed_QL'
            else:
                num_ql = int(parts[1].replace('QL', ''))
                ql_type = parts[2]  # Vanilla or Hysteretic
                opponent_type = parts[-1]
            
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
                
                writer.writerow([
                    scenario_name, total_agents, num_ql, ql_type, opponent_type,
                    p_avg_coop, p_final_coop, p_final_score,
                    n_avg_coop, n_final_coop, n_final_score
                ])


def create_multi_agent_summary(all_results, output_dir):
    """Create summary of multi-agent experiments"""
    summary_path = os.path.join(output_dir, "multi_agent_summary.txt")
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MULTI-AGENT EXPERIMENT SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Organize results by experiment type
        by_opponent = defaultdict(dict)
        
        for scenario_name, results in all_results.items():
            parts = scenario_name.split('_')
            
            if 'AllQL' in scenario_name:
                opponent_type = "AllQL"
            elif 'Mixed' in scenario_name:
                opponent_type = "MixedQL"
            else:
                # Find opponent type
                for opp in ['AllC', 'AllD', 'Random', 'TFT']:
                    if opp in scenario_name:
                        opponent_type = opp
                        break
            
            by_opponent[opponent_type][scenario_name] = results
        
        # Print results organized by opponent type
        for opponent_type, scenarios in by_opponent.items():
            f.write(f"\n{opponent_type} Scenarios:\n")
            f.write("-"*60 + "\n")
            
            for scenario_name in sorted(scenarios.keys()):
                p_data, n_data = scenarios[scenario_name]
                
                # Find QL agents
                ql_agents = [aid for aid in p_data.keys() if 'QL' in aid]
                
                if ql_agents:
                    # Calculate averages
                    avg_p_coop = np.mean([np.mean(p_data[aid]['coop_rate']) for aid in ql_agents])
                    avg_p_score = np.mean([p_data[aid]['score'][-1] for aid in ql_agents])
                    avg_n_coop = np.mean([np.mean(n_data[aid]['coop_rate']) for aid in ql_agents])
                    avg_n_score = np.mean([n_data[aid]['score'][-1] for aid in ql_agents])
                    
                    f.write(f"\n{scenario_name}:\n")
                    f.write(f"  Pairwise - Avg Coop: {avg_p_coop:.2%}, Final Score: {avg_p_score:.0f}\n")
                    f.write(f"  N-Person - Avg Coop: {avg_n_coop:.2%}, Final Score: {avg_n_score:.0f}\n")


if __name__ == "__main__":
    NUM_ROUNDS = SIMULATION_CONFIG['num_rounds']
    NUM_RUNS = SIMULATION_CONFIG['num_runs']
    OUTPUT_DIR = "multi_agent_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Agent group sizes to test
    AGENT_COUNTS = [5, 7, 19, 25]
    
    # QL agents to test per scenario
    QL_COUNTS = [1, 3]
    
    # Opponent types
    OPPONENT_TYPES = {
        "AllC": {"strategy": "AllC", "error_rate": 0.0},
        "AllD": {"strategy": "AllD", "error_rate": 0.0},
        "Random": {"strategy": "Random", "error_rate": 0.0},
        "TFT": {"strategy": "TFT", "error_rate": 0.0},
    }
    
    # Q-learner configurations (Vanilla and Hysteretic)
    QL_CONFIGS = {
        "Vanilla": {"class": PairwiseAdaptiveQLearner, "params": VANILLA_PARAMS},
        "Hysteretic": {"class": HystereticQLearner, "params": HYSTERETIC_PARAMS},
    }
    
    USE_PARALLEL = True
    n_cores = cpu_count()
    n_processes = max(1, n_cores - 1) if USE_PARALLEL else 1
    
    print(f"Running multi-agent experiments with {NUM_ROUNDS} rounds and {NUM_RUNS} runs per scenario")
    print(f"Agent counts: {AGENT_COUNTS}")
    print(f"QL counts per scenario: {QL_COUNTS}")
    if USE_PARALLEL:
        print(f"Using {n_processes} processes on {n_cores} available CPU cores")
    
    # Save configuration
    scenario_descriptions = {
        f"{n_agents}_agents_{n_ql}QL_vs_{opp}": f"{n_agents} total agents with {n_ql} QL agents vs {n_agents-n_ql} {opp} agents"
        for n_agents in AGENT_COUNTS
        for n_ql in QL_COUNTS
        for opp in OPPONENT_TYPES.keys()
        if n_ql < n_agents
    }
    
    # Add special scenarios
    for n_agents in AGENT_COUNTS:
        scenario_descriptions[f"{n_agents}_agents_AllQL"] = f"All {n_agents} agents are the same type of QL"
        if n_agents >= 4:  # Need at least 4 agents for mixed
            n_vanilla = n_agents // 2
            n_hysteretic = n_agents - n_vanilla
            scenario_descriptions[f"{n_agents}_agents_Mixed_{n_vanilla}v{n_hysteretic}h"] = \
                f"{n_vanilla} Vanilla QL vs {n_hysteretic} Hysteretic QL"
    
    save_detailed_config(OUTPUT_DIR, scenario_descriptions)
    
    all_results = {}
    total_start_time = time.time()
    
    # --- Standard scenarios: QL agents vs opponents ---
    print("\n=== Running QL vs Opponent Scenarios ===")
    
    for n_agents in AGENT_COUNTS:
        for n_ql in QL_COUNTS:
            if n_ql >= n_agents:
                continue  # Skip if we don't have enough agents
            
            for opp_name, opp_config in OPPONENT_TYPES.items():
                for ql_type, ql_config in QL_CONFIGS.items():
                    scenario_name = f"{n_agents}_{n_ql}QL_{ql_type}_vs_{opp_name}"
                    print(f"\nScenario: {scenario_name}")
                    
                    # Create QL agents
                    ql_agents = []
                    for i in range(n_ql):
                        agent = ql_config["class"](
                            agent_id=f"{ql_type}_QL_{i+1}",
                            params=ql_config["params"]
                        )
                        ql_agents.append(agent)
                    
                    # Create opponent agents
                    opp_agents = []
                    for i in range(n_agents - n_ql):
                        agent = StaticAgent(
                            agent_id=f"{opp_name}_{i+1}",
                            strategy_name=opp_config["strategy"],
                            error_rate=opp_config["error_rate"]
                        )
                        opp_agents.append(agent)
                    
                    # Combine all agents
                    agents = ql_agents + opp_agents
                    
                    # Run experiment
                    start_time = time.time()
                    p_agg, n_agg = run_experiment_set(agents, NUM_ROUNDS, NUM_RUNS, USE_PARALLEL)
                    elapsed = time.time() - start_time
                    
                    all_results[scenario_name] = (p_agg, n_agg)
                    print(f"  Completed in {elapsed:.1f}s")
                    
                    # Save CSV and plot for this scenario
                    save_scenario_csv(scenario_name, p_agg, n_agg, OUTPUT_DIR)
                    plot_individual_scenario(scenario_name, p_agg, n_agg, OUTPUT_DIR)
    
    # --- Special scenario 1: All same type of QL ---
    print("\n=== Running All Same QL Type Scenarios ===")
    
    for n_agents in AGENT_COUNTS:
        for ql_type, ql_config in QL_CONFIGS.items():
            scenario_name = f"{n_agents}_AllQL_{ql_type}"
            print(f"\nScenario: {scenario_name}")
            
            # Create all QL agents of same type
            agents = []
            for i in range(n_agents):
                agent = ql_config["class"](
                    agent_id=f"{ql_type}_QL_{i+1}",
                    params=ql_config["params"]
                )
                agents.append(agent)
            
            # Run experiment
            start_time = time.time()
            p_agg, n_agg = run_experiment_set(agents, NUM_ROUNDS, NUM_RUNS, USE_PARALLEL)
            elapsed = time.time() - start_time
            
            all_results[scenario_name] = (p_agg, n_agg)
            print(f"  Completed in {elapsed:.1f}s")
            
            # Save CSV and plot for this scenario
            save_scenario_csv(scenario_name, p_agg, n_agg, OUTPUT_DIR)
            plot_individual_scenario(scenario_name, p_agg, n_agg, OUTPUT_DIR)
    
    # --- Special scenario 2: Mixed QL types ---
    print("\n=== Running Mixed QL Type Scenarios ===")
    
    for n_agents in AGENT_COUNTS:
        if n_agents < 4:  # Need at least 4 agents for meaningful mixed scenario
            continue
            
        # Split roughly in half, with Hysteretic having equal or more agents
        n_vanilla = n_agents // 2
        n_hysteretic = n_agents - n_vanilla
        
        scenario_name = f"{n_agents}_MixedQL_{n_vanilla}v{n_hysteretic}h"
        print(f"\nScenario: {scenario_name} ({n_vanilla} Vanilla vs {n_hysteretic} Hysteretic)")
        
        agents = []
        
        # Create Vanilla agents
        for i in range(n_vanilla):
            agent = QL_CONFIGS["Vanilla"]["class"](
                agent_id=f"Vanilla_QL_{i+1}",
                params=QL_CONFIGS["Vanilla"]["params"]
            )
            agents.append(agent)
        
        # Create Hysteretic agents
        for i in range(n_hysteretic):
            agent = QL_CONFIGS["Hysteretic"]["class"](
                agent_id=f"Hysteretic_QL_{i+1}",
                params=QL_CONFIGS["Hysteretic"]["params"]
            )
            agents.append(agent)
        
        # Run experiment
        start_time = time.time()
        p_agg, n_agg = run_experiment_set(agents, NUM_ROUNDS, NUM_RUNS, USE_PARALLEL)
        elapsed = time.time() - start_time
        
        all_results[scenario_name] = (p_agg, n_agg)
        print(f"  Completed in {elapsed:.1f}s")
    
    # --- Generate plots and summaries ---
    print("\n=== Generating Plots and Summaries ===")
    
    # Group results by experiment type for plotting
    by_opponent = defaultdict(dict)
    
    for scenario_name, results in all_results.items():
        if 'AllQL' in scenario_name:
            by_opponent['AllQL'][scenario_name] = results
        elif 'MixedQL' in scenario_name:
            by_opponent['MixedQL'][scenario_name] = results
        else:
            # Find opponent type
            for opp in OPPONENT_TYPES.keys():
                if f"vs_{opp}" in scenario_name:
                    by_opponent[opp][scenario_name] = results
                    break
    
    # Create plots for each opponent type
    for opponent_type, scenarios in by_opponent.items():
        if scenarios:
            plot_path = os.path.join(OUTPUT_DIR, f"multi_agent_{opponent_type}.png")
            plot_multi_agent_comparison(scenarios, f"Multi-Agent Experiments vs {opponent_type}", plot_path)
            print(f"  Saved plot: {plot_path}")
    
    # Create summary and additional visualizations
    create_multi_agent_summary(all_results, OUTPUT_DIR)
    create_master_summary_csv(all_results, OUTPUT_DIR)
    create_heatmap(all_results, OUTPUT_DIR)
    
    # Calculate total time
    total_elapsed = time.time() - total_start_time
    print(f"\nAll experiments complete! Results saved to '{OUTPUT_DIR}'")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    
    if USE_PARALLEL:
        speedup_estimate = n_processes * 0.8
        sequential_estimate = total_elapsed * speedup_estimate
        print(f"Estimated time saved by using {n_processes} processes: {(sequential_estimate - total_elapsed)/60:.1f} minutes")