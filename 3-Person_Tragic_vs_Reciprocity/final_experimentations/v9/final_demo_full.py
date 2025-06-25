import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import time
import csv
from datetime import datetime

from final_agents import StaticAgent, PairwiseAdaptiveQLearner, NeighborhoodAdaptiveQLearner, HystereticQLearner, LegacyQLearner, Legacy3RoundQLearner
from config import VANILLA_PARAMS, ADAPTIVE_PARAMS, HYSTERETIC_PARAMS, LEGACY_PARAMS, LEGACY_3ROUND_PARAMS, SIMULATION_CONFIG
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
    
    # Aggregate results (same as before)
    p_agg = {aid: {m: np.mean([r[aid][m] for r in p_runs if aid in r], axis=0) for m in ['coop_rate', 'score']} 
             for aid in p_runs[0]}
    n_agg = {aid: {m: np.mean([r[aid][m] for r in n_runs if aid in r], axis=0) for m in ['coop_rate', 'score']} 
             for aid in n_runs[0]}
    
    return p_agg, n_agg


def smooth_data(data, window_size=50):
    """Apply rolling average to smooth data"""
    if len(data) < window_size:
        window_size = max(1, len(data) // 10)
    
    # Use pandas for efficient rolling average
    import pandas as pd
    series = pd.Series(data)
    smoothed = series.rolling(window=window_size, center=True, min_periods=1).mean()
    return smoothed.values

def plot_scenario_comparison(results, title, save_path, num_rounds=None):
    """Generates a 4-panel plot comparing Q-learning variants for one scenario."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle(f"Comparison: {title}", fontsize=22)
    colors = {
        "Vanilla": "#1f77b4", 
        "Adaptive": "#ff7f0e", 
        "Hysteretic": "#2ca02c",
        "Legacy": "#17becf",
        "Legacy3Round": "#d62728"
    }
    
    # Determine smoothing window based on number of rounds
    if num_rounds is None:
        num_rounds = SIMULATION_CONFIG['num_rounds']
    smooth_window = max(50, num_rounds // 100)  # At least 50, or 1% of total rounds
    
    for agent_type, (p_data, n_data) in results.items():
        # Find the QL agents in the results
        ql_agents = [aid for aid in p_data.keys() if agent_type in aid and 'QL' in aid]
        if not ql_agents:
            continue
            
        # Average across all QL agents of this type
        p_coop = np.mean([p_data[aid]['coop_rate'] for aid in ql_agents], axis=0)
        p_score = np.mean([p_data[aid]['score'] for aid in ql_agents], axis=0)
        n_coop = np.mean([n_data[aid]['coop_rate'] for aid in ql_agents], axis=0)
        n_score = np.mean([n_data[aid]['score'] for aid in ql_agents], axis=0)
        
        # Apply smoothing
        p_coop_smooth = smooth_data(p_coop, smooth_window)
        n_coop_smooth = smooth_data(n_coop, smooth_window)
        
        # Get color for this agent type
        color = colors.get(agent_type, "#808080")  # Default to gray if not found
        
        # Plot raw data with low alpha and smoothed data with high alpha
        # Cooperation rates
        axes[0, 0].plot(p_coop, color=color, alpha=0.2, linewidth=0.5)
        axes[0, 0].plot(p_coop_smooth, label=agent_type, color=color, linewidth=2.5)
        
        axes[1, 0].plot(n_coop, color=color, alpha=0.2, linewidth=0.5)
        axes[1, 0].plot(n_coop_smooth, label=agent_type, color=color, linewidth=2.5)
        
        # Scores (no smoothing needed as they're cumulative)
        axes[0, 1].plot(p_score, label=agent_type, color=color, linewidth=2)
        axes[1, 1].plot(n_score, label=agent_type, color=color, linewidth=2)
    
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


def create_heatmap(all_results, save_path):
    """Generates a heatmap comparing all agent types against Vanilla baseline."""
    # Get all unique agent types
    all_agent_types = set()
    for results in all_results.values():
        all_agent_types.update(results.keys())
    
    # Remove Vanilla from comparison list (it's the baseline)
    agent_types_to_compare = sorted([a for a in all_agent_types if a != 'Vanilla'])
    
    if 'Vanilla' not in all_agent_types:
        print("No Vanilla baseline found for heatmap comparison")
        return
    
    # Create heatmap data for each agent type
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data structure
    comparison_data = {}
    
    for agent_type in agent_types_to_compare:
        comparison_data[agent_type] = []
        
        for scenario_name in sorted(all_results.keys()):
            results = all_results[scenario_name]
            
            if 'Vanilla' not in results or agent_type not in results:
                comparison_data[agent_type].append([0, 0, 0, 0])  # No data
                continue
            
            v_p, v_n = results['Vanilla']
            a_p, a_n = results[agent_type]
            
            # Find QL agents
            v_ql = [aid for aid in v_p.keys() if 'Vanilla' in aid and 'QL' in aid]
            a_ql = [aid for aid in a_p.keys() if agent_type in aid and 'QL' in aid]
            
            if not v_ql or not a_ql:
                comparison_data[agent_type].append([0, 0, 0, 0])
                continue
            
            # Calculate metrics
            v_metrics = [
                np.mean([np.mean(v_p[aid]['coop_rate']) for aid in v_ql]),
                np.mean([v_p[aid]['score'][-1] for aid in v_ql]),
                np.mean([np.mean(v_n[aid]['coop_rate']) for aid in v_ql]),
                np.mean([v_n[aid]['score'][-1] for aid in v_ql])
            ]
            
            a_metrics = [
                np.mean([np.mean(a_p[aid]['coop_rate']) for aid in a_ql]),
                np.mean([a_p[aid]['score'][-1] for aid in a_ql]),
                np.mean([np.mean(a_n[aid]['coop_rate']) for aid in a_ql]),
                np.mean([a_n[aid]['score'][-1] for aid in a_ql])
            ]
            
            # Calculate percentage improvements
            improvements = []
            for v, a in zip(v_metrics, a_metrics):
                if v != 0:
                    improvement = ((a - v) / abs(v)) * 100
                else:
                    improvement = 100 if a > 0 else 0
                improvements.append(improvement)
            
            comparison_data[agent_type].append(improvements)
    
    # Create combined heatmap
    scenario_names = sorted(all_results.keys())
    n_scenarios = len(scenario_names)
    n_agents = len(agent_types_to_compare)
    
    # Reshape data for heatmap
    heatmap_data = []
    row_labels = []
    
    for i, agent_type in enumerate(agent_types_to_compare):
        for j, scenario in enumerate(scenario_names):
            heatmap_data.append(comparison_data[agent_type][j])
            row_labels.append(f"{agent_type} - {scenario}")
    
    df = pd.DataFrame(heatmap_data,
                      index=row_labels,
                      columns=['PW Coop %', 'PW Score %', 'NH Coop %', 'NH Score %'])
    
    plt.figure(figsize=(8, max(8, len(row_labels) * 0.3)))
    sns.heatmap(df, annot=True, fmt=".1f", cmap="RdYlGn", center=0, 
                cbar_kws={'label': '% Improvement over Vanilla'})
    plt.title("Performance Comparison: All Agents vs. Vanilla Baseline", fontsize=14)
    plt.xlabel("Metrics", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_results_to_csv(all_scenario_results, output_dir, ql_configs):
    """Save detailed results to CSV files in a subfolder"""
    csv_dir = os.path.join(output_dir, "csv_results")
    os.makedirs(csv_dir, exist_ok=True)
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary statistics
    summary_data = []
    for scenario_name, scenario_results in all_scenario_results.items():
        for agent_type, (p_data, n_data) in scenario_results.items():
            # Find QL agents
            ql_agents = [aid for aid in p_data.keys() if agent_type in aid and 'QL' in aid]
            if not ql_agents:
                continue
            
            # Calculate final metrics
            for aid in ql_agents:
                summary_data.append({
                    'scenario': scenario_name,
                    'agent_type': agent_type,
                    'agent_id': aid,
                    'pairwise_final_coop': np.mean(p_data[aid]['coop_rate'][-100:]),  # Last 100 rounds
                    'pairwise_total_score': p_data[aid]['score'][-1],
                    'neighborhood_final_coop': np.mean(n_data[aid]['coop_rate'][-100:]),
                    'neighborhood_total_score': n_data[aid]['score'][-1]
                })
    
    # Save summary to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(csv_dir, f"summary_results_{timestamp}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary results saved to: {summary_path}")
    
    # Save detailed time series for each scenario
    for scenario_name, scenario_results in all_scenario_results.items():
        scenario_data = []
        
        for agent_type, (p_data, n_data) in scenario_results.items():
            ql_agents = [aid for aid in p_data.keys() if agent_type in aid and 'QL' in aid]
            if not ql_agents:
                continue
            
            # Average across QL agents of this type
            p_coop = np.mean([p_data[aid]['coop_rate'] for aid in ql_agents], axis=0)
            p_score = np.mean([p_data[aid]['score'] for aid in ql_agents], axis=0)
            n_coop = np.mean([n_data[aid]['coop_rate'] for aid in ql_agents], axis=0)
            n_score = np.mean([n_data[aid]['score'] for aid in ql_agents], axis=0)
            
            # Create time series data
            for round_num in range(len(p_coop)):
                scenario_data.append({
                    'round': round_num,
                    'agent_type': agent_type,
                    'pairwise_coop_rate': p_coop[round_num],
                    'pairwise_score': p_score[round_num],
                    'neighborhood_coop_rate': n_coop[round_num],
                    'neighborhood_score': n_score[round_num]
                })
        
        # Save scenario data to CSV
        if scenario_data:
            scenario_df = pd.DataFrame(scenario_data)
            scenario_path = os.path.join(csv_dir, f"{scenario_name}_{timestamp}.csv")
            scenario_df.to_csv(scenario_path, index=False)
            print(f"Detailed results for {scenario_name} saved to: {scenario_path}")
    
    # Save configuration information
    config_info_path = os.path.join(csv_dir, f"config_info_{timestamp}.txt")
    with open(config_info_path, 'w') as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Number of rounds: {SIMULATION_CONFIG['num_rounds']}\n")
        f.write(f"Number of runs: {SIMULATION_CONFIG['num_runs']}\n")
        f.write(f"Agents tested: {', '.join(ql_configs.keys())}\n")
    
    return csv_dir


def save_config_to_file(output_dir):
    """Save all configuration parameters to a text file"""
    config_path = os.path.join(output_dir, "simulation_config.txt")
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("SIMULATION CONFIGURATION\n")
        f.write("="*60 + "\n\n")
        
        # Simulation parameters
        f.write("SIMULATION PARAMETERS:\n")
        f.write("-"*30 + "\n")
        for key, value in SIMULATION_CONFIG.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Vanilla Q-learner parameters
        f.write("VANILLA Q-LEARNER PARAMETERS:\n")
        f.write("-"*30 + "\n")
        for key, value in VANILLA_PARAMS.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Adaptive Q-learner parameters
        f.write("ADAPTIVE Q-LEARNER PARAMETERS:\n")
        f.write("-"*30 + "\n")
        for key, value in ADAPTIVE_PARAMS.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Timestamp
        from datetime import datetime
        f.write("="*60 + "\n")
        f.write(f"Configuration saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n")
    
    print(f"Configuration saved to: {config_path}")

if __name__ == "__main__":
    NUM_ROUNDS = SIMULATION_CONFIG['num_rounds']
    NUM_RUNS = SIMULATION_CONFIG['num_runs']
    OUTPUT_DIR = "final_comparison_charts"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Detect parallelization capability
    USE_PARALLEL = True  # Now works on Windows with our picklable agents
    n_cores = cpu_count()
    n_processes = max(1, n_cores - 1) if USE_PARALLEL else 1
    
    print(f"Running simulations with {NUM_ROUNDS} rounds and {NUM_RUNS} runs per scenario")
    if USE_PARALLEL:
        print(f"Using {n_processes} processes on {n_cores} available CPU cores")
    else:
        print(f"Running in sequential mode (parallelization disabled on Windows due to pickling limitations)")
    
    # Define scenario descriptions for documentation
    scenario_descriptions = {
        "2QL_vs_1AllC": "Two Q-learners compete against one Always Cooperate agent",
        "2QL_vs_1AllD": "Two Q-learners compete against one Always Defect agent",
        "2QL_vs_1Random": "Two Q-learners compete against one Random agent",
        "2QL_vs_1TFT": "Two Q-learners compete against one Tit-for-Tat agent",
        "2QL_vs_1TFT-E": "Two Q-learners compete against one TFT with 10% error rate",
        "1QL_vs_2AllC": "One Q-learner competes against two Always Cooperate agents",
        "1QL_vs_2AllD": "One Q-learner competes against two Always Defect agents",
        "1QL_vs_2Random": "One Q-learner competes against two Random agents",
        "1QL_vs_2TFT": "One Q-learner competes against two Tit-for-Tat agents",
        "1QL_vs_2TFT-E": "One Q-learner competes against two TFT with 10% error rate"
    }
    
    # Save configuration to file
    save_detailed_config(OUTPUT_DIR, scenario_descriptions)
    
    # Start total timer
    total_start_time = time.time()
    
    # Define opponent strategies
    opponent_configs = {
        "AllC": {"strategy": "AllC", "error_rate": 0.0},
        "AllD": {"strategy": "AllD", "error_rate": 0.0},
        "Random": {"strategy": "Random", "error_rate": 0.0},
        "TFT": {"strategy": "TFT", "error_rate": 0.0},
        "TFT-E": {"strategy": "TFT-E", "error_rate": 0.1},
    }
    
    # Define Q-learner configurations
    ql_configs = {
        # "Vanilla": {"class": PairwiseAdaptiveQLearner,  # Using PairwiseAdaptiveQLearner for both modes
        #             "params": VANILLA_PARAMS},
        # "Adaptive": {"class": PairwiseAdaptiveQLearner,  # Using PairwiseAdaptiveQLearner for both modes
        #              "params": ADAPTIVE_PARAMS},
        "Hysteretic": {"class": HystereticQLearner,      # New Hysteretic Q-learner
                       "params": HYSTERETIC_PARAMS},
        "Legacy": {"class": LegacyQLearner,              # Legacy Q-learner with sophisticated state
                   "params": LEGACY_PARAMS},
        "Legacy3Round": {"class": Legacy3RoundQLearner,  # Legacy Q-learner with 3-round history
                         "params": LEGACY_3ROUND_PARAMS},
    }
    # Removed Adaptive+Soft and Adaptive+StatSoft as requested
    
    all_scenario_results = {}
    
    # --- Scenario 1: 2 QL vs. 1 Opponent ---
    print("\n=== Running 2 QL vs. 1 Opponent Scenarios ===")
    for opp_name, opp_config in opponent_configs.items():
        scenario_name = f"2QL_vs_1{opp_name}"
        print(f"\nScenario: {scenario_name}")
        scenario_results = {}
        
        for ql_name, ql_config in ql_configs.items():
            print(f"  Testing {ql_name} Q-learners...", end='', flush=True)
            start_time = time.time()
            agents = [
                ql_config["class"](agent_id=f"{ql_name}_QL_1", params=ql_config["params"]),
                ql_config["class"](agent_id=f"{ql_name}_QL_2", params=ql_config["params"]),
                StaticAgent(agent_id=f"{opp_name}_1", strategy_name=opp_config["strategy"], 
                           error_rate=opp_config["error_rate"])
            ]
            scenario_results[ql_name] = run_experiment_set(agents, NUM_ROUNDS, NUM_RUNS, USE_PARALLEL)
            elapsed = time.time() - start_time
            print(f" done in {elapsed:.1f}s")
        
        all_scenario_results[scenario_name] = scenario_results
        plot_path = os.path.join(OUTPUT_DIR, f"{scenario_name}.png")
        plot_scenario_comparison(scenario_results, scenario_name, plot_path, NUM_ROUNDS)
        print(f"  Saved plot: {plot_path}")
    
    # --- Scenario 2: 1 QL vs. 2 Opponents ---
    print("\n=== Running 1 QL vs. 2 Opponents Scenarios ===")
    for opp_name, opp_config in opponent_configs.items():
        scenario_name = f"1QL_vs_2{opp_name}"
        print(f"\nScenario: {scenario_name}")
        scenario_results = {}
        
        for ql_name, ql_config in ql_configs.items():
            print(f"  Testing {ql_name} Q-learner...", end='', flush=True)
            start_time = time.time()
            agents = [
                ql_config["class"](agent_id=f"{ql_name}_QL_1", params=ql_config["params"]),
                StaticAgent(agent_id=f"{opp_name}_1", strategy_name=opp_config["strategy"],
                           error_rate=opp_config["error_rate"]),
                StaticAgent(agent_id=f"{opp_name}_2", strategy_name=opp_config["strategy"],
                           error_rate=opp_config["error_rate"])
            ]
            scenario_results[ql_name] = run_experiment_set(agents, NUM_ROUNDS, NUM_RUNS, USE_PARALLEL)
            elapsed = time.time() - start_time
            print(f" done in {elapsed:.1f}s")
        
        all_scenario_results[scenario_name] = scenario_results
        plot_path = os.path.join(OUTPUT_DIR, f"{scenario_name}.png")
        plot_scenario_comparison(scenario_results, scenario_name, plot_path, NUM_ROUNDS)
        print(f"  Saved plot: {plot_path}")
    
    # --- Generate Summary Heatmap ---
    print("\n=== Generating Summary Heatmap ===")
    heatmap_path = os.path.join(OUTPUT_DIR, "summary_heatmap_all_scenarios.png")
    create_heatmap(all_scenario_results, heatmap_path)
    print(f"Heatmap saved: {heatmap_path}")
    
    # --- Save Results to CSV ---
    print("\n=== Saving Results to CSV ===")
    csv_dir = save_results_to_csv(all_scenario_results, OUTPUT_DIR, ql_configs)
    print(f"CSV results saved to: {csv_dir}")
    
    # Calculate total time
    total_elapsed = time.time() - total_start_time
    print(f"\nAll experiments complete! Results saved to '{OUTPUT_DIR}'")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    
    if USE_PARALLEL:
        speedup_estimate = n_processes * 0.8  # Rough estimate of speedup
        sequential_estimate = total_elapsed * speedup_estimate
        print(f"Estimated time saved by using {n_processes} processes: {(sequential_estimate - total_elapsed)/60:.1f} minutes")