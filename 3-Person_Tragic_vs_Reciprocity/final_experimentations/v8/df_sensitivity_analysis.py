"""
Discount Factor Sensitivity Analysis Script
Tests the effect of different discount factor values on cooperation rates
Focuses on measuring cooperation between Q-Learning agents and TFT agents
Tests DF values: 0.99, 0.95, 0.9, 0.7, 0.4, 0.0
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from collections import defaultdict
import time
import copy

from final_agents import StaticAgent, LegacyQLearner, Legacy3RoundQLearner
from config import LEGACY_PARAMS, LEGACY_3ROUND_PARAMS, SIMULATION_CONFIG

# --- Payoff Logic ---
PAYOFFS_2P = {(0, 0): (3, 3), (0, 1): (0, 5), (1, 0): (5, 0), (1, 1): (1, 1)}
T, R, P, S = 5, 3, 1, 0
COOPERATE, DEFECT = 0, 1

# Discount factors to test
DF_VALUES = [0.99, 0.95, 0.9, 0.7, 0.4, 0.0]


def smooth_data(data, window_size=50):
    """Apply rolling average to smooth data"""
    if len(data) < window_size:
        window_size = max(1, len(data) // 10)
    
    series = pd.Series(data)
    smoothed = series.rolling(window=window_size, center=True, min_periods=1).mean()
    return smoothed.values


def nperson_payoff(my_move, num_cooperators, total_agents):
    """Calculate payoff for n-person game"""
    others_coop = num_cooperators - (1 - my_move)
    if my_move == 0:  # cooperate
        return S + (R - S) * (others_coop / (total_agents - 1))
    else:  # defect
        return P + (T - P) * (others_coop / (total_agents - 1))


def run_pairwise_tournament_with_groups(agents, num_rounds):
    """Run pairwise tournament tracking QL and TFT groups separately"""
    for agent in agents:
        agent.reset()
    
    # Identify QL and TFT agents
    ql_agents = [a for a in agents if 'QL' in a.strategy_name]
    tft_agents = [a for a in agents if a.strategy_name == 'TFT']
    
    # Initialize history tracking
    history = {
        'ql_coop_rate': [],
        'tft_coop_rate': []
    }
    
    agent_map = {a.agent_id: a for a in agents}
    
    for round_num in range(num_rounds):
        moves = {}
        # Play all pairwise games
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                move1 = agent1.choose_pairwise_action(agent2.agent_id)
                move2 = agent2.choose_pairwise_action(agent1.agent_id)
                moves[(agent1.agent_id, agent2.agent_id)] = (move1, move2)
        
        # Record outcomes and collect moves by group
        ql_moves = []
        tft_moves = []
        
        for (id1, id2), (move1, move2) in moves.items():
            payoff1, payoff2 = PAYOFFS_2P[(move1, move2)]
            agent_map[id1].record_pairwise_outcome(id2, move1, move2, payoff1)
            agent_map[id2].record_pairwise_outcome(id1, move2, move1, payoff2)
            
            # Collect moves by group
            if agent_map[id1] in ql_agents:
                ql_moves.append(move1)
            elif agent_map[id1] in tft_agents:
                tft_moves.append(move1)
                
            if agent_map[id2] in ql_agents:
                ql_moves.append(move2)
            elif agent_map[id2] in tft_agents:
                tft_moves.append(move2)
        
        # Calculate cooperation rates
        ql_coop_rate = (ql_moves.count(COOPERATE) / len(ql_moves)) if ql_moves else 0
        tft_coop_rate = (tft_moves.count(COOPERATE) / len(tft_moves)) if tft_moves else 0
        
        history['ql_coop_rate'].append(ql_coop_rate)
        history['tft_coop_rate'].append(tft_coop_rate)
    
    return history


def run_nperson_simulation_with_groups(agents, num_rounds):
    """Run n-person simulation tracking QL and TFT groups separately"""
    for agent in agents:
        agent.reset()
    
    # Identify QL and TFT agents
    ql_agents = [a for a in agents if 'QL' in a.strategy_name]
    tft_agents = [a for a in agents if a.strategy_name == 'TFT']
    
    # Initialize history tracking
    history = {
        'ql_coop_rate': [],
        'tft_coop_rate': []
    }
    
    coop_ratio = None
    
    for round_num in range(num_rounds):
        # Get moves from all agents
        moves = {a.agent_id: a.choose_neighborhood_action(coop_ratio) for a in agents}
        
        # Calculate payoffs
        num_cooperators = list(moves.values()).count(COOPERATE)
        current_coop_ratio = num_cooperators / len(agents)
        
        # Update agents
        for agent in agents:
            my_move = moves[agent.agent_id]
            payoff = nperson_payoff(my_move, num_cooperators, len(agents))
            agent.record_neighborhood_outcome(current_coop_ratio, payoff)
        
        # Calculate group cooperation rates
        ql_moves = [moves[a.agent_id] for a in ql_agents]
        tft_moves = [moves[a.agent_id] for a in tft_agents]
        
        ql_coop_rate = (ql_moves.count(COOPERATE) / len(ql_moves)) if ql_moves else 0
        tft_coop_rate = (tft_moves.count(COOPERATE) / len(tft_moves)) if tft_moves else 0
        
        history['ql_coop_rate'].append(ql_coop_rate)
        history['tft_coop_rate'].append(tft_coop_rate)
        
        coop_ratio = current_coop_ratio
    
    return history


def run_multiple_simulations(agents, num_rounds, num_runs):
    """Run multiple simulations and average results"""
    pairwise_runs = []
    nperson_runs = []
    
    for run in range(num_runs):
        # Create fresh copies of agents
        fresh_agents = []
        for agent in agents:
            if isinstance(agent, StaticAgent):
                fresh_agents.append(StaticAgent(agent.agent_id, agent.strategy_name))
            elif isinstance(agent, LegacyQLearner):
                fresh_agents.append(LegacyQLearner(agent.agent_id, agent.params))
            elif isinstance(agent, Legacy3RoundQLearner):
                fresh_agents.append(Legacy3RoundQLearner(agent.agent_id, agent.params))
        
        pairwise_runs.append(run_pairwise_tournament_with_groups(fresh_agents, num_rounds))
        nperson_runs.append(run_nperson_simulation_with_groups(fresh_agents, num_rounds))
    
    # Average the results
    avg_pairwise = {
        'ql_coop_rate': np.mean([run['ql_coop_rate'] for run in pairwise_runs], axis=0),
        'tft_coop_rate': np.mean([run['tft_coop_rate'] for run in pairwise_runs], axis=0)
    }
    
    avg_nperson = {
        'ql_coop_rate': np.mean([run['ql_coop_rate'] for run in nperson_runs], axis=0),
        'tft_coop_rate': np.mean([run['tft_coop_rate'] for run in nperson_runs], axis=0)
    }
    
    return avg_pairwise, avg_nperson


def plot_df_comparison(df_results, scenario_name, save_path):
    """Plot cooperation rates for different DF values"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{scenario_name}: Cooperation Rates Across Different Discount Factors', fontsize=16)
    
    # Create color map for different DF values
    colors = plt.cm.viridis(np.linspace(0, 1, len(DF_VALUES)))
    
    # Determine smoothing window
    first_df_result = df_results[DF_VALUES[0]]
    num_rounds = len(first_df_result[0]['ql_coop_rate'])
    smooth_window = max(50, num_rounds // 100)
    rounds = np.arange(num_rounds)
    
    # Plot pairwise results
    for idx, (df_value, (pairwise, nperson)) in enumerate(df_results.items()):
        # Smooth the data
        pairwise_ql_smooth = smooth_data(pairwise['ql_coop_rate'], smooth_window)
        pairwise_tft_smooth = smooth_data(pairwise['tft_coop_rate'], smooth_window)
        
        # Plot QL cooperation rates
        ax1.plot(rounds, pairwise_ql_smooth, 
                label=f'QL (DF={df_value})', 
                color=colors[idx], linewidth=2)
        # Plot TFT cooperation rates with dashed line
        ax1.plot(rounds, pairwise_tft_smooth, 
                label=f'TFT (DF={df_value})', 
                color=colors[idx], linewidth=2, linestyle='--', alpha=0.7)
    
    ax1.set_title('Pairwise Cooperation Rates', fontsize=14)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Cooperation Rate')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Plot neighborhood results
    for idx, (df_value, (pairwise, nperson)) in enumerate(df_results.items()):
        # Smooth the data
        nperson_ql_smooth = smooth_data(nperson['ql_coop_rate'], smooth_window)
        nperson_tft_smooth = smooth_data(nperson['tft_coop_rate'], smooth_window)
        
        # Plot QL cooperation rates
        ax2.plot(rounds, nperson_ql_smooth, 
                label=f'QL (DF={df_value})', 
                color=colors[idx], linewidth=2)
        # Plot TFT cooperation rates with dashed line
        ax2.plot(rounds, nperson_tft_smooth, 
                label=f'TFT (DF={df_value})', 
                color=colors[idx], linewidth=2, linestyle='--', alpha=0.7)
    
    ax2.set_title('Neighborhood Cooperation Rates', fontsize=14)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Cooperation Rate')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_final_comparison_legacy3round(all_df_results, save_path):
    """Plot final comparison for 2 Legacy 3 round QL vs 1 TFT scenario across all DF values"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract data for Legacy3Round_2QL_vs_1TFT scenario
    scenario_key = "Legacy3Round_2QL_vs_1TFT"
    
    # Create color map for different DF values
    colors = plt.cm.viridis(np.linspace(0, 1, len(DF_VALUES)))
    
    # Determine smoothing window
    first_df_data = all_df_results[DF_VALUES[0]][scenario_key]
    num_rounds = len(first_df_data[0]['ql_coop_rate'])
    smooth_window = max(50, num_rounds // 100)
    rounds = np.arange(num_rounds)
    
    # Plot both pairwise and neighborhood for each DF
    for idx, df_value in enumerate(DF_VALUES):
        pairwise, nperson = all_df_results[df_value][scenario_key]
        
        # Smooth the data
        pairwise_ql_smooth = smooth_data(pairwise['ql_coop_rate'], smooth_window)
        nperson_ql_smooth = smooth_data(nperson['ql_coop_rate'], smooth_window)
        
        # Plot pairwise with solid line
        ax.plot(rounds, pairwise_ql_smooth, 
               label=f'Pairwise (DF={df_value})', 
               color=colors[idx], linewidth=2.5)
        
        # Plot neighborhood with dashed line
        ax.plot(rounds, nperson_ql_smooth, 
               label=f'Neighborhood (DF={df_value})', 
               color=colors[idx], linewidth=2.5, linestyle='--')
    
    ax.set_title('2 Legacy3Round QL vs 1 TFT: Effect of Discount Factor on QL Cooperation', fontsize=16)
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('QL Cooperation Rate', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_df_results_to_csv(all_df_results, output_dir):
    """Save results for each DF value to CSV files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed time series for each DF and scenario
    for df_value in DF_VALUES:
        df_dir = os.path.join(output_dir, f"df_{df_value}")
        os.makedirs(df_dir, exist_ok=True)
        
        for scenario_name, (pairwise, nperson) in all_df_results[df_value].items():
            data = []
            rounds = len(pairwise['ql_coop_rate'])
            
            for i in range(rounds):
                data.append({
                    'round': i,
                    'pairwise_ql_coop': pairwise['ql_coop_rate'][i],
                    'pairwise_tft_coop': pairwise['tft_coop_rate'][i],
                    'nperson_ql_coop': nperson['ql_coop_rate'][i],
                    'nperson_tft_coop': nperson['tft_coop_rate'][i]
                })
            
            df = pd.DataFrame(data)
            filename = f"{scenario_name}_df{df_value}_{timestamp}.csv"
            df.to_csv(os.path.join(df_dir, filename), index=False)
    
    # Save summary statistics across all DF values
    summary_data = []
    for df_value in DF_VALUES:
        for scenario_name, (pairwise, nperson) in all_df_results[df_value].items():
            # Calculate final averages (last 100 rounds)
            summary_data.append({
                'discount_factor': df_value,
                'scenario': scenario_name,
                'pairwise_ql_final_coop': np.mean(pairwise['ql_coop_rate'][-100:]),
                'pairwise_tft_final_coop': np.mean(pairwise['tft_coop_rate'][-100:]),
                'nperson_ql_final_coop': np.mean(nperson['ql_coop_rate'][-100:]),
                'nperson_tft_final_coop': np.mean(nperson['tft_coop_rate'][-100:])
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_filename = f"df_sensitivity_summary_{timestamp}.csv"
    summary_df.to_csv(os.path.join(output_dir, summary_filename), index=False)
    print(f"Saved summary to {summary_filename}")


def main():
    # Configuration
    NUM_ROUNDS = SIMULATION_CONFIG['num_rounds']
    NUM_RUNS = SIMULATION_CONFIG['num_runs']
    OUTPUT_DIR = "df_sensitivity_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Running discount factor sensitivity analysis")
    print(f"DF values to test: {DF_VALUES}")
    print(f"Rounds: {NUM_ROUNDS}, Runs: {NUM_RUNS}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Store all results across DF values
    all_df_results = {}
    
    for df_value in DF_VALUES:
        print(f"\n{'='*60}")
        print(f"Testing Discount Factor: {df_value}")
        print(f"{'='*60}")
        
        # Create modified parameters for this DF value
        legacy_params_df = copy.deepcopy(LEGACY_PARAMS)
        legacy_params_df['df'] = df_value
        
        legacy3_params_df = copy.deepcopy(LEGACY_3ROUND_PARAMS)
        legacy3_params_df['df'] = df_value
        
        # Store results for this DF value
        df_results = {}
        
        # Scenario 1: 2 Legacy QL vs 1 TFT
        print(f"\nRunning: 2 Legacy QL vs 1 TFT (DF={df_value})")
        agents_legacy_2v1 = [
            LegacyQLearner("LegacyQL_1", legacy_params_df),
            LegacyQLearner("LegacyQL_2", legacy_params_df),
            StaticAgent("TFT_1", "TFT")
        ]
        results_legacy_2v1 = run_multiple_simulations(agents_legacy_2v1, NUM_ROUNDS, NUM_RUNS)
        df_results["Legacy_2QL_vs_1TFT"] = results_legacy_2v1
        
        # Scenario 2: 2 Legacy3Round QL vs 1 TFT
        print(f"Running: 2 Legacy3Round QL vs 1 TFT (DF={df_value})")
        agents_legacy3_2v1 = [
            Legacy3RoundQLearner("Legacy3QL_1", legacy3_params_df),
            Legacy3RoundQLearner("Legacy3QL_2", legacy3_params_df),
            StaticAgent("TFT_1", "TFT")
        ]
        results_legacy3_2v1 = run_multiple_simulations(agents_legacy3_2v1, NUM_ROUNDS, NUM_RUNS)
        df_results["Legacy3Round_2QL_vs_1TFT"] = results_legacy3_2v1
        
        # Scenario 3: 1 Legacy QL vs 2 TFT
        print(f"Running: 1 Legacy QL vs 2 TFT (DF={df_value})")
        agents_legacy_1v2 = [
            LegacyQLearner("LegacyQL_1", legacy_params_df),
            StaticAgent("TFT_1", "TFT"),
            StaticAgent("TFT_2", "TFT")
        ]
        results_legacy_1v2 = run_multiple_simulations(agents_legacy_1v2, NUM_ROUNDS, NUM_RUNS)
        df_results["Legacy_1QL_vs_2TFT"] = results_legacy_1v2
        
        # Scenario 4: 1 Legacy3Round QL vs 2 TFT
        print(f"Running: 1 Legacy3Round QL vs 2 TFT (DF={df_value})")
        agents_legacy3_1v2 = [
            Legacy3RoundQLearner("Legacy3QL_1", legacy3_params_df),
            StaticAgent("TFT_1", "TFT"),
            StaticAgent("TFT_2", "TFT")
        ]
        results_legacy3_1v2 = run_multiple_simulations(agents_legacy3_1v2, NUM_ROUNDS, NUM_RUNS)
        df_results["Legacy3Round_1QL_vs_2TFT"] = results_legacy3_1v2
        
        # Store results for this DF value
        all_df_results[df_value] = df_results
    
    print("\n\nCreating visualizations...")
    
    # Create comparison plots for each scenario
    for scenario_name in ["Legacy_2QL_vs_1TFT", "Legacy3Round_2QL_vs_1TFT", 
                         "Legacy_1QL_vs_2TFT", "Legacy3Round_1QL_vs_2TFT"]:
        scenario_df_results = {}
        for df_value in DF_VALUES:
            scenario_df_results[df_value] = all_df_results[df_value][scenario_name]
        
        plot_df_comparison(scenario_df_results, scenario_name, 
                          os.path.join(OUTPUT_DIR, f"{scenario_name}_df_comparison.png"))
    
    # Create final comparison plot for Legacy3Round_2QL_vs_1TFT
    print("Creating final comparison plot for 2 Legacy3Round QL vs 1 TFT...")
    plot_final_comparison_legacy3round(all_df_results, 
                                     os.path.join(OUTPUT_DIR, "legacy3round_2ql_vs_1tft_all_df.png"))
    
    # Save CSV files
    print("\nSaving CSV files...")
    save_df_results_to_csv(all_df_results, OUTPUT_DIR)
    
    print(f"\nAll experiments complete! Results saved to '{OUTPUT_DIR}'")
    print(f"Total scenarios tested: {len(DF_VALUES)} DF values × 4 scenarios = {len(DF_VALUES) * 4} experiments")


if __name__ == "__main__":
    main()