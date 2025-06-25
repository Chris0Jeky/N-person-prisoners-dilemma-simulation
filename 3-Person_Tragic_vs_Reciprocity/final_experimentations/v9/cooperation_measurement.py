"""
Cooperation Measurement Script
Focuses on measuring cooperation between Q-Learning agents and TFT agents
Scenarios:
- 2 QL vs 1 TFT (both Legacy and Legacy3Round versions)
- 1 QL vs 2 TFT (both Legacy and Legacy3Round versions)
Tracks cooperation rates for QL group vs TFT group separately
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from collections import defaultdict
import time

from final_agents import StaticAgent, LegacyQLearner, Legacy3RoundQLearner
from config import LEGACY_PARAMS, LEGACY_3ROUND_PARAMS, SIMULATION_CONFIG

# --- Payoff Logic ---
PAYOFFS_2P = {(0, 0): (3, 3), (0, 1): (0, 5), (1, 0): (5, 0), (1, 1): (1, 1)}
T, R, P, S = 5, 3, 1, 0
COOPERATE, DEFECT = 0, 1


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


def plot_cooperation_comparison(results, title, save_path):
    """Plot cooperation rates for QL vs TFT groups with smoothing"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title, fontsize=16)
    
    pairwise, nperson = results
    rounds = np.arange(len(pairwise['ql_coop_rate']))
    
    # Determine smoothing window based on number of rounds
    smooth_window = max(50, len(rounds) // 100)
    
    # Smooth the data
    pairwise_ql_smooth = smooth_data(pairwise['ql_coop_rate'], smooth_window)
    pairwise_tft_smooth = smooth_data(pairwise['tft_coop_rate'], smooth_window)
    nperson_ql_smooth = smooth_data(nperson['ql_coop_rate'], smooth_window)
    nperson_tft_smooth = smooth_data(nperson['tft_coop_rate'], smooth_window)
    
    # Pairwise cooperation rates
    # Plot raw data with low alpha
    ax1.plot(rounds, pairwise['ql_coop_rate'], color='blue', alpha=0.2, linewidth=0.5)
    ax1.plot(rounds, pairwise['tft_coop_rate'], color='green', alpha=0.2, linewidth=0.5)
    # Plot smoothed data with high alpha
    ax1.plot(rounds, pairwise_ql_smooth, label='QL Group', color='blue', linewidth=2.5)
    ax1.plot(rounds, pairwise_tft_smooth, label='TFT Group', color='green', linewidth=2.5)
    ax1.set_title('Pairwise Cooperation Rates', fontsize=14)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Cooperation Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Neighborhood cooperation rates
    # Plot raw data with low alpha
    ax2.plot(rounds, nperson['ql_coop_rate'], color='blue', alpha=0.2, linewidth=0.5)
    ax2.plot(rounds, nperson['tft_coop_rate'], color='green', alpha=0.2, linewidth=0.5)
    # Plot smoothed data with high alpha
    ax2.plot(rounds, nperson_ql_smooth, label='QL Group', color='blue', linewidth=2.5)
    ax2.plot(rounds, nperson_tft_smooth, label='TFT Group', color='green', linewidth=2.5)
    ax2.set_title('Neighborhood Cooperation Rates', fontsize=14)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Cooperation Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_combined_4lines(all_results, save_path):
    """Plot all 4 lines (pairwise QL, pairwise TFT, neighborhood QL, neighborhood TFT) on one graph"""
    fig = plt.figure(figsize=(14, 8))
    ax = plt.subplot(111)
    
    colors = {'Legacy': 'blue', 'Legacy3Round': 'purple', 'TFT': 'green'}
    base_colors = ['blue', 'purple']
    
    # Determine smoothing window
    first_scenario = list(all_results.values())[0]
    num_rounds = len(first_scenario[0]['ql_coop_rate'])
    smooth_window = max(50, num_rounds // 100)
    
    # Plot cooperation rates with smoothing
    color_idx = 0
    for scenario_name, (pairwise, nperson) in all_results.items():
        # Determine if this is Legacy or Legacy3Round
        if 'Legacy3Round' in scenario_name:
            base_color = colors['Legacy3Round']
        else:
            base_color = colors['Legacy']
        
        rounds = np.arange(len(pairwise['ql_coop_rate']))
        
        # Smooth the data
        pairwise_ql_smooth = smooth_data(pairwise['ql_coop_rate'], smooth_window)
        pairwise_tft_smooth = smooth_data(pairwise['tft_coop_rate'], smooth_window)
        nperson_ql_smooth = smooth_data(nperson['ql_coop_rate'], smooth_window)
        nperson_tft_smooth = smooth_data(nperson['tft_coop_rate'], smooth_window)
        
        # Determine line styles based on scenario type
        if '2QL_vs_1TFT' in scenario_name:
            ql_style = '-'
            tft_style = '--'
            alpha = 1.0
        else:  # 1QL_vs_2TFT
            ql_style = '-.'
            tft_style = ':'
            alpha = 0.8
        
        # Plot smoothed lines with labels
        label_suffix = ' (2v1)' if '2QL_vs_1TFT' in scenario_name else ' (1v2)'
        ql_type = 'Legacy3Round' if 'Legacy3Round' in scenario_name else 'Legacy'
        
        ax.plot(rounds, pairwise_ql_smooth, 
                label=f'{ql_type} QL - Pairwise{label_suffix}', 
                color=base_color, linestyle=ql_style, linewidth=2.5, alpha=alpha)
        ax.plot(rounds, pairwise_tft_smooth, 
                label=f'TFT - Pairwise{label_suffix}', 
                color=colors['TFT'], linestyle=ql_style, linewidth=2.5, alpha=alpha)
        ax.plot(rounds, nperson_ql_smooth, 
                label=f'{ql_type} QL - Neighborhood{label_suffix}', 
                color=base_color, linestyle=tft_style, linewidth=2.5, alpha=alpha*0.8)
        ax.plot(rounds, nperson_tft_smooth, 
                label=f'TFT - Neighborhood{label_suffix}', 
                color=colors['TFT'], linestyle=tft_style, linewidth=2.5, alpha=alpha*0.8)
    
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Cooperation Rate', fontsize=12)
    ax.set_title('Cooperation Rates: All Scenarios Combined', fontsize=16)
    
    # Move legend outside plot area
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_results_to_csv(all_results, output_dir):
    """Save detailed results to CSV files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed time series
    for scenario_name, (pairwise, nperson) in all_results.items():
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
        filename = f"{scenario_name}_{timestamp}.csv"
        df.to_csv(os.path.join(output_dir, filename), index=False)
        print(f"Saved {filename}")
    
    # Save summary statistics
    summary_data = []
    for scenario_name, (pairwise, nperson) in all_results.items():
        # Calculate final averages (last 100 rounds)
        summary_data.append({
            'scenario': scenario_name,
            'pairwise_ql_final_coop': np.mean(pairwise['ql_coop_rate'][-100:]),
            'pairwise_tft_final_coop': np.mean(pairwise['tft_coop_rate'][-100:]),
            'nperson_ql_final_coop': np.mean(nperson['ql_coop_rate'][-100:]),
            'nperson_tft_final_coop': np.mean(nperson['tft_coop_rate'][-100:])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_filename = f"summary_cooperation_{timestamp}.csv"
    summary_df.to_csv(os.path.join(output_dir, summary_filename), index=False)
    print(f"Saved {summary_filename}")


def main():
    # Configuration
    NUM_ROUNDS = SIMULATION_CONFIG['num_rounds']
    NUM_RUNS = SIMULATION_CONFIG['num_runs']
    OUTPUT_DIR = "cooperation_measurement_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Running cooperation measurement experiments")
    print(f"Rounds: {NUM_ROUNDS}, Runs: {NUM_RUNS}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Store all results
    all_results = {}
    
    # Scenario 1: 2 Legacy QL vs 1 TFT
    print("Running Scenario 1: 2 Legacy QL vs 1 TFT")
    agents_legacy_2v1 = [
        LegacyQLearner("LegacyQL_1", LEGACY_PARAMS),
        LegacyQLearner("LegacyQL_2", LEGACY_PARAMS),
        StaticAgent("TFT_1", "TFT")
    ]
    results_legacy_2v1 = run_multiple_simulations(agents_legacy_2v1, NUM_ROUNDS, NUM_RUNS)
    all_results["Legacy_2QL_vs_1TFT"] = results_legacy_2v1
    plot_cooperation_comparison(results_legacy_2v1, "2 Legacy QL vs 1 TFT", 
                               os.path.join(OUTPUT_DIR, "legacy_2ql_vs_1tft.png"))
    
    # Scenario 2: 2 Legacy3Round QL vs 1 TFT
    print("Running Scenario 2: 2 Legacy3Round QL vs 1 TFT")
    agents_legacy3_2v1 = [
        Legacy3RoundQLearner("Legacy3QL_1", LEGACY_3ROUND_PARAMS),
        Legacy3RoundQLearner("Legacy3QL_2", LEGACY_3ROUND_PARAMS),
        StaticAgent("TFT_1", "TFT")
    ]
    results_legacy3_2v1 = run_multiple_simulations(agents_legacy3_2v1, NUM_ROUNDS, NUM_RUNS)
    all_results["Legacy3Round_2QL_vs_1TFT"] = results_legacy3_2v1
    plot_cooperation_comparison(results_legacy3_2v1, "2 Legacy3Round QL vs 1 TFT", 
                               os.path.join(OUTPUT_DIR, "legacy3round_2ql_vs_1tft.png"))
    
    # Scenario 3: 1 Legacy QL vs 2 TFT
    print("Running Scenario 3: 1 Legacy QL vs 2 TFT")
    agents_legacy_1v2 = [
        LegacyQLearner("LegacyQL_1", LEGACY_PARAMS),
        StaticAgent("TFT_1", "TFT"),
        StaticAgent("TFT_2", "TFT")
    ]
    results_legacy_1v2 = run_multiple_simulations(agents_legacy_1v2, NUM_ROUNDS, NUM_RUNS)
    all_results["Legacy_1QL_vs_2TFT"] = results_legacy_1v2
    plot_cooperation_comparison(results_legacy_1v2, "1 Legacy QL vs 2 TFT", 
                               os.path.join(OUTPUT_DIR, "legacy_1ql_vs_2tft.png"))
    
    # Scenario 4: 1 Legacy3Round QL vs 2 TFT
    print("Running Scenario 4: 1 Legacy3Round QL vs 2 TFT")
    agents_legacy3_1v2 = [
        Legacy3RoundQLearner("Legacy3QL_1", LEGACY_3ROUND_PARAMS),
        StaticAgent("TFT_1", "TFT"),
        StaticAgent("TFT_2", "TFT")
    ]
    results_legacy3_1v2 = run_multiple_simulations(agents_legacy3_1v2, NUM_ROUNDS, NUM_RUNS)
    all_results["Legacy3Round_1QL_vs_2TFT"] = results_legacy3_1v2
    plot_cooperation_comparison(results_legacy3_1v2, "1 Legacy3Round QL vs 2 TFT", 
                               os.path.join(OUTPUT_DIR, "legacy3round_1ql_vs_2tft.png"))
    
    # Create combined plot
    print("\nCreating combined 4-line plot...")
    plot_combined_4lines(all_results, os.path.join(OUTPUT_DIR, "combined_all_scenarios.png"))
    
    # Save CSV files
    print("\nSaving CSV files...")
    save_results_to_csv(all_results, OUTPUT_DIR)
    
    print(f"\nAll experiments complete! Results saved to '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()