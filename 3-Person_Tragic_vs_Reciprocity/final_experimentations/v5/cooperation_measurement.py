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
        'tft_coop_rate': np.mean([run['tft_coop_rate'] for run in pairwise_runs], axis=0),
        'ql_avg_score': np.mean([np.mean([run['ql_scores'][aid] for aid in run['ql_scores']], axis=0) 
                                 for run in pairwise_runs], axis=0) if pairwise_runs[0]['ql_scores'] else [],
        'tft_avg_score': np.mean([np.mean([run['tft_scores'][aid] for aid in run['tft_scores']], axis=0) 
                                  for run in pairwise_runs], axis=0) if pairwise_runs[0]['tft_scores'] else []
    }
    
    avg_nperson = {
        'ql_coop_rate': np.mean([run['ql_coop_rate'] for run in nperson_runs], axis=0),
        'tft_coop_rate': np.mean([run['tft_coop_rate'] for run in nperson_runs], axis=0),
        'ql_avg_score': np.mean([np.mean([run['ql_scores'][aid] for aid in run['ql_scores']], axis=0) 
                                for run in nperson_runs], axis=0) if nperson_runs[0]['ql_scores'] else [],
        'tft_avg_score': np.mean([np.mean([run['tft_scores'][aid] for aid in run['tft_scores']], axis=0) 
                                 for run in nperson_runs], axis=0) if nperson_runs[0]['tft_scores'] else []
    }
    
    return avg_pairwise, avg_nperson


def plot_cooperation_comparison(results, title, save_path):
    """Plot cooperation rates for QL vs TFT groups"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16)
    
    pairwise, nperson = results
    rounds = np.arange(len(pairwise['ql_coop_rate']))
    
    # Pairwise cooperation rates
    ax1.plot(rounds, pairwise['ql_coop_rate'], label='QL Group', color='blue', linewidth=2)
    ax1.plot(rounds, pairwise['tft_coop_rate'], label='TFT Group', color='green', linewidth=2)
    ax1.set_title('Pairwise Cooperation Rates')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Cooperation Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Neighborhood cooperation rates
    ax2.plot(rounds, nperson['ql_coop_rate'], label='QL Group', color='blue', linewidth=2)
    ax2.plot(rounds, nperson['tft_coop_rate'], label='TFT Group', color='green', linewidth=2)
    ax2.set_title('Neighborhood Cooperation Rates')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Cooperation Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    # Pairwise scores
    if len(pairwise['ql_avg_score']) > 0:
        ax3.plot(rounds, pairwise['ql_avg_score'], label='QL Group', color='blue', linewidth=2)
    if len(pairwise['tft_avg_score']) > 0:
        ax3.plot(rounds, pairwise['tft_avg_score'], label='TFT Group', color='green', linewidth=2)
    ax3.set_title('Pairwise Average Scores')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Cumulative Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Neighborhood scores
    if len(nperson['ql_avg_score']) > 0:
        ax4.plot(rounds, nperson['ql_avg_score'], label='QL Group', color='blue', linewidth=2)
    if len(nperson['tft_avg_score']) > 0:
        ax4.plot(rounds, nperson['tft_avg_score'], label='TFT Group', color='green', linewidth=2)
    ax4.set_title('Neighborhood Average Scores')
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Cumulative Score')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_combined_4lines(all_results, save_path):
    """Plot all 4 lines (pairwise QL, pairwise TFT, neighborhood QL, neighborhood TFT) on one graph"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Combined Cooperation Rates: All Scenarios', fontsize=16)
    
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':']
    
    # Plot cooperation rates
    for idx, (scenario_name, (pairwise, nperson)) in enumerate(all_results.items()):
        color = colors[idx % len(colors)]
        
        rounds = np.arange(len(pairwise['ql_coop_rate']))
        
        # Cooperation rates
        ax1.plot(rounds, pairwise['ql_coop_rate'], 
                label=f'{scenario_name} - Pairwise QL', 
                color=color, linestyle='-', linewidth=2, alpha=0.8)
        ax1.plot(rounds, pairwise['tft_coop_rate'], 
                label=f'{scenario_name} - Pairwise TFT', 
                color=color, linestyle='--', linewidth=2, alpha=0.8)
        ax1.plot(rounds, nperson['ql_coop_rate'], 
                label=f'{scenario_name} - Neighborhood QL', 
                color=color, linestyle='-.', linewidth=2, alpha=0.8)
        ax1.plot(rounds, nperson['tft_coop_rate'], 
                label=f'{scenario_name} - Neighborhood TFT', 
                color=color, linestyle=':', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Cooperation Rate')
    ax1.set_title('Cooperation Rates Across All Scenarios')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Plot final cooperation rates as bar chart
    scenario_names = []
    pairwise_ql_final = []
    pairwise_tft_final = []
    nperson_ql_final = []
    nperson_tft_final = []
    
    for scenario_name, (pairwise, nperson) in all_results.items():
        scenario_names.append(scenario_name)
        # Average last 100 rounds
        pairwise_ql_final.append(np.mean(pairwise['ql_coop_rate'][-100:]))
        pairwise_tft_final.append(np.mean(pairwise['tft_coop_rate'][-100:]))
        nperson_ql_final.append(np.mean(nperson['ql_coop_rate'][-100:]))
        nperson_tft_final.append(np.mean(nperson['tft_coop_rate'][-100:]))
    
    x = np.arange(len(scenario_names))
    width = 0.2
    
    ax2.bar(x - 1.5*width, pairwise_ql_final, width, label='Pairwise QL', color='blue', alpha=0.8)
    ax2.bar(x - 0.5*width, pairwise_tft_final, width, label='Pairwise TFT', color='green', alpha=0.8)
    ax2.bar(x + 0.5*width, nperson_ql_final, width, label='Neighborhood QL', color='red', alpha=0.8)
    ax2.bar(x + 1.5*width, nperson_tft_final, width, label='Neighborhood TFT', color='orange', alpha=0.8)
    
    ax2.set_xlabel('Scenario')
    ax2.set_ylabel('Final Cooperation Rate')
    ax2.set_title('Final Cooperation Rates (Last 100 Rounds)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
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
                'nperson_tft_coop': nperson['tft_coop_rate'][i],
                'pairwise_ql_score': pairwise['ql_avg_score'][i] if i < len(pairwise['ql_avg_score']) else None,
                'pairwise_tft_score': pairwise['tft_avg_score'][i] if i < len(pairwise['tft_avg_score']) else None,
                'nperson_ql_score': nperson['ql_avg_score'][i] if i < len(nperson['ql_avg_score']) else None,
                'nperson_tft_score': nperson['tft_avg_score'][i] if i < len(nperson['tft_avg_score']) else None
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
            'nperson_tft_final_coop': np.mean(nperson['tft_coop_rate'][-100:]),
            'pairwise_ql_total_score': pairwise['ql_avg_score'][-1] if len(pairwise['ql_avg_score']) > 0 else 0,
            'pairwise_tft_total_score': pairwise['tft_avg_score'][-1] if len(pairwise['tft_avg_score']) > 0 else 0,
            'nperson_ql_total_score': nperson['ql_avg_score'][-1] if len(nperson['ql_avg_score']) > 0 else 0,
            'nperson_tft_total_score': nperson['tft_avg_score'][-1] if len(nperson['tft_avg_score']) > 0 else 0
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