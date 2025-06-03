"""
Enhanced Analysis Script for 3-Person Tragic Valley vs Reciprocity Hill Results
===============================================================================
This script loads the experimental data and creates comparative visualizations
that distinguish between TFT cooperation rates when playing against TFT vs AllD.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import glob
from datetime import datetime
from collections import defaultdict


def load_results(results_dir: str) -> Dict:
    """Load all result files from the specified directory."""
    results = {}
    
    # Find all JSON files in the directory
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    
    for file_path in json_files:
        filename = os.path.basename(file_path)
        if filename != "all_results_combined.json":
            with open(file_path, 'r') as f:
                data = json.load(f)
                results[filename] = data
    
    return results


def calculate_moving_average(data: List[float], window: int = 5) -> List[float]:
    """Calculate moving average for smoothing."""
    if len(data) < window:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2 + 1)
        smoothed.append(np.mean(data[start:end]))
    
    return smoothed


def extract_pairwise_cooperation_by_opponent(results: Dict) -> Dict[str, Dict[str, List[float]]]:
    """
    Extract TFT cooperation rates separated by opponent type for pairwise mode.
    Returns: {round_num: {'TFT_vs_TFT': rate, 'TFT_vs_AllD': rate, 'TFT_overall': rate}}
    """
    round_data = results.get('round_data', [])
    if not round_data:
        return {}
    
    rounds_per_episode = results['summary']['rounds_per_episode']
    num_episodes = results['summary']['num_episodes']
    
    # Initialize storage for cooperation data by round
    coop_by_round = defaultdict(lambda: {'TFT_vs_TFT': [], 'TFT_vs_AllD': [], 'TFT_overall': []})
    
    for data_point in round_data:
        episode = data_point['episode']
        round_num = data_point['round']
        
        # For pairwise mode, check if we have interactions data
        if 'interactions' in data_point:
            for interaction in data_point['interactions']:
                agent1_name = interaction['agent1_name']
                agent2_name = interaction['agent2_name']
                action1 = interaction['agent1_action']
                action2 = interaction['agent2_action']
                
                # Track TFT cooperation rates
                if 'TFT' in agent1_name:
                    coop = 1 if action1 == 'C' else 0
                    coop_by_round[round_num]['TFT_overall'].append(coop)
                    
                    if 'TFT' in agent2_name:
                        coop_by_round[round_num]['TFT_vs_TFT'].append(coop)
                    elif 'AllD' in agent2_name:
                        coop_by_round[round_num]['TFT_vs_AllD'].append(coop)
                
                if 'TFT' in agent2_name:
                    coop = 1 if action2 == 'C' else 0
                    coop_by_round[round_num]['TFT_overall'].append(coop)
                    
                    if 'TFT' in agent1_name:
                        coop_by_round[round_num]['TFT_vs_TFT'].append(coop)
                    elif 'AllD' in agent1_name:
                        coop_by_round[round_num]['TFT_vs_AllD'].append(coop)
        else:
            # Fallback for data without detailed interactions
            actions = data_point.get('actions', {})
            for agent_name, action in actions.items():
                if 'TFT' in agent_name:
                    coop = 1 if action == 'C' else 0
                    coop_by_round[round_num]['TFT_overall'].append(coop)
    
    # Calculate averages for each round
    avg_by_round = {}
    for round_num in range(rounds_per_episode):
        avg_by_round[round_num] = {}
        for category in ['TFT_vs_TFT', 'TFT_vs_AllD', 'TFT_overall']:
            data = coop_by_round[round_num][category]
            avg_by_round[round_num][category] = np.mean(data) if data else 0.0
    
    return avg_by_round


def extract_tft_cooperation_over_rounds(results: Dict) -> Tuple[List[float], List[float]]:
    """Extract overall TFT agents' cooperation rates over rounds, averaged across episodes."""
    round_data = results.get('round_data', [])
    if not round_data:
        return [], []
    
    rounds_per_episode = results['summary']['rounds_per_episode']
    tft_cooperation_by_round = [[] for _ in range(rounds_per_episode)]
    
    for data_point in round_data:
        episode = data_point['episode']
        round_num = data_point['round']
        actions = data_point['actions']
        
        tft_actions = []
        for agent_name, action in actions.items():
            if 'TFT' in agent_name:
                tft_actions.append(1 if action == 'C' else 0)
        
        if tft_actions:
            tft_coop_rate = np.mean(tft_actions)
            tft_cooperation_by_round[round_num].append(tft_coop_rate)
    
    avg_tft_cooperation = []
    std_tft_cooperation = []
    
    for round_cooperations in tft_cooperation_by_round:
        if round_cooperations:
            avg_tft_cooperation.append(np.mean(round_cooperations))
            std_tft_cooperation.append(np.std(round_cooperations))
        else:
            avg_tft_cooperation.append(0)
            std_tft_cooperation.append(0)
    
    return avg_tft_cooperation, std_tft_cooperation


def extract_tft_scores_by_opponent(results: Dict) -> Dict[str, Dict[str, List[float]]]:
    """
    Extract TFT scores separated by interaction type for pairwise mode.
    This is more complex as we need to track cumulative scores from interactions.
    """
    # For now, return overall scores - this would require modifying the data collection
    # to track scores per interaction type
    return extract_tft_scores_over_rounds(results)


def extract_tft_scores_over_rounds(results: Dict) -> Tuple[List[float], List[float]]:
    """Extract TFT agents' scores over rounds, averaged across episodes."""
    round_data = results.get('round_data', [])
    if not round_data:
        return [], []
    
    rounds_per_episode = results['summary']['rounds_per_episode']
    
    # Track cumulative scores by episode and round
    tft_scores_by_episode = defaultdict(lambda: defaultdict(float))
    tft_agents_by_episode = defaultdict(set)
    
    for data_point in round_data:
        episode = data_point['episode']
        round_num = data_point['round']
        
        # For pairwise mode with interactions
        if 'interactions' in data_point:
            for interaction in data_point['interactions']:
                agent1_name = interaction['agent1_name']
                agent2_name = interaction['agent2_name']
                payoff1 = interaction['payoff1']
                payoff2 = interaction['payoff2']
                
                if 'TFT' in agent1_name:
                    tft_scores_by_episode[episode][agent1_name] += payoff1
                    tft_agents_by_episode[episode].add(agent1_name)
                if 'TFT' in agent2_name:
                    tft_scores_by_episode[episode][agent2_name] += payoff2
                    tft_agents_by_episode[episode].add(agent2_name)
        else:
            # Fallback for N-person mode
            payoffs = data_point.get('payoffs', {})
            for agent_name, payoff in payoffs.items():
                if 'TFT' in agent_name:
                    tft_scores_by_episode[episode][agent_name] += payoff
                    tft_agents_by_episode[episode].add(agent_name)
    
    # Calculate average scores per round
    avg_scores_by_round = []
    std_scores_by_round = []
    
    for round_num in range(rounds_per_episode):
        round_scores = []
        for episode in tft_scores_by_episode:
            if tft_agents_by_episode[episode]:
                # Average score per TFT agent in this episode up to this round
                total_score = sum(tft_scores_by_episode[episode].values())
                avg_score = total_score / len(tft_agents_by_episode[episode]) / (round_num + 1)
                round_scores.append(avg_score)
        
        if round_scores:
            avg_scores_by_round.append(np.mean(round_scores))
            std_scores_by_round.append(np.std(round_scores))
        else:
            avg_scores_by_round.append(0)
            std_scores_by_round.append(0)
    
    return avg_scores_by_round, std_scores_by_round


def create_enhanced_comparison_plots(results_dir: str):
    """Create enhanced comparison plots with separated TFT cooperation rates."""
    all_results = load_results(results_dir)
    
    if not all_results:
        print("No results found in directory!")
        return
    
    scenarios = [
        {
            'name': '2TFT + 1AllD (No Exploration)',
            'pairwise': 'results_2TFT_1AllD_NoExpl_Pairwise.json',
            'nperson': 'results_2TFT_1AllD_NoExpl_NPerson.json'
        },
        {
            'name': '2TFT + 1AllD (10% Exploration)',
            'pairwise': 'results_2TFT_1AllD_10Expl_Pairwise.json',
            'nperson': 'results_2TFT_1AllD_10Expl_NPerson.json'
        },
        {
            'name': '3TFT (No Exploration)',
            'pairwise': 'results_3TFT_NoExpl_Pairwise.json',
            'nperson': 'results_3TFT_NoExpl_NPerson.json'
        },
        {
            'name': '3TFT (10% Exploration)',
            'pairwise': 'results_3TFT_10Expl_Pairwise.json',
            'nperson': 'results_3TFT_10Expl_NPerson.json'
        }
    ]
    
    # Create figure with 2 rows x 4 columns for cooperation and scores
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle('3-Person IPD: Detailed Analysis of Pairwise vs N-Person Dynamics', fontsize=18)
    
    for idx, scenario in enumerate(scenarios):
        if scenario['pairwise'] not in all_results or scenario['nperson'] not in all_results:
            print(f"Missing data for scenario: {scenario['name']}")
            continue
        
        pw_results = all_results[scenario['pairwise']]
        np_results = all_results[scenario['nperson']]
        
        rounds_per_episode = pw_results['summary']['rounds_per_episode']
        rounds = list(range(rounds_per_episode))
        
        # --- Cooperation Rate Plot ---
        ax_coop = axes[0, idx]
        
        # For pairwise mode with AllD scenario, extract separated cooperation rates
        if '2TFT' in scenario['name']:
            pw_coop_by_opponent = extract_pairwise_cooperation_by_opponent(pw_results)
            
            # Extract the different cooperation rates
            pw_tft_vs_tft = [pw_coop_by_opponent[r]['TFT_vs_TFT'] for r in rounds]
            pw_tft_vs_alld = [pw_coop_by_opponent[r]['TFT_vs_AllD'] for r in rounds]
            pw_tft_overall = [pw_coop_by_opponent[r]['TFT_overall'] for r in rounds]
            
            # Smooth the data
            pw_tft_vs_tft_smooth = calculate_moving_average(pw_tft_vs_tft, window=3)
            pw_tft_vs_alld_smooth = calculate_moving_average(pw_tft_vs_alld, window=3)
            pw_tft_overall_smooth = calculate_moving_average(pw_tft_overall, window=3)
            
            # Plot pairwise lines
            ax_coop.plot(rounds, pw_tft_vs_tft_smooth, 'b-', label='PW: TFT vs TFT', linewidth=2.5)
            ax_coop.plot(rounds, pw_tft_vs_alld_smooth, 'b--', label='PW: TFT vs AllD', linewidth=2.5)
            ax_coop.plot(rounds, pw_tft_overall_smooth, 'b:', label='PW: TFT Overall', linewidth=2, alpha=0.7)
        else:
            # For 3TFT scenario, just show overall
            pw_coop, pw_coop_std = extract_tft_cooperation_over_rounds(pw_results)
            pw_coop_smooth = calculate_moving_average(pw_coop, window=3)
            ax_coop.plot(rounds, pw_coop_smooth, 'b-', label='Pairwise', linewidth=2.5)
        
        # N-Person cooperation (always overall)
        np_coop, np_coop_std = extract_tft_cooperation_over_rounds(np_results)
        np_coop_smooth = calculate_moving_average(np_coop, window=3)
        ax_coop.plot(rounds, np_coop_smooth, 'r-', label='N-Person', linewidth=2.5)
        
        # Add episode boundaries (faint vertical lines)
        for ep in range(1, pw_results['summary']['num_episodes']):
            ax_coop.axvline(x=0, color='gray', alpha=0.2, linestyle='--')
        
        ax_coop.set_title(f'{scenario["name"]}\nTFT Cooperation Rate', fontsize=12)
        ax_coop.set_xlabel('Round within Episode')
        ax_coop.set_ylabel('Cooperation Rate')
        ax_coop.set_ylim(-0.05, 1.05)
        ax_coop.grid(True, alpha=0.3)
        ax_coop.legend(fontsize=9)
        
        # --- Score Plot ---
        ax_score = axes[1, idx]
        
        pw_scores, pw_scores_std = extract_tft_scores_over_rounds(pw_results)
        np_scores, np_scores_std = extract_tft_scores_over_rounds(np_results)
        
        pw_scores_smooth = calculate_moving_average(pw_scores, window=3)
        np_scores_smooth = calculate_moving_average(np_scores, window=3)
        
        ax_score.plot(rounds, pw_scores_smooth, 'b-', label='Pairwise', linewidth=2.5)
        ax_score.plot(rounds, np_scores_smooth, 'r-', label='N-Person', linewidth=2.5)
        
        ax_score.set_title('TFT Average Score per Round', fontsize=12)
        ax_score.set_xlabel('Round within Episode')
        ax_score.set_ylabel('Average Score')
        ax_score.grid(True, alpha=0.3)
        ax_score.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'enhanced_tft_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced plots saved to {results_dir}")


def print_detailed_statistics(results_dir: str):
    """Print detailed summary statistics for each simulation."""
    all_results = load_results(results_dir)
    
    print("\n" + "="*80)
    print("DETAILED STATISTICS: RECIPROCITY HILL VS TRAGIC VALLEY")
    print("="*80)
    
    scenarios = [
        ('2TFT + 1AllD (No Exploration)', 'results_2TFT_1AllD_NoExpl_Pairwise.json', 'results_2TFT_1AllD_NoExpl_NPerson.json'),
        ('2TFT + 1AllD (10% Exploration)', 'results_2TFT_1AllD_10Expl_Pairwise.json', 'results_2TFT_1AllD_10Expl_NPerson.json'),
        ('3TFT (No Exploration)', 'results_3TFT_NoExpl_Pairwise.json', 'results_3TFT_NoExpl_NPerson.json'),
        ('3TFT (10% Exploration)', 'results_3TFT_10Expl_Pairwise.json', 'results_3TFT_10Expl_NPerson.json')
    ]
    
    for scenario_name, pw_file, np_file in scenarios:
        if pw_file not in all_results or np_file not in all_results:
            continue
        
        print(f"\n{scenario_name}:")
        print("-" * len(scenario_name))
        
        pw_summary = all_results[pw_file]['summary']
        np_summary = all_results[np_file]['summary']
        
        # Overall TFT cooperation rates
        pw_tft_coops = [rate for name, rate in pw_summary['cooperation_rates'].items() if 'TFT' in name]
        np_tft_coops = [rate for name, rate in np_summary['cooperation_rates'].items() if 'TFT' in name]
        
        pw_tft_avg = np.mean(pw_tft_coops) if pw_tft_coops else 0
        np_tft_avg = np.mean(np_tft_coops) if np_tft_coops else 0
        
        print(f"\nOverall TFT Cooperation Rates:")
        print(f"  Pairwise: {pw_tft_avg:.1%}")
        print(f"  N-Person: {np_tft_avg:.1%}")
        print(f"  Difference: {pw_tft_avg - np_tft_avg:+.1%}")
        
        # For 2TFT+1AllD scenarios, calculate separated cooperation rates
        if '2TFT' in scenario_name:
            pw_results = all_results[pw_file]
            pw_coop_by_opponent = extract_pairwise_cooperation_by_opponent(pw_results)
            
            # Calculate averages across all rounds
            all_tft_vs_tft = []
            all_tft_vs_alld = []
            
            for round_data in pw_coop_by_opponent.values():
                all_tft_vs_tft.append(round_data['TFT_vs_TFT'])
                all_tft_vs_alld.append(round_data['TFT_vs_AllD'])
            
            avg_tft_vs_tft = np.mean(all_tft_vs_tft) if all_tft_vs_tft else 0
            avg_tft_vs_alld = np.mean(all_tft_vs_alld) if all_tft_vs_alld else 0
            
            print(f"\nPairwise Mode - TFT Cooperation by Opponent Type:")
            print(f"  TFT vs TFT: {avg_tft_vs_tft:.1%}")
            print(f"  TFT vs AllD: {avg_tft_vs_alld:.1%}")
            print(f"  Difference: {avg_tft_vs_tft - avg_tft_vs_alld:+.1%}")
        
        # Average scores
        pw_tft_scores = [score for name, score in pw_summary['average_scores_per_round'].items() if 'TFT' in name]
        np_tft_scores = [score for name, score in np_summary['average_scores_per_round'].items() if 'TFT' in name]
        pw_alld_scores = [score for name, score in pw_summary['average_scores_per_round'].items() if 'AllD' in name]
        np_alld_scores = [score for name, score in np_summary['average_scores_per_round'].items() if 'AllD' in name]
        
        pw_tft_score_avg = np.mean(pw_tft_scores) if pw_tft_scores else 0
        np_tft_score_avg = np.mean(np_tft_scores) if np_tft_scores else 0
        pw_alld_score_avg = np.mean(pw_alld_scores) if pw_alld_scores else 0
        np_alld_score_avg = np.mean(np_alld_scores) if np_alld_scores else 0
        
        print(f"\nAverage Scores per Round:")
        print(f"  TFT agents:")
        print(f"    Pairwise: {pw_tft_score_avg:.2f}")
        print(f"    N-Person: {np_tft_score_avg:.2f}")
        print(f"    Difference: {pw_tft_score_avg - np_tft_score_avg:+.2f}")
        
        if pw_alld_scores:
            print(f"  AllD agent:")
            print(f"    Pairwise: {pw_alld_score_avg:.2f}")
            print(f"    N-Person: {np_alld_score_avg:.2f}")
            print(f"    Difference: {pw_alld_score_avg - np_alld_score_avg:+.2f}")
        
        # Final episode scores
        pw_final_episode = pw_summary.get('final_episode_avg_score', {})
        np_final_episode = np_summary.get('final_episode_avg_score', {})
        
        if pw_final_episode:
            print(f"\nFinal Episode Average Scores:")
            for agent_type in ['TFT', 'AllD']:
                pw_agents = [score for name, score in pw_final_episode.items() if agent_type in name]
                np_agents = [score for name, score in np_final_episode.items() if agent_type in name]
                
                if pw_agents:
                    pw_avg = np.mean(pw_agents)
                    np_avg = np.mean(np_agents) if np_agents else 0
                    print(f"  {agent_type} agents:")
                    print(f"    Pairwise: {pw_avg:.2f}")
                    print(f"    N-Person: {np_avg:.2f}")
        
        # Interpretation
        print(f"\nInterpretation:")
        if pw_tft_avg - np_tft_avg > 0.15:
            print(f"  → STRONG RECIPROCITY HILL: Pairwise enables much better cooperation")
        elif pw_tft_avg - np_tft_avg > 0.05:
            print(f"  → MODERATE RECIPROCITY HILL: Pairwise maintains cooperation better")
        elif np_tft_avg - pw_tft_avg > 0.05:
            print(f"  → UNEXPECTED: N-Person performs better")
        else:
            print(f"  → SIMILAR: Both modes yield comparable cooperation")
        
        if '2TFT' in scenario_name and avg_tft_vs_tft > 0.7 and avg_tft_vs_alld < 0.3:
            print(f"  → DISCRIMINATORY SUCCESS: TFTs cooperate with each other but not with AllD")
        
        if np_tft_avg < 0.2:
            print(f"  → TRAGIC VALLEY DETECTED: N-Person cooperation collapsed")


def create_summary_statistics_plot(results_dir: str):
    """Create a comprehensive summary plot showing key statistics."""
    all_results = load_results(results_dir)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Summary Statistics: Pairwise vs N-Person Dynamics', fontsize=16)
    
    scenarios = ['2TFT+1AllD\nNo Expl', '2TFT+1AllD\n10% Expl', '3TFT\nNo Expl', '3TFT\n10% Expl']
    files = [
        ('results_2TFT_1AllD_NoExpl_Pairwise.json', 'results_2TFT_1AllD_NoExpl_NPerson.json'),
        ('results_2TFT_1AllD_10Expl_Pairwise.json', 'results_2TFT_1AllD_10Expl_NPerson.json'),
        ('results_3TFT_NoExpl_Pairwise.json', 'results_3TFT_NoExpl_NPerson.json'),
        ('results_3TFT_10Expl_Pairwise.json', 'results_3TFT_10Expl_NPerson.json')
    ]
    
    # Data collection
    pw_overall_coop = []
    np_overall_coop = []
    pw_tft_vs_tft = []
    pw_tft_vs_alld = []
    pw_tft_scores = []
    np_tft_scores = []
    pw_alld_scores = []
    np_alld_scores = []
    
    for pw_file, np_file in files:
        if pw_file in all_results and np_file in all_results:
            # Overall cooperation
            pw_summary = all_results[pw_file]['summary']
            np_summary = all_results[np_file]['summary']
            
            pw_tft_coops = [rate for name, rate in pw_summary['cooperation_rates'].items() if 'TFT' in name]
            np_tft_coops = [rate for name, rate in np_summary['cooperation_rates'].items() if 'TFT' in name]
            
            pw_overall_coop.append(np.mean(pw_tft_coops) if pw_tft_coops else 0)
            np_overall_coop.append(np.mean(np_tft_coops) if np_tft_coops else 0)
            
            # Separated cooperation for 2TFT scenarios
            if '2TFT' in pw_file:
                pw_coop_by_opponent = extract_pairwise_cooperation_by_opponent(all_results[pw_file])
                all_tft_tft = [d['TFT_vs_TFT'] for d in pw_coop_by_opponent.values()]
                all_tft_alld = [d['TFT_vs_AllD'] for d in pw_coop_by_opponent.values()]
                pw_tft_vs_tft.append(np.mean(all_tft_tft) if all_tft_tft else 0)
                pw_tft_vs_alld.append(np.mean(all_tft_alld) if all_tft_alld else 0)
            else:
                pw_tft_vs_tft.append(np.nan)
                pw_tft_vs_alld.append(np.nan)
            
            # Scores
            pw_tft_sc = [score for name, score in pw_summary['average_scores_per_round'].items() if 'TFT' in name]
            np_tft_sc = [score for name, score in np_summary['average_scores_per_round'].items() if 'TFT' in name]
            pw_alld_sc = [score for name, score in pw_summary['average_scores_per_round'].items() if 'AllD' in name]
            np_alld_sc = [score for name, score in np_summary['average_scores_per_round'].items() if 'AllD' in name]
            
            pw_tft_scores.append(np.mean(pw_tft_sc) if pw_tft_sc else 0)
            np_tft_scores.append(np.mean(np_tft_sc) if np_tft_sc else 0)
            pw_alld_scores.append(np.mean(pw_alld_sc) if pw_alld_sc else np.nan)
            np_alld_scores.append(np.mean(np_alld_sc) if np_alld_sc else np.nan)
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    # Plot 1: Overall TFT Cooperation
    bars1 = ax1.bar(x - width/2, pw_overall_coop, width, label='Pairwise', color='blue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, np_overall_coop, width, label='N-Person', color='red', alpha=0.8)
    
    ax1.set_ylabel('Cooperation Rate')
    ax1.set_title('Overall TFT Cooperation Rates')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.0%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Pairwise TFT Cooperation by Opponent (only for 2TFT scenarios)
    x_2tft = np.array([0, 1])  # Only first two scenarios
    bars3 = ax2.bar(x_2tft - width/2, pw_tft_vs_tft[:2], width, label='TFT vs TFT', color='green', alpha=0.8)
    bars4 = ax2.bar(x_2tft + width/2, pw_tft_vs_alld[:2], width, label='TFT vs AllD', color='orange', alpha=0.8)
    
    ax2.set_ylabel('Cooperation Rate')
    ax2.set_title('Pairwise: TFT Cooperation by Opponent Type')
    ax2.set_xticks(x_2tft)
    ax2.set_xticklabels(['2TFT+1AllD\nNo Expl', '2TFT+1AllD\n10% Expl'])
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.0%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # Plot 3: TFT Average Scores
    bars5 = ax3.bar(x - width/2, pw_tft_scores, width, label='Pairwise', color='blue', alpha=0.8)
    bars6 = ax3.bar(x + width/2, np_tft_scores, width, label='N-Person', color='red', alpha=0.8)
    
    ax3.set_ylabel('Average Score per Round')
    ax3.set_title('TFT Performance (Average Score)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(scenarios)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars5, bars6]:
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # Plot 4: AllD Scores (only for 2TFT scenarios)
    valid_indices = [i for i, v in enumerate(pw_alld_scores) if not np.isnan(v)]
    if valid_indices:
        x_alld = np.array(valid_indices)
        pw_alld_valid = [pw_alld_scores[i] for i in valid_indices]
        np_alld_valid = [np_alld_scores[i] for i in valid_indices]
        
        bars7 = ax4.bar(x_alld - width/2, pw_alld_valid, width, label='Pairwise', color='blue', alpha=0.8)
        bars8 = ax4.bar(x_alld + width/2, np_alld_valid, width, label='N-Person', color='red', alpha=0.8)
        
        ax4.set_ylabel('Average Score per Round')
        ax4.set_title('AllD Performance (Average Score)')
        ax4.set_xticks(x_alld)
        ax4.set_xticklabels(['2TFT+1AllD\nNo Expl', '2TFT+1AllD\n10% Expl'])
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars7, bars8]:
            for bar in bars:
                height = bar.get_height()
                ax4.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'summary_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main analysis function."""
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Find the most recent results directory
        results_dirs = glob.glob("results/chris_huyck_*")
        if not results_dirs:
            print("No results directories found!")
            print("Please run chris_huyck_experiments.py first.")
            return
        
        # Sort by timestamp and get the most recent
        results_dirs.sort()
        results_dir = results_dirs[-1]
    
    print(f"Analyzing results from: {results_dir}")
    
    # Create enhanced plots
    create_enhanced_comparison_plots(results_dir)
    
    # Create summary statistics plot
    create_summary_statistics_plot(results_dir)
    
    # Print detailed statistics
    print_detailed_statistics(results_dir)
    
    # Also create the original plots for comparison
    from analyze_3_person_results import create_comparison_plots
    create_comparison_plots(results_dir)
    
    print(f"\nAnalysis complete! All plots saved to: {results_dir}")


if __name__ == "__main__":
    main()
