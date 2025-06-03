"""
Analysis Script for 3-Person Tragic Valley vs Reciprocity Hill Results
======================================================================
This script loads the experimental data and creates comparative visualizations.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import glob
from datetime import datetime


def load_results(results_dir: str) -> Dict:
    """Load all result files from the specified directory."""
    results = {}
    
    # Find all JSON files in the directory
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    
    for file_path in json_files:
        filename = os.path.basename(file_path)
        if filename != "all_results_combined.json":
            # Extract scenario info from filename
            parts = filename.replace("results_", "").replace(".json", "").split("_")
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                results[filename] = data
    
    return results


def calculate_moving_average(data: List[float], window: int = 10) -> List[float]:
    """Calculate moving average for smoothing."""
    if len(data) < window:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2 + 1)
        smoothed.append(np.mean(data[start:end]))
    
    return smoothed


def extract_tft_cooperation_over_rounds(results: Dict) -> Tuple[List[float], List[float]]:
    """Extract TFT agents' cooperation rates over rounds, averaged across episodes."""
    
    # Get round data
    round_data = results.get('round_data', [])
    if not round_data:
        return [], []
    
    # Get number of rounds per episode
    rounds_per_episode = results['summary']['rounds_per_episode']
    num_episodes = results['summary']['num_episodes']
    
    # Initialize arrays to store cooperation data
    tft_cooperation_by_round = [[] for _ in range(rounds_per_episode)]
    
    # Process each round
    for data_point in round_data:
        episode = data_point['episode']
        round_num = data_point['round']
        actions = data_point['actions']
        
        # Get TFT agents' actions
        tft_actions = []
        for agent_name, action in actions.items():
            if 'TFT' in agent_name:
                tft_actions.append(1 if action == 'C' else 0)
        
        if tft_actions:
            # Average cooperation rate among TFT agents for this round
            tft_coop_rate = np.mean(tft_actions)
            tft_cooperation_by_round[round_num].append(tft_coop_rate)
    
    # Average across episodes for each round
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


def extract_tft_scores_over_rounds(results: Dict) -> Tuple[List[float], List[float]]:
    """Extract TFT agents' scores over rounds, averaged across episodes."""
    
    # Get episode data
    episode_data = results.get('episode_data', [])
    if not episode_data:
        return [], []
    
    # Get number of rounds per episode
    rounds_per_episode = results['summary']['rounds_per_episode']
    
    # Initialize arrays to store score data
    tft_scores_by_round = [[] for _ in range(rounds_per_episode)]
    
    # Process each episode
    for episode in episode_data:
        scores = episode['scores']
        
        # Get TFT agents' scores
        tft_agent_names = [name for name in scores.keys() if 'TFT' in name]
        
        if tft_agent_names:
            # For each round
            for round_num in range(rounds_per_episode):
                # Average score among TFT agents for this round
                round_scores = []
                for agent_name in tft_agent_names:
                    if round_num < len(scores[agent_name]):
                        round_scores.append(scores[agent_name][round_num])
                
                if round_scores:
                    tft_scores_by_round[round_num].append(np.mean(round_scores))
    
    # Average across episodes for each round
    avg_tft_scores = []
    std_tft_scores = []
    
    for round_scores in tft_scores_by_round:
        if round_scores:
            avg_tft_scores.append(np.mean(round_scores))
            std_tft_scores.append(np.std(round_scores))
        else:
            avg_tft_scores.append(0)
            std_tft_scores.append(0)
    
    return avg_tft_scores, std_tft_scores


def create_comparison_plots(results_dir: str):
    """Create comparison plots for all scenarios."""
    
    # Load all results
    all_results = load_results(results_dir)
    
    if not all_results:
        print("No results found in directory!")
        return
    
    # Define scenario groups
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
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('3-Person Iterated Prisoner\'s Dilemma: Pairwise vs N-Person Dynamics', fontsize=16)
    
    for idx, scenario in enumerate(scenarios):
        if scenario['pairwise'] not in all_results or scenario['nperson'] not in all_results:
            print(f"Missing data for scenario: {scenario['name']}")
            continue
        
        # Get data
        pw_results = all_results[scenario['pairwise']]
        np_results = all_results[scenario['nperson']]
        
        # Extract cooperation rates
        pw_coop, pw_coop_std = extract_tft_cooperation_over_rounds(pw_results)
        np_coop, np_coop_std = extract_tft_cooperation_over_rounds(np_results)
        
        # Extract scores
        pw_scores, pw_scores_std = extract_tft_scores_over_rounds(pw_results)
        np_scores, np_scores_std = extract_tft_scores_over_rounds(np_results)
        
        # Plot cooperation rates
        ax_coop = axes[0, idx]
        rounds = list(range(len(pw_coop)))
        
        # Smooth the data for clearer visualization
        pw_coop_smooth = calculate_moving_average(pw_coop, window=5)
        np_coop_smooth = calculate_moving_average(np_coop, window=5)
        
        ax_coop.plot(rounds, pw_coop_smooth, 'b-', label='Pairwise', linewidth=2)
        ax_coop.plot(rounds, np_coop_smooth, 'r-', label='N-Person', linewidth=2)
        
        # Add confidence bands
        ax_coop.fill_between(rounds, 
                            np.array(pw_coop) - np.array(pw_coop_std),
                            np.array(pw_coop) + np.array(pw_coop_std),
                            alpha=0.2, color='blue')
        ax_coop.fill_between(rounds, 
                            np.array(np_coop) - np.array(np_coop_std),
                            np.array(np_coop) + np.array(np_coop_std),
                            alpha=0.2, color='red')
        
        ax_coop.set_title(f'{scenario["name"]}\nTFT Cooperation Rate')
        ax_coop.set_xlabel('Round')
        ax_coop.set_ylabel('Cooperation Rate')
        ax_coop.set_ylim(-0.05, 1.05)
        ax_coop.grid(True, alpha=0.3)
        ax_coop.legend()
        
        # Plot scores
        ax_score = axes[1, idx]
        
        # Smooth the scores
        pw_scores_smooth = calculate_moving_average(pw_scores, window=5)
        np_scores_smooth = calculate_moving_average(np_scores, window=5)
        
        ax_score.plot(rounds, pw_scores_smooth, 'b-', label='Pairwise', linewidth=2)
        ax_score.plot(rounds, np_scores_smooth, 'r-', label='N-Person', linewidth=2)
        
        # Add confidence bands
        ax_score.fill_between(rounds, 
                             np.array(pw_scores) - np.array(pw_scores_std),
                             np.array(pw_scores) + np.array(pw_scores_std),
                             alpha=0.2, color='blue')
        ax_score.fill_between(rounds, 
                             np.array(np_scores) - np.array(np_scores_std),
                             np.array(np_scores) + np.array(np_scores_std),
                             alpha=0.2, color='red')
        
        ax_score.set_title('TFT Average Score per Round')
        ax_score.set_xlabel('Round')
        ax_score.set_ylabel('Average Score')
        ax_score.grid(True, alpha=0.3)
        ax_score.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'tft_cooperation_and_scores_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary comparison plot
    create_summary_plot(all_results, scenarios, results_dir)
    
    print(f"Plots saved to {results_dir}")


def create_summary_plot(all_results: Dict, scenarios: List[Dict], results_dir: str):
    """Create a summary bar plot comparing final outcomes."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Summary: Pairwise vs N-Person Outcomes', fontsize=16)
    
    scenario_names = []
    pw_final_coop = []
    np_final_coop = []
    pw_avg_scores = []
    np_avg_scores = []
    
    for scenario in scenarios:
        if scenario['pairwise'] not in all_results or scenario['nperson'] not in all_results:
            continue
        
        scenario_names.append(scenario['name'].replace(' ', '\n'))
        
        # Get final cooperation rates
        pw_summary = all_results[scenario['pairwise']]['summary']
        np_summary = all_results[scenario['nperson']]['summary']
        
        # Calculate TFT-only cooperation rates
        pw_tft_coops = [rate for name, rate in pw_summary['cooperation_rates'].items() if 'TFT' in name]
        np_tft_coops = [rate for name, rate in np_summary['cooperation_rates'].items() if 'TFT' in name]
        
        pw_final_coop.append(np.mean(pw_tft_coops) if pw_tft_coops else 0)
        np_final_coop.append(np.mean(np_tft_coops) if np_tft_coops else 0)
        
        # Get average scores
        pw_tft_scores = [score for name, score in pw_summary['average_scores_per_round'].items() if 'TFT' in name]
        np_tft_scores = [score for name, score in np_summary['average_scores_per_round'].items() if 'TFT' in name]
        
        pw_avg_scores.append(np.mean(pw_tft_scores) if pw_tft_scores else 0)
        np_avg_scores.append(np.mean(np_tft_scores) if np_tft_scores else 0)
    
    # Plot cooperation rates
    x = np.arange(len(scenario_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, pw_final_coop, width, label='Pairwise', color='blue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, np_final_coop, width, label='N-Person', color='red', alpha=0.8)
    
    ax1.set_ylabel('Average TFT Cooperation Rate')
    ax1.set_title('Final Cooperation Rates')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.2%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    # Plot average scores
    bars3 = ax2.bar(x - width/2, pw_avg_scores, width, label='Pairwise', color='blue', alpha=0.8)
    bars4 = ax2.bar(x + width/2, np_avg_scores, width, label='N-Person', color='red', alpha=0.8)
    
    ax2.set_ylabel('Average Score per Round')
    ax2.set_title('TFT Performance (Average Score)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenario_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'summary_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def analyze_reciprocity_vs_tragedy(results_dir: str):
    """Analyze and print key findings about reciprocity hill vs tragic valley."""
    
    all_results = load_results(results_dir)
    
    print("\n" + "="*80)
    print("RECIPROCITY HILL VS TRAGIC VALLEY ANALYSIS")
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
        
        pw_summary = all_results[pw_file]['summary']
        np_summary = all_results[np_file]['summary']
        
        # TFT cooperation rates
        pw_tft_coops = [rate for name, rate in pw_summary['cooperation_rates'].items() if 'TFT' in name]
        np_tft_coops = [rate for name, rate in np_summary['cooperation_rates'].items() if 'TFT' in name]
        
        pw_tft_avg = np.mean(pw_tft_coops) if pw_tft_coops else 0
        np_tft_avg = np.mean(np_tft_coops) if np_tft_coops else 0
        
        print(f"  TFT Cooperation Rates:")
        print(f"    Pairwise: {pw_tft_avg:.2%}")
        print(f"    N-Person: {np_tft_avg:.2%}")
        print(f"    Difference: {pw_tft_avg - np_tft_avg:+.2%}")
        
        # Interpretation
        if pw_tft_avg - np_tft_avg > 0.1:
            print(f"    → RECIPROCITY HILL: Pairwise maintains cooperation better")
        elif np_tft_avg - pw_tft_avg > 0.1:
            print(f"    → Unexpected: N-Person performs better")
        else:
            print(f"    → Similar performance in both modes")
        
        # Average scores
        pw_tft_scores = [score for name, score in pw_summary['average_scores_per_round'].items() if 'TFT' in name]
        np_tft_scores = [score for name, score in np_summary['average_scores_per_round'].items() if 'TFT' in name]
        
        pw_score_avg = np.mean(pw_tft_scores) if pw_tft_scores else 0
        np_score_avg = np.mean(np_tft_scores) if np_tft_scores else 0
        
        print(f"  TFT Average Scores:")
        print(f"    Pairwise: {pw_score_avg:.2f}")
        print(f"    N-Person: {np_score_avg:.2f}")
        print(f"    Difference: {pw_score_avg - np_score_avg:+.2f}")
        
        # Check for tragic valley in N-Person with defector
        if 'AllD' in scenario_name and np_tft_avg < 0.2:
            print(f"    → TRAGIC VALLEY DETECTED: N-Person TFT agents collapsed to defection")


def main():
    """Main analysis function."""
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Find the most recent results directory
        results_dirs = glob.glob("3-Person_Tragic_vs_Reciprocity/results/chris_huyck_*")
        if not results_dirs:
            print("No results directories found!")
            print("Please run chris_huyck_experiments.py first.")
            return
        
        # Sort by timestamp and get the most recent
        results_dirs.sort()
        results_dir = results_dirs[-1]
    
    print(f"Analyzing results from: {results_dir}")
    
    # Create plots
    create_comparison_plots(results_dir)
    
    # Analyze findings
    analyze_reciprocity_vs_tragedy(results_dir)
    
    print(f"\nAnalysis complete! Plots saved to: {results_dir}")


if __name__ == "__main__":
    main()
