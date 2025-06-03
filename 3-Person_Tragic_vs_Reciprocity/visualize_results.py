"""
Simple Visualization Script for 3-Person Prisoner's Dilemma Results
===================================================================
This script creates visualizations from the JSON result files.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import glob


def load_single_result_file(filepath: str) -> Dict:
    """Load a single JSON result file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_cooperation_over_rounds(data: Dict) -> Tuple[List[float], List[float]]:
    """Extract TFT agents' cooperation rates over rounds from the data."""
    
    # Check if we have round_data
    if 'round_data' not in data:
        print(f"No round_data found in file")
        return [], []
    
    round_data = data['round_data']
    if not round_data:
        return [], []
    
    # Get number of rounds per episode
    if 'summary' in data:
        rounds_per_episode = data['summary'].get('rounds_per_episode', 100)
    else:
        # Infer from data
        rounds_per_episode = max([d['round'] for d in round_data]) + 1
    
    # Initialize arrays to store cooperation data
    tft_cooperation_by_round = [[] for _ in range(rounds_per_episode)]
    
    # Process each round
    for data_point in round_data:
        round_num = data_point['round']
        actions = data_point['actions']
        
        # Get TFT agents' actions (agents with TFT in their name)
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


def create_cooperation_plot(results_dir: str):
    """Create comparison plots for cooperation rates."""
    
    # Define the files we're looking for
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
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    fig.suptitle('3-Person IPD: TFT Cooperation Rates - Pairwise vs N-Person', fontsize=16)
    
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        
        # Load pairwise data
        pw_path = os.path.join(results_dir, scenario['pairwise'])
        np_path = os.path.join(results_dir, scenario['nperson'])
        
        if os.path.exists(pw_path) and os.path.exists(np_path):
            pw_data = load_single_result_file(pw_path)
            np_data = load_single_result_file(np_path)
            
            # Extract cooperation rates
            pw_coop, pw_std = extract_cooperation_over_rounds(pw_data)
            np_coop, np_std = extract_cooperation_over_rounds(np_data)
            
            if pw_coop and np_coop:
                rounds = list(range(len(pw_coop)))
                
                # Plot the data
                ax.plot(rounds, pw_coop, 'b-', label='Pairwise', linewidth=2)
                ax.plot(rounds, np_coop, 'r-', label='N-Person', linewidth=2)
                
                # Add shaded error regions
                ax.fill_between(rounds, 
                               np.array(pw_coop) - np.array(pw_std),
                               np.array(pw_coop) + np.array(pw_std),
                               alpha=0.2, color='blue')
                ax.fill_between(rounds, 
                               np.array(np_coop) - np.array(np_std),
                               np.array(np_coop) + np.array(np_std),
                               alpha=0.2, color='red')
                
                ax.set_title(scenario['name'])
                ax.set_xlabel('Round')
                ax.set_ylabel('TFT Cooperation Rate')
                ax.set_ylim(-0.05, 1.05)
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Add text showing final cooperation rates
                if len(pw_coop) > 10:
                    final_pw = np.mean(pw_coop[-10:])
                    final_np = np.mean(np_coop[-10:])
                    ax.text(0.02, 0.98, f'Final rates:\nPW: {final_pw:.2%}\nNP: {final_np:.2%}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'Files not found', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    output_path = os.path.join(results_dir, 'cooperation_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved cooperation comparison plot to: {output_path}")
    plt.close()


def create_summary_statistics(results_dir: str):
    """Create a summary of key findings."""
    
    print("\n" + "="*80)
    print("SUMMARY: TRAGIC VALLEY VS RECIPROCITY HILL")
    print("="*80)
    
    scenarios = [
        ('2TFT + 1AllD (No Exploration)', 'results_2TFT_1AllD_NoExpl_Pairwise.json', 'results_2TFT_1AllD_NoExpl_NPerson.json'),
        ('2TFT + 1AllD (10% Exploration)', 'results_2TFT_1AllD_10Expl_Pairwise.json', 'results_2TFT_1AllD_10Expl_NPerson.json'),
        ('3TFT (No Exploration)', 'results_3TFT_NoExpl_Pairwise.json', 'results_3TFT_NoExpl_NPerson.json'),
        ('3TFT (10% Exploration)', 'results_3TFT_10Expl_Pairwise.json', 'results_3TFT_10Expl_NPerson.json')
    ]
    
    for scenario_name, pw_file, np_file in scenarios:
        pw_path = os.path.join(results_dir, pw_file)
        np_path = os.path.join(results_dir, np_file)
        
        if os.path.exists(pw_path) and os.path.exists(np_path):
            print(f"\n{scenario_name}:")
            
            pw_data = load_single_result_file(pw_path)
            np_data = load_single_result_file(np_path)
            
            # Get cooperation rates
            pw_coop, _ = extract_cooperation_over_rounds(pw_data)
            np_coop, _ = extract_cooperation_over_rounds(np_data)
            
            if pw_coop and np_coop:
                # Calculate average cooperation in last 20% of rounds
                last_portion = int(len(pw_coop) * 0.2)
                if last_portion > 0:
                    pw_final = np.mean(pw_coop[-last_portion:])
                    np_final = np.mean(np_coop[-last_portion:])
                else:
                    pw_final = np.mean(pw_coop)
                    np_final = np.mean(np_coop)
                
                print(f"  Final TFT Cooperation Rates (last 20% of rounds):")
                print(f"    Pairwise: {pw_final:.2%}")
                print(f"    N-Person: {np_final:.2%}")
                print(f"    Difference: {pw_final - np_final:+.2%}")
                
                # Interpretation
                if pw_final - np_final > 0.1:
                    print(f"    → RECIPROCITY HILL: Pairwise maintains significantly higher cooperation")
                elif np_final - pw_final > 0.1:
                    print(f"    → Unexpected: N-Person maintains higher cooperation")
                else:
                    print(f"    → Similar performance in both modes")
                
                # Check for tragic valley
                if 'AllD' in scenario_name and np_final < 0.2:
                    print(f"    → TRAGIC VALLEY CONFIRMED: N-Person TFT agents collapsed to defection")


def main():
    """Main function to create visualizations."""
    # Get the most recent results directory
    results_dir = "3-Person_Tragic_vs_Reciprocity/results/chris_huyck_20250603_014939"
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
    
    print(f"Analyzing results from: {results_dir}")
    
    # Create cooperation plot
    create_cooperation_plot(results_dir)
    
    # Print summary statistics
    create_summary_statistics(results_dir)
    
    print(f"\nVisualization complete!")


if __name__ == "__main__":
    main()
