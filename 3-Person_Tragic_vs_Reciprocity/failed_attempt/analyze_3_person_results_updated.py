"""
Analysis script for 3-Person Reciprocity Hill vs Tragic Valley Results (Updated)
==============================================================================
This script analyzes and visualizes the results from the updated experiments,
with special focus on pairwise interaction breakdowns.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List
import seaborn as sns


def load_results(filepath: str) -> Dict:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_cooperation_evolution(results: Dict, title: str, save_path: str = None):
    """Plot cooperation rate evolution over rounds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract cooperation rates per episode
    episodes = results['episode_data']
    num_episodes = len(episodes)
    rounds_per_episode = len(episodes[0]['cooperation_rates']) if episodes else 0
    
    # Calculate average cooperation rate per round across episodes
    avg_cooperation_per_round = []
    for round_idx in range(rounds_per_episode):
        round_cooperations = [ep['cooperation_rates'][round_idx] for ep in episodes]
        avg_cooperation_per_round.append(np.mean(round_cooperations))
    
    # Plot overall cooperation
    rounds = list(range(1, rounds_per_episode + 1))
    ax.plot(rounds, avg_cooperation_per_round, 'k-', linewidth=2, label='Overall Average')
    
    # Plot individual agent cooperation rates if available
    agent_names = list(results['summary']['cooperation_rates'].keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(agent_names)))
    
    for agent_idx, agent_name in enumerate(agent_names):
        agent_cooperation_by_round = []
        
        for round_idx in range(rounds_per_episode):
            round_cooperations = []
            for ep in episodes:
                if agent_name in ep['moves']:
                    move = ep['moves'][agent_name][round_idx]
                    round_cooperations.append(1 if move == 'C' else 0)
            
            if round_cooperations:
                agent_cooperation_by_round.append(np.mean(round_cooperations))
        
        if agent_cooperation_by_round:
            ax.plot(rounds[:len(agent_cooperation_by_round)], agent_cooperation_by_round, 
                   color=colors[agent_idx], linewidth=1.5, alpha=0.7, label=agent_name)
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Cooperation Rate')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_pairwise_breakdown(results: Dict, title: str, save_path: str = None):
    """Plot breakdown of cooperation in pairwise interactions."""
    if 'pairwise_analysis' not in results or not results['pairwise_analysis']:
        print(f"No pairwise analysis data available for {title}")
        return None
    
    analysis = results['pairwise_analysis']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Evolution of TFT-TFT vs TFT-AllD cooperation
    rounds = list(range(1, len(analysis['round_by_round_tft_tft']) + 1))
    
    if analysis['round_by_round_tft_tft']:
        ax1.plot(rounds, analysis['round_by_round_tft_tft'], 'g-', linewidth=2, 
                label='TFT vs TFT', marker='o', markersize=4)
    
    if analysis['round_by_round_tft_alld']:
        rounds_alld = list(range(1, len(analysis['round_by_round_tft_alld']) + 1))
        ax1.plot(rounds_alld, analysis['round_by_round_tft_alld'], 'r-', linewidth=2, 
                label='TFT vs AllD', marker='s', markersize=4)
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Cooperation Rate')
    ax1.set_title(f'{title} - Pairwise Interaction Types')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Plot 2: Bar chart of average cooperation rates
    interaction_types = []
    cooperation_rates = []
    
    if 'tft_tft_cooperation' in analysis:
        interaction_types.append('TFT vs TFT')
        cooperation_rates.append(analysis['tft_tft_cooperation'])
    
    if 'tft_alld_cooperation' in analysis:
        interaction_types.append('TFT vs AllD')
        cooperation_rates.append(analysis['tft_alld_cooperation'])
    
    if interaction_types:
        bars = ax2.bar(interaction_types, cooperation_rates, color=['green', 'red'])
        ax2.set_ylabel('Average Cooperation Rate')
        ax2.set_title(f'{title} - Average Cooperation by Interaction Type')
        ax2.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, rate in zip(bars, cooperation_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{rate:.2%}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_comparison(pairwise_results: Dict, nperson_results: Dict, 
                   scenario_name: str, save_path: str = None):
    """Plot comparison between pairwise and N-person modes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract cooperation evolution
    pw_episodes = pairwise_results['episode_data']
    np_episodes = nperson_results['episode_data']
    
    rounds_per_episode = len(pw_episodes[0]['cooperation_rates']) if pw_episodes else 0
    rounds = list(range(1, rounds_per_episode + 1))
    
    # Calculate average cooperation per round
    pw_avg_coop = []
    np_avg_coop = []
    
    for round_idx in range(rounds_per_episode):
        pw_round_coop = [ep['cooperation_rates'][round_idx] for ep in pw_episodes]
        np_round_coop = [ep['cooperation_rates'][round_idx] for ep in np_episodes]
        
        pw_avg_coop.append(np.mean(pw_round_coop))
        np_avg_coop.append(np.mean(np_round_coop))
    
    # Plot 1: Overall cooperation comparison
    ax1.plot(rounds, pw_avg_coop, 'b-', linewidth=2, label='Pairwise', marker='o', markersize=4)
    ax1.plot(rounds, np_avg_coop, 'r-', linewidth=2, label='N-Person', marker='s', markersize=4)
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Average Cooperation Rate')
    ax1.set_title(f'{scenario_name} - Mode Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Plot 2: Final cooperation rates by agent type
    pw_summary = pairwise_results['summary']
    np_summary = nperson_results['summary']
    
    # Group by agent type
    agent_types = set()
    for agent_name in pw_summary['cooperation_rates']:
        agent_type = agent_name.split('_')[0] if '_' in agent_name else agent_name.rstrip('0123456789')
        agent_types.add(agent_type)
    
    agent_types = sorted(list(agent_types))
    
    # Calculate average cooperation by type
    pw_coop_by_type = {}
    np_coop_by_type = {}
    
    for agent_type in agent_types:
        pw_agents = [name for name in pw_summary['cooperation_rates'] if agent_type in name]
        np_agents = [name for name in np_summary['cooperation_rates'] if agent_type in name]
        
        if pw_agents:
            pw_coop_by_type[agent_type] = np.mean([pw_summary['cooperation_rates'][name] for name in pw_agents])
        if np_agents:
            np_coop_by_type[agent_type] = np.mean([np_summary['cooperation_rates'][name] for name in np_agents])
    
    # Create grouped bar chart
    x = np.arange(len(agent_types))
    width = 0.35
    
    pw_values = [pw_coop_by_type.get(at, 0) for at in agent_types]
    np_values = [np_coop_by_type.get(at, 0) for at in agent_types]
    
    bars1 = ax2.bar(x - width/2, pw_values, width, label='Pairwise', color='blue', alpha=0.7)
    bars2 = ax2.bar(x + width/2, np_values, width, label='N-Person', color='red', alpha=0.7)
    
    ax2.set_xlabel('Agent Type')
    ax2.set_ylabel('Final Cooperation Rate')
    ax2.set_title(f'{scenario_name} - Final Cooperation by Agent Type')
    ax2.set_xticks(x)
    ax2.set_xticklabels(agent_types)
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def analyze_experiment_set(results_dir: str):
    """Analyze a complete set of experiment results."""
    # Load all results
    all_results_path = os.path.join(results_dir, 'all_results_combined.json')
    if not os.path.exists(all_results_path):
        print(f"Results file not found: {all_results_path}")
        return
    
    all_results = load_results(all_results_path)
    
    # Create plots directory
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Analyze each scenario set
    scenario_sets = {
        'A1': ('2TFT_1AllD_NoExpl', '2 TFT + 1 AllD (No Exploration)'),
        'A2': ('2TFT_1AllD_10Expl', '2 TFT + 1 AllD (10% Exploration)'),
        'B1': ('3TFT_NoExpl', '3 TFT (No Exploration)'),
        'B2': ('3TFT_10Expl', '3 TFT (10% Exploration)')
    }
    
    for scenario_key, (scenario_name, display_name) in scenario_sets.items():
        pw_key = f'{scenario_key}_Pairwise'
        np_key = f'{scenario_key}_NPerson'
        
        if pw_key in all_results and np_key in all_results:
            pw_results = all_results[pw_key]
            np_results = all_results[np_key]
            
            # Plot cooperation evolution for each mode
            plot_cooperation_evolution(
                pw_results, 
                f'{display_name} - Pairwise Mode',
                os.path.join(plots_dir, f'{scenario_name}_pairwise_evolution.png')
            )
            
            plot_cooperation_evolution(
                np_results, 
                f'{display_name} - N-Person Mode',
                os.path.join(plots_dir, f'{scenario_name}_nperson_evolution.png')
            )
            
            # Plot pairwise breakdown if available
            if 'A' in scenario_key:  # Only for scenarios with AllD
                plot_pairwise_breakdown(
                    pw_results,
                    display_name,
                    os.path.join(plots_dir, f'{scenario_name}_pairwise_breakdown.png')
                )
            
            # Plot comparison
            plot_comparison(
                pw_results, 
                np_results,
                display_name,
                os.path.join(plots_dir, f'{scenario_name}_comparison.png')
            )
            
            plt.close('all')  # Close all figures to free memory
    
    # Create summary plot
    create_summary_plot(all_results, plots_dir)
    
    print(f"Analysis complete! Plots saved to: {plots_dir}")


def create_summary_plot(all_results: Dict, plots_dir: str):
    """Create a summary plot showing key differences."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    scenarios = ['A1', 'A2', 'B1', 'B2']
    scenario_names = ['2TFT+1AllD\n(No Expl)', '2TFT+1AllD\n(10% Expl)', 
                     '3TFT\n(No Expl)', '3TFT\n(10% Expl)']
    
    # Extract final cooperation rates
    pw_final_coop = []
    np_final_coop = []
    pw_tft_tft_coop = []
    pw_tft_alld_coop = []
    
    for scenario in scenarios:
        pw_results = all_results.get(f'{scenario}_Pairwise', {})
        np_results = all_results.get(f'{scenario}_NPerson', {})
        
        pw_final_coop.append(pw_results.get('summary', {}).get('overall_cooperation_rate', 0))
        np_final_coop.append(np_results.get('summary', {}).get('overall_cooperation_rate', 0))
        
        # Get pairwise breakdown if available
        if 'pairwise_analysis' in pw_results:
            analysis = pw_results['pairwise_analysis']
            pw_tft_tft_coop.append(analysis.get('tft_tft_cooperation', 0))
            pw_tft_alld_coop.append(analysis.get('tft_alld_cooperation', 0))
        else:
            pw_tft_tft_coop.append(0)
            pw_tft_alld_coop.append(0)
    
    # Plot 1: Overall cooperation comparison
    x = np.arange(len(scenarios))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, pw_final_coop, width, label='Pairwise', color='blue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, np_final_coop, width, label='N-Person', color='red', alpha=0.7)
    
    ax1.set_xlabel('Scenario')
    ax1.set_ylabel('Final Cooperation Rate')
    ax1.set_title('Overall Cooperation: Pairwise vs N-Person')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_names)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Pairwise interaction breakdown (for A scenarios)
    a_scenarios = ['A1', 'A2']
    a_names = ['No Exploration', '10% Exploration']
    a_indices = [0, 1]
    
    x2 = np.arange(len(a_scenarios))
    
    bars3 = ax2.bar(x2 - width/2, [pw_tft_tft_coop[i] for i in a_indices], 
                    width, label='TFT vs TFT', color='green', alpha=0.7)
    bars4 = ax2.bar(x2 + width/2, [pw_tft_alld_coop[i] for i in a_indices], 
                    width, label='TFT vs AllD', color='orange', alpha=0.7)
    
    ax2.set_xlabel('Scenario')
    ax2.set_ylabel('Cooperation Rate')
    ax2.set_title('Pairwise Mode: Cooperation by Interaction Type')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(a_names)
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    
    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Cooperation difference (Pairwise - N-Person)
    cooperation_diff = [pw - np for pw, np in zip(pw_final_coop, np_final_coop)]
    colors = ['green' if diff > 0 else 'red' for diff in cooperation_diff]
    
    bars5 = ax3.bar(x, cooperation_diff, color=colors, alpha=0.7)
    ax3.set_xlabel('Scenario')
    ax3.set_ylabel('Cooperation Difference (Pairwise - N-Person)')
    ax3.set_title('Reciprocity Hill Effect: Cooperation Advantage in Pairwise Mode')
    ax3.set_xticks(x)
    ax3.set_xticklabels(scenario_names)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_ylim(-0.5, 0.8)
    
    # Add value labels
    for bar in bars5:
        height = bar.get_height()
        y_pos = height + 0.02 if height > 0 else height - 0.02
        va = 'bottom' if height > 0 else 'top'
        ax3.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{height:.2f}', ha='center', va=va, fontsize=9)
    
    # Plot 4: Text summary
    ax4.axis('off')
    summary_text = """
Key Findings:
    
1. Reciprocity Hill (Pairwise Mode):
   • TFT agents maintain cooperation with each other
   • TFT agents learn to defect against Always Defect
   • Overall cooperation stabilizes around 50% in 2TFT+1AllD
   
2. Tragedy Valley (N-Person Mode):
   • Any defection causes all TFT agents to defect
   • Cooperation collapses to near 0% with defector present
   • Even with exploration, recovery is difficult
   
3. Exploration Effects:
   • 10% exploration reduces cooperation in both modes
   • Pairwise mode is more resilient to noise
   • N-Person mode shows catastrophic breakdown
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'summary_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Default to most recent results
        results_dirs = [d for d in os.listdir('results') if d.startswith('chris_huyck_updated_')]
        if results_dirs:
            results_dir = os.path.join('results', sorted(results_dirs)[-1])
        else:
            print("No results directory found. Please run experiments first.")
            sys.exit(1)
    
    print(f"Analyzing results from: {results_dir}")
    analyze_experiment_set(results_dir)
