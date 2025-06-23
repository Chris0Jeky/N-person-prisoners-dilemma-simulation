#!/usr/bin/env python3
"""
Generate cooperation plots for 2 QL vs 1 TFT with different discount factors:
1. DF = 0.95 for both modes
2. DF = 0.4 for both modes
3. DF = 0.4 for neighborhood, DF = 0.95 for pairwise (mixed)
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
sys.path.append('.')

from final_agents_v2 import VanillaQLearner, StaticAgent
from final_simulation import run_pairwise_tournament, run_nperson_simulation

# Simulation parameters
NUM_ROUNDS = 100000
NUM_RUNS = 10

# Q-learner configurations
QL_CONFIG_095 = {
    'lr': 0.08,
    'df': 0.95,
    'eps': 0.1,
}

QL_CONFIG_02 = {
    'lr': 0.08,
    'df': 0.2,
    'eps': 0.1,
}

def smooth_data(data, window_size=100):
    """Apply rolling average"""
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        smoothed.append(np.mean(data[start:end]))
    return smoothed

def run_single_df_experiment(config, experiment_name):
    """Run experiment with same DF for both modes"""
    pairwise_ql_coop = []
    pairwise_tft_coop = []
    neighbor_ql_coop = []
    neighbor_tft_coop = []
    
    print(f"\nRunning {experiment_name}...")
    
    for run in range(NUM_RUNS):
        if run % 10 == 0:
            print(f"  Run {run+1}/{NUM_RUNS}")
        
        # Create agents for pairwise
        agents = [
            VanillaQLearner("QL1", config),
            VanillaQLearner("QL2", config),
            StaticAgent("TFT", "TFT", 0.0)
        ]
        
        # Run pairwise
        p_results = run_pairwise_tournament(agents, NUM_ROUNDS)
        
        # Average QL cooperation
        ql1_p_coop = np.array(p_results['QL1']['coop_rate'])
        ql2_p_coop = np.array(p_results['QL2']['coop_rate'])
        avg_ql_p_coop = (ql1_p_coop + ql2_p_coop) / 2
        
        pairwise_ql_coop.append(avg_ql_p_coop)
        pairwise_tft_coop.append(p_results['TFT']['coop_rate'])
        
        # Create fresh agents for neighborhood
        agents = [
            VanillaQLearner("QL1", config),
            VanillaQLearner("QL2", config),
            StaticAgent("TFT", "TFT", 0.0)
        ]
        
        # Run neighborhood
        n_results = run_nperson_simulation(agents, NUM_ROUNDS)
        
        # Average QL cooperation
        ql1_n_coop = np.array(n_results['QL1']['coop_rate'])
        ql2_n_coop = np.array(n_results['QL2']['coop_rate'])
        avg_ql_n_coop = (ql1_n_coop + ql2_n_coop) / 2
        
        neighbor_ql_coop.append(avg_ql_n_coop)
        neighbor_tft_coop.append(n_results['TFT']['coop_rate'])
    
    # Average across all runs
    avg_pairwise_ql = np.mean(pairwise_ql_coop, axis=0)
    avg_pairwise_tft = np.mean(pairwise_tft_coop, axis=0)
    avg_neighbor_ql = np.mean(neighbor_ql_coop, axis=0)
    avg_neighbor_tft = np.mean(neighbor_tft_coop, axis=0)
    
    # Smooth the data
    smooth_p_ql = smooth_data(avg_pairwise_ql)
    smooth_p_tft = smooth_data(avg_pairwise_tft)
    smooth_n_ql = smooth_data(avg_neighbor_ql)
    smooth_n_tft = smooth_data(avg_neighbor_tft)
    
    return smooth_p_ql, smooth_p_tft, smooth_n_ql, smooth_n_tft

def run_mixed_df_experiment():
    """Run experiment with different DFs for each mode"""
    pairwise_ql_coop = []
    pairwise_tft_coop = []
    neighbor_ql_coop = []
    neighbor_tft_coop = []
    
    print(f"\nRunning Mixed DF experiment (Pairwise=0.95, Neighborhood=0.2)...")
    
    for run in range(NUM_RUNS):
        if run % 10 == 0:
            print(f"  Run {run+1}/{NUM_RUNS}")
        
        # Create agents for pairwise (DF=0.95)
        agents_pairwise = [
            VanillaQLearner("QL1", QL_CONFIG_095),
            VanillaQLearner("QL2", QL_CONFIG_095),
            StaticAgent("TFT", "TFT", 0.0)
        ]
        
        # Create agents for neighborhood (DF=0.2)
        agents_neighbor = [
            VanillaQLearner("QL1", QL_CONFIG_02),
            VanillaQLearner("QL2", QL_CONFIG_02),
            StaticAgent("TFT", "TFT", 0.0)
        ]
        
        # Run pairwise with DF=0.95
        p_results = run_pairwise_tournament(agents_pairwise, NUM_ROUNDS)
        
        # Average QL cooperation
        ql1_p_coop = np.array(p_results['QL1']['coop_rate'])
        ql2_p_coop = np.array(p_results['QL2']['coop_rate'])
        avg_ql_p_coop = (ql1_p_coop + ql2_p_coop) / 2
        
        pairwise_ql_coop.append(avg_ql_p_coop)
        pairwise_tft_coop.append(p_results['TFT']['coop_rate'])
        
        # Run neighborhood with DF=0.2
        n_results = run_nperson_simulation(agents_neighbor, NUM_ROUNDS)
        
        # Average QL cooperation
        ql1_n_coop = np.array(n_results['QL1']['coop_rate'])
        ql2_n_coop = np.array(n_results['QL2']['coop_rate'])
        avg_ql_n_coop = (ql1_n_coop + ql2_n_coop) / 2
        
        neighbor_ql_coop.append(avg_ql_n_coop)
        neighbor_tft_coop.append(n_results['TFT']['coop_rate'])
    
    # Average across all runs
    avg_pairwise_ql = np.mean(pairwise_ql_coop, axis=0)
    avg_pairwise_tft = np.mean(pairwise_tft_coop, axis=0)
    avg_neighbor_ql = np.mean(neighbor_ql_coop, axis=0)
    avg_neighbor_tft = np.mean(neighbor_tft_coop, axis=0)
    
    # Smooth the data
    smooth_p_ql = smooth_data(avg_pairwise_ql)
    smooth_p_tft = smooth_data(avg_pairwise_tft)
    smooth_n_ql = smooth_data(avg_neighbor_ql)
    smooth_n_tft = smooth_data(avg_neighbor_tft)
    
    return smooth_p_ql, smooth_p_tft, smooth_n_ql, smooth_n_tft

def create_individual_plot(p_ql, p_tft, n_ql, n_tft, title, filename):
    """Create a single plot for one scenario"""
    plt.figure(figsize=(12, 8))
    
    rounds = np.arange(len(p_ql))
    
    # Plot the 4 lines
    plt.plot(rounds, p_ql, label='Pairwise: QL agents', color='#1f77b4', linewidth=2)
    plt.plot(rounds, p_tft, label='Pairwise: TFT', color='#ff7f0e', linewidth=2)
    plt.plot(rounds, n_ql, label='Neighborhood: QL agents', color='#2ca02c', linewidth=2)
    plt.plot(rounds, n_tft, label='Neighborhood: TFT', color='#d62728', linewidth=2)
    
    # Formatting
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Cooperation Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{filename}.pdf', bbox_inches='tight')
    plt.close()

def create_comparison_plot(all_results):
    """Create a comprehensive comparison plot"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('2 QL vs 1 TFT: Discount Factor Comparison', fontsize=16)
    
    scenarios = ['DF=0.95 (Both)', 'DF=0.2 (Both)', 'Mixed (P=0.95, N=0.2)']
    
    for idx, (scenario_name, (p_ql, p_tft, n_ql, n_tft)) in enumerate(all_results.items()):
        rounds = np.arange(len(p_ql))
        
        # Top row: QL agents
        ax_top = axes[0, idx]
        ax_top.plot(rounds, p_ql, label='Pairwise QL', color='#1f77b4', linewidth=2)
        ax_top.plot(rounds, n_ql, label='Neighborhood QL', color='#2ca02c', linewidth=2)
        ax_top.set_title(f'{scenarios[idx]}\nQL Agents Cooperation', fontsize=12)
        ax_top.set_ylabel('Cooperation Rate', fontsize=10)
        ax_top.set_ylim(-0.05, 1.05)
        ax_top.legend(fontsize=9)
        ax_top.grid(True, alpha=0.3)
        
        # Bottom row: TFT
        ax_bottom = axes[1, idx]
        ax_bottom.plot(rounds, p_tft, label='Pairwise TFT', color='#ff7f0e', linewidth=2)
        ax_bottom.plot(rounds, n_tft, label='Neighborhood TFT', color='#d62728', linewidth=2)
        ax_bottom.set_title(f'TFT Cooperation', fontsize=12)
        ax_bottom.set_xlabel('Round', fontsize=10)
        ax_bottom.set_ylabel('Cooperation Rate', fontsize=10)
        ax_bottom.set_ylim(-0.05, 1.05)
        ax_bottom.legend(fontsize=9)
        ax_bottom.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('df_comparison_all.png', dpi=300, bbox_inches='tight')
    plt.savefig('df_comparison_all.pdf', bbox_inches='tight')
    plt.close()

def save_summary_csv(all_results):
    """Save summary statistics for all scenarios"""
    with open('df_comparison_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Scenario', 'Mode', 'Agent', 'Average_Cooperation', 'Final_Cooperation'])
        
        for scenario_name, (p_ql, p_tft, n_ql, n_tft) in all_results.items():
            # Pairwise
            writer.writerow([scenario_name, 'Pairwise', 'QL_agents', np.mean(p_ql), p_ql[-1]])
            writer.writerow([scenario_name, 'Pairwise', 'TFT', np.mean(p_tft), p_tft[-1]])
            # Neighborhood
            writer.writerow([scenario_name, 'Neighborhood', 'QL_agents', np.mean(n_ql), n_ql[-1]])
            writer.writerow([scenario_name, 'Neighborhood', 'TFT', np.mean(n_tft), n_tft[-1]])

def main():
    print("=" * 70)
    print("DISCOUNT FACTOR COMPARISON EXPERIMENT")
    print("2 QL vs 1 TFT with different discount factors")
    print("=" * 70)
    
    all_results = {}
    
    # Scenario 1: DF = 0.95 for both
    p_ql_095, p_tft_095, n_ql_095, n_tft_095 = run_single_df_experiment(
        QL_CONFIG_095, "DF=0.95 for both modes"
    )
    all_results['df_095'] = (p_ql_095, p_tft_095, n_ql_095, n_tft_095)
    create_individual_plot(
        p_ql_095, p_tft_095, n_ql_095, n_tft_095,
        '2 QL vs 1 TFT: Discount Factor = 0.95 (Both Modes)',
        'cooperation_df_095'
    )
    
    # Scenario 2: DF = 0.2 for both
    p_ql_02, p_tft_02, n_ql_02, n_tft_02 = run_single_df_experiment(
        QL_CONFIG_02, "DF=0.2 for both modes"
    )
    all_results['df_02'] = (p_ql_02, p_tft_02, n_ql_02, n_tft_02)
    create_individual_plot(
        p_ql_02, p_tft_02, n_ql_02, n_tft_02,
        '2 QL vs 1 TFT: Discount Factor = 0.2 (Both Modes)',
        'cooperation_df_02'
    )
    
    # Scenario 3: Mixed DF
    p_ql_mixed, p_tft_mixed, n_ql_mixed, n_tft_mixed = run_mixed_df_experiment()
    all_results['df_mixed'] = (p_ql_mixed, p_tft_mixed, n_ql_mixed, n_tft_mixed)
    create_individual_plot(
        p_ql_mixed, p_tft_mixed, n_ql_mixed, n_tft_mixed,
        '2 QL vs 1 TFT: Mixed DF (Pairwise=0.95, Neighborhood=0.2)',
        'cooperation_df_mixed'
    )
    
    # Create comparison plot
    create_comparison_plot(all_results)
    
    # Save summary
    save_summary_csv(all_results)
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY:")
    print("=" * 70)
    
    for scenario_name, (p_ql, p_tft, n_ql, n_tft) in all_results.items():
        if scenario_name == 'df_095':
            print("\nDF = 0.95 (Both modes):")
        elif scenario_name == 'df_02':
            print("\nDF = 0.2 (Both modes):")
        else:
            print("\nMixed DF (Pairwise=0.95, Neighborhood=0.2):")
        
        print(f"  Pairwise - QL agents: {np.mean(p_ql):.3f}")
        print(f"  Pairwise - TFT: {np.mean(p_tft):.3f}")
        print(f"  Neighborhood - QL agents: {np.mean(n_ql):.3f}")
        print(f"  Neighborhood - TFT: {np.mean(n_tft):.3f}")
    
    print("\n" + "=" * 70)
    print("FILES GENERATED:")
    print("=" * 70)
    print("Individual plots:")
    print("  - cooperation_df_095.png/pdf (DF=0.95 for both modes)")
    print("  - cooperation_df_02.png/pdf (DF=0.2 for both modes)")
    print("  - cooperation_df_mixed.png/pdf (Mixed: P=0.95, N=0.2)")
    print("\nComparison plot:")
    print("  - df_comparison_all.png/pdf (All scenarios side by side)")
    print("\nSummary data:")
    print("  - df_comparison_summary.csv")
    print("\nKey observation: Mixed DF allows optimization for each mode!")

if __name__ == "__main__":
    main()