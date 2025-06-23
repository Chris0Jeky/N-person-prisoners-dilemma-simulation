#!/usr/bin/env python3
"""
Generate simple cooperation data for plotting: 2 QL vs 1 TFT
Creates 4 lines: QL and TFT cooperation in both pairwise and neighborhood modes
"""

import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
sys.path.append('.')

from final_agents_v2 import VanillaQLearner, StaticAgent
from final_simulation import run_pairwise_tournament, run_nperson_simulation

# Simulation parameters
NUM_ROUNDS = 10000
NUM_RUNS = 30

# Single discount factor configuration (0.95 as mentioned for chapter 4)
QL_CONFIG = {
    'lr': 0.08,
    'df': 0.95,  # Using 0.95 as mentioned for chapter 4
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

def run_experiment():
    """Run the experiment and collect cooperation data"""
    
    # Storage for all runs
    pairwise_ql_coop = []
    pairwise_tft_coop = []
    neighbor_ql_coop = []
    neighbor_tft_coop = []
    
    print(f"Running {NUM_RUNS} simulations...")
    
    for run in range(NUM_RUNS):
        if run % 5 == 0:
            print(f"  Run {run+1}/{NUM_RUNS}")
        
        # Create fresh agents for each run
        agents = [
            VanillaQLearner("QL1", QL_CONFIG),
            VanillaQLearner("QL2", QL_CONFIG),
            StaticAgent("TFT", "TFT", 0.0)
        ]
        
        # Run pairwise
        p_results = run_pairwise_tournament(agents, NUM_ROUNDS)
        
        # Average QL cooperation (both QL agents)
        ql1_p_coop = np.array(p_results['QL1']['coop_rate'])
        ql2_p_coop = np.array(p_results['QL2']['coop_rate'])
        avg_ql_p_coop = (ql1_p_coop + ql2_p_coop) / 2
        
        pairwise_ql_coop.append(avg_ql_p_coop)
        pairwise_tft_coop.append(p_results['TFT']['coop_rate'])
        
        # Create fresh agents for neighborhood
        agents = [
            VanillaQLearner("QL1", QL_CONFIG),
            VanillaQLearner("QL2", QL_CONFIG),
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

def save_to_csv(p_ql, p_tft, n_ql, n_tft):
    """Save the cooperation data to CSV"""
    
    # Save detailed data
    with open('cooperation_plot_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Round', 'Pairwise_QL', 'Pairwise_TFT', 'Neighborhood_QL', 'Neighborhood_TFT'])
        
        for i in range(len(p_ql)):
            writer.writerow([i, p_ql[i], p_tft[i], n_ql[i], n_tft[i]])
    
    # Save summary
    with open('cooperation_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Mode', 'Agent', 'Average_Cooperation'])
        writer.writerow(['Pairwise', 'QL_agents', np.mean(p_ql)])
        writer.writerow(['Pairwise', 'TFT', np.mean(p_tft)])
        writer.writerow(['Neighborhood', 'QL_agents', np.mean(n_ql)])
        writer.writerow(['Neighborhood', 'TFT', np.mean(n_tft)])
    
    print("\nSummary:")
    print(f"Pairwise - QL agents: {np.mean(p_ql):.3f}")
    print(f"Pairwise - TFT: {np.mean(p_tft):.3f}")
    print(f"Neighborhood - QL agents: {np.mean(n_ql):.3f}")
    print(f"Neighborhood - TFT: {np.mean(n_tft):.3f}")

def main():
    print("=" * 60)
    print("2 QL vs 1 TFT Cooperation Data Generator")
    print(f"Discount Factor: 0.95")
    print(f"Rounds: {NUM_ROUNDS}, Runs: {NUM_RUNS}")
    print("=" * 60)
    
    # Run experiment
    p_ql, p_tft, n_ql, n_tft = run_experiment()
    
    # Save results
    save_to_csv(p_ql, p_tft, n_ql, n_tft)
    
    # Create plot
    create_plot(p_ql, p_tft, n_ql, n_tft)
    
    print("\nFiles created:")
    print("- cooperation_plot_data.csv (4 columns: round number + 4 cooperation lines)")
    print("- cooperation_summary.csv (average cooperation rates)")
    print("- cooperation_plot.png (visual plot)")
    print("- cooperation_plot.pdf (publication-ready)")
    
    print("\nThe cooperation_plot_data.csv file contains exactly 4 lines of data:")
    print("1. Pairwise_QL: Average cooperation of both QL agents in pairwise mode")
    print("2. Pairwise_TFT: TFT cooperation in pairwise mode")
    print("3. Neighborhood_QL: Average cooperation of both QL agents in neighborhood mode")
    print("4. Neighborhood_TFT: TFT cooperation in neighborhood mode")

def create_plot(p_ql, p_tft, n_ql, n_tft):
    """Create the cooperation plot with 4 lines"""
    
    plt.figure(figsize=(12, 8))
    
    # Plot the 4 lines
    rounds = np.arange(len(p_ql))
    
    # Pairwise lines
    plt.plot(rounds, p_ql, label='Pairwise: QL agents', color='#1f77b4', linewidth=2)
    plt.plot(rounds, p_tft, label='Pairwise: TFT', color='#ff7f0e', linewidth=2)
    
    # Neighborhood lines
    plt.plot(rounds, n_ql, label='Neighborhood: QL agents', color='#2ca02c', linewidth=2)
    plt.plot(rounds, n_tft, label='Neighborhood: TFT', color='#d62728', linewidth=2)
    
    # Formatting
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Cooperation Rate', fontsize=12)
    plt.title('2 QL vs 1 TFT: Cooperation in Pairwise and Neighborhood Modes\n(Discount Factor = 0.95)', fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('cooperation_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig('cooperation_plot.pdf', bbox_inches='tight')  # Also save as PDF for papers
    plt.close()
    
    print("\nPlot saved as:")
    print("- cooperation_plot.png")
    print("- cooperation_plot.pdf")

if __name__ == "__main__":
    main()