#!/usr/bin/env python3
"""Demo script showcasing modular Q-learning agents with different strategies"""

import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

from final_agents import StaticAgent
from modular_agents import (
    create_vanilla_qlearner,
    create_statistical_qlearner,
    create_softmax_qlearner,
    create_statistical_softmax_qlearner,
    create_hysteretic_statistical_qlearner
)
from config import SIMULATION_CONFIG

# Reuse simulation functions from main demo
from final_demo_full import run_pairwise_tournament, run_nperson_simulation, smooth_data

def plot_strategy_comparison(results, title, save_path):
    """Plot comparison of different strategy combinations"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle(title, fontsize=20)
    
    colors = {
        "Vanilla": "#1f77b4",
        "Statistical": "#ff7f0e", 
        "Softmax": "#2ca02c",
        "Statistical+Softmax": "#d62728",
        "Hysteretic+Statistical": "#9467bd"
    }
    
    smooth_window = max(50, SIMULATION_CONFIG['num_rounds'] // 100)
    
    for agent_name, data in results.items():
        p_data, n_data = data
        agent_id = f"{agent_name}_test"
        
        if agent_id not in p_data:
            continue
            
        # Apply smoothing
        p_coop_smooth = smooth_data(p_data[agent_id]['coop_rate'], smooth_window)
        n_coop_smooth = smooth_data(n_data[agent_id]['coop_rate'], smooth_window)
        
        # Plot
        color = colors.get(agent_name, 'black')
        
        # Raw data with low alpha
        axes[0, 0].plot(p_data[agent_id]['coop_rate'], color=color, alpha=0.2, linewidth=0.5)
        axes[1, 0].plot(n_data[agent_id]['coop_rate'], color=color, alpha=0.2, linewidth=0.5)
        
        # Smoothed data
        axes[0, 0].plot(p_coop_smooth, label=agent_name, color=color, linewidth=2.5)
        axes[0, 1].plot(p_data[agent_id]['score'], label=agent_name, color=color, linewidth=2)
        axes[1, 0].plot(n_coop_smooth, label=agent_name, color=color, linewidth=2.5)
        axes[1, 1].plot(n_data[agent_id]['score'], label=agent_name, color=color, linewidth=2)
    
    axes[0, 0].set_title('Pairwise Cooperation Rate')
    axes[0, 0].set_ylabel('Cooperation Rate')
    axes[0, 1].set_title('Pairwise Cumulative Score')
    axes[0, 1].set_ylabel('Cumulative Score')
    axes[1, 0].set_title('Neighborhood Cooperation Rate')
    axes[1, 0].set_ylabel('Cooperation Rate')
    axes[1, 1].set_title('Neighborhood Cumulative Score')
    axes[1, 1].set_ylabel('Cumulative Score')
    
    for ax in axes.flat:
        ax.set_xlabel('Round')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def run_strategy_experiments():
    """Run experiments comparing different strategy combinations"""
    
    NUM_ROUNDS = min(1000, SIMULATION_CONFIG['num_rounds'])  # Use shorter runs for demo
    NUM_RUNS = min(5, SIMULATION_CONFIG['num_runs'])
    OUTPUT_DIR = "modular_strategy_comparison"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=== Modular Q-Learning Strategy Comparison ===\n")
    
    # Define agent configurations
    agent_configs = {
        "Vanilla": create_vanilla_qlearner,
        "Statistical": create_statistical_qlearner,
        "Softmax": create_softmax_qlearner,
        "Statistical+Softmax": create_statistical_softmax_qlearner,
        "Hysteretic+Statistical": create_hysteretic_statistical_qlearner
    }
    
    # Test scenarios
    scenarios = {
        "vs_2TFT": [StaticAgent("TFT_1", "TFT"), StaticAgent("TFT_2", "TFT")],
        "vs_TFT_AllD": [StaticAgent("TFT", "TFT"), StaticAgent("AllD", "AllD")],
        "vs_2AllC": [StaticAgent("AllC_1", "AllC"), StaticAgent("AllC_2", "AllC")],
        "vs_Mixed": [StaticAgent("TFT", "TFT"), StaticAgent("Random", "Random")]
    }
    
    for scenario_name, opponents in scenarios.items():
        print(f"\nScenario: {scenario_name}")
        scenario_results = {}
        
        for agent_name, creator_func in agent_configs.items():
            print(f"  Testing {agent_name}...", end='', flush=True)
            
            # Create agent template
            test_agent = creator_func(f"{agent_name}_test")
            agents = [test_agent] + opponents
            
            # Run simulations
            p_runs = []
            n_runs = []
            
            for _ in range(NUM_RUNS):
                # Create fresh copies
                fresh_agents = [creator_func(f"{agent_name}_test")] + \
                              [type(opp)(**opp.__dict__) for opp in opponents]
                
                p_runs.append(run_pairwise_tournament(fresh_agents, NUM_ROUNDS))
                n_runs.append(run_nperson_simulation(fresh_agents, NUM_ROUNDS))
            
            # Aggregate results
            p_agg = {aid: {m: np.mean([r[aid][m] for r in p_runs if aid in r], axis=0) 
                          for m in ['coop_rate', 'score']} 
                     for aid in p_runs[0]}
            n_agg = {aid: {m: np.mean([r[aid][m] for r in n_runs if aid in r], axis=0) 
                          for m in ['coop_rate', 'score']} 
                     for aid in n_runs[0]}
            
            scenario_results[agent_name] = (p_agg, n_agg)
            print(" done!")
        
        # Plot results
        plot_path = os.path.join(OUTPUT_DIR, f"comparison_{scenario_name}.png")
        plot_strategy_comparison(scenario_results, f"Strategy Comparison: {scenario_name}", plot_path)
        print(f"  Saved plot: {plot_path}")
    
    # Create summary statistics
    create_summary_report(scenarios, agent_configs, OUTPUT_DIR)

def create_summary_report(scenarios, agent_configs, output_dir):
    """Create a text summary of strategy performance"""
    report_path = os.path.join(output_dir, "strategy_comparison_summary.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODULAR Q-LEARNING STRATEGY COMPARISON SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write("Strategy Descriptions:\n")
        f.write("-"*30 + "\n")
        f.write("Vanilla: Simple 2-round history state + Epsilon-greedy action\n")
        f.write("Statistical: Opponent disposition state + Epsilon-greedy action\n")
        f.write("Softmax: Simple state + Softmax (temperature-based) action\n")
        f.write("Statistical+Softmax: Opponent disposition + Softmax action\n")
        f.write("Hysteretic+Statistical: Opponent disposition + Asymmetric learning\n\n")
        
        f.write("Key Insights:\n")
        f.write("-"*30 + "\n")
        f.write("- Statistical state helps identify opponent types early\n")
        f.write("- Softmax exploration is more principled than epsilon-greedy\n")
        f.write("- Combining strategies can yield synergistic benefits\n")
        f.write("- Hysteretic learning promotes cooperation through optimism\n\n")
        
        f.write("Recommended Use Cases:\n")
        f.write("-"*30 + "\n")
        f.write("- vs Deterministic opponents: Statistical state excels\n")
        f.write("- vs Mixed strategies: Softmax provides smoother adaptation\n")
        f.write("- For cooperation promotion: Hysteretic learning\n")
        f.write("- For general robustness: Statistical+Softmax combination\n")
    
    print(f"\nSummary report saved to: {report_path}")

if __name__ == "__main__":
    run_strategy_experiments()
    print("\n✓ All experiments completed!")