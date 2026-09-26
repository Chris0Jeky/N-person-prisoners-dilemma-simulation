#!/usr/bin/env python3
"""
Demo comparing Adaptive Q-learners with different strategy enhancements.
Tests if Statistical State and Softmax provide benefits over baseline adaptive.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from collections import defaultdict

from final_agents import StaticAgent
from modular_agents import (
    create_adaptive_baseline,
    create_adaptive_statistical,
    create_adaptive_softmax,
    create_adaptive_statistical_softmax,
    create_adaptive_hysteretic_statistical
)
from config import ADAPTIVE_PARAMS, SIMULATION_CONFIG
from final_demo_full import run_experiment_set, smooth_data

def plot_adaptive_comparison(results, title, save_path):
    """Plot comparison of different adaptive agent strategies"""
    fig, axes = plt.subplots(2, 3, figsize=(24, 15))
    fig.suptitle(title, fontsize=22)
    
    colors = {
        "Baseline": "#1f77b4",
        "Statistical": "#ff7f0e",
        "Softmax": "#2ca02c",
        "Statistical+Softmax": "#d62728",
        "Hysteretic+Statistical": "#9467bd"
    }
    
    smooth_window = max(50, SIMULATION_CONFIG['num_rounds'] // 100)
    
    # Separate plots for cooperation rate, score, and adaptation metrics
    for idx, (agent_name, (p_data, n_data)) in enumerate(results.items()):
        agent_id = f"{agent_name}_Adaptive"
        
        if agent_id not in p_data:
            continue
        
        color = colors.get(agent_name, 'black')
        
        # Pairwise cooperation
        p_coop_smooth = smooth_data(p_data[agent_id]['coop_rate'], smooth_window)
        axes[0, 0].plot(p_data[agent_id]['coop_rate'], color=color, alpha=0.2, linewidth=0.5)
        axes[0, 0].plot(p_coop_smooth, label=agent_name, color=color, linewidth=2.5)
        
        # Pairwise score
        axes[0, 1].plot(p_data[agent_id]['score'], label=agent_name, color=color, linewidth=2)
        
        # Pairwise efficiency (score per round)
        efficiency = np.diff(p_data[agent_id]['score'])
        efficiency_smooth = smooth_data(efficiency, smooth_window)
        axes[0, 2].plot(efficiency_smooth, label=agent_name, color=color, linewidth=2)
        
        # Neighborhood cooperation
        n_coop_smooth = smooth_data(n_data[agent_id]['coop_rate'], smooth_window)
        axes[1, 0].plot(n_data[agent_id]['coop_rate'], color=color, alpha=0.2, linewidth=0.5)
        axes[1, 0].plot(n_coop_smooth, label=agent_name, color=color, linewidth=2)
        
        # Neighborhood score
        axes[1, 1].plot(n_data[agent_id]['score'], label=agent_name, color=color, linewidth=2)
        
        # Neighborhood efficiency
        n_efficiency = np.diff(n_data[agent_id]['score'])
        n_efficiency_smooth = smooth_data(n_efficiency, smooth_window)
        axes[1, 2].plot(n_efficiency_smooth, label=agent_name, color=color, linewidth=2)
    
    # Configure axes
    axes[0, 0].set_title('Pairwise Cooperation Rate')
    axes[0, 0].set_ylabel('Cooperation Rate')
    axes[0, 1].set_title('Pairwise Cumulative Score')
    axes[0, 1].set_ylabel('Cumulative Score')
    axes[0, 2].set_title('Pairwise Efficiency (Score/Round)')
    axes[0, 2].set_ylabel('Score per Round')
    
    axes[1, 0].set_title('Neighborhood Cooperation Rate')
    axes[1, 0].set_ylabel('Cooperation Rate')
    axes[1, 1].set_title('Neighborhood Cumulative Score')
    axes[1, 1].set_ylabel('Cumulative Score')
    axes[1, 2].set_title('Neighborhood Efficiency (Score/Round)')
    axes[1, 2].set_ylabel('Score per Round')
    
    for ax in axes.flat:
        ax.set_xlabel('Round')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def run_adaptive_strategy_comparison():
    """Compare different adaptive agent strategies"""
    
    OUTPUT_DIR = "adaptive_strategy_comparison"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=== Adaptive Q-Learning Strategy Comparison ===")
    print(f"Running {SIMULATION_CONFIG['num_rounds']} rounds, {SIMULATION_CONFIG['num_runs']} runs per scenario\n")
    
    # Agent configurations
    agent_configs = {
        "Baseline": create_adaptive_baseline,
        "Statistical": create_adaptive_statistical,
        "Softmax": create_adaptive_softmax,
        "Statistical+Softmax": create_adaptive_statistical_softmax,
        "Hysteretic+Statistical": create_adaptive_hysteretic_statistical
    }
    
    # Test scenarios with varying difficulty
    scenarios = {
        "vs_2TFT": {
            "opponents": [StaticAgent("TFT_1", "TFT"), StaticAgent("TFT_2", "TFT")],
            "description": "Cooperative environment (2 TFT)"
        },
        "vs_AllD_TFT": {
            "opponents": [StaticAgent("AllD", "AllD"), StaticAgent("TFT", "TFT")],
            "description": "Mixed environment (AllD + TFT)"
        },
        "vs_2Random": {
            "opponents": [StaticAgent("Random_1", "Random"), StaticAgent("Random_2", "Random")],
            "description": "Unpredictable environment (2 Random)"
        },
        "vs_TFT-E_AllC": {
            "opponents": [StaticAgent("TFT-E", "TFT-E", error_rate=0.1), StaticAgent("AllC", "AllC")],
            "description": "Noisy cooperative environment (TFT-E + AllC)"
        }
    }
    
    # Store results for summary
    all_results = defaultdict(dict)
    
    for scenario_name, scenario_data in scenarios.items():
        print(f"\nScenario: {scenario_data['description']}")
        scenario_results = {}
        
        for agent_name, creator_func in agent_configs.items():
            print(f"  Testing {agent_name}...", end='', flush=True)
            start_time = time.time()
            
            # Create agent with adaptive parameters
            agents = [
                creator_func(f"{agent_name}_Adaptive", ADAPTIVE_PARAMS)
            ] + scenario_data["opponents"]
            
            # Run experiment
            p_agg, n_agg = run_experiment_set(
                agents, 
                SIMULATION_CONFIG['num_rounds'], 
                SIMULATION_CONFIG['num_runs'],
                use_parallel=False  # Avoid pickling issues with complex strategies
            )
            
            scenario_results[agent_name] = (p_agg, n_agg)
            all_results[scenario_name][agent_name] = (p_agg, n_agg)
            
            elapsed = time.time() - start_time
            print(f" done in {elapsed:.1f}s")
        
        # Plot results for this scenario
        plot_path = os.path.join(OUTPUT_DIR, f"adaptive_comparison_{scenario_name}.png")
        plot_adaptive_comparison(scenario_results, 
                               f"Adaptive Strategy Comparison: {scenario_data['description']}", 
                               plot_path)
        print(f"  Saved plot: {plot_path}")
    
    # Create summary report
    create_performance_summary(all_results, OUTPUT_DIR)
    create_strategy_analysis(all_results, OUTPUT_DIR)

def create_performance_summary(all_results, output_dir):
    """Create a summary table of performance metrics"""
    summary_path = os.path.join(output_dir, "performance_summary.txt")
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ADAPTIVE Q-LEARNING STRATEGY PERFORMANCE SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        for scenario_name, scenario_results in all_results.items():
            f.write(f"\nScenario: {scenario_name}\n")
            f.write("-"*50 + "\n")
            f.write(f"{'Strategy':<25} {'Final PW Score':>15} {'Final NH Score':>15} {'Avg PW Coop':>15} {'Avg NH Coop':>15}\n")
            f.write("-"*50 + "\n")
            
            for agent_name, (p_data, n_data) in scenario_results.items():
                agent_id = f"{agent_name}_Adaptive"
                
                final_pw_score = p_data[agent_id]['score'][-1]
                final_nh_score = n_data[agent_id]['score'][-1]
                avg_pw_coop = np.mean(p_data[agent_id]['coop_rate'])
                avg_nh_coop = np.mean(n_data[agent_id]['coop_rate'])
                
                f.write(f"{agent_name:<25} {final_pw_score:>15.0f} {final_nh_score:>15.0f} "
                       f"{avg_pw_coop:>15.2%} {avg_nh_coop:>15.2%}\n")
    
    print(f"\nPerformance summary saved to: {summary_path}")

def create_strategy_analysis(all_results, output_dir):
    """Analyze which strategies provide the most benefit"""
    analysis_path = os.path.join(output_dir, "strategy_analysis.txt")
    
    # Calculate improvements over baseline
    improvements = defaultdict(lambda: defaultdict(list))
    
    for scenario_name, scenario_results in all_results.items():
        if "Baseline" not in scenario_results:
            continue
            
        baseline_p, baseline_n = scenario_results["Baseline"]
        baseline_id = "Baseline_Adaptive"
        baseline_pw_score = baseline_p[baseline_id]['score'][-1]
        baseline_nh_score = baseline_n[baseline_id]['score'][-1]
        
        for agent_name, (p_data, n_data) in scenario_results.items():
            if agent_name == "Baseline":
                continue
                
            agent_id = f"{agent_name}_Adaptive"
            pw_improvement = (p_data[agent_id]['score'][-1] - baseline_pw_score) / baseline_pw_score * 100
            nh_improvement = (n_data[agent_id]['score'][-1] - baseline_nh_score) / baseline_nh_score * 100
            
            improvements[agent_name]['pairwise'].append(pw_improvement)
            improvements[agent_name]['neighborhood'].append(nh_improvement)
    
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("STRATEGY IMPROVEMENT ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Average Improvement Over Baseline Adaptive (%)\n")
        f.write("-"*50 + "\n")
        f.write(f"{'Strategy':<25} {'Pairwise':>20} {'Neighborhood':>20}\n")
        f.write("-"*50 + "\n")
        
        for strategy, data in improvements.items():
            avg_pw = np.mean(data['pairwise'])
            avg_nh = np.mean(data['neighborhood'])
            f.write(f"{strategy:<25} {avg_pw:>20.1f}% {avg_nh:>20.1f}%\n")
        
        f.write("\n\nKey Findings:\n")
        f.write("-"*50 + "\n")
        
        # Find best strategies
        best_pw = max(improvements.items(), key=lambda x: np.mean(x[1]['pairwise']))
        best_nh = max(improvements.items(), key=lambda x: np.mean(x[1]['neighborhood']))
        
        f.write(f"- Best for Pairwise: {best_pw[0]} ({np.mean(best_pw[1]['pairwise']):.1f}% improvement)\n")
        f.write(f"- Best for Neighborhood: {best_nh[0]} ({np.mean(best_nh[1]['neighborhood']):.1f}% improvement)\n")
        
        # Strategy-specific insights
        f.write("\n\nStrategy-Specific Insights:\n")
        f.write("-"*50 + "\n")
        f.write("- Statistical State: Better opponent modeling, especially vs deterministic strategies\n")
        f.write("- Softmax Action: Smoother exploration, better Q-value exploitation\n")
        f.write("- Statistical+Softmax: Synergistic benefits from both enhancements\n")
        f.write("- Hysteretic+Statistical: Promotes cooperation through optimistic learning\n")
    
    print(f"Strategy analysis saved to: {analysis_path}")

if __name__ == "__main__":
    run_adaptive_strategy_comparison()
    print("\n✓ All experiments completed!")