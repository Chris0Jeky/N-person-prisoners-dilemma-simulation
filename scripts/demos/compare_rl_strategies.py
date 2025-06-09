"""
Compare Standard RL vs N-Person RL Strategies

This script demonstrates the differences between standard RL agents
and N-person aware RL agents in multi-agent scenarios.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from npdl.core.environment import Environment
from npdl.core.agents import Agent
from npdl.simulation.runner import SimulationRunner


def create_comparison_scenarios():
    """Create scenarios comparing standard vs N-person RL."""
    scenarios = []
    
    # Varying group sizes
    for N in [5, 10, 20]:
        # Standard Q-learning
        scenarios.append({
            "name": f"Standard_QL_N{N}",
            "num_agents": N,
            "num_rounds": 500,
            "interaction_mode": "neighborhood",
            "agents": {
                "q_learning": 1,
                "tit_for_tat": N - 1
            },
            "agent_params": {
                "q_learning": {
                    "epsilon": 0.1,
                    "learning_rate": 0.1,
                    "state_type": "proportion_discretized"
                }
            }
        })
        
        # N-person Q-learning
        scenarios.append({
            "name": f"NPerson_QL_N{N}",
            "num_agents": N,
            "num_rounds": 500,
            "interaction_mode": "neighborhood",
            "agents": {
                "n_person_q_learning": 1,
                "tit_for_tat": N - 1
            },
            "agent_params": {
                "n_person_q_learning": {
                    "epsilon": 0.1,
                    "learning_rate": 0.1,
                    "state_type": "n_person_basic",
                    "N": N
                }
            }
        })
    
    # Compare all three N-person variants
    for strategy in ["q_learning", "hysteretic_q", "wolf_phc"]:
        # Standard version
        scenarios.append({
            "name": f"Standard_{strategy}_mixed",
            "num_agents": 15,
            "num_rounds": 500,
            "interaction_mode": "neighborhood",
            "agents": {
                strategy: 3,
                "tit_for_tat": 6,
                "generous_tit_for_tat": 3,
                "always_defect": 3
            }
        })
        
        # N-person version
        n_person_strategy = f"n_person_{strategy}"
        scenarios.append({
            "name": f"NPerson_{strategy}_mixed",
            "num_agents": 15,
            "num_rounds": 500,
            "interaction_mode": "neighborhood",
            "agents": {
                n_person_strategy: 3,
                "tit_for_tat": 6,
                "generous_tit_for_tat": 3,
                "always_defect": 3
            },
            "agent_params": {
                n_person_strategy: {"N": 15}
            }
        })
    
    return scenarios


def run_comparison_experiments(scenarios):
    """Run the comparison experiments."""
    results = {}
    
    for scenario in scenarios:
        print(f"\nRunning scenario: {scenario['name']}")
        
        # Create environment
        env = Environment(
            num_agents=scenario['num_agents'],
            interaction_mode=scenario['interaction_mode'],
            payoff_matrix="standard"  # Standard PD payoffs
        )
        
        # Create agents
        agents = []
        agent_id = 0
        
        for strategy, count in scenario['agents'].items():
            for _ in range(count):
                # Get agent parameters if specified
                params = scenario.get('agent_params', {}).get(strategy, {})
                
                agent = Agent(
                    agent_id=agent_id,
                    strategy=strategy,
                    **params
                )
                agents.append(agent)
                agent_id += 1
        
        env.agents = agents
        
        # Run simulation
        runner = SimulationRunner(env)
        history = runner.run(num_rounds=scenario['num_rounds'])
        
        # Store results
        results[scenario['name']] = {
            'history': history,
            'final_cooperation': history['cooperation_rate'][-1] if history['cooperation_rate'] else 0,
            'avg_cooperation': np.mean(history['cooperation_rate']) if history['cooperation_rate'] else 0,
            'scenario': scenario
        }
        
        print(f"Final cooperation rate: {results[scenario['name']]['final_cooperation']:.3f}")
        print(f"Average cooperation rate: {results[scenario['name']]['avg_cooperation']:.3f}")
    
    return results


def analyze_results(results):
    """Analyze and visualize the comparison results."""
    
    # 1. Group size scaling comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    group_sizes = [5, 10, 20]
    standard_coop = []
    nperson_coop = []
    
    for N in group_sizes:
        standard_key = f"Standard_QL_N{N}"
        nperson_key = f"NPerson_QL_N{N}"
        
        if standard_key in results:
            standard_coop.append(results[standard_key]['avg_cooperation'])
        if nperson_key in results:
            nperson_coop.append(results[nperson_key]['avg_cooperation'])
    
    # Plot cooperation vs group size
    ax1.plot(group_sizes, standard_coop, 'b-o', label='Standard Q-Learning', linewidth=2)
    ax1.plot(group_sizes, nperson_coop, 'r-s', label='N-Person Q-Learning', linewidth=2)
    ax1.set_xlabel('Group Size (N)')
    ax1.set_ylabel('Average Cooperation Rate')
    ax1.set_title('Cooperation vs Group Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Strategy comparison in mixed population
    strategies = ['q_learning', 'hysteretic_q', 'wolf_phc']
    standard_perf = []
    nperson_perf = []
    
    for strategy in strategies:
        standard_key = f"Standard_{strategy}_mixed"
        nperson_key = f"NPerson_{strategy}_mixed"
        
        if standard_key in results:
            standard_perf.append(results[standard_key]['avg_cooperation'])
        if nperson_key in results:
            nperson_perf.append(results[nperson_key]['avg_cooperation'])
    
    # Bar chart comparison
    x = np.arange(len(strategies))
    width = 0.35
    
    ax2.bar(x - width/2, standard_perf, width, label='Standard', color='blue', alpha=0.7)
    ax2.bar(x + width/2, nperson_perf, width, label='N-Person', color='red', alpha=0.7)
    ax2.set_xlabel('Strategy Type')
    ax2.set_ylabel('Average Cooperation Rate')
    ax2.set_title('Strategy Performance in Mixed Population')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Q-Learning', 'Hysteretic-Q', 'Wolf-PHC'])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('rl_comparison_results.png', dpi=150)
    print("\nResults saved to rl_comparison_results.png")
    
    # 3. Learning curves comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Plot learning curves for different scenarios
    plot_idx = 0
    for N in [5, 10, 20]:
        standard_key = f"Standard_QL_N{N}"
        nperson_key = f"NPerson_QL_N{N}"
        
        if standard_key in results and nperson_key in results:
            ax = axes[plot_idx]
            
            # Smooth the cooperation rates
            window = 20
            standard_smooth = smooth_data(results[standard_key]['history']['cooperation_rate'], window)
            nperson_smooth = smooth_data(results[nperson_key]['history']['cooperation_rate'], window)
            
            ax.plot(standard_smooth, 'b-', label='Standard QL', alpha=0.7)
            ax.plot(nperson_smooth, 'r-', label='N-Person QL', alpha=0.7)
            ax.set_title(f'Learning Curves (N={N})')
            ax.set_xlabel('Round')
            ax.set_ylabel('Cooperation Rate')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
    
    # Plot strategy comparisons
    for idx, strategy in enumerate(['q_learning', 'hysteretic_q', 'wolf_phc']):
        standard_key = f"Standard_{strategy}_mixed"
        nperson_key = f"NPerson_{strategy}_mixed"
        
        if standard_key in results and nperson_key in results:
            ax = axes[plot_idx]
            
            # Smooth the cooperation rates
            standard_smooth = smooth_data(results[standard_key]['history']['cooperation_rate'], window)
            nperson_smooth = smooth_data(results[nperson_key]['history']['cooperation_rate'], window)
            
            ax.plot(standard_smooth, 'b-', label=f'Standard', alpha=0.7)
            ax.plot(nperson_smooth, 'r-', label=f'N-Person', alpha=0.7)
            ax.set_title(f'{strategy.replace("_", " ").title()} in Mixed Population')
            ax.set_xlabel('Round')
            ax.set_ylabel('Cooperation Rate')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('rl_learning_curves.png', dpi=150)
    print("Learning curves saved to rl_learning_curves.png")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print("\nGroup Size Scaling (Average Cooperation):")
    print(f"{'N':<5} {'Standard QL':<15} {'N-Person QL':<15} {'Improvement':<15}")
    print("-"*50)
    for i, N in enumerate(group_sizes):
        if i < len(standard_coop) and i < len(nperson_coop):
            improvement = ((nperson_coop[i] - standard_coop[i]) / standard_coop[i]) * 100
            print(f"{N:<5} {standard_coop[i]:<15.3f} {nperson_coop[i]:<15.3f} {improvement:<15.1f}%")
    
    print("\nStrategy Comparison in Mixed Population:")
    print(f"{'Strategy':<15} {'Standard':<15} {'N-Person':<15} {'Improvement':<15}")
    print("-"*60)
    for i, strategy in enumerate(strategies):
        if i < len(standard_perf) and i < len(nperson_perf):
            improvement = ((nperson_perf[i] - standard_perf[i]) / standard_perf[i]) * 100
            print(f"{strategy:<15} {standard_perf[i]:<15.3f} {nperson_perf[i]:<15.3f} {improvement:<15.1f}%")


def smooth_data(data, window):
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2 + 1)
        smoothed.append(np.mean(data[start:end]))
    
    return smoothed


def main():
    """Main function to run the comparison."""
    print("Comparing Standard RL vs N-Person RL Strategies")
    print("="*60)
    
    # Create scenarios
    scenarios = create_comparison_scenarios()
    print(f"Created {len(scenarios)} comparison scenarios")
    
    # Run experiments
    results = run_comparison_experiments(scenarios)
    
    # Analyze and visualize
    analyze_results(results)
    
    # Save detailed results
    summary = {}
    for name, result in results.items():
        summary[name] = {
            'final_cooperation': result['final_cooperation'],
            'avg_cooperation': result['avg_cooperation'],
            'scenario': result['scenario']
        }
    
    with open('rl_comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nDetailed results saved to rl_comparison_summary.json")
    print("\nComparison complete!")


if __name__ == "__main__":
    main()