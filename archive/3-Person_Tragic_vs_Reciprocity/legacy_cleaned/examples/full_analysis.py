#!/usr/bin/env python3
"""
Full analysis example - Comprehensive experiment with visualization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import ExperimentRunner

# Try importing visualization libraries
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False
    print("Note: Install matplotlib for visualizations (pip install matplotlib numpy)")

def analyze_cooperation_dynamics():
    """Run experiments and analyze cooperation dynamics"""
    print("=== Full Cooperation Dynamics Analysis ===\n")
    
    # Custom configurations for analysis
    compositions = [
        {
            "name": "Baseline_3TFT",
            "agents": [
                {"id": "TFT_1", "strategy": "pTFT"},
                {"id": "TFT_2", "strategy": "pTFT"},
                {"id": "TFT_3", "strategy": "pTFT"}
            ]
        },
        {
            "name": "QL_Learning",
            "agents": [
                {"id": "QL_1", "strategy": "NPDLQL"},
                {"id": "TFT_1", "strategy": "pTFT"},
                {"id": "TFT_2", "strategy": "pTFT"}
            ]
        },
        {
            "name": "QL_vs_Defector",
            "agents": [
                {"id": "QL_1", "strategy": "NPDLQL"},
                {"id": "TFT_1", "strategy": "pTFT"},
                {"id": "AllD_1", "strategy": "AllD"}
            ]
        },
        {
            "name": "Multi_QL",
            "agents": [
                {"id": "QL_1", "strategy": "NPDLQL"},
                {"id": "QL_2", "strategy": "NPDLQL"},
                {"id": "TFT_1", "strategy": "pTFT"}
            ]
        }
    ]
    
    # Create experiment runner
    runner = ExperimentRunner(base_output_dir="analysis_results")
    runner.set_configurations(
        exploration_rates=[0.0],  # Q-learning uses epsilon
        agent_compositions=compositions
    )
    runner.num_runs = 10  # Multiple runs for statistical significance
    
    # Run experiments
    print("Running experiments...")
    runner.run_all_experiments()
    
    print("\nExperiments completed! Results saved to 'analysis_results/'")
    
    # Generate visualizations if available
    if HAS_VIZ:
        generate_cooperation_plot(compositions)

def generate_cooperation_plot(compositions):
    """Generate a cooperation rate comparison plot"""
    print("\nGenerating visualization...")
    
    # Sample data for demonstration (in real use, load from results)
    comp_names = [comp['name'] for comp in compositions]
    pairwise_rates = [0.85, 0.72, 0.45, 0.68]  # Example data
    nperson_rates = [0.90, 0.78, 0.52, 0.71]   # Example data
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(comp_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pairwise_rates, width, label='Pairwise', alpha=0.8)
    bars2 = ax.bar(x + width/2, nperson_rates, width, label='N-Person', alpha=0.8)
    
    ax.set_xlabel('Agent Composition')
    ax.set_ylabel('Cooperation Rate')
    ax.set_title('Cooperation Rates Across Different Agent Compositions')
    ax.set_xticks(x)
    ax.set_xticklabels(comp_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('cooperation_comparison.png', dpi=300)
    print("Plot saved as 'cooperation_comparison.png'")

def parameter_sensitivity_analysis():
    """Analyze sensitivity to Q-learning parameters"""
    print("\n\n=== Parameter Sensitivity Analysis ===\n")
    
    from src import create_agent, NPersonGame
    
    # Parameters to test
    learning_rates = [0.05, 0.1, 0.2]
    epsilons = [0.05, 0.1, 0.15]
    
    results = {}
    
    for lr in learning_rates:
        for eps in epsilons:
            print(f"Testing: learning_rate={lr}, epsilon={eps}")
            
            # Create agents
            agents = [
                create_agent("QL_1", "NPDLQL", 
                           learning_rate=lr, epsilon=eps),
                create_agent("TFT_1", "pTFT"),
                create_agent("TFT_2", "pTFT")
            ]
            
            # Run multiple simulations
            scores = []
            for _ in range(5):
                game = NPersonGame(agents, num_rounds=300)
                game_results = game.run_simulation()
                scores.append(game_results['agents']['QL_1']['total_score'])
            
            avg_score = sum(scores) / len(scores)
            results[(lr, eps)] = avg_score
            print(f"  Average score: {avg_score:.1f}")
    
    # Find best parameters
    best_params = max(results.items(), key=lambda x: x[1])
    print(f"\nBest parameters: learning_rate={best_params[0][0]}, "
          f"epsilon={best_params[0][1]}")
    print(f"Best average score: {best_params[1]:.1f}")

if __name__ == "__main__":
    analyze_cooperation_dynamics()
    parameter_sensitivity_analysis()