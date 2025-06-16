"""
Demo script showing how to use the NPD Simulator
"""

import sys
from pathlib import Path

# Add npd_simulator to path
sys.path.insert(0, str(Path(__file__).parent))

from main import NPDSimulator
from experiments.scenarios.scenario_generator import ScenarioGenerator
from utils.config_loader import create_sample_config


def demo_single_experiment():
    """Run a simple single experiment."""
    print("=== Demo: Single Experiment ===")
    
    # Create simulator
    simulator = NPDSimulator("demo_results")
    
    # Create a simple configuration
    config = {
        "name": "demo_3agent_mixed",
        "agents": [
            {"id": 0, "type": "TFT", "exploration_rate": 0.1},
            {"id": 1, "type": "AllD", "exploration_rate": 0.0},
            {"id": 2, "type": "AllC", "exploration_rate": 0.0}
        ],
        "num_rounds": 500
    }
    
    # Run experiment
    results = simulator.run_single_experiment(config)
    
    print(f"Average cooperation: {results['average_cooperation']:.2%}")
    print("\nAgent scores:")
    for stat in results['agent_stats']:
        print(f"  Agent {stat['agent_id']}: {stat['total_score']:.1f} (coop rate: {stat['cooperation_rate']:.2%})")


def demo_qlearning_comparison():
    """Compare basic and enhanced Q-learning."""
    print("\n=== Demo: Q-Learning Comparison ===")
    
    simulator = NPDSimulator("demo_results")
    
    configs = [
        {
            "name": "basic_qlearning",
            "agents": [
                {"id": 0, "type": "QLearning", "exploration_rate": 0.0,
                 "learning_rate": 0.15, "epsilon": 0.1},
                {"id": 1, "type": "AllC", "exploration_rate": 0.0},
                {"id": 2, "type": "AllC", "exploration_rate": 0.0}
            ],
            "num_rounds": 1000
        },
        {
            "name": "enhanced_qlearning",
            "agents": [
                {"id": 0, "type": "EnhancedQLearning", "exploration_rate": 0.0,
                 "learning_rate": 0.15, "epsilon": 0.1, "epsilon_decay": 0.995,
                 "exclude_self": True},
                {"id": 1, "type": "AllC", "exploration_rate": 0.0},
                {"id": 2, "type": "AllC", "exploration_rate": 0.0}
            ],
            "num_rounds": 1000
        }
    ]
    
    results = simulator.run_batch_experiments(configs)
    
    print("\nQ-Learning Comparison Results:")
    for result in results:
        q_agent_stat = result['agent_stats'][0]  # Q-learning is agent 0
        exploitation_rate = 1 - q_agent_stat['cooperation_rate']
        print(f"{result['config']['name']}:")
        print(f"  Exploitation rate: {exploitation_rate:.2%}")
        print(f"  Score: {q_agent_stat['total_score']:.1f}")


def demo_scaling_analysis():
    """Demonstrate scaling analysis."""
    print("\n=== Demo: Scaling Analysis ===")
    
    simulator = NPDSimulator("demo_results")
    
    base_config = {
        "name": "scaling_test",
        "agents": [
            {"id": 0, "type": "pTFT", "exploration_rate": 0.05},
            {"id": 1, "type": "AllD", "exploration_rate": 0.0},
            {"id": 2, "type": "AllC", "exploration_rate": 0.0}
        ],
        "num_rounds": 500
    }
    
    # Test with different agent counts
    agent_counts = [3, 6, 9, 12]
    
    analysis = simulator.run_comparative_analysis(base_config, agent_counts)
    
    print("\nScaling Results:")
    for i, n in enumerate(analysis['agent_counts']):
        print(f"{n} agents: avg cooperation = {analysis['avg_cooperation'][i]:.2%}, "
              f"avg score = {analysis['avg_score'][i]:.1f}")


def demo_scenario_generation():
    """Demonstrate automatic scenario generation."""
    print("\n=== Demo: Scenario Generation ===")
    
    generator = ScenarioGenerator()
    
    # Generate Q-learning test scenarios
    scenarios = generator.generate_qlearning_test_scenarios(num_agents=4)
    
    print(f"Generated {len(scenarios)} Q-learning test scenarios:")
    for scenario in scenarios:
        print(f"  - {scenario['name']}")
    
    # Generate balanced scenarios
    balanced = generator.generate_balanced_scenarios(num_agents=5, num_scenarios=3)
    
    print(f"\nGenerated {len(balanced)} balanced scenarios:")
    for scenario in balanced:
        agent_types = [a['type'] for a in scenario['agents']]
        print(f"  - {scenario['name']}: {agent_types}")


if __name__ == "__main__":
    print("NPD Simulator Demo\n")
    
    # Run all demos
    demo_single_experiment()
    demo_qlearning_comparison()
    demo_scaling_analysis()
    demo_scenario_generation()
    
    print("\nDemo complete! Check 'demo_results' directory for outputs.")
    print("\nTo run your own experiments, use:")
    print("  python main.py single -c <config_file>")
    print("  python main.py batch -c <config_file>")
    print("  python main.py compare -c <config_file> -n 3 5 10")
    print("  python main.py sweep -c <config_file>")