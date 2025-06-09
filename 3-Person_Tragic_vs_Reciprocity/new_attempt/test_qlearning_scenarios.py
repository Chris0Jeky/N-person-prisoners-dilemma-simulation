"""
Test scenarios for Q-learning in 3-person Prisoner's Dilemma

Tests both Simple and NPDL Q-learning implementations in various scenarios
for both neighborhood and pairwise modes.
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Import game structures
from main_neighbourhood import NPersonPrisonersDilemma, NPersonAgent
from main_pairwise import PairwiseIteratedPrisonersDilemma, PairwiseAgent

# Import extended agents
from extended_agents import (
    ExtendedNPersonAgent, ExtendedPairwiseAgent,
    QLearningNPersonWrapper, QLearningPairwiseWrapper
)

# Import Q-learning implementations
from qlearning_agents import (
    SimpleQLearningAgent, NPDLQLearningAgent,
    create_simple_qlearning, create_npdl_qlearning
)


def create_agent_neighborhood(agent_id, strategy, qlearning_type="simple"):
    """Create an agent for neighborhood mode."""
    if strategy == "QL":
        if qlearning_type == "simple":
            ql_agent = create_simple_qlearning(agent_id, epsilon=0.1, learning_rate=0.1)
        else:  # NPDL
            ql_agent = create_npdl_qlearning(agent_id, epsilon=0.1, learning_rate=0.1)
        return QLearningNPersonWrapper(agent_id, ql_agent)
    else:
        # Use extended agent for AllC support
        return ExtendedNPersonAgent(agent_id, strategy, exploration_rate=0.01)


def create_agent_pairwise(agent_id, strategy, qlearning_type="simple"):
    """Create an agent for pairwise mode."""
    if strategy == "QL":
        if qlearning_type == "simple":
            ql_agent = create_simple_qlearning(agent_id, epsilon=0.1, learning_rate=0.1)
        else:  # NPDL
            ql_agent = create_npdl_qlearning(agent_id, epsilon=0.1, learning_rate=0.1)
        return QLearningPairwiseWrapper(agent_id, ql_agent)
    else:
        return ExtendedPairwiseAgent(agent_id, strategy, exploration_rate=0.01)


def run_scenario(scenario_name, strategies, mode="neighborhood", 
                 num_rounds=1000, num_episodes=10, rounds_per_episode=100,
                 qlearning_type="simple", num_runs=10):
    """
    Run a single scenario multiple times and collect results.
    
    Args:
        scenario_name: Name of the scenario
        strategies: List of 3 strategy names
        mode: "neighborhood" or "pairwise"
        num_rounds: Number of rounds for neighborhood mode
        num_episodes: Number of episodes for pairwise mode
        rounds_per_episode: Rounds per episode for pairwise mode
        qlearning_type: "simple" or "npdl"
        num_runs: Number of independent runs
    
    Returns:
        Dictionary with results
    """
    results = {
        'scenario': scenario_name,
        'strategies': strategies,
        'mode': mode,
        'qlearning_type': qlearning_type,
        'scores': {i: [] for i in range(3)},
        'cooperation_rates': {i: [] for i in range(3)},
        'avg_scores': {},
        'avg_coop_rates': {},
        'cooperation_evolution': []  # Track cooperation over time
    }
    
    for run in range(num_runs):
        # Create agents
        if mode == "neighborhood":
            agents = [create_agent_neighborhood(i, strategies[i], qlearning_type) 
                     for i in range(3)]
            
            # Run simulation
            game = NPersonPrisonersDilemma(agents, num_rounds)
            
            # Track cooperation evolution
            coop_history = []
            for round_num in range(num_rounds):
                # Run one round
                prev_coop_ratio = None
                if round_num > 0 and coop_history:
                    prev_coop_ratio = coop_history[-1]
                
                round_coops = 0
                for agent in agents:
                    _, actual = agent.choose_action(prev_coop_ratio, round_num)
                    if actual == 0:  # COOPERATE
                        round_coops += 1
                    # Simple payoff calculation for tracking
                    agent.record_round_outcome(actual, 1)  # Dummy payoff
                
                coop_history.append(round_coops / 3.0)
            
            # Reset and run full simulation
            for agent in agents:
                agent.reset()
            game.run_simulation()
            
            results['cooperation_evolution'].append(coop_history)
            
        else:  # pairwise
            agents = [create_agent_pairwise(i, strategies[i], qlearning_type) 
                     for i in range(3)]
            
            # Run tournament
            game = PairwiseIteratedPrisonersDilemma(agents, num_episodes, rounds_per_episode)
            game.run_tournament()
        
        # Collect results
        for i, agent in enumerate(agents):
            results['scores'][i].append(agent.total_score)
            results['cooperation_rates'][i].append(agent.get_cooperation_rate())
    
    # Calculate averages
    for i in range(3):
        results['avg_scores'][i] = np.mean(results['scores'][i])
        results['avg_coop_rates'][i] = np.mean(results['cooperation_rates'][i])
    
    return results


def plot_results(all_results, qlearning_type):
    """Plot results for all scenarios."""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Average scores by scenario
    ax = axes[0, 0]
    scenarios = list(all_results.keys())
    x = np.arange(len(scenarios))
    width = 0.25
    
    for i in range(3):
        scores = [all_results[s]['avg_scores'][i] for s in scenarios]
        strategies = [all_results[s]['strategies'][i] for s in scenarios]
        label = f"Agent {i}"
        ax.bar(x + i*width - width, scores, width, label=label)
    
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Average Score')
    ax.set_title(f'Average Scores by Scenario ({qlearning_type} Q-Learning)')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Cooperation rates by scenario
    ax = axes[0, 1]
    for i in range(3):
        coop_rates = [all_results[s]['avg_coop_rates'][i] for s in scenarios]
        ax.bar(x + i*width - width, coop_rates, width, label=f"Agent {i}")
    
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Cooperation Rate')
    ax.set_title(f'Cooperation Rates by Scenario ({qlearning_type} Q-Learning)')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Q-Learning performance across scenarios
    ax = axes[1, 0]
    ql_scores = []
    ql_coop = []
    scenario_labels = []
    
    for scenario, result in all_results.items():
        for i, strategy in enumerate(result['strategies']):
            if strategy == "QL":
                ql_scores.append(result['avg_scores'][i])
                ql_coop.append(result['avg_coop_rates'][i])
                scenario_labels.append(scenario)
    
    x_ql = np.arange(len(ql_scores))
    ax.scatter(x_ql, ql_scores, label='Score', s=100, alpha=0.7)
    ax2 = ax.twinx()
    ax2.scatter(x_ql, ql_coop, label='Cooperation Rate', color='red', s=100, alpha=0.7)
    
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Score', color='blue')
    ax2.set_ylabel('Cooperation Rate', color='red')
    ax.set_title(f'Q-Learning Performance Across Scenarios ({qlearning_type})')
    ax.set_xticks(x_ql)
    ax.set_xticklabels(scenario_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # 4. Cooperation evolution (for neighborhood mode scenarios)
    ax = axes[1, 1]
    for scenario, result in all_results.items():
        if result['mode'] == 'neighborhood' and result['cooperation_evolution']:
            # Average across runs
            avg_evolution = np.mean(result['cooperation_evolution'], axis=0)
            # Plot every 10th point to reduce clutter
            ax.plot(avg_evolution[::10], label=scenario, alpha=0.7)
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Cooperation Rate')
    ax.set_title(f'Cooperation Evolution Over Time ({qlearning_type} Q-Learning)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def print_detailed_results(results):
    """Print detailed results for a scenario."""
    print(f"\n{'='*60}")
    print(f"Scenario: {results['scenario']}")
    print(f"Mode: {results['mode']}")
    print(f"Q-Learning Type: {results['qlearning_type']}")
    print(f"Strategies: {results['strategies']}")
    print(f"{'='*60}")
    
    print("\nAverage Scores:")
    for i in range(3):
        strategy = results['strategies'][i]
        score = results['avg_scores'][i]
        std = np.std(results['scores'][i])
        print(f"  Agent {i} ({strategy}): {score:.2f} ± {std:.2f}")
    
    print("\nCooperation Rates:")
    for i in range(3):
        strategy = results['strategies'][i]
        coop_rate = results['avg_coop_rates'][i]
        std = np.std(results['cooperation_rates'][i])
        print(f"  Agent {i} ({strategy}): {coop_rate:.3f} ± {std:.3f}")
    
    # Find Q-learning agent and report its performance
    for i, strategy in enumerate(results['strategies']):
        if strategy == "QL":
            print(f"\nQ-Learning Agent Performance:")
            print(f"  Position: Agent {i}")
            print(f"  Score Rank: {sorted(results['avg_scores'].values(), reverse=True).index(results['avg_scores'][i]) + 1}/3")
            print(f"  Cooperation Rank: {sorted(results['avg_coop_rates'].values(), reverse=True).index(results['avg_coop_rates'][i]) + 1}/3")


def main():
    """Run all test scenarios."""
    # Define scenarios
    scenarios = [
        ("QL vs TFT vs AllD", ["QL", "pTFT", "AllD"]),      # For neighborhood
        ("QL vs TFT vs AllC", ["QL", "pTFT", "AllC"]),
        ("QL vs AllC vs AllC", ["QL", "AllC", "AllC"]),
        ("QL vs AllD vs AllD", ["QL", "AllD", "AllD"]),
        ("QL vs AllD vs AllC", ["QL", "AllD", "AllC"]),
        ("QL vs TFT vs TFT", ["QL", "pTFT", "pTFT"]),
        ("QL vs QL vs TFT", ["QL", "QL", "pTFT"]),
    ]
    
    scenarios_pairwise = [
        ("QL vs TFT vs AllD", ["QL", "TFT", "AllD"]),       # For pairwise
        ("QL vs TFT vs AllC", ["QL", "TFT", "AllC"]),
        ("QL vs AllC vs AllC", ["QL", "AllC", "AllC"]),
        ("QL vs AllD vs AllD", ["QL", "AllD", "AllD"]),
        ("QL vs AllD vs AllC", ["QL", "AllD", "AllC"]),
        ("QL vs TFT vs TFT", ["QL", "TFT", "TFT"]),
        ("QL vs QL vs TFT", ["QL", "QL", "TFT"]),
    ]
    
    # Test both Q-learning implementations
    for ql_type in ["simple", "npdl"]:
        print(f"\n{'#'*80}")
        print(f"# Testing {ql_type.upper()} Q-Learning Implementation")
        print(f"{'#'*80}")
        
        # Run neighborhood mode scenarios
        print(f"\n{'='*60}")
        print("NEIGHBORHOOD MODE")
        print(f"{'='*60}")
        
        neighborhood_results = {}
        for name, strategies in scenarios:
            print(f"\nRunning: {name}")
            results = run_scenario(name, strategies, mode="neighborhood", 
                                 num_rounds=1000, qlearning_type=ql_type,
                                 num_runs=5)
            neighborhood_results[name] = results
            print_detailed_results(results)
        
        # Run pairwise mode scenarios
        print(f"\n{'='*60}")
        print("PAIRWISE MODE")
        print(f"{'='*60}")
        
        pairwise_results = {}
        for name, strategies in scenarios_pairwise:
            print(f"\nRunning: {name}")
            results = run_scenario(name, strategies, mode="pairwise",
                                 num_episodes=10, rounds_per_episode=100,
                                 qlearning_type=ql_type, num_runs=5)
            pairwise_results[name] = results
            print_detailed_results(results)
        
        # Plot results
        print(f"\nGenerating plots for {ql_type} Q-Learning...")
        
        # Neighborhood plots
        fig_neighborhood = plot_results(neighborhood_results, f"{ql_type} (Neighborhood)")
        fig_neighborhood.savefig(f'qlearning_{ql_type}_neighborhood_results.png', dpi=150, bbox_inches='tight')
        
        # Pairwise plots
        fig_pairwise = plot_results(pairwise_results, f"{ql_type} (Pairwise)")
        fig_pairwise.savefig(f'qlearning_{ql_type}_pairwise_results.png', dpi=150, bbox_inches='tight')
        
        print(f"Plots saved as qlearning_{ql_type}_neighborhood_results.png and qlearning_{ql_type}_pairwise_results.png")
    
    # Compare Simple vs NPDL Q-Learning
    print(f"\n{'#'*80}")
    print("# COMPARISON: Simple vs NPDL Q-Learning")
    print(f"{'#'*80}")
    
    # Run a focused comparison on key scenarios
    comparison_scenarios = [
        ("QL vs TFT vs AllD", ["QL", "pTFT", "AllD"]),
        ("QL vs AllC vs AllC", ["QL", "AllC", "AllC"]),
        ("QL vs TFT vs TFT", ["QL", "pTFT", "pTFT"]),
    ]
    
    print("\nNeighborhood Mode Comparison:")
    for name, strategies in comparison_scenarios:
        print(f"\n{name}:")
        
        # Simple QL
        simple_result = run_scenario(name, strategies, mode="neighborhood",
                                   num_rounds=1000, qlearning_type="simple", 
                                   num_runs=10)
        simple_ql_idx = strategies.index("QL")
        simple_score = simple_result['avg_scores'][simple_ql_idx]
        simple_coop = simple_result['avg_coop_rates'][simple_ql_idx]
        
        # NPDL QL
        npdl_result = run_scenario(name, strategies, mode="neighborhood",
                                 num_rounds=1000, qlearning_type="npdl",
                                 num_runs=10)
        npdl_ql_idx = strategies.index("QL")
        npdl_score = npdl_result['avg_scores'][npdl_ql_idx]
        npdl_coop = npdl_result['avg_coop_rates'][npdl_ql_idx]
        
        print(f"  Simple QL: Score={simple_score:.2f}, Cooperation={simple_coop:.3f}")
        print(f"  NPDL QL:   Score={npdl_score:.2f}, Cooperation={npdl_coop:.3f}")
        print(f"  Difference: Score={npdl_score-simple_score:+.2f}, Cooperation={npdl_coop-simple_coop:+.3f}")
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)


if __name__ == "__main__":
    main()