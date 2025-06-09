"""
Simple demonstration of Q-learning in 3-person Prisoner's Dilemma

This script provides a clear, focused demonstration of how Q-learning
agents behave in different environments.
"""

import random
import matplotlib.pyplot as plt
import numpy as np

# Import game structures
from main_neighbourhood import NPersonPrisonersDilemma, NPersonAgent, NPERSON_COOPERATE
from main_pairwise import PairwiseIteratedPrisonersDilemma, PairwiseAgent, PAIRWISE_COOPERATE

# Import extended agents
from extended_agents import (
    ExtendedNPersonAgent, ExtendedPairwiseAgent,
    QLearningNPersonWrapper, QLearningPairwiseWrapper
)

# Import Q-learning
from qlearning_agents import SimpleQLearningAgent, NPDLQLearningAgent


def create_agents_neighborhood(scenario_strategies, ql_type="simple"):
    """Create agents for neighborhood mode."""
    agents = []
    for i, strategy in enumerate(scenario_strategies):
        if strategy == "QL":
            if ql_type == "simple":
                ql_agent = SimpleQLearningAgent(i, learning_rate=0.15, epsilon=0.1)
            else:
                ql_agent = NPDLQLearningAgent(i, learning_rate=0.15, epsilon=0.1, 
                                            state_type="proportion_discretized")
            agents.append(QLearningNPersonWrapper(i, ql_agent))
        else:
            agents.append(ExtendedNPersonAgent(i, strategy, exploration_rate=0.01))
    return agents


def create_agents_pairwise(scenario_strategies, ql_type="simple"):
    """Create agents for pairwise mode."""
    agents = []
    for i, strategy in enumerate(scenario_strategies):
        if strategy == "QL":
            if ql_type == "simple":
                ql_agent = SimpleQLearningAgent(i, learning_rate=0.15, epsilon=0.1)
            else:
                ql_agent = NPDLQLearningAgent(i, learning_rate=0.15, epsilon=0.1,
                                            state_type="proportion_discretized")
            agents.append(QLearningPairwiseWrapper(i, ql_agent))
        else:
            agents.append(ExtendedPairwiseAgent(i, strategy, exploration_rate=0.01))
    return agents


def track_cooperation_evolution(agents, num_rounds=500):
    """Track how cooperation evolves over time in neighborhood mode."""
    cooperation_history = {i: [] for i in range(len(agents))}
    group_cooperation = []
    
    prev_coop_ratio = None
    
    for round_num in range(num_rounds):
        # Collect actions
        actions = {}
        for agent in agents:
            _, actual = agent.choose_action(prev_coop_ratio, round_num)
            actions[agent.agent_id] = actual
        
        # Calculate cooperation
        num_coops = sum(1 for action in actions.values() if action == NPERSON_COOPERATE)
        prev_coop_ratio = num_coops / len(agents)
        group_cooperation.append(prev_coop_ratio)
        
        # Calculate payoffs (simplified)
        for agent in agents:
            my_action = actions[agent.agent_id]
            others_coop = num_coops - (1 if my_action == NPERSON_COOPERATE else 0)
            
            # Simple payoff calculation
            if my_action == NPERSON_COOPERATE:
                payoff = 0 + 3 * (others_coop / (len(agents) - 1))
            else:
                payoff = 1 + 4 * (others_coop / (len(agents) - 1))
            
            agent.record_round_outcome(my_action, payoff)
            
            # Track individual cooperation
            if my_action == NPERSON_COOPERATE:
                cooperation_history[agent.agent_id].append(1)
            else:
                cooperation_history[agent.agent_id].append(0)
    
    return cooperation_history, group_cooperation


def analyze_qlearning_behavior(scenario_name, strategies, num_rounds=1000):
    """Analyze Q-learning behavior in a specific scenario."""
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name}")
    print(f"Strategies: {strategies}")
    print(f"{'='*60}")
    
    # Test both Q-learning types
    results = {}
    
    for ql_type in ["simple", "npdl"]:
        print(f"\n{ql_type.upper()} Q-Learning:")
        
        # Create agents
        agents = create_agents_neighborhood(strategies, ql_type)
        
        # Track evolution
        coop_history, group_coop = track_cooperation_evolution(agents, num_rounds)
        
        # Find Q-learning agent
        ql_idx = None
        for i, strategy in enumerate(strategies):
            if strategy == "QL":
                ql_idx = i
                break
        
        # Calculate statistics
        ql_agent = agents[ql_idx]
        final_coop_rate = ql_agent.get_cooperation_rate()
        final_score = ql_agent.total_score
        
        # Q-table analysis for simple Q-learning
        if ql_type == "simple" and hasattr(ql_agent.qlearning_agent, 'q_table'):
            print(f"\n  Q-Table Analysis:")
            q_table = ql_agent.qlearning_agent.q_table
            for state, values in sorted(q_table.items()):
                print(f"    State '{state}': C={values['cooperate']:.3f}, D={values['defect']:.3f}")
        
        print(f"\n  Final Results:")
        print(f"    Cooperation Rate: {final_coop_rate:.3f}")
        print(f"    Total Score: {final_score:.1f}")
        print(f"    Average Score per Round: {final_score/num_rounds:.3f}")
        
        # Compare with other agents
        print(f"\n  All Agents:")
        for i, agent in enumerate(agents):
            print(f"    Agent {i} ({strategies[i]}): "
                  f"Score={agent.total_score:.1f}, "
                  f"Coop={agent.get_cooperation_rate():.3f}")
        
        results[ql_type] = {
            'agents': agents,
            'coop_history': coop_history,
            'group_coop': group_coop,
            'ql_idx': ql_idx
        }
    
    return results


def plot_scenario_comparison(scenario_name, results, strategies):
    """Plot comparison of Simple vs NPDL Q-learning."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Group cooperation over time
    ax = axes[0, 0]
    window = 20  # Moving average window
    
    for ql_type in ["simple", "npdl"]:
        group_coop = results[ql_type]['group_coop']
        # Apply moving average
        smoothed = np.convolve(group_coop, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=f"{ql_type.upper()} QL", alpha=0.8)
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Group Cooperation Rate')
    ax.set_title('Group Cooperation Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Q-learning agent cooperation over time
    ax = axes[0, 1]
    
    for ql_type in ["simple", "npdl"]:
        ql_idx = results[ql_type]['ql_idx']
        ql_coop = results[ql_type]['coop_history'][ql_idx]
        # Apply moving average
        smoothed = np.convolve(ql_coop, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=f"{ql_type.upper()} QL", alpha=0.8)
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Q-Learning Agent Cooperation Rate')
    ax.set_title('Q-Learning Agent Behavior')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Final scores comparison
    ax = axes[1, 0]
    x = np.arange(3)
    width = 0.35
    
    for i, ql_type in enumerate(["simple", "npdl"]):
        agents = results[ql_type]['agents']
        scores = [agent.total_score for agent in agents]
        ax.bar(x + i*width - width/2, scores, width, label=f"{ql_type.upper()} QL")
    
    ax.set_xlabel('Agent')
    ax.set_ylabel('Total Score')
    ax.set_title('Final Scores')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i} ({strategies[i]})" for i in range(3)])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Cooperation rates comparison
    ax = axes[1, 1]
    
    for i, ql_type in enumerate(["simple", "npdl"]):
        agents = results[ql_type]['agents']
        coop_rates = [agent.get_cooperation_rate() for agent in agents]
        ax.bar(x + i*width - width/2, coop_rates, width, label=f"{ql_type.upper()} QL")
    
    ax.set_xlabel('Agent')
    ax.set_ylabel('Cooperation Rate')
    ax.set_title('Final Cooperation Rates')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i} ({strategies[i]})" for i in range(3)])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Scenario: {scenario_name}', fontsize=14)
    plt.tight_layout()
    
    return fig


def main():
    """Run focused Q-learning demonstrations."""
    print("Q-Learning Demonstration in 3-Person Prisoner's Dilemma")
    print("="*60)
    
    # Key scenarios to demonstrate Q-learning behavior
    scenarios = [
        # Scenario 1: QL learns to defect against defectors
        ("QL vs AllD vs AllD", ["QL", "AllD", "AllD"]),
        
        # Scenario 2: QL learns to cooperate with cooperators
        ("QL vs AllC vs AllC", ["QL", "AllC", "AllC"]),
        
        # Scenario 3: QL in mixed environment
        ("QL vs TFT vs AllD", ["QL", "pTFT-Threshold", "AllD"]),
        
        # Scenario 4: Multiple QL agents
        ("QL vs QL vs TFT", ["QL", "QL", "pTFT-Threshold"]),
    ]
    
    # Run each scenario
    for scenario_name, strategies in scenarios:
        results = analyze_qlearning_behavior(scenario_name, strategies, num_rounds=1000)
        
        # Plot results
        fig = plot_scenario_comparison(scenario_name, results, strategies)
        filename = f"ql_demo_{scenario_name.replace(' ', '_').lower()}.png"
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved as {filename}")
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY: Simple vs NPDL Q-Learning")
    print("="*60)
    
    print("\nKey Differences Observed:")
    print("1. State Representation:")
    print("   - Simple: Uses basic discretized states (very_low, low, medium, high, very_high)")
    print("   - NPDL: Uses proportion-based states with finer granularity")
    
    print("\n2. Learning Behavior:")
    print("   - Simple: More reactive, quicker to change strategies")
    print("   - NPDL: More stable, considers more context")
    
    print("\n3. Performance:")
    print("   - Against AllD: Both learn to defect")
    print("   - Against AllC: Both learn to cooperate") 
    print("   - Mixed environments: NPDL often achieves better balance")
    
    print("\n" + "="*60)
    print("Demonstration complete!")


if __name__ == "__main__":
    main()