"""
Demonstration script for the true pairwise implementation.

This script shows the key differences between aggregate and individual pairwise modes.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from npdl.core.true_pairwise import (
    TruePairwiseTFT, TruePairwiseGTFT, TruePairwisePavlov,
    TruePairwiseQLearning, TruePairwiseAdaptive, TruePairwiseEnvironment
)
from npdl.core.true_pairwise_adapter import create_true_pairwise_agent


def demonstrate_individual_decisions():
    """Show how agents make different decisions for different opponents."""
    print("=== Demonstrating Individual Decision-Making ===\n")
    
    # Create a TFT agent
    tft = TruePairwiseTFT("TFT_Agent")
    
    # Create three different opponents
    nice_opponent = TruePairwiseTFT("Nice_Opponent")
    mean_opponent = create_true_pairwise_agent({'type': 'always_defect', 'id': 'Mean_Opponent'})
    random_opponent = create_true_pairwise_agent({'type': 'random', 'id': 'Random_Opponent'})
    
    # Set up environment
    agents = [tft, nice_opponent, mean_opponent, random_opponent]
    env = TruePairwiseEnvironment(agents, rounds_per_episode=10)
    
    # Run simulation
    print("Running 10 rounds of interaction...")
    for round_num in range(10):
        round_results = env.run_round(round_num)
        
    # Show TFT's different relationships
    print("\nTFT's relationships with different opponents:")
    for opp_id, memory in tft.opponent_memories.items():
        print(f"\n{opp_id}:")
        print(f"  - Cooperation rate: {memory.get_cooperation_rate():.2%}")
        print(f"  - Total interactions: {memory.total_interactions}")
        print(f"  - Average reward: {memory.cumulative_reward / memory.total_interactions:.2f}")
        print(f"  - TFT's next move: {tft.choose_action_for_opponent(opp_id, 11)}")
        

def compare_aggregate_vs_individual():
    """Compare performance in aggregate vs individual pairwise modes."""
    print("\n=== Comparing Aggregate vs Individual Pairwise ===\n")
    
    # Note: This would require running the full simulation framework
    # Here we'll simulate the key difference conceptually
    
    print("Scenario: 1 TFT agent faces 1 cooperator and 1 defector\n")
    
    print("Aggregate Pairwise Mode:")
    print("- TFT must choose ONE action for both opponents")
    print("- If TFT cooperates: gets 3 from cooperator, 0 from defector = 3 total")
    print("- If TFT defects: gets 5 from cooperator, 1 from defector = 6 total")
    print("- Rational choice: DEFECT (exploiting the cooperator)")
    print("- Result: Destroys cooperation with the nice opponent\n")
    
    print("Individual Pairwise Mode:")
    print("- TFT can choose different actions for each opponent")
    print("- Cooperates with cooperator: gets 3")
    print("- Defects against defector: gets 1")
    print("- Total: 4 per round")
    print("- Result: Maintains cooperation with nice opponent while punishing defector")
    

def demonstrate_adaptive_strategy():
    """Show how adaptive agents identify and exploit different opponents."""
    print("\n=== Demonstrating Adaptive Strategy ===\n")
    
    # Create adaptive agent
    adaptive = TruePairwiseAdaptive("Adaptive_Agent", assessment_period=5)
    
    # Create various opponent types
    opponents = [
        TruePairwiseTFT("TFT_1"),
        TruePairwiseTFT("TFT_2"),
        create_true_pairwise_agent({'type': 'always_cooperate', 'id': 'Sucker'}),
        create_true_pairwise_agent({'type': 'always_defect', 'id': 'Meanie'}),
        TruePairwisePavlov("Pavlov_1")
    ]
    
    # Set up environment
    all_agents = [adaptive] + opponents
    env = TruePairwiseEnvironment(all_agents, rounds_per_episode=20)
    
    # Run simulation
    print("Running 20 rounds for adaptive agent to learn...")
    for round_num in range(20):
        env.run_round(round_num)
        
        # Show strategy assessments every 5 rounds
        if (round_num + 1) % 5 == 0:
            print(f"\nAfter round {round_num + 1}, adaptive agent's assessments:")
            for opp_id in adaptive.opponent_strategies:
                strategy = adaptive.opponent_strategies[opp_id]
                memory = adaptive.get_opponent_memory(opp_id)
                print(f"  {opp_id}: identified as '{strategy}' (coop rate: {memory.get_cooperation_rate():.2%})")
                

def demonstrate_qlearning_individual():
    """Show Q-learning with separate Q-tables per opponent."""
    print("\n=== Demonstrating Q-Learning with Individual Q-Tables ===\n")
    
    # Create Q-learning agents with different state representations
    ql_agents = [
        TruePairwiseQLearning("QL_Basic", state_type="basic", epsilon=0.1),
        TruePairwiseQLearning("QL_Memory", state_type="memory_enhanced", epsilon=0.1),
        TruePairwiseQLearning("QL_Reciprocity", state_type="reciprocity", epsilon=0.1)
    ]
    
    # Add some fixed-strategy opponents
    opponents = [
        TruePairwiseTFT("TFT_Opponent"),
        create_true_pairwise_agent({'type': 'always_cooperate', 'id': 'Cooperator'}),
        create_true_pairwise_agent({'type': 'always_defect', 'id': 'Defector'})
    ]
    
    all_agents = ql_agents + opponents
    env = TruePairwiseEnvironment(all_agents, rounds_per_episode=100)
    
    print("Training Q-learning agents for 100 rounds...")
    cooperation_rates = {agent.agent_id: [] for agent in ql_agents}
    
    for round_num in range(100):
        env.run_round(round_num)
        
        # Track cooperation rates every 10 rounds
        if (round_num + 1) % 10 == 0:
            for ql_agent in ql_agents:
                total_coops = sum(
                    sum(1 for h in mem.interaction_history if h['my_move'] == 'cooperate')
                    for mem in ql_agent.opponent_memories.values()
                )
                total_moves = sum(
                    len(mem.interaction_history)
                    for mem in ql_agent.opponent_memories.values()
                )
                coop_rate = total_coops / total_moves if total_moves > 0 else 0
                cooperation_rates[ql_agent.agent_id].append(coop_rate)
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    rounds = list(range(10, 101, 10))
    for agent_id, rates in cooperation_rates.items():
        plt.plot(rounds, rates, marker='o', label=agent_id)
    
    plt.xlabel('Round')
    plt.ylabel('Cooperation Rate')
    plt.title('Q-Learning Agents: Learning to Cooperate Selectively')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ql_individual_learning.png')
    plt.close()
    
    print("\nFinal Q-learning statistics:")
    for ql_agent in ql_agents:
        print(f"\n{ql_agent.agent_id}:")
        for opp_id, memory in ql_agent.opponent_memories.items():
            print(f"  vs {opp_id}: coop rate = {memory.get_cooperation_rate():.2%}, "
                  f"avg reward = {memory.cumulative_reward / memory.total_interactions:.2f}")
            

def visualize_relationship_network():
    """Visualize the different relationships between agents."""
    print("\n=== Visualizing Agent Relationships ===\n")
    
    # Create diverse agent population
    agents = [
        TruePairwiseTFT("TFT_1"),
        TruePairwiseTFT("TFT_2", forgiving_probability=0.1),
        TruePairwiseGTFT("GTFT_1", generosity=0.2),
        TruePairwisePavlov("Pavlov_1"),
        TruePairwiseAdaptive("Adaptive_1"),
        create_true_pairwise_agent({'type': 'always_cooperate', 'id': 'AllC'}),
        create_true_pairwise_agent({'type': 'always_defect', 'id': 'AllD'})
    ]
    
    env = TruePairwiseEnvironment(agents, rounds_per_episode=50)
    
    print("Running 50 rounds to establish relationships...")
    for round_num in range(50):
        env.run_round(round_num)
        
    # Create relationship matrix
    n_agents = len(agents)
    coop_matrix = np.zeros((n_agents, n_agents))
    
    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents):
            if i != j and agent2.agent_id in agent1.opponent_memories:
                memory = agent1.get_opponent_memory(agent2.agent_id)
                # Calculate cooperation rate from agent1 to agent2
                my_coops = sum(1 for h in memory.interaction_history 
                             if h['my_move'] == 'cooperate')
                coop_matrix[i, j] = my_coops / len(memory.interaction_history) if memory.interaction_history else 0
    
    # Visualize
    plt.figure(figsize=(10, 8))
    im = plt.imshow(coop_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    
    # Add labels
    agent_names = [a.agent_id for a in agents]
    plt.xticks(range(n_agents), agent_names, rotation=45, ha='right')
    plt.yticks(range(n_agents), agent_names)
    
    # Add colorbar
    plt.colorbar(im, label='Cooperation Rate')
    
    # Add text annotations
    for i in range(n_agents):
        for j in range(n_agents):
            if i != j:
                text = plt.text(j, i, f'{coop_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)
    
    plt.title('Agent Cooperation Relationships\n(row cooperates with column at rate shown)')
    plt.tight_layout()
    plt.savefig('agent_relationships.png')
    plt.close()
    
    print("Relationship visualization saved as 'agent_relationships.png'")
    
    # Print interesting relationships
    print("\nInteresting relationships found:")
    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents):
            if i < j:  # Only check each pair once
                rate1to2 = coop_matrix[i, j]
                rate2to1 = coop_matrix[j, i]
                
                if abs(rate1to2 - rate2to1) < 0.1 and rate1to2 > 0.8:
                    print(f"  Mutual cooperation: {agent1.agent_id} <-> {agent2.agent_id}")
                elif rate1to2 < 0.2 and rate2to1 < 0.2:
                    print(f"  Mutual defection: {agent1.agent_id} <-> {agent2.agent_id}")
                elif abs(rate1to2 - rate2to1) > 0.5:
                    print(f"  Asymmetric: {agent1.agent_id} ({rate1to2:.2f}) -> {agent2.agent_id} ({rate2to1:.2f})")


if __name__ == "__main__":
    print("True Pairwise Implementation Demonstration")
    print("=" * 50)
    
    # Run demonstrations
    demonstrate_individual_decisions()
    compare_aggregate_vs_individual()
    demonstrate_adaptive_strategy()
    demonstrate_qlearning_individual()
    visualize_relationship_network()
    
    print("\n" + "=" * 50)
    print("Demonstration complete!")
    print("\nKey takeaways:")
    print("1. Individual pairwise mode allows agents to maintain different relationships")
    print("2. Agents can cooperate with cooperators while punishing defectors")
    print("3. Learning agents maintain separate knowledge about each opponent")
    print("4. This leads to more nuanced and realistic social dynamics")
    print("\nCheck the generated PNG files for visualizations!")