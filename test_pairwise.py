"""
Test script for validating the pairwise interaction model implementation.

This script includes tests to verify:
1. The correct payoff calculation in pairwise interactions
2. The proper state representation handling for RL agents
3. Performance comparison between different strategies in the pairwise model
"""

import random
import logging
import numpy as np
from npdl.core.agents import Agent
from npdl.core.environment import Environment
from npdl.core.utils import create_payoff_matrix, get_pairwise_payoffs

def test_pairwise_basic():
    """Test basic functionality of the pairwise interaction model."""
    print("Testing pairwise interaction model basic functionality...")
    
    # Setup a small environment with a few agents
    agents = [
        Agent(agent_id=0, strategy="always_cooperate"),
        Agent(agent_id=1, strategy="always_defect"),
        Agent(agent_id=2, strategy="tit_for_tat")
    ]
    
    # Create payoff matrix (though not directly used in pairwise mode)
    payoff_matrix = create_payoff_matrix(len(agents))
    
    # Create environment with pairwise interaction mode
    env = Environment(
        agents,
        payoff_matrix,
        network_type="fully_connected",
        interaction_mode="pairwise",
        R=3, S=0, T=5, P=1  # Standard PD payoff values
    )
    
    # Run a single round
    moves, payoffs = env.run_round()
    
    print("First round moves:", moves)
    print("First round payoffs:", payoffs)
    
    # Verify expected outcomes
    # Agent 0 (always cooperate) should cooperate against both others
    assert moves[0] == "cooperate", "Always cooperate agent should cooperate"
    
    # Agent 1 (always defect) should defect against both others
    assert moves[1] == "defect", "Always defect agent should defect"
    
    # Agent 2 (tit-for-tat) should cooperate in first round
    assert moves[2] == "cooperate", "Tit-for-tat agent should cooperate in first round"
    
    # Expected payoffs (manually calculated):
    # Agent 0 (C) against Agent 1 (D): 0 points
    # Agent 0 (C) against Agent 2 (C): 3 points
    # Total for Agent 0: 3 points
    #
    # Agent 1 (D) against Agent 0 (C): 5 points
    # Agent 1 (D) against Agent 2 (C): 5 points
    # Total for Agent 1: 10 points
    #
    # Agent 2 (C) against Agent 0 (C): 3 points
    # Agent 2 (C) against Agent 1 (D): 0 points
    # Total for Agent 2: 3 points
    
    # Allow some floating point tolerance
    assert abs(payoffs[0] - 3) < 0.001, f"Expected payoff 3 for agent 0, got {payoffs[0]}"
    assert abs(payoffs[1] - 10) < 0.001, f"Expected payoff 10 for agent 1, got {payoffs[1]}"
    assert abs(payoffs[2] - 3) < 0.001, f"Expected payoff 3 for agent 2, got {payoffs[2]}"
    
    # Run a second round
    moves, payoffs = env.run_round()
    
    print("Second round moves:", moves)
    print("Second round payoffs:", payoffs)
    
    # Now TFT agent should defect against agent 1 (who defected in previous round)
    assert moves[2] == "defect", "Tit-for-tat agent should defect against previous defector"
    
    print("Basic pairwise test passed!")

def test_pairwise_against_neighborhood():
    """Compare pairwise and neighborhood interaction models with same agents."""
    print("\nComparing pairwise and neighborhood interaction models...")
    
    # Setup agents with various strategies
    agents_pairwise = [
        Agent(agent_id=0, strategy="always_cooperate"),
        Agent(agent_id=1, strategy="always_defect"),
        Agent(agent_id=2, strategy="tit_for_tat"),
        Agent(agent_id=3, strategy="tit_for_two_tats"),
        Agent(agent_id=4, strategy="q_learning", epsilon=0.1)
    ]
    
    agents_neighborhood = [
        Agent(agent_id=0, strategy="always_cooperate"),
        Agent(agent_id=1, strategy="always_defect"),
        Agent(agent_id=2, strategy="tit_for_tat"),
        Agent(agent_id=3, strategy="tit_for_two_tats"),
        Agent(agent_id=4, strategy="q_learning", epsilon=0.1)
    ]
    
    # Set fixed seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create payoff matrix
    payoff_matrix = create_payoff_matrix(len(agents_pairwise))
    
    # Create environments
    env_pairwise = Environment(
        agents_pairwise,
        payoff_matrix,
        network_type="fully_connected",
        interaction_mode="pairwise",
        R=3, S=0, T=5, P=1
    )
    
    env_neighborhood = Environment(
        agents_neighborhood,
        payoff_matrix,
        network_type="fully_connected",
        interaction_mode="neighborhood"  # Default
    )
    
    # Run simulations
    num_rounds = 20
    
    print("Running pairwise simulation...")
    results_pairwise = env_pairwise.run_simulation(num_rounds, logging_interval=5)
    
    # Reset random seeds for neighborhood model
    random.seed(42)
    np.random.seed(42)
    
    print("Running neighborhood simulation...")
    results_neighborhood = env_neighborhood.run_simulation(num_rounds, logging_interval=5)
    
    # Compare final scores
    print("\nFinal scores comparison:")
    print(f"{'Strategy':<20} {'Pairwise':<15} {'Neighborhood':<15}")
    print("-" * 50)
    
    for i in range(len(agents_pairwise)):
        strategy = agents_pairwise[i].strategy_type
        pairwise_score = agents_pairwise[i].score
        neighborhood_score = agents_neighborhood[i].score
        print(f"{strategy:<20} {pairwise_score:<15.2f} {neighborhood_score:<15.2f}")
    
    # Compare final cooperation rates
    print("\nFinal round cooperation rates:")
    
    pairwise_coop_rate = sum(1 for move in results_pairwise[-1]['moves'].values() 
                             if move == "cooperate") / len(results_pairwise[-1]['moves'])
    
    neighborhood_coop_rate = sum(1 for move in results_neighborhood[-1]['moves'].values() 
                                if move == "cooperate") / len(results_neighborhood[-1]['moves'])
    
    print(f"Pairwise model: {pairwise_coop_rate:.2f}")
    print(f"Neighborhood model: {neighborhood_coop_rate:.2f}")
    
    print("Comparison test completed!")

def test_pairwise_q_learning():
    """Test that Q-learning works properly in the pairwise interaction model."""
    print("\nTesting Q-learning in pairwise interaction model...")
    
    # Setup agents with only Q-learning
    agents = [Agent(agent_id=i, strategy="q_learning", epsilon=0.3, state_type="proportion_discretized") 
              for i in range(5)]
    
    # Create payoff matrix (not directly used)
    payoff_matrix = create_payoff_matrix(len(agents))
    
    # Create environment
    env = Environment(
        agents,
        payoff_matrix,
        network_type="fully_connected",
        interaction_mode="pairwise",
        R=3, S=0, T=5, P=1
    )
    
    # Run simulation
    num_rounds = 50
    results = env.run_simulation(num_rounds, logging_interval=10)
    
    # Check that Q-values are being updated
    q_values_sizes = [len(agent.q_values) for agent in agents]
    print(f"Q-value table sizes: {q_values_sizes}")
    
    # At least some Q-values should have been learned
    assert all(size > 0 for size in q_values_sizes), "Q-values not being learned"
    
    # The most common states should include those with high/low cooperation
    # Print a sample of Q-values from the first agent
    print("\nSample Q-values from agent 0:")
    for state, actions in list(agents[0].q_values.items())[:5]:
        print(f"State: {state}, Actions: {actions}")
    
    # Check cooperation rate trend - print cooperation rate for each round
    coop_rates = []
    for round_idx in range(0, num_rounds, 10):
        if round_idx < len(results):
            coop_rate = sum(1 for move in results[round_idx]['moves'].values() 
                          if move == "cooperate") / len(results[round_idx]['moves'])
            coop_rates.append(coop_rate)
    
    print(f"\nCooperation rates: {coop_rates}")
    
    print("Q-learning in pairwise mode test completed!")

def main():
    """Run all tests."""
    print("=== Starting pairwise interaction model tests ===\n")
    
    test_pairwise_basic()
    test_pairwise_q_learning()
    test_pairwise_against_neighborhood()
    
    print("\n=== All pairwise tests completed! ===")

if __name__ == "__main__":
    main()
