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
import sys
import os

# Add the project root to the Python path to fix imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from npdl.core.agents import Agent
from npdl.core.environment import Environment
from npdl.core.utils import create_payoff_matrix, get_pairwise_payoffs

# Note: The TitForTat implementation in agents.py handles pairwise mode correctly.
# It checks for 'specific_opponent_moves' first, then falls back to 'opponent_coop_proportion'.
# In pairwise mode with opponent_coop_proportion, it uses a threshold of 0.99 to decide.

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
    
    # Run the first round (everyone uses their default strategy)
    moves, payoffs = env.run_round()
    
    print("First round moves:", moves)
    print("First round payoffs:", payoffs)
    
    # Verify first round outcomes
    assert moves[0] == "cooperate", "Always cooperate agent should cooperate"
    assert moves[1] == "defect", "Always defect agent should defect"
    assert moves[2] == "cooperate", "Tit-for-tat agent should cooperate in first round"
    
    # Run the second round
    # Note: In the second round, the TFT agent has now observed agent 1's defection
    # However, it may not respond correctly in the current implementation
    moves, payoffs = env.run_round()
    
    print("Second round moves:", moves)
    print("Second round payoffs:", payoffs)
    
    # Check TFT agent's memory and cooperation proportion
    tft_agent = agents[2]
    neighbor_moves = tft_agent.memory[-1]['neighbor_moves']
    coop_prop = neighbor_moves.get('opponent_coop_proportion', 1.0)
    
    # Print debugging information
    print(f"TFT agent memory: {tft_agent.memory[-1]}")
    print(f"Opponent cooperation proportion: {coop_prop}")
    
    # The TFT implementation uses a threshold of 0.99, so with coop_prop = 0.5,
    # it should defect in the next round
    assert coop_prop == 0.5, "Cooperation proportion should be 0.5 (1 of 2 opponents cooperated)"
    
    # In the third round, TFT should defect because coop_prop < 0.99
    moves, payoffs = env.run_round()
    print("Third round moves:", moves)
    assert moves[2] == "defect", "Tit-for-tat agent should defect when cooperation proportion < 0.99"
    
    print("Basic pairwise test passed with manual override!")

def test_explicit_tft_behavior():
    """Test TFT behavior directly with explicit memory manipulation."""
    print("\nTesting explicit TFT behavior in pairwise mode...")
    
    # Create a TFT agent
    tft_agent = Agent(agent_id=0, strategy="tit_for_tat")
    
    # First action with no memory
    move = tft_agent.choose_move([])
    assert move == "cooperate", "First move should be cooperate"
    
    # Now create a memory entry with different cooperation proportions
    
    # Case 1: All opponents cooperated (1.0)
    tft_agent.memory.clear()
    tft_agent.update_memory("cooperate", {"opponent_coop_proportion": 1.0}, 3.0)
    move = tft_agent.choose_move([])
    print(f"TFT with all cooperators (1.0): {move}")
    assert move == "cooperate", "TFT should cooperate when all opponents cooperated"
    
    # Case 2: Some opponents defected (0.5)
    tft_agent.memory.clear()
    tft_agent.update_memory("cooperate", {"opponent_coop_proportion": 0.5}, 1.5)
    move = tft_agent.choose_move([])
    print(f"TFT with some defectors (0.5): {move}")
    assert move == "defect", "TFT should defect when cooperation proportion < 0.99"
    
    # Case 3: All opponents defected (0.0)
    tft_agent.memory.clear()
    tft_agent.update_memory("cooperate", {"opponent_coop_proportion": 0.0}, 0.0)
    move = tft_agent.choose_move([])
    print(f"TFT with all defectors (0.0): {move}")
    assert move == "defect", "TFT should defect when all opponents defected"
    
    # Case 4: Almost all cooperated (0.99)
    tft_agent.memory.clear()
    tft_agent.update_memory("cooperate", {"opponent_coop_proportion": 0.99}, 2.97)
    move = tft_agent.choose_move([])
    print(f"TFT with 99% cooperators (0.99): {move}")
    assert move == "cooperate", "TFT should cooperate when cooperation proportion >= 0.99"
    
    # Case 5: Test with specific_opponent_moves (preferred format)
    tft_agent.memory.clear()
    tft_agent.update_memory("cooperate", {
        "specific_opponent_moves": {"agent1": "cooperate", "agent2": "defect"},
        "opponent_coop_proportion": 0.5
    }, 1.5)
    move = tft_agent.choose_move([])
    print(f"TFT with specific moves (one defector): {move}")
    assert move == "defect", "TFT should defect when ANY specific opponent defected"
    
    print("Explicit TFT behavior test passed!")

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
    test_explicit_tft_behavior()
    test_pairwise_q_learning()
    test_pairwise_against_neighborhood()
    
    print("\n=== All pairwise tests completed! ===")

if __name__ == "__main__":
    main()
