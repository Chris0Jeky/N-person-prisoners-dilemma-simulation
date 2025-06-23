#!/usr/bin/env python3
"""Test script for Hysteretic Q-learning implementation"""

import numpy as np
from final_agents import HystereticQLearner, StaticAgent, PairwiseAdaptiveQLearner
from config import HYSTERETIC_PARAMS, VANILLA_PARAMS

def test_hysteretic_update():
    """Test that hysteretic updates work correctly"""
    print("=== Testing Hysteretic Q-Learning Updates ===")
    
    # Create agent with specific parameters
    params = {
        'lr': 0.2,     # High learning rate for positive
        'beta': 0.02,  # Low learning rate for negative
        'df': 0.9,
        'eps': 0.0    # No exploration for testing
    }
    
    agent = HystereticQLearner("test_hysteretic", params)
    
    # Simulate some interactions
    print("\nInitial Q-values should be 0:")
    state = agent._get_state("opponent1")
    agent.q_tables["opponent1"] = {state: {'cooperate': 0.0, 'defect': 0.0}}
    print(f"Q-values: {agent.q_tables['opponent1'][state]}")
    
    # Test positive update (good news)
    print("\nPositive update test (reward=3, should use lr=0.2):")
    agent.last_contexts["opponent1"] = {'state': state, 'action': 'cooperate'}
    agent.record_pairwise_outcome("opponent1", 0, 0, 3)  # Both cooperated
    print(f"After positive update: {agent.q_tables['opponent1'][state]['cooperate']:.3f}")
    print(f"Expected: {0.0 + 0.2 * (3 + 0.9 * 0 - 0.0):.3f} = 0.600")
    
    # Test negative update (bad news)
    print("\nNegative update test (reward=0, should use beta=0.02):")
    agent.last_contexts["opponent1"] = {'state': state, 'action': 'cooperate'}
    agent.record_pairwise_outcome("opponent1", 0, 1, 0)  # I cooperated, they defected
    print(f"After negative update: {agent.q_tables['opponent1'][state]['cooperate']:.3f}")
    # Current Q is 0.6, target is 0, delta is -0.6, so: 0.6 + 0.02 * (-0.6) = 0.588
    print(f"Expected: {0.6 + 0.02 * (0 + 0.9 * 0.6 - 0.6):.3f} = 0.588")

def compare_hysteretic_vs_vanilla():
    """Compare Hysteretic vs Vanilla Q-learning against different opponents"""
    print("\n\n=== Comparing Hysteretic vs Vanilla Q-Learning ===")
    
    scenarios = [
        ("vs AllD", StaticAgent("AllD", "AllD")),
        ("vs TFT", StaticAgent("TFT", "TFT")),
        ("vs AllC", StaticAgent("AllC", "AllC"))
    ]
    
    for scenario_name, opponent in scenarios:
        print(f"\n{scenario_name}:")
        
        # Test Hysteretic
        hysteretic = HystereticQLearner("Hysteretic", HYSTERETIC_PARAMS)
        h_score = run_quick_match(hysteretic, opponent, rounds=100)
        
        # Test Vanilla
        vanilla = PairwiseAdaptiveQLearner("Vanilla", VANILLA_PARAMS)
        v_score = run_quick_match(vanilla, opponent, rounds=100)
        
        print(f"  Hysteretic score: {h_score}")
        print(f"  Vanilla score: {v_score}")
        print(f"  Hysteretic advantage: {h_score - v_score:+d}")

def run_quick_match(agent1, agent2, rounds=100):
    """Run a quick match between two agents"""
    agent1.reset()
    agent2.reset()
    
    for _ in range(rounds):
        move1 = agent1.choose_pairwise_action(agent2.agent_id)
        move2 = agent2.choose_pairwise_action(agent1.agent_id)
        
        # Calculate payoffs
        payoffs = {(0, 0): (3, 3), (0, 1): (0, 5), (1, 0): (5, 0), (1, 1): (1, 1)}
        p1, p2 = payoffs[(move1, move2)]
        
        agent1.record_pairwise_outcome(agent2.agent_id, move1, move2, p1)
        agent2.record_pairwise_outcome(agent1.agent_id, move2, move1, p2)
    
    return agent1.total_score

def test_neighborhood_hysteretic():
    """Test hysteretic learning in neighborhood games"""
    print("\n\n=== Testing Neighborhood Hysteretic Learning ===")
    
    agent = HystereticQLearner("test", HYSTERETIC_PARAMS)
    
    # Test different cooperation ratios
    for coop_ratio in [0.2, 0.5, 0.8]:
        print(f"\nCooperation ratio: {coop_ratio}")
        for _ in range(5):
            action = agent.choose_neighborhood_action(coop_ratio)
            # Simulate reward based on action and group cooperation
            if action == 0:  # Cooperate
                reward = 0 + 3 * coop_ratio
            else:  # Defect
                reward = 1 + 4 * coop_ratio
            
            agent.record_neighborhood_outcome(coop_ratio, reward)
        
        state = agent._get_neighborhood_state(coop_ratio)
        if state in agent.neighborhood_q_table:
            print(f"  Q-values: Coop={agent.neighborhood_q_table[state]['cooperate']:.3f}, "
                  f"Defect={agent.neighborhood_q_table[state]['defect']:.3f}")

if __name__ == "__main__":
    test_hysteretic_update()
    compare_hysteretic_vs_vanilla()
    test_neighborhood_hysteretic()
    print("\n✓ All tests completed!")