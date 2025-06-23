#!/usr/bin/env python3
"""Test script for modular Q-learning agents"""

import numpy as np
from modular_agents import (
    create_vanilla_qlearner,
    create_statistical_qlearner,
    create_softmax_qlearner,
    create_statistical_softmax_qlearner,
    create_hysteretic_statistical_qlearner
)
from final_agents import StaticAgent

def test_statistical_state():
    """Test that statistical state representation works correctly"""
    print("=== Testing Statistical State Representation ===\n")
    
    agent = create_statistical_qlearner("test_statistical")
    
    # Simulate interactions with different opponent behaviors
    print("Testing state discretization:")
    
    # Test with a very cooperative opponent
    for i in range(20):
        if i < 18:  # 90% cooperation
            agent.state_strategy.update_stats("coop_opponent", 0)  # COOPERATE
        else:
            agent.state_strategy.update_stats("coop_opponent", 1)  # DEFECT
    
    state = agent.state_strategy.get_state(agent, "coop_opponent")
    print(f"After 90% cooperation: State = {state}")
    
    # Test with a mostly defecting opponent
    for i in range(20):
        if i < 4:  # 20% cooperation
            agent.state_strategy.update_stats("defect_opponent", 0)  # COOPERATE
        else:
            agent.state_strategy.update_stats("defect_opponent", 1)  # DEFECT
    
    state = agent.state_strategy.get_state(agent, "defect_opponent")
    print(f"After 20% cooperation: State = {state}")
    
    # Test with unknown opponent
    state = agent.state_strategy.get_state(agent, "unknown_opponent")
    print(f"Unknown opponent: State = {state}")

def test_softmax_action_selection():
    """Test softmax action selection probabilities"""
    print("\n\n=== Testing Softmax Action Selection ===\n")
    
    agent = create_softmax_qlearner("test_softmax", temperature=1.0)
    
    # Set up some Q-values
    test_cases = [
        {"cooperate": 5.0, "defect": 1.0, "description": "Strong preference for cooperation"},
        {"cooperate": 1.0, "defect": 5.0, "description": "Strong preference for defection"},
        {"cooperate": 3.0, "defect": 3.0, "description": "Equal Q-values"},
        {"cooperate": 3.0, "defect": 2.8, "description": "Slight preference for cooperation"}
    ]
    
    for test_case in test_cases:
        q_values = {"cooperate": test_case["cooperate"], "defect": test_case["defect"]}
        print(f"\n{test_case['description']}:")
        print(f"Q-values: {q_values}")
        
        # Sample actions to estimate probabilities
        actions = {"cooperate": 0, "defect": 0}
        for _ in range(1000):
            action = agent.action_strategy.choose_action(q_values)
            actions[action] += 1
        
        print(f"Action distribution: Cooperate {actions['cooperate']/10:.1%}, Defect {actions['defect']/10:.1%}")

def test_temperature_decay():
    """Test that softmax temperature decays over time"""
    print("\n\n=== Testing Temperature Decay ===\n")
    
    agent = create_softmax_qlearner("test_decay", temperature=2.0)
    q_values = {"cooperate": 1.0, "defect": 0.0}
    
    print("Initial temperature:", agent.action_strategy.temperature)
    
    # Make many choices to trigger decay
    for i in range(100):
        agent.action_strategy.choose_action(q_values)
        if i % 20 == 19:  # Every 20 steps
            print(f"After {i+1} steps: temperature = {agent.action_strategy.temperature:.3f}")

def compare_all_variants():
    """Compare different agent variants against various opponents"""
    print("\n\n=== Comparing All Agent Variants ===\n")
    
    agent_creators = [
        ("Vanilla", create_vanilla_qlearner),
        ("Statistical", create_statistical_qlearner),
        ("Softmax", create_softmax_qlearner),
        ("Statistical+Softmax", create_statistical_softmax_qlearner),
        ("Hysteretic+Statistical", create_hysteretic_statistical_qlearner)
    ]
    
    opponents = [
        ("TFT", StaticAgent("TFT", "TFT")),
        ("AllD", StaticAgent("AllD", "AllD")),
        ("AllC", StaticAgent("AllC", "AllC"))
    ]
    
    print("Average scores over 200 rounds:")
    print("-" * 60)
    
    for opp_name, opponent in opponents:
        print(f"\nVs {opp_name}:")
        for agent_name, creator in agent_creators:
            agent = creator(f"{agent_name}_test")
            score = run_match(agent, opponent, rounds=200)
            print(f"  {agent_name:20s}: {score:4d}")

def run_match(agent1, agent2, rounds=100):
    """Run a match between two agents"""
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

if __name__ == "__main__":
    test_statistical_state()
    test_softmax_action_selection()
    test_temperature_decay()
    compare_all_variants()
    print("\n✓ All tests completed!")