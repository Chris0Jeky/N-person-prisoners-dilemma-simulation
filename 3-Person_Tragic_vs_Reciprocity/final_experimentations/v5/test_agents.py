#!/usr/bin/env python3
"""Simple test script to verify agent behaviors"""

from final_agents import StaticAgent, PairwiseAdaptiveQLearner, NeighborhoodAdaptiveQLearner
from config import VANILLA_PARAMS, ADAPTIVE_PARAMS

def test_static_agents():
    print("\n=== Testing Static Agents ===")
    
    # Test AllC
    allc = StaticAgent("test_allc", "AllC")
    print(f"AllC pairwise action: {allc.choose_pairwise_action('opponent1')}")
    print(f"AllC neighborhood action (coop_ratio=0.2): {allc.choose_neighborhood_action(0.2)}")
    
    # Test AllD
    alld = StaticAgent("test_alld", "AllD")
    print(f"\nAllD pairwise action: {alld.choose_pairwise_action('opponent1')}")
    print(f"AllD neighborhood action (coop_ratio=0.8): {alld.choose_neighborhood_action(0.8)}")
    
    # Test Random (run multiple times to see randomness)
    random_agent = StaticAgent("test_random", "Random")
    print(f"\nRandom pairwise actions (5 samples): ", end="")
    print([random_agent.choose_pairwise_action('opponent1') for _ in range(5)])
    
    # Test TFT
    tft = StaticAgent("test_tft", "TFT")
    print(f"\nTFT initial action: {tft.choose_pairwise_action('opponent1')}")
    tft.record_pairwise_outcome('opponent1', 0, 1, 0)  # opponent defected
    print(f"TFT after opponent defected: {tft.choose_pairwise_action('opponent1')}")
    
    # Test TFT-E (with errors)
    tfte = StaticAgent("test_tfte", "TFT-E", error_rate=0.1)
    print(f"\nTFT-E actions (10 samples, should see ~1 error): ", end="")
    actions = []
    for _ in range(10):
        tfte.opponent_last_moves['opponent1'] = 0  # opponent cooperated
        actions.append(tfte.choose_pairwise_action('opponent1'))
    print(actions)
    print(f"Expected ~1 defection due to 10% error rate, got {actions.count(1)}")

def test_qlearners():
    print("\n\n=== Testing Q-Learners ===")
    
    # Test Vanilla Q-learner
    vanilla_ql = PairwiseAdaptiveQLearner("vanilla_test", VANILLA_PARAMS)
    print(f"Vanilla QL initial action: {vanilla_ql.choose_pairwise_action('opponent1')}")
    print(f"Vanilla QL epsilon: {vanilla_ql.epsilons['opponent1']}")
    
    # Test Adaptive Q-learner
    adaptive_ql = PairwiseAdaptiveQLearner("adaptive_test", ADAPTIVE_PARAMS)
    print(f"\nAdaptive QL initial action: {adaptive_ql.choose_pairwise_action('opponent1')}")
    print(f"Adaptive QL initial epsilon: {adaptive_ql.epsilons['opponent1']}")
    
    # Simulate some good outcomes to see adaptation
    for i in range(25):
        adaptive_ql.record_pairwise_outcome('opponent1', 0, 0, 3)  # mutual cooperation
    
    print(f"Adaptive QL epsilon after 25 good outcomes: {adaptive_ql.epsilons['opponent1']}")
    print("(Should be lower due to adaptation)")

if __name__ == "__main__":
    test_static_agents()
    test_qlearners()
    print("\n✓ All tests completed!")