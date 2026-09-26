#!/usr/bin/env python3
"""Comparative analysis of Q-learning performance across different scenarios"""

import numpy as np
import matplotlib.pyplot as plt
from final_agents import StaticAgent, PairwiseAdaptiveQLearner
from config import VANILLA_PARAMS, ADAPTIVE_PARAMS, SIMULATION_CONFIG

def quick_pairwise_test(agent1, agent2, rounds=100):
    """Quick pairwise test between two agents"""
    agent1.reset()
    agent2.reset()
    
    scores = {agent1.agent_id: [], agent2.agent_id: []}
    coop_rates = {agent1.agent_id: [], agent2.agent_id: []}
    
    for _ in range(rounds):
        move1 = agent1.choose_pairwise_action(agent2.agent_id)
        move2 = agent2.choose_pairwise_action(agent1.agent_id)
        
        # Calculate payoffs
        payoffs = {(0, 0): (3, 3), (0, 1): (0, 5), (1, 0): (5, 0), (1, 1): (1, 1)}
        p1, p2 = payoffs[(move1, move2)]
        
        agent1.record_pairwise_outcome(agent2.agent_id, move1, move2, p1)
        agent2.record_pairwise_outcome(agent1.agent_id, move2, move1, p2)
        
        scores[agent1.agent_id].append(agent1.total_score)
        scores[agent2.agent_id].append(agent2.total_score)
        coop_rates[agent1.agent_id].append(1 - move1)
        coop_rates[agent2.agent_id].append(1 - move2)
    
    return scores, coop_rates

def plot_quick_comparison():
    """Generate a quick visual comparison of agent behaviors"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Q-Learner Performance Against Different Strategies", fontsize=16)
    
    opponents = [
        ("AllC", StaticAgent("AllC", "AllC")),
        ("AllD", StaticAgent("AllD", "AllD")),
        ("Random", StaticAgent("Random", "Random")),
        ("TFT", StaticAgent("TFT", "TFT")),
        ("TFT-E", StaticAgent("TFT-E", "TFT-E", error_rate=0.1))
    ]
    
    for idx, (opp_name, opponent) in enumerate(opponents):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Test Vanilla QL
        vanilla_ql = PairwiseAdaptiveQLearner("Vanilla_QL", VANILLA_PARAMS)
        v_scores, v_coop = quick_pairwise_test(vanilla_ql, opponent, rounds=200)
        
        # Test Adaptive QL
        adaptive_ql = PairwiseAdaptiveQLearner("Adaptive_QL", ADAPTIVE_PARAMS)
        a_scores, a_coop = quick_pairwise_test(adaptive_ql, opponent, rounds=200)
        
        # Plot cooperation rates
        ax.plot(v_coop["Vanilla_QL"], label="Vanilla QL", color='blue', alpha=0.7)
        ax.plot(a_coop["Adaptive_QL"], label="Adaptive QL", color='green', alpha=0.7)
        ax.plot(v_coop[opponent.agent_id], label=opp_name, color='red', linestyle='--', alpha=0.7)
        
        ax.set_title(f"vs {opp_name}")
        ax.set_xlabel("Round")
        ax.set_ylabel("Cooperation Rate")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()
    
    # Clear the last subplot if we have an odd number
    if len(opponents) % 3 != 0:
        fig.delaxes(axes.flatten()[-1])
    
    plt.tight_layout()
    plt.savefig("final_comparison_charts/quick_behavior_comparison.png", dpi=150)
    plt.close()
    
    print("Quick comparison plot saved!")

def analyze_adaptation_effectiveness():
    """Analyze how well the adaptive agent adapts to different opponents"""
    print("\n=== Adaptation Effectiveness Analysis ===")
    
    opponents = {
        "AllC": StaticAgent("AllC", "AllC"),
        "AllD": StaticAgent("AllD", "AllD"),
        "TFT": StaticAgent("TFT", "TFT")
    }
    
    for opp_name, opponent in opponents.items():
        print(f"\nAgainst {opp_name}:")
        
        # Test adaptive agent
        adaptive_ql = PairwiseAdaptiveQLearner("Adaptive_QL", ADAPTIVE_PARAMS)
        scores, coop = quick_pairwise_test(adaptive_ql, opponent, rounds=500)
        
        # Analyze adaptation
        early_coop = np.mean(coop["Adaptive_QL"][:100])
        late_coop = np.mean(coop["Adaptive_QL"][400:])
        
        print(f"  Early cooperation rate: {early_coop:.2f}")
        print(f"  Late cooperation rate: {late_coop:.2f}")
        print(f"  Final score: {scores['Adaptive_QL'][-1]}")
        print(f"  Final learning rate: {adaptive_ql.learning_rates[opponent.agent_id]:.3f}")
        print(f"  Final epsilon: {adaptive_ql.epsilons[opponent.agent_id]:.3f}")

if __name__ == "__main__":
    import os
    os.makedirs("final_comparison_charts", exist_ok=True)
    
    print("Running comparative analysis...")
    plot_quick_comparison()
    analyze_adaptation_effectiveness()
    print("\nAnalysis complete!")