#!/usr/bin/env python3
"""
Test pairwise TFT to see if it's working correctly.
"""

import sys
from pathlib import Path

# Add npd_simulator to path
sys.path.insert(0, str(Path(__file__).parent))

from npd_simulator.agents.strategies.tft import TFTAgent
from npd_simulator.core.game.pairwise_game import PairwiseGame

def test_pairwise_tft():
    """Test if TFT agents cooperate in pairwise game."""
    print("Testing pairwise TFT cooperation...")
    
    # Create 3 TFT agents
    agents = [TFTAgent(i) for i in range(3)]
    
    # Create pairwise game
    game = PairwiseGame(
        num_agents=3,
        rounds_per_pair=10,
        num_episodes=1
    )
    
    # Play one episode
    cooperation_count = 0
    total_actions = 0
    
    for episode in range(1):
        # Reset agents for new episode
        for agent in agents:
            agent.reset_memory()
        
        # Play all pairwise rounds
        for round_num in range(10):
            print(f"\nRound {round_num + 1}:")
            
            # Each pair plays
            for i in range(3):
                for j in range(i + 1, 3):
                    # Get actions
                    _, action1 = agents[i].choose_action(j, round_num)
                    _, action2 = agents[j].choose_action(i, round_num)
                    
                    # Play round
                    payoff1, payoff2 = game.play_pairwise_round(
                        agents[i], agents[j],
                        action1, action2,
                        round_num, episode
                    )
                    
                    # Update agent memories
                    agents[i].update(j, action2, payoff1)
                    agents[j].update(i, action1, payoff2)
                    
                    # Track cooperation
                    cooperation_count += (0 if action1 else 1) + (0 if action2 else 1)
                    total_actions += 2
                    
                    print(f"  Agent {i} vs Agent {j}: "
                          f"{'C' if action1 == 0 else 'D'} vs {'C' if action2 == 0 else 'D'}, "
                          f"payoffs: {payoff1:.1f}, {payoff2:.1f}")
    
    cooperation_rate = cooperation_count / total_actions if total_actions > 0 else 0
    print(f"\nOverall cooperation rate: {cooperation_rate:.2%}")
    print(f"Cooperation count: {cooperation_count}/{total_actions}")
    
    # Check tournament scores
    print("\nTournament scores:")
    for i, score in game.tournament_scores.items():
        coop_count = game.tournament_cooperation_counts[i]
        defect_count = game.tournament_defection_counts[i]
        total = coop_count + defect_count
        coop_rate = coop_count / total if total > 0 else 0
        print(f"  Agent {i}: score={score:.1f}, cooperation={coop_count}/{total} ({coop_rate:.2%})")

if __name__ == "__main__":
    test_pairwise_tft()