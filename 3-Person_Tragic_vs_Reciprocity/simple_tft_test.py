#!/usr/bin/env python3
"""
Simple test to verify TFT behavior.
"""

# Test basic TFT logic
def test_tft_logic():
    """Test basic TFT cooperation logic."""
    print("Testing TFT cooperation logic:")
    print("=" * 40)
    
    # Simulate TFT agents
    class SimpleTFT:
        def __init__(self, agent_id):
            self.agent_id = agent_id
            self.opponent_last_moves = {}
        
        def choose_action(self, opponent_id, round_num):
            # First round or no history: cooperate
            if round_num == 0 or opponent_id not in self.opponent_last_moves:
                return 0  # Cooperate
            else:
                # Copy opponent's last move
                return self.opponent_last_moves[opponent_id]
        
        def update(self, opponent_id, opponent_action):
            self.opponent_last_moves[opponent_id] = opponent_action
    
    # Create 3 TFT agents
    agents = [SimpleTFT(i) for i in range(3)]
    
    # Simulate pairwise rounds
    print("Round 1 (all should cooperate):")
    for i in range(3):
        for j in range(i + 1, 3):
            action1 = agents[i].choose_action(j, 0)
            action2 = agents[j].choose_action(i, 0)
            print(f"  Agent {i} vs Agent {j}: {'C' if action1 == 0 else 'D'} vs {'C' if action2 == 0 else 'D'}")
            
            # Update memories
            agents[i].update(j, action2)
            agents[j].update(i, action1)
    
    print("\nRound 2 (all should still cooperate):")
    for i in range(3):
        for j in range(i + 1, 3):
            action1 = agents[i].choose_action(j, 1)
            action2 = agents[j].choose_action(i, 1)
            print(f"  Agent {i} vs Agent {j}: {'C' if action1 == 0 else 'D'} vs {'C' if action2 == 0 else 'D'}")
            
            agents[i].update(j, action2)
            agents[j].update(i, action1)
    
    # Test with one defector
    print("\nRound 3 (inject one defection):")
    # Force agent 2 to defect against agent 0
    agents[0].update(2, 1)  # Agent 0 sees agent 2 defect
    
    for i in range(3):
        for j in range(i + 1, 3):
            action1 = agents[i].choose_action(j, 2)
            action2 = agents[j].choose_action(i, 2)
            print(f"  Agent {i} vs Agent {j}: {'C' if action1 == 0 else 'D'} vs {'C' if action2 == 0 else 'D'}")
            
            agents[i].update(j, action2)
            agents[j].update(i, action1)
    
    print("\nExpected behavior:")
    print("- Round 1: All cooperate (C vs C)")
    print("- Round 2: All cooperate (C vs C)")
    print("- Round 3: Agent 0 defects against Agent 2 (reciprocating)")

if __name__ == "__main__":
    test_tft_logic()