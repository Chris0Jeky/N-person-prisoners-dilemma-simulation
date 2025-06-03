"""
Test script to verify correct pairwise TFT behavior
===================================================
This script runs a simple test to ensure TFT agents maintain cooperation
with each other while defecting against Always Defect in pairwise mode.
"""

from unified_simulation import (
    Agent, UnifiedSimulation, InteractionMode, 
    Strategy, Action
)


def test_pairwise_tft():
    """Test that TFT agents cooperate with each other but defect against AllD."""
    
    print("Testing Pairwise TFT Behavior")
    print("="*60)
    
    # Create agents with no exploration
    agents = [
        Agent(0, "TFT1", Strategy.TIT_FOR_TAT, exploration_rate=0.0),
        Agent(1, "TFT2", Strategy.TIT_FOR_TAT, exploration_rate=0.0),
        Agent(2, "AllD", Strategy.ALWAYS_DEFECT, exploration_rate=0.0)
    ]
    
    # Create simulation
    sim = UnifiedSimulation(agents, InteractionMode.PAIRWISE)
    
    # Run a few rounds and track specific interactions
    print("\nRunning 10 rounds...")
    
    for round_num in range(10):
        print(f"\nRound {round_num + 1}:")
        
        # Get planned actions before the round
        planned_actions = {}
        for agent in agents:
            opponent_ids = [other.id for other in agents if other.id != agent.id]
            actions = agent.choose_actions_for_current_round_pairwise(opponent_ids)
            planned_actions[agent.id] = actions
            
            print(f"  {agent.name} plans:")
            for opp_id, action in actions.items():
                opp_name = next(a.name for a in agents if a.id == opp_id)
                print(f"    vs {opp_name}: {action.value}")
        
        # Run the round
        round_result = sim._run_pairwise_round(round_num, verbose=False)
        
        # Analyze interactions
        print("\n  Interactions:")
        for interaction in round_result['interactions']:
            print(f"    {interaction['agent1']} vs {interaction['agent2']}: "
                  f"{interaction['action1']} vs {interaction['action2']}")
        
        # Check if TFT agents are behaving correctly
        tft_vs_tft_cooperating = all(
            interaction['action1'] == 'C' and interaction['action2'] == 'C'
            for interaction in round_result['interactions']
            if 'TFT' in interaction['agent1'] and 'TFT' in interaction['agent2']
        )
        
        tft_vs_alld_defecting = all(
            (interaction['action1'] == 'D' if 'TFT' in interaction['agent1'] else True) and
            (interaction['action2'] == 'D' if 'TFT' in interaction['agent2'] else True)
            for interaction in round_result['interactions']
            if ('TFT' in interaction['agent1'] and 'AllD' in interaction['agent2']) or
               ('AllD' in interaction['agent1'] and 'TFT' in interaction['agent2'])
        ) if round_num > 0 else True  # After first round, TFT should defect against AllD
        
        print(f"\n  TFT vs TFT cooperating: {tft_vs_tft_cooperating}")
        print(f"  TFT vs AllD defecting (after round 1): {tft_vs_alld_defecting if round_num > 0 else 'N/A'}")
    
    # Calculate final cooperation rates
    print("\n" + "="*60)
    print("Final Results:")
    
    # Count cooperation in each dyad type
    tft_tft_coops = 0
    tft_tft_total = 0
    tft_alld_coops = 0
    tft_alld_total = 0
    
    for agent in agents:
        if 'TFT' in agent.name:
            for opp_id, history in agent.pairwise_history.items():
                opp_name = next(a.name for a in agents if a.id == opp_id)
                coop_count = sum(1 for action in history if action == Action.COOPERATE)
                
                if 'TFT' in opp_name:
                    tft_tft_coops += coop_count
                    tft_tft_total += len(history)
                elif 'AllD' in opp_name:
                    tft_alld_coops += coop_count
                    tft_alld_total += len(history)
    
    print(f"TFT vs TFT cooperation rate: {tft_tft_coops/tft_tft_total:.2%} ({tft_tft_coops}/{tft_tft_total})")
    print(f"TFT vs AllD cooperation rate: {tft_alld_coops/tft_alld_total:.2%} ({tft_alld_coops}/{tft_alld_total})")
    
    # Test N-Person mode for comparison
    print("\n" + "="*60)
    print("Testing N-Person Mode for Comparison")
    print("="*60)
    
    # Create new agents for N-person test
    agents_np = [
        Agent(0, "TFT1", Strategy.TIT_FOR_TAT, exploration_rate=0.0),
        Agent(1, "TFT2", Strategy.TIT_FOR_TAT, exploration_rate=0.0),
        Agent(2, "AllD", Strategy.ALWAYS_DEFECT, exploration_rate=0.0)
    ]
    
    sim_np = UnifiedSimulation(agents_np, InteractionMode.N_PERSON)
    
    # Run 10 rounds
    for round_num in range(10):
        round_result = sim_np._run_nperson_round(round_num, verbose=(round_num < 3))
    
    # Calculate cooperation rates
    print("\nN-Person Final Results:")
    for agent in agents_np:
        if agent.my_history:
            coop_rate = sum(1 for action in agent.my_history if action == Action.COOPERATE) / len(agent.my_history)
            print(f"{agent.name} cooperation rate: {coop_rate:.2%}")
    
    print("\n" + "="*60)
    print("Expected Results:")
    print("- Pairwise: TFT agents should maintain ~100% cooperation with each other")
    print("            TFT agents should have ~10% cooperation with AllD (only first round)")
    print("- N-Person: TFT agents should quickly drop to 0% cooperation")
    print("="*60)


if __name__ == "__main__":
    test_pairwise_tft()
