"""
Simple test of enhanced Q-learning to verify basic functionality
"""

from enhanced_qlearning_agents import EnhancedQLearningAgent
from extended_agents import ExtendedNPersonAgent, QLearningNPersonWrapper
from main_neighbourhood import NPERSON_COOPERATE

def test_basic_enhanced_qlearning():
    """Test basic enhanced Q-learning functionality."""
    print("Testing Enhanced Q-Learning Basic Functionality...")
    
    # Test different configurations
    configs = [
        {"name": "baseline", "exclude_self": False, "epsilon": 0.1},
        {"name": "exclude_self", "exclude_self": True, "epsilon": 0.1},
        {"name": "decay_epsilon", "exclude_self": False, "epsilon": 0.1, "epsilon_decay": 0.99, "epsilon_min": 0.01},
        {"name": "all_improvements", "exclude_self": True, "epsilon": 0.1, "epsilon_decay": 0.99, "epsilon_min": 0.01, "opponent_modeling": True}
    ]
    
    # Test each configuration
    for config in configs:
        name = config.pop("name")
        print(f"\n--- Testing {name} ---")
        
        # Create agents
        ql_agent = EnhancedQLearningAgent(0, **config)
        ql_wrapper = QLearningNPersonWrapper(0, ql_agent)
        allc_agent = ExtendedNPersonAgent(1, "AllC", exploration_rate=0.01)
        allc_agent2 = ExtendedNPersonAgent(2, "AllC", exploration_rate=0.01)
        
        agents = [ql_wrapper, allc_agent, allc_agent2]
        
        # Run short simulation
        prev_coop_ratio = None
        for round_num in range(100):
            # Collect actions
            actions = {}
            for agent in agents:
                _, actual = agent.choose_action(prev_coop_ratio, round_num)
                actions[agent.agent_id] = actual
            
            # Calculate cooperation
            num_coops = sum(1 for action in actions.values() if action == NPERSON_COOPERATE)
            prev_coop_ratio = num_coops / len(agents)
            
            # Calculate payoffs
            for agent in agents:
                my_action = actions[agent.agent_id]
                others_coop = num_coops - (1 if my_action == NPERSON_COOPERATE else 0)
                
                if my_action == NPERSON_COOPERATE:
                    payoff = 0 + 3 * (others_coop / (len(agents) - 1))
                else:
                    payoff = 1 + 4 * (others_coop / (len(agents) - 1))
                
                agent.record_round_outcome(my_action, payoff)
        
        # Report results
        ql_coop_rate = ql_wrapper.get_cooperation_rate()
        ql_score = ql_wrapper.total_score
        print(f"QL Agent - Cooperation Rate: {ql_coop_rate:.3f}, Score: {ql_score:.1f}")
        
        # Show Q-table if available
        if hasattr(ql_wrapper.qlearning_agent, 'q_table'):
            print("Q-Table:")
            for state, values in ql_wrapper.qlearning_agent.q_table.items():
                print(f"  {state}: C={values['cooperate']:.2f}, D={values['defect']:.2f}")
        
        print(f"Final epsilon: {ql_wrapper.qlearning_agent.epsilon:.4f}")


if __name__ == "__main__":
    test_basic_enhanced_qlearning()
    print("\nBasic test complete!")