"""
Quick Test Script for All Q-Learning Changes

This script provides a fast comparison of:
1. Original Simple QL and NPDL QL 
2. Enhanced QL with individual improvements
3. Enhanced QL with best combinations

Focuses on the key exploitation scenario: QL vs AllC vs AllC
"""

import random
from qlearning_agents import SimpleQLearningAgent, NPDLQLearningAgent
from enhanced_qlearning_agents import EnhancedQLearningAgent
from extended_agents import ExtendedNPersonAgent, QLearningNPersonWrapper
from main_neighbourhood import NPERSON_COOPERATE


def run_exploitation_test(agent_name, ql_agent, num_rounds=1000):
    """Run exploitation test with given Q-learning agent."""
    
    # Create wrapper and opponents
    ql_wrapper = QLearningNPersonWrapper(0, ql_agent)
    allc_agent1 = ExtendedNPersonAgent(1, "AllC", exploration_rate=0.01)
    allc_agent2 = ExtendedNPersonAgent(2, "AllC", exploration_rate=0.01)
    
    agents = [ql_wrapper, allc_agent1, allc_agent2]
    
    # Run simulation
    prev_coop_ratio = None
    for round_num in range(num_rounds):
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
    
    # Calculate results
    ql_coop_rate = ql_wrapper.get_cooperation_rate()
    ql_score = ql_wrapper.total_score
    exploitation_rate = (1 - ql_coop_rate) * 100
    final_epsilon = getattr(ql_agent, 'epsilon', 'N/A')
    
    return {
        'name': agent_name,
        'cooperation_rate': ql_coop_rate,
        'score': ql_score,
        'exploitation_rate': exploitation_rate,
        'final_epsilon': final_epsilon
    }


def main():
    """Run comprehensive comparison of all Q-learning variants."""
    
    print("QUICK TEST: ALL Q-LEARNING CHANGES")
    print("="*50)
    print("Testing QL vs AllC vs AllC (1000 rounds)")
    print("="*50)
    
    random.seed(42)  # Reproducibility
    
    # Define all test configurations
    test_configs = [
        # Original implementations
        ("Original Simple QL", "simple", {}),
        ("Original NPDL QL", "npdl", {}),
        
        # Enhanced baseline
        ("Enhanced Baseline", "enhanced", {
            "exclude_self": False,
            "epsilon": 0.1,
            "epsilon_decay": 1.0,
            "opponent_modeling": False,
            "state_type": "basic"
        }),
        
        # Individual improvements
        ("+ Exclude Self", "enhanced", {
            "exclude_self": True,
            "epsilon": 0.1,
            "epsilon_decay": 1.0,
            "opponent_modeling": False,
            "state_type": "basic"
        }),
        
        ("+ Epsilon Decay", "enhanced", {
            "exclude_self": False,
            "epsilon": 0.1,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01,
            "opponent_modeling": False,
            "state_type": "basic"
        }),
        
        ("+ Opponent Modeling", "enhanced", {
            "exclude_self": False,
            "epsilon": 0.1,
            "epsilon_decay": 1.0,
            "opponent_modeling": True,
            "state_type": "basic"
        }),
        
        ("+ Fine State", "enhanced", {
            "exclude_self": False,
            "epsilon": 0.1,
            "epsilon_decay": 1.0,
            "opponent_modeling": False,
            "state_type": "fine"
        }),
        
        # Best combinations
        ("Exclude Self + Decay", "enhanced", {
            "exclude_self": True,
            "epsilon": 0.1,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01,
            "opponent_modeling": False,
            "state_type": "basic"
        }),
        
        ("Optimal Exploitation", "enhanced", {
            "exclude_self": True,
            "epsilon": 0.1,
            "epsilon_decay": 0.99,
            "epsilon_min": 0.001,
            "opponent_modeling": False,
            "state_type": "basic"
        }),
        
        ("All Improvements", "enhanced", {
            "exclude_self": True,
            "epsilon": 0.1,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.001,
            "opponent_modeling": True,
            "state_type": "fine"
        })
    ]
    
    results = []
    
    # Run tests
    for agent_name, agent_type, config in test_configs:
        print(f"\nTesting: {agent_name}")
        
        # Create appropriate agent
        if agent_type == "simple":
            ql_agent = SimpleQLearningAgent(0, learning_rate=0.15, epsilon=0.1)
        elif agent_type == "npdl":
            ql_agent = NPDLQLearningAgent(0, learning_rate=0.15, epsilon=0.1, 
                                        state_type="proportion_discretized")
        elif agent_type == "enhanced":
            ql_agent = EnhancedQLearningAgent(0, **config)
        
        # Run test
        result = run_exploitation_test(agent_name, ql_agent)
        results.append(result)
        
        print(f"  Cooperation: {result['cooperation_rate']:.3f}")
        print(f"  Score: {result['score']:.0f}")
        print(f"  Exploitation: {result['exploitation_rate']:.1f}%")
    
    # Summary table
    print("\n" + "="*70)
    print("COMPLETE RESULTS SUMMARY")
    print("="*70)
    print(f"{'Configuration':<25} {'Coop Rate':<10} {'Score':<8} {'Exploit %':<10} {'Final ε':<8}")
    print("-"*70)
    
    for result in results:
        print(f"{result['name']:<25} {result['cooperation_rate']:<10.3f} "
              f"{result['score']:<8.0f} {result['exploitation_rate']:<10.1f}% "
              f"{result['final_epsilon']}")
    
    # Find best and worst
    best_exploit = max(results, key=lambda x: x['exploitation_rate'])
    worst_exploit = min(results, key=lambda x: x['exploitation_rate'])
    best_score = max(results, key=lambda x: x['score'])
    
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    print(f"Best Exploitation: {best_exploit['name']} ({best_exploit['exploitation_rate']:.1f}%)")
    print(f"Worst Exploitation: {worst_exploit['name']} ({worst_exploit['exploitation_rate']:.1f}%)")
    print(f"Highest Score: {best_score['name']} ({best_score['score']:.0f})")
    
    improvement = best_exploit['exploitation_rate'] - worst_exploit['exploitation_rate']
    print(f"Total Improvement: +{improvement:.1f}% exploitation")
    
    theoretical_max = 1000 * (1 + 4 * 1.0)  # Always defect vs 2 cooperators
    best_efficiency = (best_score['score'] / theoretical_max) * 100
    print(f"Best Efficiency: {best_efficiency:.1f}% of theoretical maximum")
    
    print(f"\nTheoretical Maximum: 100% exploitation, score {theoretical_max:.0f}")
    
    # Show improvement categories
    print("\n" + "="*70)
    print("IMPROVEMENT ANALYSIS")
    print("="*70)
    
    # Find baseline enhanced for comparison
    baseline = next((r for r in results if "Enhanced Baseline" in r['name']), None)
    if baseline:
        print(f"Enhanced Baseline: {baseline['exploitation_rate']:.1f}% exploitation")
        
        improvements = [
            ("Exclude Self", next((r for r in results if r['name'] == "+ Exclude Self"), None)),
            ("Epsilon Decay", next((r for r in results if r['name'] == "+ Epsilon Decay"), None)),
            ("Opponent Modeling", next((r for r in results if r['name'] == "+ Opponent Modeling"), None)),
            ("Combined Best", next((r for r in results if r['name'] == "Optimal Exploitation"), None))
        ]
        
        for improvement_name, result in improvements:
            if result:
                delta = result['exploitation_rate'] - baseline['exploitation_rate']
                print(f"{improvement_name:<20}: {result['exploitation_rate']:.1f}% ({delta:+.1f}%)")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()