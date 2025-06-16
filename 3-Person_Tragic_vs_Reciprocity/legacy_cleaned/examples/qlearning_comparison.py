#!/usr/bin/env python3
"""
Q-Learning comparison example - Simple vs NPDL implementations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import create_agent, NPersonGame, PairwiseGame

def run_qlearning_comparison():
    print("=== Q-Learning Algorithm Comparison ===\n")
    
    # Test parameters
    num_rounds = 500
    num_simulations = 5
    
    # Results storage
    simple_scores = []
    npdl_scores = []
    simple_coop = []
    npdl_coop = []
    
    print(f"Running {num_simulations} simulations of {num_rounds} rounds each...\n")
    
    for sim in range(num_simulations):
        # Create agents
        agents = [
            create_agent("SimpleQL", "SimpleQL", 
                        learning_rate=0.1, epsilon=0.1),
            create_agent("NPDLQL", "NPDLQL",
                        learning_rate=0.1, epsilon=0.1, 
                        epsilon_decay=0.995, state_type="proportion_discretized"),
            create_agent("TFT", "pTFT")  # Baseline
        ]
        
        # Run N-person game
        game = NPersonGame(agents, num_rounds=num_rounds)
        results = game.run_simulation()
        
        # Collect results
        simple_scores.append(results['agents']['SimpleQL']['total_score'])
        npdl_scores.append(results['agents']['NPDLQL']['total_score'])
        simple_coop.append(results['agents']['SimpleQL']['cooperation_rate'])
        npdl_coop.append(results['agents']['NPDLQL']['cooperation_rate'])
        
        print(f"Simulation {sim+1}/{num_simulations} complete")
    
    # Calculate averages
    avg_simple_score = sum(simple_scores) / len(simple_scores)
    avg_npdl_score = sum(npdl_scores) / len(npdl_scores)
    avg_simple_coop = sum(simple_coop) / len(simple_coop)
    avg_npdl_coop = sum(npdl_coop) / len(npdl_coop)
    
    print("\n=== Results Summary ===")
    print(f"\nSimple Q-Learning:")
    print(f"  Average Score: {avg_simple_score:.1f}")
    print(f"  Average Cooperation Rate: {avg_simple_coop:.3f}")
    
    print(f"\nNPDL Q-Learning:")
    print(f"  Average Score: {avg_npdl_score:.1f}")
    print(f"  Average Cooperation Rate: {avg_npdl_coop:.3f}")
    
    print(f"\nPerformance Difference:")
    print(f"  Score: {((avg_npdl_score - avg_simple_score) / avg_simple_score * 100):.1f}%")
    print(f"  Cooperation: {((avg_npdl_coop - avg_simple_coop) / avg_simple_coop * 100):.1f}%")

def test_different_environments():
    print("\n\n=== Testing in Different Environments ===\n")
    
    # Test against different opponent compositions
    test_cases = [
        ("vs 2 Cooperators", ["QL", "TFT", "TFT"]),
        ("vs 2 Defectors", ["QL", "AllD", "AllD"]),
        ("vs Mixed", ["QL", "TFT", "AllD"])
    ]
    
    for test_name, agent_types in test_cases:
        print(f"\nTest: {test_name}")
        
        for ql_type in ["SimpleQL", "NPDLQL"]:
            # Replace first agent with Q-learning variant
            agents = [create_agent(ql_type, ql_type)]
            for i, agent_type in enumerate(agent_types[1:], 1):
                agents.append(create_agent(f"{agent_type}_{i}", agent_type))
            
            # Run game
            game = NPersonGame(agents, num_rounds=200)
            results = game.run_simulation()
            
            ql_results = results['agents'][ql_type]
            print(f"  {ql_type}: Score={ql_results['total_score']:.1f}, "
                  f"Cooperation={ql_results['cooperation_rate']:.3f}")

if __name__ == "__main__":
    run_qlearning_comparison()
    test_different_environments()