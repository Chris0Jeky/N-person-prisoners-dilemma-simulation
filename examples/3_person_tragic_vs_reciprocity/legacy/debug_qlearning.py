#!/usr/bin/env python3
"""
Debug Q-Learning behavior with simple experiments
Shows individual choices each round to understand convergence to ~20% cooperation
"""

import random
import sys
import os

# Add the legacy directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qlearning_demo_generator import (
    QLearningAgent, StaticAgent, COOPERATE, DEFECT, 
    run_pairwise_simulation_extended, run_nperson_simulation_extended,
    PAYOFFS_2P, nperson_payoff
)

def detailed_pairwise_simulation(agents, num_rounds=20, verbose=True):
    """Run pairwise simulation with detailed output of each choice."""
    for agent in agents:
        agent.reset()
    
    print("\n=== PAIRWISE SIMULATION ===")
    print(f"Agents: {[f'{a.agent_id} ({a.strategy_name})' for a in agents]}")
    print(f"Payoff matrix: CC=(3,3), CD=(0,5), DC=(5,0), DD=(1,1)")
    print("-" * 80)
    
    cumulative_scores = {agent.agent_id: 0 for agent in agents}
    
    for round_num in range(num_rounds):
        if verbose:
            print(f"\nRound {round_num + 1}:")
        
        round_moves = {}
        round_payoffs = {agent.agent_id: 0 for agent in agents}
        
        # All pairs play
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                
                # Get Q-table states for QL agents before moves
                if verbose and agent1.strategy_name == "QLearning":
                    state1 = agent1._get_state_pairwise(agent2.agent_id)
                    q_vals1 = agent1.q_table_pairwise.get(state1, {})
                    print(f"  {agent1.agent_id} state vs {agent2.agent_id}: '{state1}', Q-values: C={q_vals1.get(COOPERATE, 0):.3f}, D={q_vals1.get(DEFECT, 0):.3f}")
                
                if verbose and agent2.strategy_name == "QLearning":
                    state2 = agent2._get_state_pairwise(agent1.agent_id)
                    q_vals2 = agent2.q_table_pairwise.get(state2, {})
                    print(f"  {agent2.agent_id} state vs {agent1.agent_id}: '{state2}', Q-values: C={q_vals2.get(COOPERATE, 0):.3f}, D={q_vals2.get(DEFECT, 0):.3f}")
                
                # Make moves
                move1 = agent1.choose_pairwise_action(agent2.agent_id)
                move2 = agent2.choose_pairwise_action(agent1.agent_id)
                
                # Store moves
                round_moves[(agent1.agent_id, agent2.agent_id)] = (move1, move2)
                
                # Update histories
                if hasattr(agent1, 'opponent_last_moves'):
                    agent1.opponent_last_moves[agent2.agent_id] = move2
                if hasattr(agent2, 'opponent_last_moves'):
                    agent2.opponent_last_moves[agent1.agent_id] = move1
                
                # Calculate payoffs
                payoff1, payoff2 = PAYOFFS_2P[(move1, move2)]
                round_payoffs[agent1.agent_id] += payoff1
                round_payoffs[agent2.agent_id] += payoff2
                
                # Update Q-values
                if agent1.strategy_name == "QLearning":
                    agent1.update_q_value_pairwise(agent2.agent_id, move2, payoff1)
                if agent2.strategy_name == "QLearning":
                    agent2.update_q_value_pairwise(agent1.agent_id, move1, payoff2)
                
                if verbose:
                    move1_str = "C" if move1 == COOPERATE else "D"
                    move2_str = "C" if move2 == COOPERATE else "D"
                    print(f"  {agent1.agent_id} vs {agent2.agent_id}: {move1_str} vs {move2_str}, payoffs: ({payoff1}, {payoff2})")
        
        # Update cumulative scores
        for agent_id in cumulative_scores:
            cumulative_scores[agent_id] += round_payoffs[agent_id]
        
        if verbose:
            print(f"  Round scores: {round_payoffs}")
            print(f"  Cumulative: {cumulative_scores}")
    
    # Final cooperation rates
    print("\n=== FINAL RESULTS ===")
    for agent in agents:
        total_moves = (len(agents) - 1) * num_rounds
        coop_moves = 0
        for (a1, a2), (m1, m2) in round_moves.items():
            if a1 == agent.agent_id and m1 == COOPERATE:
                coop_moves += 1
        coop_rate = coop_moves / total_moves if total_moves > 0 else 0
        print(f"{agent.agent_id}: Score={cumulative_scores[agent.agent_id]}, Cooperation rate={coop_rate:.2%}")
        
        # Show final Q-table for QL agents
        if agent.strategy_name == "QLearning":
            print(f"  Final Q-table: {dict(agent.q_table_pairwise)}")


def detailed_nperson_simulation(agents, num_rounds=20, verbose=True):
    """Run N-person simulation with detailed output."""
    for agent in agents:
        agent.reset()
    
    print("\n=== N-PERSON SIMULATION ===")
    print(f"Agents: {[f'{a.agent_id} ({a.strategy_name})' for a in agents]}")
    print(f"Payoff formula: S=0, P=1, R=3, T=5")
    print("-" * 80)
    
    cumulative_scores = {agent.agent_id: 0 for agent in agents}
    prev_round_coop_ratio = None
    num_total_agents = len(agents)
    
    for round_num in range(num_rounds):
        if verbose:
            print(f"\nRound {round_num + 1}:")
            if prev_round_coop_ratio is not None:
                print(f"  Previous cooperation ratio: {prev_round_coop_ratio:.2f}")
        
        # Show Q-values for QL agents
        for agent in agents:
            if verbose and agent.strategy_name == "QLearning":
                state = agent._get_state_nperson(prev_round_coop_ratio)
                q_vals = agent.q_table_nperson.get(state, {})
                print(f"  {agent.agent_id} state: '{state}', Q-values: C={q_vals.get(COOPERATE, 0):.3f}, D={q_vals.get(DEFECT, 0):.3f}")
        
        # All agents make moves
        moves = {}
        for agent in agents:
            move = agent.choose_nperson_action(prev_round_coop_ratio)
            moves[agent.agent_id] = move
            if verbose:
                move_str = "C" if move == COOPERATE else "D"
                print(f"  {agent.agent_id} plays: {move_str}")
        
        # Calculate cooperation ratio
        num_cooperators = list(moves.values()).count(COOPERATE)
        current_coop_ratio = num_cooperators / num_total_agents if num_total_agents > 0 else 0
        
        if verbose:
            print(f"  Cooperation count: {num_cooperators}/{num_total_agents} = {current_coop_ratio:.2f}")
        
        # Calculate payoffs
        round_payoffs = {}
        for agent in agents:
            my_move = moves[agent.agent_id]
            num_other_cooperators = num_cooperators - (1 if my_move == COOPERATE else 0)
            payoff = nperson_payoff(my_move, num_other_cooperators, num_total_agents)
            round_payoffs[agent.agent_id] = payoff
            cumulative_scores[agent.agent_id] += payoff
            
            # Update Q-values for QL agents
            if agent.strategy_name == "QLearning":
                agent.update_q_value_nperson(my_move, payoff, current_coop_ratio)
        
        if verbose:
            print(f"  Payoffs: {round_payoffs}")
            print(f"  Cumulative: {cumulative_scores}")
        
        prev_round_coop_ratio = current_coop_ratio
    
    # Final results
    print("\n=== FINAL RESULTS ===")
    total_rounds = num_rounds
    for agent in agents:
        coop_count = sum(1 for r in range(num_rounds) if moves.get(agent.agent_id) == COOPERATE)
        coop_rate = coop_count / total_rounds if total_rounds > 0 else 0
        print(f"{agent.agent_id}: Score={cumulative_scores[agent.agent_id]}, Cooperation rate={coop_rate:.2%}")
        
        # Show final Q-table for QL agents
        if agent.strategy_name == "QLearning":
            print(f"  Final Q-table: {dict(agent.q_table_nperson)}")


def main():
    """Run debugging experiments."""
    print("Q-Learning Debugging Experiments")
    print("=" * 80)
    
    # Experiment 1: 1 QL vs 2 TFT (Pairwise)
    print("\n### Experiment 1: 1 QL vs 2 TFT (Pairwise) ###")
    agents1 = [
        QLearningAgent(agent_id="QL_1", epsilon=0.1),
        StaticAgent(agent_id="TFT_1", strategy_name="TFT"),
        StaticAgent(agent_id="TFT_2", strategy_name="TFT")
    ]
    detailed_pairwise_simulation(agents1, num_rounds=20, verbose=True)
    
    # Experiment 2: 1 QL vs 2 TFT (N-person)
    print("\n\n### Experiment 2: 1 QL vs 2 TFT (N-person) ###")
    agents2 = [
        QLearningAgent(agent_id="QL_1", epsilon=0.1),
        StaticAgent(agent_id="TFT_1", strategy_name="TFT"),
        StaticAgent(agent_id="TFT_2", strategy_name="TFT")
    ]
    detailed_nperson_simulation(agents2, num_rounds=20, verbose=True)
    
    # Experiment 3: 2 QL vs 1 TFT (Pairwise)
    print("\n\n### Experiment 3: 2 QL vs 1 TFT (Pairwise) ###")
    agents3 = [
        QLearningAgent(agent_id="QL_1", epsilon=0.1),
        QLearningAgent(agent_id="QL_2", epsilon=0.1),
        StaticAgent(agent_id="TFT_1", strategy_name="TFT")
    ]
    detailed_pairwise_simulation(agents3, num_rounds=20, verbose=True)
    
    # Experiment 4: 2 QL vs 1 TFT (N-person) 
    print("\n\n### Experiment 4: 2 QL vs 1 TFT (N-person) ###")
    agents4 = [
        QLearningAgent(agent_id="QL_1", epsilon=0.1),
        QLearningAgent(agent_id="QL_2", epsilon=0.1),
        StaticAgent(agent_id="TFT_1", strategy_name="TFT")
    ]
    detailed_nperson_simulation(agents4, num_rounds=20, verbose=True)
    
    # Run longer simulation to see convergence
    print("\n\n### Experiment 5: 2 QL vs 1 TFT (Pairwise, 100 rounds, less verbose) ###")
    agents5 = [
        QLearningAgent(agent_id="QL_1", epsilon=0.1),
        QLearningAgent(agent_id="QL_2", epsilon=0.1),
        StaticAgent(agent_id="TFT_1", strategy_name="TFT")
    ]
    detailed_pairwise_simulation(agents5, num_rounds=100, verbose=False)


if __name__ == "__main__":
    main()