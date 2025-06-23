"""
Unified Simulation Runner

This module provides simplified and robust simulation functions that work with
the unified agent API. It eliminates all agent type checking and branching logic.
"""

import random
import numpy as np
from unified_agents import StaticAgent, SimpleQLearningAgent, EnhancedQLearningAgent

# --- Constants and Payoffs ---
COOPERATE = 0
DEFECT = 1
PAYOFFS_2P = {
    (COOPERATE, COOPERATE): (3, 3), (COOPERATE, DEFECT): (0, 5),
    (DEFECT, COOPERATE): (5, 0), (DEFECT, DEFECT): (1, 1),
}
T, R, P, S = 5, 3, 1, 0

def nperson_payoff(my_move, num_other_cooperators, total_agents):
    """Calculates N-Person payoff based on the linear formula."""
    if total_agents <= 1: return R if my_move == COOPERATE else P
    if my_move == COOPERATE:
        return S + (R - S) * (num_other_cooperators / (total_agents - 1))
    else:
        return P + (T - P) * (num_other_cooperators / (total_agents - 1))

# --- Simplified Simulation Functions ---

def run_pairwise_simulation(agents, num_rounds):
    """Runs a single pairwise tournament using the unified API."""
    for agent in agents: agent.reset()

    all_coop_history = {agent.agent_id: [] for agent in agents}
    score_history = {agent.agent_id: [] for agent in agents}
    
    for round_num in range(num_rounds):
        round_moves = {agent.agent_id: {} for agent in agents}
        
        # All pairs choose actions
        moves_this_round = {}
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                move1 = agent1.choose_action(opponent_id=agent2.agent_id, round=round_num)
                move2 = agent2.choose_action(opponent_id=agent1.agent_id, round=round_num)
                moves_this_round[(agent1.agent_id, agent2.agent_id)] = (move1, move2)

        # All pairs record outcomes
        round_payoffs = {agent.agent_id: 0 for agent in agents}
        for (id1, id2), (move1, move2) in moves_this_round.items():
            agent1 = next(a for a in agents if a.agent_id == id1)
            agent2 = next(a for a in agents if a.agent_id == id2)
            
            payoff1, payoff2 = PAYOFFS_2P[(move1, move2)]
            round_payoffs[id1] += payoff1
            round_payoffs[id2] += payoff2
            
            round_moves[agent1.agent_id][agent2.agent_id] = move1
            round_moves[agent2.agent_id][agent1.agent_id] = move2

            agent1.record_outcome(opponent_id=id2, opponent_move=move2, payoff=payoff1, my_move=move1)
            agent2.record_outcome(opponent_id=id1, opponent_move=move1, payoff=payoff2, my_move=move2)

        # Track cooperation history and cumulative scores
        for agent in agents:
            coop_count = sum(1 for move in round_moves[agent.agent_id].values() if move == COOPERATE)
            coop_rate = coop_count / (len(agents) - 1) if len(agents) > 1 else 0
            all_coop_history[agent.agent_id].append(coop_rate)
            score_history[agent.agent_id].append(agent.total_score)
            
    return all_coop_history, score_history


def run_nperson_simulation(agents, num_rounds):
    """Runs a single N-Person simulation using the unified API."""
    for agent in agents: agent.reset()
    
    all_coop_history = {agent.agent_id: [] for agent in agents}
    score_history = {agent.agent_id: [] for agent in agents}
    prev_round_coop_ratio = None
    num_total_agents = len(agents)

    for round_num in range(num_rounds):
        # All agents choose actions based on the same context
        moves = {agent.agent_id: agent.choose_action(prev_round_group_coop_ratio=prev_round_coop_ratio) for agent in agents}
        
        num_cooperators = list(moves.values()).count(COOPERATE)
        current_coop_ratio = num_cooperators / num_total_agents if num_total_agents > 0 else 0

        # All agents record outcomes
        for agent in agents:
            my_move = moves[agent.agent_id]
            num_other_cooperators = num_cooperators - (1 if my_move == COOPERATE else 0)
            payoff = nperson_payoff(my_move, num_other_cooperators, num_total_agents)
            
            agent.record_outcome(payoff=payoff, my_move=my_move, prev_round_group_coop_ratio=current_coop_ratio)
            all_coop_history[agent.agent_id].append(1 if my_move == COOPERATE else 0)
            score_history[agent.agent_id].append(agent.total_score)

        prev_round_coop_ratio = current_coop_ratio
        
    return all_coop_history, score_history


def run_multiple_simulations(simulation_func, agents, num_rounds, num_runs=100, training_rounds=0):
    """Run multiple simulations with optional training phase."""
    all_coop_runs = {agent.agent_id: [] for agent in agents}
    all_score_runs = {agent.agent_id: [] for agent in agents}
    
    for run in range(num_runs):
        # Create fresh agents for each run
        fresh_agents = []
        for agent in agents:
            if agent.strategy_name == "EnhancedQLearning":
                fresh_agents.append(EnhancedQLearningAgent(
                    agent_id=agent.agent_id,
                    learning_rate=agent.learning_rate,
                    discount_factor=agent.discount_factor,
                    epsilon=agent.initial_epsilon,
                    epsilon_decay=agent.epsilon_decay,
                    epsilon_min=agent.epsilon_min,
                    state_type=getattr(agent, 'state_type', 'basic')
                ))
            elif agent.strategy_name == "SimpleQLearning":
                fresh_agents.append(SimpleQLearningAgent(
                    agent_id=agent.agent_id,
                    learning_rate=agent.learning_rate,
                    discount_factor=agent.discount_factor,
                    epsilon=agent.epsilon
                ))
            else:  # Static agents
                fresh_agents.append(StaticAgent(
                    agent_id=agent.agent_id,
                    strategy_name=agent.strategy_name,
                    exploration_rate=agent.exploration_rate
                ))
        
        # Training phase for Q-learning agents
        if training_rounds > 0:
            for _ in range(training_rounds // num_rounds):
                simulation_func(fresh_agents, num_rounds)
                # Reset triggers epsilon decay for enhanced agents
                for agent in fresh_agents:
                    agent.reset()
        
        # Main simulation run
        coop_history, score_history = simulation_func(fresh_agents, num_rounds)
        
        for agent_id in coop_history:
            all_coop_runs[agent_id].append(coop_history[agent_id])
            all_score_runs[agent_id].append(score_history[agent_id])
    
    return all_coop_runs, all_score_runs