# Modified version of Environment class with pairwise interactions
from npdl.core.utils import get_pairwise_payoffs

def _run_pairwise_round(self, rewiring_prob=0.0):
    """Run a single round using pairwise interactions between all agents.
    
    In this mode, each agent plays a separate 2-player PD game against every other agent.
    The agent's score for the round is the sum of payoffs from all its pairwise games.
    
    Args:
        rewiring_prob: Probability to rewire network edges after this round
        
    Returns:
        Tuple of (moves, payoffs) dictionaries
    """
    # Step 1: Each agent chooses one move for the round (used against all opponents)
    agent_moves_for_round = {agent.agent_id: agent.choose_move([])
                           for agent in self.agents}
    
    # Step 2: Simulate Pairwise Games between all agent pairs
    pairwise_payoffs = {agent.agent_id: [] for agent in self.agents}  # Store all payoffs per agent
    opponent_coop_counts = {agent.agent_id: 0 for agent in self.agents}  # Track how many opponents cooperated *against* this agent
    opponent_total_counts = {agent.agent_id: 0 for agent in self.agents}
    
    agent_ids = [a.agent_id for a in self.agents]
    for i in range(len(agent_ids)):
        for j in range(i + 1, len(agent_ids)):
            agent_i_id = agent_ids[i]
            agent_j_id = agent_ids[j]
            move_i = agent_moves_for_round[agent_i_id]
            move_j = agent_moves_for_round[agent_j_id]
            
            # Get 2-player payoffs
            payoff_i, payoff_j = get_pairwise_payoffs(move_i, move_j, self.R, self.S, self.T, self.P)
            
            pairwise_payoffs[agent_i_id].append(payoff_i)
            pairwise_payoffs[agent_j_id].append(payoff_j)
            
            # Track opponent cooperation for state calculation later
            opponent_total_counts[agent_i_id] += 1
            opponent_total_counts[agent_j_id] += 1
            if move_j == "cooperate":
                opponent_coop_counts[agent_i_id] += 1
            if move_i == "cooperate":
                opponent_coop_counts[agent_j_id] += 1
    
    # Step 3: Aggregate Results & Update Agents
    round_payoffs_avg = {}
    round_opponent_coop_proportion = {}
    round_payoffs_total = {}
    
    for agent in self.agents:
        agent_id = agent.agent_id
        agent_total_payoff = sum(pairwise_payoffs[agent_id])
        num_games = len(pairwise_payoffs[agent_id])
        agent_avg_payoff = agent_total_payoff / num_games if num_games > 0 else 0
        
        round_payoffs_avg[agent_id] = agent_avg_payoff
        round_payoffs_total[agent_id] = agent_total_payoff
        
        # Calculate aggregate opponent behavior for this round (used for *next* state)
        opp_total = opponent_total_counts[agent_id]
        opp_coop = opponent_coop_counts[agent_id]
        opp_coop_prop = opp_coop / opp_total if opp_total > 0 else 0.5  # Default if isolated
        round_opponent_coop_proportion[agent_id] = opp_coop_prop
        
        # Update agent score (using total payoff)
        agent.score += agent_total_payoff
        
        # Store the aggregate state info instead of individual neighbor moves
        neighbor_moves = {'opponent_coop_proportion': opp_coop_prop}
        
        # Update memory with average reward and aggregate opponent state
        agent.update_memory(
            my_move=agent_moves_for_round[agent_id],
            neighbor_moves=neighbor_moves,
            reward=agent_avg_payoff  # Use average reward for memory/learning signal
        )
    
    # Step 4: Q-Value Updates (after memory is updated for all agents)
    for agent in self.agents:
        if agent.strategy_type in ("q_learning", "q_learning_adaptive", 
                                  "lra_q", "hysteretic_q", "wolf_phc", "ucb1_q"):
            # Pass opponent cooperation proportion for next state calculation
            agent.update_q_value(
                action=agent_moves_for_round[agent.agent_id],
                reward=round_payoffs_avg[agent.agent_id],
                next_state_actions=neighbor_moves
            )
    
    # Optionally update network structure (for visualization, not functional in pairwise mode)
    if rewiring_prob > 0:
        self.update_network(rewiring_prob=rewiring_prob)
    
    return agent_moves_for_round, round_payoffs_total  # Return total payoffs for consistency with original model