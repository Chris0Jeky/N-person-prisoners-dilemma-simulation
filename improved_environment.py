"""
Patch file to update Environment._run_pairwise_round

This fixes the TitForTat strategy issues in pairwise mode by enhancing how
memory is stored and used for decision making.
"""

def _run_pairwise_round_improved(self, rewiring_prob=0.0):
    """Run a single round using pairwise interactions with improved TFT support.
    
    This method fixes issues with TFT strategy in pairwise mode by:
    1. Storing specific opponent moves in memory
    2. Using this data for TFT decision making
    3. Maintaining compatibility with all strategies
    
    Args:
        rewiring_prob: Probability to rewire network edges
        
    Returns:
        Tuple of (moves, payoffs) dictionaries
    """
    # Step 1: Each agent chooses one move for the round
    agent_moves_for_round = {}
    
    # For TFT strategies, we need to be more careful about how they make decisions
    for agent in self.agents:
        if agent.strategy_type in ["tit_for_tat", "generous_tit_for_tat", 
                                 "suspicious_tit_for_tat", "tit_for_two_tats"]:
            # These strategies need special handling
            # If they have memory, examine it to guide decision
            if agent.memory:
                last_round = agent.memory[-1]
                
                if 'specific_opponent_moves' in last_round.get('neighbor_moves', {}):
                    # Use specific opponent moves if available
                    specific_moves = last_round['neighbor_moves']['specific_opponent_moves']
                    # Check if any opponent defected
                    if any(move == "defect" for move in specific_moves.values()):
                        move = "defect"  # TFT should defect if any opponent defected
                    else:
                        move = "cooperate"
                elif isinstance(last_round.get('neighbor_moves', {}), dict) and 'opponent_coop_proportion' in last_round['neighbor_moves']:
                    # Fall back to cooperation proportion
                    coop_proportion = last_round['neighbor_moves']['opponent_coop_proportion']
                    # Standard TFT should defect if ANY opponent defected (not just majority)
                    # So we defect unless 100% cooperation
                    move = "cooperate" if coop_proportion >= 0.99 else "defect"
                else:
                    # Default behavior with normal memory format
                    move = agent.choose_move([])
            else:
                # First round behavior from strategy
                move = agent.choose_move([])
        else:
            # For non-TFT strategies, use normal choose_move
            move = agent.choose_move([])
        
        agent_moves_for_round[agent.agent_id] = move
    
    # Step 2: Simulate Pairwise Games between all agent pairs
    pairwise_payoffs = {agent.agent_id: [] for agent in self.agents}
    pairwise_opponent_moves = {agent.agent_id: {} for agent in self.agents}
    opponent_coop_counts = {agent.agent_id: 0 for agent in self.agents}
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
            
            # Track opponent moves for specific opponents (used by TFT)
            pairwise_opponent_moves[agent_i_id][agent_j_id] = move_j
            pairwise_opponent_moves[agent_j_id][agent_i_id] = move_i
            
            # Track aggregated opponent cooperation
            opponent_total_counts[agent_i_id] += 1
            opponent_total_counts[agent_j_id] += 1
            if move_j == "cooperate":
                opponent_coop_counts[agent_i_id] += 1
            if move_i == "cooperate":
                opponent_coop_counts[agent_j_id] += 1
    
    # Step 3: Update agents with results
    round_payoffs_avg = {}
    round_payoffs_total = {}
    
    for agent in self.agents:
        agent_id = agent.agent_id
        agent_total_payoff = sum(pairwise_payoffs[agent_id])
        num_games = len(pairwise_payoffs[agent_id])
        agent_avg_payoff = agent_total_payoff / num_games if num_games > 0 else 0
        
        round_payoffs_avg[agent_id] = agent_avg_payoff
        round_payoffs_total[agent_id] = agent_total_payoff
        
        # Calculate aggregate opponent behavior
        opp_total = opponent_total_counts[agent_id]
        opp_coop = opponent_coop_counts[agent_id]
        opp_coop_prop = opp_coop / opp_total if opp_total > 0 else 0.5
        
        # Update agent score
        agent.score += agent_total_payoff
        
        # Create enhanced memory structure with both formats
        neighbor_moves = {
            'opponent_coop_proportion': opp_coop_prop,
            'specific_opponent_moves': pairwise_opponent_moves[agent_id]
        }
        
        # Update memory with both aggregate and specific opponent data
        agent.update_memory(
            my_move=agent_moves_for_round[agent_id],
            neighbor_moves=neighbor_moves,
            reward=agent_avg_payoff
        )
    
    # Step 4: Q-Value Updates
    for agent in self.agents:
        if agent.strategy_type in ("q_learning", "q_learning_adaptive", 
                                  "lra_q", "hysteretic_q", "wolf_phc", "ucb1_q"):
            # Q-learning strategies use the aggregate format
            neighbor_moves = {'opponent_coop_proportion': 
                             opponent_coop_counts[agent.agent_id] / opponent_total_counts[agent.agent_id] 
                             if opponent_total_counts[agent.agent_id] > 0 else 0.5}
            
            agent.update_q_value(
                action=agent_moves_for_round[agent.agent_id],
                reward=round_payoffs_avg[agent.agent_id],
                next_state_actions=neighbor_moves
            )
    
    # Optional network rewiring  
    if rewiring_prob > 0:
        self.update_network(rewiring_prob=rewiring_prob)
    
    return agent_moves_for_round, round_payoffs_total
