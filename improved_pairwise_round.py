"""
Replace the current _run_pairwise_round method in the Environment class with this improved version.
"""

def _run_pairwise_round(self, rewiring_prob=0.0):
    """Run a single round using pairwise interactions with improved TFT support.
    
    This method fixes issues with TFT strategy in pairwise mode by:
    1. Tracking per-opponent moves in memory
    2. Using specific opponent moves for TFT strategies
    3. Maintaining compatibility with all strategies
    
    Args:
        rewiring_prob: Probability to rewire network edges
        
    Returns:
        Tuple of (moves, payoffs) dictionaries
    """
    # Step 1: Each agent chooses one move for the round
    agent_moves_for_round = {}
    
    # Special handling for TFT-like strategies
    for agent in self.agents:
        if agent.strategy_type in ["tit_for_tat", "generous_tit_for_tat", 
                                "suspicious_tit_for_tat", "tit_for_two_tats"]:
            # These strategies need special handling
            if agent.memory:
                last_round = agent.memory[-1]
                neighbor_moves = last_round.get('neighbor_moves', {})
                
                # Check if we have specific opponent moves
                if isinstance(neighbor_moves, dict) and 'specific_opponent_moves' in neighbor_moves:
                    specific_moves = neighbor_moves['specific_opponent_moves']
                    # For standard TFT, defect if ANY opponent defected
                    if any(move == "defect" for move in specific_moves.values()):
                        move = "defect"
                    else:
                        move = "cooperate"
                elif agent.strategy_type == "tit_for_tat":
                    # Default to cooperate for first round or if no specific data
                    move = agent.choose_move([])
                else:
                    # For other strategies just use normal choose_move
                    move = agent.choose_move([])
            else:
                # First round - use strategy's default behavior
                move = agent.choose_move([])
        else:
            # Non-TFT strategies - use normal choose_move
            move = agent.choose_move([])
        
        agent_moves_for_round[agent.agent_id] = move
    
    # Step 2: Simulate Pairwise Games and track specific opponent moves
    pairwise_payoffs = {agent.agent_id: [] for agent in self.agents}
    specific_opponent_moves = {agent.agent_id: {} for agent in self.agents}
    opponent_coop_counts = {agent.agent_id: 0 for agent in self.agents}
    opponent_total_counts = {agent.agent_id: 0 for agent in self.agents}
    
    agent_ids = [a.agent_id for a in self.agents]
    for i in range(len(agent_ids)):
        for j in range(i + 1, len(agent_ids)):
            agent_i_id = agent_ids[i]
            agent_j_id = agent_ids[j]
            move_i = agent_moves_for_round[agent_i_id]
            move_j = agent_moves_for_round[agent_j_id]
            
            # Calculate payoffs
            payoff_i, payoff_j = get_pairwise_payoffs(move_i, move_j, self.R, self.S, self.T, self.P)
            
            # Store payoffs
            pairwise_payoffs[agent_i_id].append(payoff_i)
            pairwise_payoffs[agent_j_id].append(payoff_j)
            
            # Track specific opponent moves (crucial for TFT)
            specific_opponent_moves[agent_i_id][agent_j_id] = move_j
            specific_opponent_moves[agent_j_id][agent_i_id] = move_i
            
            # Also track aggregate cooperation stats
            opponent_total_counts[agent_i_id] += 1
            opponent_total_counts[agent_j_id] += 1
            if move_j == "cooperate":
                opponent_coop_counts[agent_i_id] += 1
            if move_i == "cooperate":
                opponent_coop_counts[agent_j_id] += 1
    
    # Step 3: Update agent memories with BOTH specific and aggregate data
    round_payoffs_total = {}
    
    for agent in self.agents:
        agent_id = agent.agent_id
        agent_total_payoff = sum(pairwise_payoffs[agent_id])
        num_games = len(pairwise_payoffs[agent_id])
        agent_avg_payoff = agent_total_payoff / num_games if num_games > 0 else 0
        
        # Store total payoffs for return value
        round_payoffs_total[agent_id] = agent_total_payoff
        
        # Calculate aggregate opponent behavior
        opp_total = opponent_total_counts[agent_id]
        opp_coop = opponent_coop_counts[agent_id]
        opp_coop_prop = opp_coop / opp_total if opp_total > 0 else 0.5
        
        # Update agent score
        agent.score += agent_total_payoff
        
        # Create a hybrid memory format with both aggregate and specific data
        neighbor_moves = {
            'opponent_coop_proportion': opp_coop_prop,
            'specific_opponent_moves': specific_opponent_moves[agent_id]
        }
        
        # Update memory with both data formats
        agent.update_memory(
            my_move=agent_moves_for_round[agent_id],
            neighbor_moves=neighbor_moves,
            reward=agent_avg_payoff
        )
    
    # Step 4: Q-Value Updates (using aggregate format for RL strategies)
    for agent in self.agents:
        if agent.strategy_type in ("q_learning", "q_learning_adaptive", 
                                  "lra_q", "hysteretic_q", "wolf_phc", "ucb1_q"):
            # Create the expected format for Q-learning
            neighbor_moves = {'opponent_coop_proportion': 
                             opponent_coop_counts[agent.agent_id] / opponent_total_counts[agent.agent_id] 
                             if opponent_total_counts[agent.agent_id] > 0 else 0.5}
            
            # Update Q-values
            agent.update_q_value(
                action=agent_moves_for_round[agent.agent_id],
                reward=round_payoffs_total[agent.agent_id] / max(1, len(specific_opponent_moves[agent.agent_id])),
                next_state_actions=neighbor_moves
            )
    
    # Handle network visualization (has no functional effect in pairwise mode)
    if rewiring_prob > 0:
        self.update_network(rewiring_prob=rewiring_prob)
    
    return agent_moves_for_round, round_payoffs_total
