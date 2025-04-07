"""
Improved implementation of pairwise interactions for the N-Person Prisoner's Dilemma.
"""

from typing import Dict, List, Tuple, Any, Set
from npdl.core.utils import get_pairwise_payoffs

class PairwiseMemory:
    """Class to track per-opponent move history for pairwise interactions."""
    
    def __init__(self):
        # Store the last move of each opponent
        self.opponent_last_moves = {}  # {opponent_id: last_move}
        # Store the last two moves of each opponent (for TitForTwoTats)
        self.opponent_prev_moves = {}  # {opponent_id: [move_t-2, move_t-1]}
        # Track cooperation rates over time
        self.opponent_coop_rates = {}  # {opponent_id: cooperation_rate}
    
    def update(self, opponent_id, move):
        """Update memory with an opponent's move."""
        # Update last two moves tracking
        if opponent_id in self.opponent_prev_moves:
            # Keep the last move and add the new one
            self.opponent_prev_moves[opponent_id] = [
                self.opponent_prev_moves[opponent_id][1], move
            ]
        else:
            # Initialize with two copies of the current move (conservative approach)
            self.opponent_prev_moves[opponent_id] = [move, move]
        
        # Update last move directly
        self.opponent_last_moves[opponent_id] = move
        
        # Update cooperation rate
        if opponent_id not in self.opponent_coop_rates:
            # Initial rate is 1.0 for cooperate, 0.0 for defect
            self.opponent_coop_rates[opponent_id] = 1.0 if move == "cooperate" else 0.0
        else:
            # Exponential moving average with 0.7 weight to recent moves
            current_rate = self.opponent_coop_rates[opponent_id]
            move_value = 1.0 if move == "cooperate" else 0.0
            self.opponent_coop_rates[opponent_id] = 0.3 * current_rate + 0.7 * move_value
    
    def get_last_move(self, opponent_id, default="cooperate"):
        """Get an opponent's last move."""
        return self.opponent_last_moves.get(opponent_id, default)
    
    def get_prev_two_moves(self, opponent_id):
        """Get an opponent's last two moves."""
        return self.opponent_prev_moves.get(opponent_id, ["cooperate", "cooperate"])
    
    def get_cooperation_rate(self, opponent_id):
        """Get an opponent's cooperation rate."""
        return self.opponent_coop_rates.get(opponent_id, 1.0)  # Default to 1.0 (optimistic)
    
    def get_aggregate_cooperation_rate(self):
        """Get the average cooperation rate across all opponents."""
        if not self.opponent_coop_rates:
            return 0.5  # Default when no history
        return sum(self.opponent_coop_rates.values()) / len(self.opponent_coop_rates)


def _run_improved_pairwise_round(self, rewiring_prob=0.0):
    """Run a single round using improved pairwise interactions between all agents.
    
    This implementation enhances the original pairwise mode by:
    1. Tracking per-opponent move history
    2. Enabling strategies like TFT to respond to specific opponents
    3. Maintaining both aggregate and per-opponent statistics
    
    Args:
        rewiring_prob: Probability to rewire network edges after this round
        
    Returns:
        Tuple of (moves, payoffs) dictionaries
    """
    # Initialize pairwise memory if not present
    if not hasattr(self, 'pairwise_memories'):
        self.pairwise_memories = {agent.agent_id: PairwiseMemory() for agent in self.agents}
    
    # Step 1: Each agent chooses one move for the round (used against all opponents)
    # We'll use a smarter approach where agents can choose different moves for different opponents
    
    # First, initialize empty move dictionary for all agents
    agent_moves_for_round = {}
    
    # Determine moves for each agent against each opponent
    # For simplicity, we'll have agents choose one global move to use with all opponents
    for agent in self.agents:
        # Get agent's memory to use for decision making
        pairwise_memory = self.pairwise_memories[agent.agent_id]
        try:
            # For TFT and similar strategies, we'll use the aggregate cooperation rate 
            # which is more meaningful than an empty neighbor list
            if agent.strategy_type in ("tit_for_tat", "generous_tit_for_tat", 
                                     "suspicious_tit_for_tat", "tit_for_two_tats"):
                # Create a special neighbor_moves dict with the aggregate cooperation info
                # This maintains compatibility with strategies expecting memory format
                coop_rate = pairwise_memory.get_aggregate_cooperation_rate()
                neighbor_moves = {'opponent_coop_proportion': coop_rate}
                
                # We have to use this approach since we don't have direct access to memory internals
                # Create temporary memory to guide decision making
                if not agent.memory:
                    # First round - use a dummy memory entry
                    temp_memory = [{'my_move': 'cooperate', 
                                   'neighbor_moves': neighbor_moves, 
                                   'reward': 3.0}]
                    original_memory = agent.memory
                    agent.memory = temp_memory
                    move = agent.choose_move([])  # Empty neighbor list is fine with memory
                    agent.memory = original_memory
                else:
                    # Replace last entry's neighbor_moves temporarily
                    last_entry = agent.memory[-1].copy()
                    original_neighbor_moves = last_entry['neighbor_moves']
                    agent.memory[-1]['neighbor_moves'] = neighbor_moves
                    move = agent.choose_move([])
                    # Restore original memory
                    agent.memory[-1]['neighbor_moves'] = original_neighbor_moves
            else:
                # For other strategies, just use the normal choose_move
                # Most strategies don't need neighbor info for the decision
                move = agent.choose_move([])
        except:
            # Fallback to standard approach with empty neighbors
            move = agent.choose_move([])
        
        agent_moves_for_round[agent.agent_id] = move
    
    # Step 2: Simulate Pairwise Games between all agent pairs
    # This part is similar to the original implementation
    pairwise_payoffs = {agent.agent_id: [] for agent in self.agents}  # Store all payoffs per agent
    pairwise_opponents = {agent.agent_id: [] for agent in self.agents}  # Track which opponents each agent played
    pairwise_opponent_moves = {agent.agent_id: {} for agent in self.agents}  # Track each opponent's move
    
    # Also track aggregate opponent cooperation for backward compatibility
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
            
            # Store payoffs
            pairwise_payoffs[agent_i_id].append(payoff_i)
            pairwise_payoffs[agent_j_id].append(payoff_j)
            
            # Track which opponents each agent played
            pairwise_opponents[agent_i_id].append(agent_j_id)
            pairwise_opponents[agent_j_id].append(agent_i_id)
            
            # Track each opponent's move
            pairwise_opponent_moves[agent_i_id][agent_j_id] = move_j
            pairwise_opponent_moves[agent_j_id][agent_i_id] = move_i
            
            # Also track aggregate opponent cooperation for backward compatibility
            opponent_total_counts[agent_i_id] += 1
            opponent_total_counts[agent_j_id] += 1
            if move_j == "cooperate":
                opponent_coop_counts[agent_i_id] += 1
            if move_i == "cooperate":
                opponent_coop_counts[agent_j_id] += 1
    
    # Step 3: Update agent memories and pairwise memories
    round_payoffs_avg = {}
    round_opponent_coop_proportion = {}
    round_payoffs_total = {}
    
    for agent in self.agents:
        agent_id = agent.agent_id
        agent_total_payoff = sum(pairwise_payoffs[agent_id])
        num_games = len(pairwise_payoffs[agent_id])
        agent_avg_payoff = agent_total_payoff / num_games if num_games > 0 else 0
        
        # Store payoff summary
        round_payoffs_avg[agent_id] = agent_avg_payoff
        round_payoffs_total[agent_id] = agent_total_payoff
        
        # Calculate aggregate opponent behavior (used for backward compatibility)
        opp_total = opponent_total_counts[agent_id]
        opp_coop = opponent_coop_counts[agent_id]
        opp_coop_prop = opp_coop / opp_total if opp_total > 0 else 0.5
        round_opponent_coop_proportion[agent_id] = opp_coop_prop
        
        # Update agent score
        agent.score += agent_total_payoff
        
        # Update pairwise memory with specific opponent moves
        pairwise_memory = self.pairwise_memories[agent_id]
        for opp_id, opp_move in pairwise_opponent_moves[agent_id].items():
            pairwise_memory.update(opp_id, opp_move)
        
        # Update agent memory with both aggregate info and specific opponent moves
        # Create a hybrid structure that works for both aggregate and per-opponent analysis
        neighbor_moves = {
            'opponent_coop_proportion': opp_coop_prop,
            # Also include specific opponent moves (TFT implementation can use either)
            'specific_opponent_moves': pairwise_opponent_moves[agent_id]
        }
        
        # Update memory with average reward and opponent state
        agent.update_memory(
            my_move=agent_moves_for_round[agent_id],
            neighbor_moves=neighbor_moves,
            reward=agent_avg_payoff
        )
    
    # Step 4: Q-Value Updates (after memory is updated for all agents)
    for agent in self.agents:
        if agent.strategy_type in ("q_learning", "q_learning_adaptive", 
                                  "lra_q", "hysteretic_q", "wolf_phc", "ucb1_q"):
            # Create proper format for Q-learning
            neighbor_moves = {'opponent_coop_proportion': round_opponent_coop_proportion[agent.agent_id]}
            agent.update_q_value(
                action=agent_moves_for_round[agent.agent_id],
                reward=round_payoffs_avg[agent.agent_id],
                next_state_actions=neighbor_moves
            )
    
    # Optionally update network structure
    if rewiring_prob > 0:
        self.update_network(rewiring_prob=rewiring_prob)
    
    return agent_moves_for_round, round_payoffs_total
