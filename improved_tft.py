"""
Improved TitForTatStrategy implementation to correctly handle pairwise mode.

Replace the current TitForTatStrategy.choose_move method in agents.py with this version.
"""

def tft_choose_move(self, agent, neighbors):
    """Choose the move for a TitForTat agent, handling both interaction modes.
    
    This version fixes issues in pairwise mode by:
    1. Properly checking for specific opponent moves
    2. Using the most conservative approach (defect if ANY opponent defected)
    3. Maintaining compatibility with the original format
    
    Args:
        agent: The agent making the decision
        neighbors: List of neighbor agent IDs
        
    Returns:
        String: "cooperate" or "defect"
    """
    # First round or no memory - default to cooperate
    if not agent.memory:
        return "cooperate"
    
    # Get the last round's information
    last_round = agent.memory[-1]
    neighbor_moves = last_round.get('neighbor_moves', {})
    
    # CASE 1: Check for specific opponent moves format (improved pairwise mode)
    if isinstance(neighbor_moves, dict) and 'specific_opponent_moves' in neighbor_moves:
        specific_moves = neighbor_moves['specific_opponent_moves']
        if specific_moves:
            # For true TFT, defect if ANY specific opponent defected
            if any(move == "defect" for move in specific_moves.values()):
                return "defect"
            return "cooperate"
    
    # CASE 2: Check for simple opponent_coop_proportion (old pairwise format)
    elif isinstance(neighbor_moves, dict) and 'opponent_coop_proportion' in neighbor_moves:
        # In standard TFT, we want to defect if ANY opponent defected
        # So we only cooperate if the cooperation rate is 100% (or very close)
        coop_proportion = neighbor_moves['opponent_coop_proportion']
        if coop_proportion < 0.99:  # Allow for floating point imprecision
            return "defect"
        return "cooperate"
    
    # CASE 3: Standard neighborhood format - original implementation
    elif neighbor_moves:
        # Get any neighbor's move from the last round
        random_neighbor_id = random.choice(list(neighbor_moves.keys()))
        return neighbor_moves[random_neighbor_id]
    
    # Default to cooperate if no neighbors or memory is in an unexpected format
    return "cooperate"
