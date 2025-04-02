"""
Modified QLearningStrategy _get_current_state method to handle the pairwise interaction mode.

This will be integrated into the agents.py file by modifying the QLearningStrategy class.
"""

def _get_current_state(self, agent: 'Agent') -> Hashable:
    """Get the current state representation for Q-learning.

    Constructs a state representation based on the agent's memory of past
    interactions, particularly focusing on neighbor cooperation patterns.
    The specific representation depends on the state_type parameter.

    Args:
        agent: The agent making the decision

    Returns:
        A hashable state representation (string, tuple, or other hashable type)
        that captures relevant information from the agent's memory and neighbors
    """
    if not agent.memory:
        return 'initial'  # Special state for the first move
    
    if self.state_type == "basic":
        return 'standard'  # Default state for backward compatibility
        
    last_round_info = agent.memory[-1]  # Memory stores results of the round *leading* to this state
    neighbor_moves = last_round_info['neighbor_moves']
    
    # Handle pairwise interaction mode with 'opponent_coop_proportion' in neighbor_moves
    if isinstance(neighbor_moves, dict) and 'opponent_coop_proportion' in neighbor_moves:
        coop_proportion = neighbor_moves['opponent_coop_proportion']
        
        if self.state_type == "proportion":
            # Use the exact proportion as state
            return (coop_proportion,)

        elif self.state_type == "proportion_discretized":
            # Discretize the proportion into bins (5 bins)
            if coop_proportion <= 0.2:
                state_feature = 0.2
            elif coop_proportion <= 0.4:
                state_feature = 0.4
            elif coop_proportion <= 0.6:
                state_feature = 0.6
            elif coop_proportion <= 0.8:
                state_feature = 0.8
            else:
                state_feature = 1.0
            return (state_feature,)
        
        elif self.state_type == "memory_enhanced":
            # Example: Own last 2 moves + discretized opponent coop proportion
            # Determine own moves (default to C if not enough history)
            own_last_move = 'cooperate'
            own_prev_move = 'cooperate'
            if len(agent.memory) >= 1:
                own_last_move = agent.memory[-1]['my_move']
            if len(agent.memory) >= 2:
                own_prev_move = agent.memory[-2]['my_move']

            own_last_bin = 1 if own_last_move == "cooperate" else 0
            own_prev_bin = 1 if own_prev_move == "cooperate" else 0

            # Use discretized neighbor state (from previous state implementation)
            if coop_proportion <= 0.33:
                opponent_state = 0  # Low
            elif coop_proportion <= 0.67:
                opponent_state = 1  # Med
            else:
                opponent_state = 2  # High

            # State includes own last 2 moves and opponent cooperation summary
            return (own_last_bin, own_prev_bin, opponent_state)
        
        elif self.state_type == "count":
            # For pairwise mode, convert proportion to a discretized "count" representation
            # This helps maintain compatibility with the standard mode
            num_bins = 10  # Discretize into 10 bins
            discretized_count = int(coop_proportion * num_bins)
            return (discretized_count,)
        
        elif self.state_type == "threshold":
            # Binary feature indicating if majority cooperated
            return (coop_proportion > 0.5,)
        
        # Default fallback for pairwise mode
        return ('pairwise', coop_proportion)
    
    # Standard neighborhood-based interaction mode logic
    num_neighbors = len(neighbor_moves)
    if num_neighbors == 0:
        return 'no_neighbors'  # Special state if isolated
        
    num_cooperating_neighbors = sum(1 for move in neighbor_moves.values() if move == "cooperate")
    coop_proportion = num_cooperating_neighbors / num_neighbors
    
    if self.state_type == "proportion":
        # Use the exact proportion as state
        return (coop_proportion,)

    elif self.state_type == "memory_enhanced":
        # Example: Own last 2 moves + discretized neighbor coop proportion
        # Determine own moves (default to C if not enough history)
        own_last_move = 'cooperate'
        own_prev_move = 'cooperate'
        if len(agent.memory) >= 1:
            own_last_move = agent.memory[-1]['my_move']
        if len(agent.memory) >= 2:
            own_prev_move = agent.memory[-2]['my_move']

        own_last_bin = 1 if own_last_move == "cooperate" else 0
        own_prev_bin = 1 if own_prev_move == "cooperate" else 0

        # Use discretized neighbor state (from previous state implementation)
        if coop_proportion <= 0.33:
            neighbor_state = 0  # Low
        elif coop_proportion <= 0.67:
            neighbor_state = 1  # Med
        else:
            neighbor_state = 2  # High

        # State includes own last 2 moves and neighbor summary
        return (own_last_bin, own_prev_bin, neighbor_state)
        
    elif self.state_type == "proportion_discretized":
        # Discretize the proportion into bins (5 bins)
        if coop_proportion <= 0.2:
            state_feature = 0.2
        elif coop_proportion <= 0.4:
            state_feature = 0.4
        elif coop_proportion <= 0.6:
            state_feature = 0.6
        elif coop_proportion <= 0.8:
            state_feature = 0.8
        else:
            state_feature = 1.0
        return (state_feature,)
        
    elif self.state_type == "count":
        # Use absolute count of cooperating neighbors
        return (num_cooperating_neighbors,)
        
    elif self.state_type == "threshold":
        # Binary feature indicating if majority cooperated
        return (coop_proportion > 0.5,)
        
    # Default fallback
    return 'standard'
