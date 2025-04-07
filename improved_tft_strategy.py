"""
Improved implementation of TitForTat strategy for pairwise interactions.

This implementation enhances the TitForTat strategy to work better in pairwise mode
by properly handling per-opponent move history.
"""

class ImprovedTitForTatStrategy:
    def choose_move(self, agent, neighbors):
        """Choose the next move for a TitForTat agent, handling both interaction modes.
        
        This improved implementation handles both traditional neighborhood interactions
        and pairwise interactions with specific opponent tracking.
        
        Args:
            agent: The agent making the decision
            neighbors: List of neighbor agent IDs (may be empty in pairwise mode)
            
        Returns:
            String: "cooperate" or "defect"
        """
        if not agent.memory:
            return "cooperate"  # Cooperate on first move
        
        # Get the last round's information
        last_round = agent.memory[-1]
        neighbor_moves = last_round['neighbor_moves']
        
        # CASE 1: Check for specific opponent moves (from improved pairwise implementation)
        if isinstance(neighbor_moves, dict) and 'specific_opponent_moves' in neighbor_moves:
            # We have specific opponent moves available
            specific_moves = neighbor_moves['specific_opponent_moves']
            
            # In pairwise mode, we'll respond to the majority behavior of opponents
            if specific_moves:
                defect_count = sum(1 for move in specific_moves.values() if move == "defect")
                # Defect if any opponent defected in the last round
                if defect_count > 0:
                    return "defect"
                else:
                    return "cooperate"
            return "cooperate"  # Default if no specific moves available
        
        # CASE 2: Simple opponent_coop_proportion (basic pairwise implementation)
        elif isinstance(neighbor_moves, dict) and 'opponent_coop_proportion' in neighbor_moves:
            # Convert proportion to binary decision based on majority behavior
            # In standard TFT, we defect if ANY opponent defected (not just majority)
            coop_proportion = neighbor_moves['opponent_coop_proportion']
            return "cooperate" if coop_proportion >= 1.0 else "defect"
            
        # CASE 3: Standard neighborhood format
        elif neighbor_moves:
            # Check if any neighbor defected in the last round
            defect_count = sum(1 for move in neighbor_moves.values() if move == "defect")
            if defect_count > 0:
                # In true TFT, we should defect if ANY neighbor defected
                return "defect"
            return "cooperate"
        
        # Default for first round or no neighbors
        return "cooperate"
