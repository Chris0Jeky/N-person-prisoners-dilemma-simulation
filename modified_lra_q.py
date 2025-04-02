"""
Modified LRAQLearningStrategy update method to handle pairwise interaction mode.

This will be integrated into the agents.py file by modifying the LRAQLearningStrategy class.
"""

def lra_q_update(self, agent, action, reward, neighbor_moves):
    """Update the LRA-Q agent's Q-values with dynamic learning rate.
    
    This update method works with both neighborhood and pairwise interaction modes.
    
    Args:
        agent: The agent being updated
        action: The action that was taken
        reward: The reward received
        neighbor_moves: Dictionary of neighbor moves or opponent coop proportion
    """
    # Handle both pairwise and neighborhood modes
    if isinstance(neighbor_moves, dict) and 'opponent_coop_proportion' in neighbor_moves:
        # Pairwise mode - neighbor_moves contains opponent_coop_proportion
        coop_proportion = neighbor_moves['opponent_coop_proportion']
        coop_neighbors = coop_proportion  # Using the proportion directly
        total_neighbors = 1  # In proportional terms
    else:
        # Standard neighborhood mode
        coop_neighbors = sum(1 for move in neighbor_moves.values() if move == "cooperate")
        total_neighbors = len(neighbor_moves) if neighbor_moves else 1
        coop_proportion = coop_neighbors / total_neighbors if total_neighbors > 0 else 0
    
    # Adjust learning rate based on cooperation level and agent's own action
    if action == "cooperate" and coop_proportion > 0.5:
        # Cooperating with cooperators - increase learning rate
        self.learning_rate = min(self.max_learning_rate, 
                                self.learning_rate + self.increase_rate)
    elif action == "defect" and coop_proportion > 0.5:
        # Defecting against cooperators - decrease learning rate
        self.learning_rate = max(self.min_learning_rate, 
                                self.learning_rate - self.decrease_rate)
    
    # Use the adjusted learning rate for the standard Q-update
    super().update(agent, action, reward, neighbor_moves)
    
    # Slowly return to base rate when not adjusted (regression to mean)
    if self.learning_rate > self.base_learning_rate:
        self.learning_rate = max(self.base_learning_rate, 
                                 self.learning_rate - 0.01)
    elif self.learning_rate < self.base_learning_rate:
        self.learning_rate = min(self.base_learning_rate, 
                                 self.learning_rate + 0.01)
