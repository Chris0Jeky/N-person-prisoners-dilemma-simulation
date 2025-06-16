"""
Random strategy agent
"""

from typing import Tuple
import random
from ..base.agent import NPDAgent, PairwiseAgent


class RandomAgent(NPDAgent, PairwiseAgent):
    """
    Random agent that cooperates with a fixed probability.
    
    Works for both N-Person and Pairwise games.
    """
    
    def __init__(self, 
                 agent_id: int,
                 cooperation_probability: float = 0.5,
                 exploration_rate: float = 0.0):
        """
        Initialize random agent.
        
        Args:
            agent_id: Unique identifier
            cooperation_probability: Probability of cooperating (0.0 to 1.0)
            exploration_rate: Additional exploration (usually 0 for random agent)
        """
        # Initialize both parent classes
        NPDAgent.__init__(self, agent_id, exploration_rate)
        PairwiseAgent.__init__(self, agent_id, exploration_rate)
        self.cooperation_probability = cooperation_probability
        self.name = f"Random({cooperation_probability:.2f})"
    
    def choose_action(self, *args, **kwargs) -> Tuple[int, int]:
        """Choose random action based on cooperation probability."""
        intended_action = 0 if random.random() < self.cooperation_probability else 1
        actual_action = self.apply_exploration(intended_action)
        return intended_action, actual_action