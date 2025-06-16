"""
Tit-for-Tat and variants
"""

from typing import Tuple, Optional
import random
from ..base.agent import NPDAgent, PairwiseAgent


class TFTAgent(PairwiseAgent):
    """
    Standard Tit-for-Tat for pairwise games.
    
    Cooperates on first move, then copies opponent's last move.
    """
    
    def __init__(self, agent_id: int, exploration_rate: float = 0.0):
        super().__init__(agent_id, exploration_rate)
        self.name = "TFT"
    
    def choose_action(self,
                     opponent_id: int,
                     round_in_episode: int) -> Tuple[int, int]:
        """Choose action based on TFT strategy."""
        # Cooperate on first round or if no history with opponent
        if round_in_episode == 0 or opponent_id not in self.opponent_last_moves:
            intended_action = 0  # Cooperate
        else:
            # Copy opponent's last move
            intended_action = self.opponent_last_moves[opponent_id]
        
        actual_action = self.apply_exploration(intended_action)
        return intended_action, actual_action


class pTFTAgent(NPDAgent):
    """
    Probabilistic Tit-for-Tat for N-Person games.
    
    Cooperates with probability equal to the cooperation ratio in last round.
    """
    
    def __init__(self, agent_id: int, exploration_rate: float = 0.0):
        super().__init__(agent_id, exploration_rate)
        self.name = "pTFT"
    
    def choose_action(self,
                     cooperation_ratio: Optional[float],
                     round_number: int) -> Tuple[int, int]:
        """Choose action based on pTFT strategy."""
        if round_number == 0 or cooperation_ratio is None:
            intended_action = 0  # Cooperate on first round
        else:
            # Cooperate with probability equal to cooperation ratio
            intended_action = 0 if random.random() < cooperation_ratio else 1
        
        actual_action = self.apply_exploration(intended_action)
        return intended_action, actual_action


class pTFTThresholdAgent(NPDAgent):
    """
    Probabilistic Tit-for-Tat with Threshold for N-Person games.
    
    - Cooperates if cooperation ratio >= threshold (default 0.5)
    - Below threshold, cooperates with scaled probability
    """
    
    def __init__(self, 
                 agent_id: int,
                 exploration_rate: float = 0.0,
                 threshold: float = 0.5):
        super().__init__(agent_id, exploration_rate)
        self.name = "pTFT-Threshold"
        self.threshold = threshold
    
    def choose_action(self,
                     cooperation_ratio: Optional[float],
                     round_number: int) -> Tuple[int, int]:
        """Choose action based on pTFT-Threshold strategy."""
        if round_number == 0 or cooperation_ratio is None:
            intended_action = 0  # Cooperate on first round
        elif cooperation_ratio >= self.threshold:
            intended_action = 0  # Cooperate if above threshold
        else:
            # Below threshold: scale probability
            prob_cooperate = cooperation_ratio / self.threshold
            intended_action = 0 if random.random() < prob_cooperate else 1
        
        actual_action = self.apply_exploration(intended_action)
        return intended_action, actual_action