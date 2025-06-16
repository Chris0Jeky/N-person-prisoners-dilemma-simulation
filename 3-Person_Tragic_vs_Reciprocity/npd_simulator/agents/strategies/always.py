"""
Always Cooperate and Always Defect strategies
"""

from typing import Tuple, Optional
from ..base.agent import NPDAgent, PairwiseAgent


class AlwaysCooperateBase:
    """Base class for Always Cooperate strategy."""
    
    def _get_intended_action(self) -> int:
        """Always cooperate strategy."""
        return 0  # Always cooperate


class AlwaysDefectBase:
    """Base class for Always Defect strategy."""
    
    def _get_intended_action(self) -> int:
        """Always defect strategy."""
        return 1  # Always defect


class AllCAgent(AlwaysCooperateBase, NPDAgent, PairwiseAgent):
    """
    Always Cooperate agent for both N-Person and Pairwise games.
    """
    
    def __init__(self, agent_id: int, exploration_rate: float = 0.0):
        # Initialize both parent classes
        NPDAgent.__init__(self, agent_id, exploration_rate)
        PairwiseAgent.__init__(self, agent_id, exploration_rate)
        self.name = "AllC"
    
    def choose_action(self, *args, **kwargs) -> Tuple[int, int]:
        """Choose action - always cooperate."""
        intended_action = self._get_intended_action()
        actual_action = self.apply_exploration(intended_action)
        return intended_action, actual_action


class AllDAgent(AlwaysDefectBase, NPDAgent, PairwiseAgent):
    """
    Always Defect agent for both N-Person and Pairwise games.
    """
    
    def __init__(self, agent_id: int, exploration_rate: float = 0.0):
        # Initialize both parent classes
        NPDAgent.__init__(self, agent_id, exploration_rate)
        PairwiseAgent.__init__(self, agent_id, exploration_rate)
        self.name = "AllD"
    
    def choose_action(self, *args, **kwargs) -> Tuple[int, int]:
        """Choose action - always defect."""
        intended_action = self._get_intended_action()
        actual_action = self.apply_exploration(intended_action)
        return intended_action, actual_action