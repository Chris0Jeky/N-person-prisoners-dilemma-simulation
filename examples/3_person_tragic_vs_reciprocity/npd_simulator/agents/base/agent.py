"""
Base Agent Classes for N-Person Prisoner's Dilemma
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict
import random


class Agent(ABC):
    """
    Abstract base class for all agents.
    """
    
    def __init__(self, agent_id: int, exploration_rate: float = 0.0):
        """
        Initialize agent.
        
        Args:
            agent_id: Unique identifier for the agent
            exploration_rate: Probability of taking random action (0.0 to 1.0)
        """
        self.agent_id = agent_id
        self.exploration_rate = exploration_rate
        self.total_score = 0.0
        self.num_cooperations = 0
        self.num_defections = 0
        
    @abstractmethod
    def choose_action(self, *args, **kwargs) -> Tuple[int, int]:
        """
        Choose an action based on strategy.
        
        Returns:
            Tuple of (intended_action, actual_action) where:
            - intended_action: What the strategy dictates
            - actual_action: What is played (may differ due to exploration)
        """
        pass
    
    def apply_exploration(self, intended_action: int) -> int:
        """
        Apply exploration to the intended action.
        
        Args:
            intended_action: The action the strategy chose
            
        Returns:
            The actual action to play
        """
        if random.random() < self.exploration_rate:
            return 1 - intended_action  # Flip action
        return intended_action
    
    def record_outcome(self, action: int, payoff: float):
        """
        Record the outcome of a round.
        
        Args:
            action: The action that was played
            payoff: The payoff received
        """
        self.total_score += payoff
        if action == 0:  # Cooperate
            self.num_cooperations += 1
        else:  # Defect
            self.num_defections += 1
    
    def get_cooperation_rate(self) -> float:
        """Get the agent's cooperation rate."""
        total_moves = self.num_cooperations + self.num_defections
        return self.num_cooperations / total_moves if total_moves > 0 else 0.0
    
    def get_stats(self) -> Dict:
        """Get agent statistics."""
        return {
            'agent_id': self.agent_id,
            'total_score': self.total_score,
            'cooperation_rate': self.get_cooperation_rate(),
            'num_cooperations': self.num_cooperations,
            'num_defections': self.num_defections
        }
    
    def reset(self):
        """Reset agent state for new simulation."""
        self.total_score = 0.0
        self.num_cooperations = 0
        self.num_defections = 0


class NPDAgent(Agent):
    """
    Base class for N-Person Prisoner's Dilemma agents.
    """
    
    @abstractmethod
    def choose_action(self, 
                     cooperation_ratio: Optional[float],
                     round_number: int) -> Tuple[int, int]:
        """
        Choose action for N-Person game.
        
        Args:
            cooperation_ratio: Ratio of agents who cooperated last round (None for first round)
            round_number: Current round number
            
        Returns:
            Tuple of (intended_action, actual_action)
        """
        pass


class PairwiseAgent(Agent):
    """
    Base class for Pairwise Prisoner's Dilemma agents.
    """
    
    def __init__(self, agent_id: int, exploration_rate: float = 0.0):
        super().__init__(agent_id, exploration_rate)
        self.opponent_last_moves = {}  # Track last moves of each opponent
        
    @abstractmethod
    def choose_action(self,
                     opponent_id: int,
                     round_in_episode: int) -> Tuple[int, int]:
        """
        Choose action for pairwise game.
        
        Args:
            opponent_id: ID of the opponent
            round_in_episode: Round number within current episode
            
        Returns:
            Tuple of (intended_action, actual_action)
        """
        pass
    
    def record_interaction(self,
                          opponent_id: int,
                          opponent_action: int,
                          my_action: int,
                          my_payoff: float):
        """
        Record the outcome of a pairwise interaction.
        
        Args:
            opponent_id: ID of the opponent
            opponent_action: Action taken by opponent
            my_action: Action I took
            my_payoff: Payoff I received
        """
        self.record_outcome(my_action, my_payoff)
        self.opponent_last_moves[opponent_id] = opponent_action
    
    def clear_opponent_history(self, opponent_id: int):
        """Clear history for a specific opponent (between episodes)."""
        if opponent_id in self.opponent_last_moves:
            del self.opponent_last_moves[opponent_id]
    
    def reset(self):
        """Reset agent state for new tournament."""
        super().reset()
        self.opponent_last_moves.clear()