"""
N-Person Prisoner's Dilemma Game Implementation

Supports any number of agents with configurable payoff structures.
"""

from typing import List, Dict, Tuple, Optional
import random
from ..models.payoff_matrix import PayoffMatrix
from ..models.game_state import GameState


class NPDGame:
    """
    N-Person Prisoner's Dilemma game supporting any number of agents.
    
    Uses linear payoff functions based on the proportion of cooperators.
    """
    
    def __init__(self, 
                 num_agents: int,
                 num_rounds: int,
                 payoff_matrix: Optional[PayoffMatrix] = None):
        """
        Initialize N-Person PD game.
        
        Args:
            num_agents: Number of agents in the game
            num_rounds: Number of rounds to play
            payoff_matrix: Payoff matrix (uses default if None)
        """
        self.num_agents = num_agents
        self.num_rounds = num_rounds
        self.payoff_matrix = payoff_matrix or PayoffMatrix()
        self.game_state = GameState(num_agents)
        self.history = []
        
    def calculate_payoff(self, 
                        my_action: int, 
                        num_others_cooperating: int) -> float:
        """
        Calculate payoff for an agent based on their action and others' cooperation.
        
        Args:
            my_action: 0 for cooperate, 1 for defect
            num_others_cooperating: Number of other agents cooperating
            
        Returns:
            Payoff value
        """
        if self.num_agents <= 1:
            return self.payoff_matrix.R if my_action == 0 else self.payoff_matrix.P
            
        cooperation_ratio = num_others_cooperating / (self.num_agents - 1)
        
        if my_action == 0:  # Cooperate
            return self.payoff_matrix.linear_cooperator_payoff(cooperation_ratio)
        else:  # Defect
            return self.payoff_matrix.linear_defector_payoff(cooperation_ratio)
    
    def play_round(self, actions: Dict[int, int]) -> Dict[int, float]:
        """
        Play a single round of the game.
        
        Args:
            actions: Dictionary mapping agent_id to action (0=cooperate, 1=defect)
            
        Returns:
            Dictionary mapping agent_id to payoff
        """
        # Count total cooperators
        num_cooperators = sum(1 for action in actions.values() if action == 0)
        
        # Calculate payoffs for each agent
        payoffs = {}
        for agent_id, action in actions.items():
            num_others_cooperating = num_cooperators
            if action == 0:  # If I cooperated, subtract myself
                num_others_cooperating -= 1
                
            payoffs[agent_id] = self.calculate_payoff(action, num_others_cooperating)
        
        # Update game state
        self.game_state.update(actions, payoffs)
        self.history.append({
            'round': len(self.history),
            'actions': actions.copy(),
            'payoffs': payoffs.copy(),
            'cooperation_rate': num_cooperators / self.num_agents
        })
        
        return payoffs
    
    def get_cooperation_ratio(self) -> float:
        """Get the cooperation ratio from the last round."""
        if not self.history:
            return 0.5  # Default for first round
        return self.history[-1]['cooperation_rate']
    
    def get_results(self) -> Dict:
        """Get comprehensive game results."""
        return {
            'num_agents': self.num_agents,
            'num_rounds': self.num_rounds,
            'final_scores': self.game_state.total_scores.copy(),
            'cooperation_rates': self.game_state.get_cooperation_rates(),
            'history': self.history,
            'average_cooperation': sum(h['cooperation_rate'] for h in self.history) / len(self.history) if self.history else 0
        }
    
    def reset(self):
        """Reset the game state for a new simulation."""
        self.game_state = GameState(self.num_agents)
        self.history = []