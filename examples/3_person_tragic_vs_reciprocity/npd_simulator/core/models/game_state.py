"""
Game State Management for N-Person Prisoner's Dilemma
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class GameState:
    """
    Tracks the state of an N-Person Prisoner's Dilemma game.
    """
    
    def __init__(self, num_agents: int):
        """
        Initialize game state.
        
        Args:
            num_agents: Number of agents in the game
        """
        self.num_agents = num_agents
        self.round_number = 0
        
        # Track scores and actions
        self.total_scores = defaultdict(float)
        self.cooperation_counts = defaultdict(int)
        self.defection_counts = defaultdict(int)
        
        # Track last round info
        self.last_actions = {}
        self.last_payoffs = {}
        self.last_cooperation_rate = None
        
    def update(self, actions: Dict[int, int], payoffs: Dict[int, float]):
        """
        Update game state after a round.
        
        Args:
            actions: Dictionary mapping agent_id to action
            payoffs: Dictionary mapping agent_id to payoff
        """
        self.round_number += 1
        
        # Update scores and counts
        for agent_id, action in actions.items():
            self.total_scores[agent_id] += payoffs[agent_id]
            
            if action == 0:  # Cooperate
                self.cooperation_counts[agent_id] += 1
            else:  # Defect
                self.defection_counts[agent_id] += 1
        
        # Update last round info
        self.last_actions = actions.copy()
        self.last_payoffs = payoffs.copy()
        
        # Calculate cooperation rate
        num_cooperators = sum(1 for action in actions.values() if action == 0)
        self.last_cooperation_rate = num_cooperators / self.num_agents
    
    def get_agent_stats(self, agent_id: int) -> Dict:
        """
        Get statistics for a specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dictionary of agent statistics
        """
        total_moves = self.cooperation_counts[agent_id] + self.defection_counts[agent_id]
        cooperation_rate = self.cooperation_counts[agent_id] / total_moves if total_moves > 0 else 0
        
        return {
            'total_score': self.total_scores[agent_id],
            'cooperation_count': self.cooperation_counts[agent_id],
            'defection_count': self.defection_counts[agent_id],
            'cooperation_rate': cooperation_rate,
            'rounds_played': total_moves
        }
    
    def get_cooperation_rates(self) -> Dict[int, float]:
        """
        Get cooperation rates for all agents.
        
        Returns:
            Dictionary mapping agent_id to cooperation rate
        """
        rates = {}
        for agent_id in range(self.num_agents):
            stats = self.get_agent_stats(agent_id)
            rates[agent_id] = stats['cooperation_rate']
        return rates
    
    def get_rankings(self) -> List[Tuple[int, float]]:
        """
        Get agent rankings by total score.
        
        Returns:
            List of (agent_id, score) tuples sorted by score descending
        """
        rankings = [(agent_id, score) for agent_id, score in self.total_scores.items()]
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def reset(self):
        """Reset the game state."""
        self.round_number = 0
        self.total_scores.clear()
        self.cooperation_counts.clear()
        self.defection_counts.clear()
        self.last_actions.clear()
        self.last_payoffs.clear()
        self.last_cooperation_rate = None