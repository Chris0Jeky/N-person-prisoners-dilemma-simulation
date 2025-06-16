"""
Pairwise Prisoner's Dilemma Game Implementation

Supports tournaments where agents play bilateral games with each other.
"""

from typing import List, Dict, Tuple, Optional
import itertools
from ..models.payoff_matrix import PayoffMatrix
from ..models.game_state import GameState


class PairwiseGame:
    """
    Pairwise Prisoner's Dilemma where agents play bilateral games.
    
    In tournament mode, each agent plays against every other agent.
    """
    
    def __init__(self,
                 num_agents: int,
                 rounds_per_pair: int,
                 num_episodes: int = 1,
                 payoff_matrix: Optional[PayoffMatrix] = None):
        """
        Initialize Pairwise PD game.
        
        Args:
            num_agents: Number of agents in the tournament
            rounds_per_pair: Rounds per pairwise interaction
            num_episodes: Number of episodes (memory resets between episodes)
            payoff_matrix: Payoff matrix (uses default if None)
        """
        self.num_agents = num_agents
        self.rounds_per_pair = rounds_per_pair
        self.num_episodes = num_episodes
        self.payoff_matrix = payoff_matrix or PayoffMatrix()
        
        # Track state for each pair
        self.pair_states = {}  # (agent1_id, agent2_id) -> GameState
        self.tournament_scores = {i: 0.0 for i in range(num_agents)}
        self.tournament_cooperation_counts = {i: 0 for i in range(num_agents)}
        self.tournament_defection_counts = {i: 0 for i in range(num_agents)}
        self.history = []
        
    def _get_pair_key(self, agent1_id: int, agent2_id: int) -> Tuple[int, int]:
        """Get canonical pair key (smaller id first)."""
        return (min(agent1_id, agent2_id), max(agent1_id, agent2_id))
    
    def play_pairwise_round(self, 
                           agent1_id: int,
                           agent2_id: int,
                           action1: int,
                           action2: int) -> Tuple[float, float]:
        """
        Play a single round between two agents.
        
        Args:
            agent1_id: First agent's ID
            agent2_id: Second agent's ID
            action1: First agent's action (0=cooperate, 1=defect)
            action2: Second agent's action (0=cooperate, 1=defect)
            
        Returns:
            Tuple of (payoff1, payoff2)
        """
        payoff1 = self.payoff_matrix.get_pairwise_payoff(action1, action2)
        payoff2 = self.payoff_matrix.get_pairwise_payoff(action2, action1)
        
        # Update tournament scores
        self.tournament_scores[agent1_id] += payoff1
        self.tournament_scores[agent2_id] += payoff2
        
        # Update cooperation/defection counts
        if action1 == 0:
            self.tournament_cooperation_counts[agent1_id] += 1
        else:
            self.tournament_defection_counts[agent1_id] += 1
            
        if action2 == 0:
            self.tournament_cooperation_counts[agent2_id] += 1
        else:
            self.tournament_defection_counts[agent2_id] += 1
        
        return payoff1, payoff2
    
    def get_last_opponent_action(self, 
                                agent_id: int, 
                                opponent_id: int,
                                episode: int) -> Optional[int]:
        """
        Get the last action of an opponent in the current episode.
        
        Args:
            agent_id: ID of the agent asking
            opponent_id: ID of the opponent
            episode: Current episode number
            
        Returns:
            Last action (0 or 1) or None if no history
        """
        # Look through history for this pair in current episode
        for record in reversed(self.history):
            if record['episode'] != episode:
                continue
                
            if (record['agent1_id'] == agent_id and record['agent2_id'] == opponent_id):
                return record['action2']
            elif (record['agent1_id'] == opponent_id and record['agent2_id'] == agent_id):
                return record['action1']
                
        return None
    
    def record_pairwise_interaction(self,
                                  agent1_id: int,
                                  agent2_id: int,
                                  action1: int,
                                  action2: int,
                                  payoff1: float,
                                  payoff2: float,
                                  round_num: int,
                                  episode: int):
        """Record a pairwise interaction in history."""
        self.history.append({
            'agent1_id': agent1_id,
            'agent2_id': agent2_id,
            'action1': action1,
            'action2': action2,
            'payoff1': payoff1,
            'payoff2': payoff2,
            'round': round_num,
            'episode': episode
        })
    
    def get_tournament_results(self) -> Dict:
        """Get comprehensive tournament results."""
        results = []
        
        for agent_id in range(self.num_agents):
            total_moves = (self.tournament_cooperation_counts[agent_id] + 
                          self.tournament_defection_counts[agent_id])
            cooperation_rate = (self.tournament_cooperation_counts[agent_id] / total_moves 
                              if total_moves > 0 else 0)
            
            results.append({
                'agent_id': agent_id,
                'total_score': self.tournament_scores[agent_id],
                'cooperation_rate': cooperation_rate,
                'cooperation_count': self.tournament_cooperation_counts[agent_id],
                'defection_count': self.tournament_defection_counts[agent_id]
            })
        
        # Sort by score
        results.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Calculate overall statistics
        total_cooperations = sum(self.tournament_cooperation_counts.values())
        total_moves = sum(self.tournament_cooperation_counts.values()) + \
                     sum(self.tournament_defection_counts.values())
        overall_cooperation_rate = total_cooperations / total_moves if total_moves > 0 else 0
        
        return {
            'rankings': results,
            'overall_cooperation_rate': overall_cooperation_rate,
            'num_agents': self.num_agents,
            'rounds_per_pair': self.rounds_per_pair,
            'num_episodes': self.num_episodes,
            'history': self.history
        }
    
    def reset(self):
        """Reset the game state for a new tournament."""
        self.pair_states.clear()
        self.tournament_scores = {i: 0.0 for i in range(self.num_agents)}
        self.tournament_cooperation_counts = {i: 0 for i in range(self.num_agents)}
        self.tournament_defection_counts = {i: 0 for i in range(self.num_agents)}
        self.history.clear()