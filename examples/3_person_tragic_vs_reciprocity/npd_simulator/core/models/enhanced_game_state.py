"""
Enhanced Game State with Agent-Type Tracking for Static Style Visualization
"""

from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from .game_state import GameState


class EnhancedGameState(GameState):
    """
    Extended game state that tracks metrics by agent type for static-style visualization.
    """
    
    def __init__(self, num_agents: int, agents: List['Agent']):
        """
        Initialize enhanced game state.
        
        Args:
            num_agents: Number of agents in the game
            agents: List of agent instances
        """
        super().__init__(num_agents)
        
        # Store agent references for type tracking
        self.agents = {agent.agent_id: agent for agent in agents}
        
        # Track by agent type
        self.agent_types = {}
        self.type_cooperation_counts = defaultdict(lambda: defaultdict(int))
        self.type_defection_counts = defaultdict(lambda: defaultdict(int))
        self.type_scores = defaultdict(lambda: defaultdict(float))
        self.type_cumulative_scores = defaultdict(lambda: defaultdict(list))
        
        # Initialize agent type mapping
        for agent in agents:
            agent_type = self._get_agent_type(agent)
            self.agent_types[agent.agent_id] = agent_type
        
        # Track round-by-round data for each agent
        self.round_history = []
        
    def _get_agent_type(self, agent) -> str:
        """Extract agent type from agent instance."""
        # Get the class name without 'Agent' suffix
        class_name = agent.__class__.__name__
        if class_name.endswith('Agent'):
            return class_name[:-5]
        return class_name
    
    def update(self, actions: Dict[int, int], payoffs: Dict[int, float]):
        """
        Update game state with agent-type tracking.
        
        Args:
            actions: Dictionary mapping agent_id to action
            payoffs: Dictionary mapping agent_id to payoff
        """
        # Call parent update
        super().update(actions, payoffs)
        
        # Track by agent type
        round_data = {
            'round': self.round_number,
            'cooperation_rate': self.last_cooperation_rate,
            'agents': {}
        }
        
        for agent_id, action in actions.items():
            agent_type = self.agent_types[agent_id]
            
            # Update type-specific counts
            if action == 0:  # Cooperate
                self.type_cooperation_counts[agent_type][agent_id] += 1
            else:  # Defect
                self.type_defection_counts[agent_type][agent_id] += 1
            
            # Update type-specific scores
            self.type_scores[agent_type][agent_id] += payoffs[agent_id]
            
            # Track cumulative scores
            cumulative_score = self.total_scores[agent_id]
            self.type_cumulative_scores[agent_type][agent_id].append(cumulative_score)
            
            # Calculate agent-specific cooperation rate
            agent_coop_count = self.cooperation_counts[agent_id]
            agent_total = agent_coop_count + self.defection_counts[agent_id]
            agent_coop_rate = agent_coop_count / agent_total if agent_total > 0 else 0
            
            # Store agent data for this round
            round_data['agents'][agent_id] = {
                'type': agent_type,
                'action': action,
                'payoff': payoffs[agent_id],
                'cumulative_score': cumulative_score,
                'cooperation_rate': agent_coop_rate
            }
        
        self.round_history.append(round_data)
    
    def get_type_cooperation_rates(self) -> Dict[str, float]:
        """
        Get average cooperation rates by agent type.
        
        Returns:
            Dictionary mapping agent type to average cooperation rate
        """
        type_rates = {}
        
        for agent_type in set(self.agent_types.values()):
            total_coops = sum(self.type_cooperation_counts[agent_type].values())
            total_defects = sum(self.type_defection_counts[agent_type].values())
            total_moves = total_coops + total_defects
            
            if total_moves > 0:
                type_rates[agent_type] = total_coops / total_moves
            else:
                type_rates[agent_type] = 0.0
        
        return type_rates
    
    def get_type_average_scores(self) -> Dict[str, float]:
        """
        Get average scores by agent type.
        
        Returns:
            Dictionary mapping agent type to average score
        """
        type_avg_scores = {}
        
        for agent_type in set(self.agent_types.values()):
            agent_scores = list(self.type_scores[agent_type].values())
            if agent_scores:
                type_avg_scores[agent_type] = sum(agent_scores) / len(agent_scores)
            else:
                type_avg_scores[agent_type] = 0.0
        
        return type_avg_scores
    
    def get_type_agents(self, agent_type: str) -> List[int]:
        """
        Get list of agent IDs for a specific type.
        
        Args:
            agent_type: The agent type to filter by
            
        Returns:
            List of agent IDs
        """
        return [aid for aid, atype in self.agent_types.items() if atype == agent_type]
    
    def get_enhanced_results(self) -> Dict:
        """
        Get comprehensive results including type-specific metrics.
        
        Returns:
            Dictionary with enhanced results
        """
        results = {
            'num_agents': self.num_agents,
            'num_rounds': self.round_number,
            'average_cooperation': sum(self.get_cooperation_rates().values()) / self.num_agents,
            'type_cooperation_rates': self.get_type_cooperation_rates(),
            'type_average_scores': self.get_type_average_scores(),
            'agent_stats': [],
            'history': self.round_history
        }
        
        # Add individual agent stats with type info
        for agent_id in range(self.num_agents):
            stats = self.get_agent_stats(agent_id)
            stats['agent_id'] = agent_id
            stats['agent_type'] = self.agent_types[agent_id]
            results['agent_stats'].append(stats)
        
        return results