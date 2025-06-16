"""
Extended agents for 3-person games including AllC strategy and Q-learning integration
"""

import random

from main_neighbourhood import (
    NPersonAgent, NPERSON_COOPERATE, NPERSON_DEFECT,
    nperson_move_to_str
)
from main_pairwise import (
    PairwiseAgent, PAIRWISE_COOPERATE, PAIRWISE_DEFECT,
    pairwise_move_to_str  
)


class ExtendedNPersonAgent(NPersonAgent):
    """Extended N-person agent that supports AllC strategy."""
    
    def choose_action(self, prev_round_overall_coop_ratio, current_round_num):
        intended_move = None
        
        if self.strategy_name == "AllC":
            intended_move = NPERSON_COOPERATE
        else:
            # Use parent class logic for other strategies
            return super().choose_action(prev_round_overall_coop_ratio, current_round_num)
        
        # Apply exploration
        actual_move = intended_move
        if random.random() < self.exploration_rate:
            actual_move = 1 - intended_move
            
        return intended_move, actual_move


class ExtendedPairwiseAgent(PairwiseAgent):
    """Extended pairwise agent that supports AllC strategy."""
    
    def choose_action(self, opponent_id, current_round_in_episode):
        intended_move = None
        
        if self.strategy_name == "AllC":
            intended_move = PAIRWISE_COOPERATE
        else:
            # Use parent class logic
            return super().choose_action(opponent_id, current_round_in_episode)
        
        # Apply exploration
        actual_move = intended_move
        if random.random() < self.exploration_rate:
            actual_move = 1 - intended_move
            
        return intended_move, actual_move


# Wrapper classes to make Q-learning agents compatible with the game structure

class QLearningNPersonWrapper:
    """Wrapper to make Q-learning agents work with N-person game."""
    
    def __init__(self, agent_id, qlearning_agent):
        self.agent_id = agent_id
        self.strategy_name = f"QL-{qlearning_agent.__class__.__name__}"
        self.qlearning_agent = qlearning_agent
        
        # Delegate attributes
        self.exploration_rate = qlearning_agent.exploration_rate
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0
    
    def choose_action(self, prev_round_overall_coop_ratio, current_round_num):
        """Delegate to Q-learning agent."""
        return self.qlearning_agent.choose_action(prev_round_overall_coop_ratio, current_round_num)
    
    def record_round_outcome(self, my_actual_move, payoff):
        """Delegate to Q-learning agent."""
        self.qlearning_agent.record_round_outcome(my_actual_move, payoff)
        # Update our own counters
        self.total_score = self.qlearning_agent.total_score
        self.num_cooperations = self.qlearning_agent.num_cooperations  
        self.num_defections = self.qlearning_agent.num_defections
    
    def get_cooperation_rate(self):
        """Delegate to Q-learning agent."""
        return self.qlearning_agent.get_cooperation_rate()
    
    def reset(self):
        """Delegate to Q-learning agent."""
        self.qlearning_agent.reset()
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0


class QLearningPairwiseWrapper:
    """Wrapper to make Q-learning agents work with pairwise game."""
    
    def __init__(self, agent_id, qlearning_agent):
        self.agent_id = agent_id
        self.strategy_name = f"QL-{qlearning_agent.__class__.__name__}"
        self.qlearning_agent = qlearning_agent
        
        # Delegate attributes
        self.exploration_rate = qlearning_agent.exploration_rate
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0
        self.opponent_last_moves = {}
    
    def choose_action(self, opponent_id, current_round_in_episode):
        """Delegate to Q-learning agent."""
        # Compatibility fix - call the pairwise method
        if hasattr(self.qlearning_agent, 'choose_action_pairwise'):
            return self.qlearning_agent.choose_action_pairwise(opponent_id, current_round_in_episode)
        else:
            # Fallback for agents without explicit pairwise method
            return self.qlearning_agent.choose_action(opponent_id, current_round_in_episode)
    
    def record_interaction(self, opponent_id, opponent_actual_move, my_payoff,
                          my_intended_move, my_actual_move, round_num_in_episode):
        """Delegate to Q-learning agent."""
        self.qlearning_agent.record_interaction(
            opponent_id, opponent_actual_move, my_payoff,
            my_intended_move, my_actual_move, round_num_in_episode
        )
        # Update our counters
        self.total_score = self.qlearning_agent.total_score
        self.num_cooperations = self.qlearning_agent.num_cooperations
        self.num_defections = self.qlearning_agent.num_defections
        self.opponent_last_moves = self.qlearning_agent.opponent_last_moves
    
    def clear_opponent_history(self, opponent_id):
        """Delegate to Q-learning agent."""
        self.qlearning_agent.clear_opponent_history(opponent_id)
        if opponent_id in self.opponent_last_moves:
            del self.opponent_last_moves[opponent_id]
    
    def get_cooperation_rate(self):
        """Delegate to Q-learning agent."""
        return self.qlearning_agent.get_cooperation_rate()
    
    def reset_for_new_tournament(self):
        """Delegate to Q-learning agent."""
        self.qlearning_agent.reset_for_new_tournament()
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0
        self.opponent_last_moves = {}


# Need to import random for exploration
import random