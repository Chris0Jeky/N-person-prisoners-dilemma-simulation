"""
Unified agent module for N-Person Prisoner's Dilemma simulations.

This module contains all agent implementations:
- Static agents (TFT, AllD, AllC)
- Q-Learning agents (Simple, Enhanced, NPDL-based)
"""

import random
import math
from collections import defaultdict, deque

# Game constants
COOPERATE = 0
DEFECT = 1

class BaseAgent:
    """Base class for all agents with common functionality."""
    
    def __init__(self, agent_id, exploration_rate=0.0):
        self.agent_id = agent_id
        self.exploration_rate = exploration_rate
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0
        
    def get_cooperation_rate(self):
        """Calculate cooperation rate."""
        total_moves = self.num_cooperations + self.num_defections
        return self.num_cooperations / total_moves if total_moves > 0 else 0.0
    
    def reset_stats(self):
        """Reset statistics for new simulation."""
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0


class StaticAgent(BaseAgent):
    """Static strategy agents (TFT, AllD, AllC, Random)."""
    
    def __init__(self, agent_id, strategy_name, exploration_rate=0.0):
        super().__init__(agent_id, exploration_rate)
        self.strategy_name = strategy_name
        self.opponent_last_moves = {}
        
    def choose_action(self, context, mode='neighborhood'):
        """Choose action based on strategy and mode."""
        if mode == 'neighborhood':
            return self._choose_neighborhood_action(context)
        else:
            return self._choose_pairwise_action(context)
    
    def _choose_neighborhood_action(self, prev_coop_ratio):
        """Choose action for neighborhood mode."""
        if self.strategy_name == "pTFT":
            if prev_coop_ratio is None:
                intended_move = COOPERATE
            else:
                intended_move = COOPERATE if random.random() < prev_coop_ratio else DEFECT
        elif self.strategy_name == "pTFT-Threshold":
            if prev_coop_ratio is None:
                intended_move = COOPERATE
            elif prev_coop_ratio >= 0.5:
                intended_move = COOPERATE
            else:
                prob_coop = prev_coop_ratio / 0.5
                intended_move = COOPERATE if random.random() < prob_coop else DEFECT
        elif self.strategy_name == "AllD":
            intended_move = DEFECT
        elif self.strategy_name == "AllC":
            intended_move = COOPERATE
        elif self.strategy_name == "Random":
            intended_move = random.choice([COOPERATE, DEFECT])
        else:
            raise ValueError(f"Unknown strategy: {self.strategy_name}")
        
        # Apply exploration
        actual_move = intended_move
        if random.random() < self.exploration_rate:
            actual_move = 1 - intended_move
            
        return intended_move, actual_move
    
    def _choose_pairwise_action(self, opponent_id):
        """Choose action for pairwise mode."""
        if self.strategy_name in ["TFT", "pTFT", "pTFT-Threshold"]:
            if opponent_id not in self.opponent_last_moves:
                intended_move = COOPERATE
            else:
                intended_move = self.opponent_last_moves[opponent_id]
        elif self.strategy_name == "AllD":
            intended_move = DEFECT
        elif self.strategy_name == "AllC":
            intended_move = COOPERATE
        elif self.strategy_name == "Random":
            intended_move = random.choice([COOPERATE, DEFECT])
        else:
            raise ValueError(f"Unknown strategy: {self.strategy_name}")
        
        # Apply exploration
        actual_move = intended_move
        if random.random() < self.exploration_rate:
            actual_move = 1 - intended_move
            
        return intended_move, actual_move
    
    def update_neighborhood(self, my_move, payoff):
        """Update after neighborhood interaction."""
        self.total_score += payoff
        if my_move == COOPERATE:
            self.num_cooperations += 1
        else:
            self.num_defections += 1
    
    def update_pairwise(self, opponent_id, opponent_move, my_move, payoff):
        """Update after pairwise interaction."""
        self.total_score += payoff
        self.opponent_last_moves[opponent_id] = opponent_move
        if my_move == COOPERATE:
            self.num_cooperations += 1
        else:
            self.num_defections += 1
    
    def clear_opponent_history(self, opponent_id=None):
        """Clear opponent history."""
        if opponent_id is None:
            self.opponent_last_moves = {}
        elif opponent_id in self.opponent_last_moves:
            del self.opponent_last_moves[opponent_id]
    
    def reset(self):
        """Full reset for new simulation."""
        self.reset_stats()
        self.opponent_last_moves = {}


class SimpleQLearningAgent(BaseAgent):
    """Simple Q-Learning implementation for baseline comparison."""
    
    def __init__(self, agent_id, learning_rate=0.1, discount_factor=0.9,
                 epsilon=0.1, exploration_rate=0.0):
        super().__init__(agent_id, exploration_rate)
        
        # Q-learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Q-table
        self.q_table = {}
        
        # Memory
        self.last_state = None
        self.last_action = None
        self.episode_count = 0
        
        # Opponent tracking
        self.opponent_last_moves = {}
        if opponent_modeling:
            self.opponent_history = defaultdict(lambda: {'C': 0, 'D': 0})
            self.opponent_models = {}
    
    def _get_state(self, context, mode='neighborhood'):
        """Get state representation based on context and mode."""
        if mode == 'neighborhood':
            return self._get_neighborhood_state(context)
        else:
            return self._get_pairwise_state(context)
    
    def _get_neighborhood_state(self, coop_ratio):
        """Get state for neighborhood mode."""
        if coop_ratio is None:
            return 'initial'
        
        # Adjust for exclude_self if needed
        if self.exclude_self and hasattr(self, 'last_own_move'):
            # This would need the total agent count to properly adjust
            # For now, using approximation
            adjusted_ratio = coop_ratio
        else:
            adjusted_ratio = coop_ratio
        
        # State representation based on type
        if self.state_type == "basic":
            if adjusted_ratio <= 0.2:
                state = 'very_low'
            elif adjusted_ratio <= 0.4:
                state = 'low'
            elif adjusted_ratio <= 0.6:
                state = 'medium'
            elif adjusted_ratio <= 0.8:
                state = 'high'
            else:
                state = 'very_high'
        elif self.state_type == "fine":
            state = f"coop_{int(adjusted_ratio * 10) / 10:.1f}"
        elif self.state_type == "coarse":
            if adjusted_ratio <= 0.33:
                state = 'low'
            elif adjusted_ratio <= 0.67:
                state = 'medium'
            else:
                state = 'high'
        else:
            state = f"ratio_{adjusted_ratio:.2f}"
        
        # Add opponent modeling if enabled
        if self.opponent_modeling and hasattr(self, 'opponent_models') and self.opponent_models:
            avg_pred = sum(self.opponent_models.values()) / len(self.opponent_models)
            state = (state, f"pred_{avg_pred:.1f}")
            
        return state
    
    def _get_pairwise_state(self, opponent_id):
        """Get state for pairwise mode."""
        if opponent_id not in self.opponent_last_moves:
            return 'initial'
        
        last_move = self.opponent_last_moves[opponent_id]
        state = 'opp_coop' if last_move == COOPERATE else 'opp_defect'
        
        # Add opponent modeling if enabled
        if self.opponent_modeling and opponent_id in self.opponent_models:
            pred = self.opponent_models[opponent_id]
            state = (state, f"pred_{pred:.1f}")
            
        return state
    
    def _ensure_state_exists(self, state):
        """Initialize Q-values for new states."""
        if state not in self.q_table:
            # Small random initialization to break ties
            self.q_table[state] = {
                'cooperate': random.uniform(-0.01, 0.01),
                'defect': random.uniform(-0.01, 0.01)
            }
    
    def _choose_action_epsilon_greedy(self, state):
        """Choose action using epsilon-greedy policy."""
        self._ensure_state_exists(state)
        
        # Exploration
        if random.random() < self.epsilon:
            return random.choice(['cooperate', 'defect'])
        
        # Exploitation
        q_values = self.q_table[state]
        if q_values['cooperate'] > q_values['defect']:
            return 'cooperate'
        elif q_values['defect'] > q_values['cooperate']:
            return 'defect'
        else:
            return random.choice(['cooperate', 'defect'])
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using standard Q-learning formula."""
        self._ensure_state_exists(state)
        self._ensure_state_exists(next_state)
        
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self):
        """Apply epsilon decay."""
        if self.epsilon_decay < 1.0:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def choose_action(self, context, mode='neighborhood'):
        """Choose action based on Q-learning."""
        state = self._get_state(context, mode)
        
        # Choose action
        action = self._choose_action_epsilon_greedy(state)
        
        # Store for update
        self.last_state = state
        self.last_action = action
        
        # Convert to game format
        intended_move = COOPERATE if action == 'cooperate' else DEFECT
        
        # Apply exploration rate
        actual_move = intended_move
        if random.random() < self.exploration_rate:
            actual_move = 1 - intended_move
        
        # Track own move for exclude_self
        if self.exclude_self:
            self.last_own_move = actual_move
            
        return intended_move, actual_move
    
    def update_neighborhood(self, my_move, payoff, next_context=None):
        """Update after neighborhood interaction."""
        self.total_score += payoff
        if my_move == COOPERATE:
            self.num_cooperations += 1
        else:
            self.num_defections += 1
        
        # Update Q-value if we have previous state
        if self.last_state is not None and self.last_action is not None:
            if next_context is not None:
                next_state = self._get_state(next_context, 'neighborhood')
            else:
                next_state = 'terminal'
                self._ensure_state_exists(next_state)
            
            self.update_q_value(self.last_state, self.last_action, payoff, next_state)
    
    def update_pairwise(self, opponent_id, opponent_move, my_move, payoff):
        """Update after pairwise interaction."""
        self.total_score += payoff
        self.opponent_last_moves[opponent_id] = opponent_move
        
        if my_move == COOPERATE:
            self.num_cooperations += 1
        else:
            self.num_defections += 1
        
        # Update opponent model if enabled
        if self.opponent_modeling:
            if opponent_move == COOPERATE:
                self.opponent_history[opponent_id]['C'] += 1
            else:
                self.opponent_history[opponent_id]['D'] += 1
            
            total = self.opponent_history[opponent_id]['C'] + self.opponent_history[opponent_id]['D']
            if total > 0:
                self.opponent_models[opponent_id] = self.opponent_history[opponent_id]['C'] / total
        
        # Update Q-value
        if self.last_state is not None and self.last_action is not None:
            next_state = self._get_state(opponent_id, 'pairwise')
            self.update_q_value(self.last_state, self.last_action, payoff, next_state)
    
    def clear_opponent_history(self, opponent_id=None):
        """Clear opponent history."""
        if opponent_id is None:
            self.opponent_last_moves = {}
        elif opponent_id in self.opponent_last_moves:
            del self.opponent_last_moves[opponent_id]
    
    def reset(self):
        """Reset for new simulation."""
        self.reset_stats()
        self.last_state = None
        self.last_action = None
        self.episode_count += 1
        self.decay_epsilon()
        
    def reset_full(self):
        """Full reset including Q-table."""
        self.reset()
        self.q_table = {}
        self.opponent_last_moves = {}
        self.epsilon = self.initial_epsilon
        self.episode_count = 0
        if self.opponent_modeling:
            self.opponent_history = defaultdict(lambda: {'C': 0, 'D': 0})
            self.opponent_models = {}


def create_agent(agent_id, agent_type, **kwargs):
    """Factory function to create agents."""
    if agent_type in ["TFT", "pTFT", "pTFT-Threshold", "AllD", "AllC", "Random"]:
        return StaticAgent(agent_id, agent_type, 
                          exploration_rate=kwargs.get('exploration_rate', 0.0))
    elif agent_type in ["QL", "QLearning"]:
        return QLearningAgent(agent_id, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")