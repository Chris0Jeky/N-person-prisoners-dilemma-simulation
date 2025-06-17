"""
Enhanced Q-Learning Agents with configurable improvements

This module implements enhanced Q-learning agents that address:
1. Decaying epsilon
2. State representation excluding self
3. Longer training support
4. Opponent modeling

REVAMPED VERSION: Uses SimpleQLearning as baseline with selective enhancements
"""

import random
import math
from collections import defaultdict, deque

from main_neighbourhood import NPERSON_COOPERATE, NPERSON_DEFECT
from main_pairwise import PAIRWISE_COOPERATE, PAIRWISE_DEFECT


class EnhancedQLearningAgent:
    """
    Revamped Enhanced Q-Learning using SimpleQLearning baseline with selective improvements.
    Key changes from original enhanced version:
    - Uses random initialization instead of optimistic (was causing poor performance)
    - Exclude-self is now OFF by default (was causing issues)
    - Simpler state representation by default
    - Epsilon decay is more conservative
    """
    
    def __init__(self, agent_id, 
                 learning_rate=0.1, 
                 discount_factor=0.9,
                 epsilon=0.1,
                 epsilon_decay=0.995,  # More conservative decay
                 epsilon_min=0.01,
                 exploration_rate=0.0,
                 exclude_self=False,  # OFF by default - was hurting performance
                 opponent_modeling=False,  # OFF by default - add only if needed
                 state_type="basic",
                 use_memory=False):  # New: optional memory enhancement
        
        self.agent_id = agent_id
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.exploration_rate = exploration_rate
        self.exclude_self = exclude_self
        self.opponent_modeling = opponent_modeling
        self.state_type = state_type
        self.use_memory = use_memory
        
        # Q-table
        self.q_table = {}
        
        # Episode counter for epsilon decay
        self.episode_count = 0
        self.step_count = 0
        
        # Memory
        self.last_state = None
        self.last_action = None
        self.last_coop_ratio = None  # Simplified from last_others_coop_ratio
        
        # Statistics
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0
        
        # For opponent modeling (lightweight version)
        if opponent_modeling:
            self.opponent_coop_rates = {}  # Simplified: just track cooperation rates
        
        # For pairwise mode
        self.opponent_last_moves = {}
        
        # Memory buffer for enhanced states
        if use_memory:
            self.memory_buffer = deque(maxlen=5)  # Last 5 rounds
        
    def decay_epsilon(self):
        """Apply epsilon decay."""
        if self.epsilon_decay < 1.0:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _get_state_neighborhood(self, coop_ratio):
        """Get state for neighborhood mode - simplified version based on SimpleQLearning."""
        if coop_ratio is None:
            return 'initial'
        
        # Use cooperation ratio directly unless exclude_self is enabled
        if self.exclude_self and self.last_action is not None and coop_ratio is not None:
            # Calculate cooperation ratio excluding self
            total_agents = 3  # Fixed for 3-person game
            total_coops = coop_ratio * total_agents
            
            # Remove self contribution
            if self.last_action == 'cooperate':
                total_coops -= 1
            
            # Calculate others' cooperation ratio
            coop_ratio = total_coops / (total_agents - 1) if total_agents > 1 else 0
        
        # Store for later use
        self.last_coop_ratio = coop_ratio
        
        if self.state_type == "basic":
            # Standard discretization from SimpleQLearning
            if coop_ratio <= 0.2:
                base_state = 'very_low'
            elif coop_ratio <= 0.4:
                base_state = 'low'
            elif coop_ratio <= 0.6:
                base_state = 'medium'
            elif coop_ratio <= 0.8:
                base_state = 'high'
            else:
                base_state = 'very_high'
        elif self.state_type == "fine":
            # Finer discretization (0.1 increments)
            base_state = f"coop_{int(coop_ratio * 10) / 10:.1f}"
        elif self.state_type == "coarse":
            # Coarser discretization
            if coop_ratio <= 0.33:
                base_state = 'low'
            elif coop_ratio <= 0.67:
                base_state = 'medium'
            else:
                base_state = 'high'
        else:
            base_state = 'default'
        
        # Add memory enhancement if enabled
        if self.use_memory and self.memory_buffer:
            recent_coops = sum(1 for m in self.memory_buffer if m == 'cooperate')
            memory_state = f"_mem{recent_coops}"
            return base_state + memory_state
        
        # Add lightweight opponent modeling if enabled
        if self.opponent_modeling and hasattr(self, 'opponent_coop_rates') and self.opponent_coop_rates:
            avg_coop = sum(self.opponent_coop_rates.values()) / len(self.opponent_coop_rates)
            model_state = f"_opp{int(avg_coop * 10)}"
            return base_state + model_state
        
        return base_state
    
    def _update_opponent_model(self, opponent_id, action):
        """Lightweight opponent model update."""
        if not self.opponent_modeling:
            return
            
        # Simple exponential moving average of cooperation rate
        if opponent_id not in self.opponent_coop_rates:
            self.opponent_coop_rates[opponent_id] = 0.5  # Start neutral
        
        # Update with decay factor
        coop_value = 1.0 if action == NPERSON_COOPERATE else 0.0
        self.opponent_coop_rates[opponent_id] = (
            0.9 * self.opponent_coop_rates[opponent_id] + 0.1 * coop_value
        )
    
    def _ensure_state_exists(self, state):
        """Initialize Q-values for new states."""
        if state not in self.q_table:
            # Random initialization like SimpleQLearning (not optimistic)
            self.q_table[state] = {
                'cooperate': random.uniform(-0.01, 0.01),
                'defect': random.uniform(-0.01, 0.01)
            }
    
    def _choose_action_epsilon_greedy(self, state):
        """Choose action using epsilon-greedy policy."""
        self._ensure_state_exists(state)
        
        # Epsilon-greedy
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
        
        # Increment step count
        self.step_count += 1
    
    # Neighborhood mode methods
    def choose_action(self, prev_round_overall_coop_ratio, current_round_num):
        """Choose action for neighborhood mode."""
        # Get state
        state = self._get_state_neighborhood(prev_round_overall_coop_ratio)
        
        # Choose action
        action = self._choose_action_epsilon_greedy(state)
        
        # Store for next update
        self.last_state = state
        self.last_action = action
        
        # Update memory buffer if enabled
        if self.use_memory:
            self.memory_buffer.append(action)
        
        # Convert to game format
        intended_move = NPERSON_COOPERATE if action == 'cooperate' else NPERSON_DEFECT
        
        # Apply exploration rate (for compatibility)
        actual_move = intended_move
        if random.random() < self.exploration_rate:
            actual_move = 1 - intended_move
        
        return intended_move, actual_move
    
    def record_round_outcome(self, my_actual_move, payoff):
        """Record outcome for neighborhood mode."""
        self.total_score += payoff
        
        if my_actual_move == NPERSON_COOPERATE:
            self.num_cooperations += 1
        else:
            self.num_defections += 1
        
        # Update Q-value if we have a previous state
        if self.last_state is not None and self.last_action is not None:
            # Use stored cooperation ratio for next state
            next_state = self._get_state_neighborhood(self.last_coop_ratio)
            self.update_q_value(self.last_state, self.last_action, payoff, next_state)
    
    def get_cooperation_rate(self):
        """Get cooperation rate."""
        total_moves = self.num_cooperations + self.num_defections
        return self.num_cooperations / total_moves if total_moves > 0 else 0.0
    
    def reset(self):
        """Reset for new episode."""
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0
        self.last_state = None
        self.last_action = None
        self.last_others_coop_ratio = None
        
        # Clear memory buffer if used
        if self.use_memory:
            self.memory_buffer.clear()
        
        # Decay epsilon at episode end
        self.episode_count += 1
        self.decay_epsilon()
    
    # Methods for pairwise mode (basic compatibility)
    def choose_action_pairwise(self, opponent_id, current_round_in_episode):
        """Choose action for pairwise mode."""
        # Simple state based on opponent's last move
        if opponent_id in self.opponent_last_moves:
            last_move = self.opponent_last_moves[opponent_id]
            state = 'opp_coop' if last_move == PAIRWISE_COOPERATE else 'opp_defect'
        else:
            state = 'initial'
        
        action = self._choose_action_epsilon_greedy(state)
        intended_move = PAIRWISE_COOPERATE if action == 'cooperate' else PAIRWISE_DEFECT
        
        actual_move = intended_move
        if random.random() < self.exploration_rate:
            actual_move = 1 - intended_move
        
        self.pairwise_last_state = state
        self.pairwise_last_action = action
        
        return intended_move, actual_move
    
    def record_interaction(self, opponent_id, opponent_actual_move, my_payoff,
                          my_intended_move, my_actual_move, round_num_in_episode):
        """Record interaction for pairwise mode."""
        self.total_score += my_payoff
        
        if my_actual_move == PAIRWISE_COOPERATE:
            self.num_cooperations += 1
        else:
            self.num_defections += 1
        
        # Update history
        self.opponent_last_moves[opponent_id] = opponent_actual_move
        
        # Update Q-value
        if hasattr(self, 'pairwise_last_state'):
            next_state = 'opp_coop' if opponent_actual_move == PAIRWISE_COOPERATE else 'opp_defect'
            self.update_q_value(self.pairwise_last_state, self.pairwise_last_action, 
                              my_payoff, next_state)
    
    def clear_opponent_history(self, opponent_id):
        """Clear opponent history."""
        if opponent_id in self.opponent_last_moves:
            del self.opponent_last_moves[opponent_id]
    
    def reset_for_new_tournament(self):
        """Reset for pairwise tournament."""
        self.reset()
        self.opponent_last_moves = {}


class RevampedQLearningAgent(EnhancedQLearningAgent):
    """
    Revamped Q-Learning that combines the best of both implementations.
    
    Key features:
    - Based on SimpleQLearning's successful approach
    - Adds epsilon decay for better exploration-exploitation balance
    - Optional memory enhancement for temporal patterns
    - Lightweight opponent modeling when beneficial
    - Conservative parameter defaults
    """
    
    def __init__(self, agent_id, **kwargs):
        # Set conservative defaults that performed well
        kwargs.setdefault('epsilon_decay', 0.995)  # Gentle decay
        kwargs.setdefault('exclude_self', False)   # Don't exclude by default
        kwargs.setdefault('opponent_modeling', False)  # Off by default
        kwargs.setdefault('use_memory', False)     # Off by default
        
        super().__init__(agent_id, **kwargs)


class AdaptiveQLearningAgent(EnhancedQLearningAgent):
    """
    Adaptive Q-Learning that adjusts its parameters based on performance.
    
    This agent monitors its performance and adjusts exploration and
    learning parameters dynamically.
    """
    
    def __init__(self, agent_id, **kwargs):
        super().__init__(agent_id, **kwargs)
        
        # Performance tracking
        self.recent_scores = deque(maxlen=10)
        self.performance_trend = 0
        
        # Adaptive parameters
        self.min_learning_rate = 0.01
        self.max_learning_rate = 0.3
        
    def adapt_parameters(self):
        """Adapt learning parameters based on recent performance."""
        if len(self.recent_scores) >= 5:
            # Calculate trend
            first_half = sum(list(self.recent_scores)[:5]) / 5
            second_half = sum(list(self.recent_scores)[5:]) / 5
            self.performance_trend = second_half - first_half
            
            # Adjust learning rate based on trend
            if self.performance_trend > 0:
                # Performance improving, reduce learning rate
                self.learning_rate = max(self.min_learning_rate, 
                                       self.learning_rate * 0.95)
            else:
                # Performance declining, increase learning rate
                self.learning_rate = min(self.max_learning_rate,
                                       self.learning_rate * 1.05)
    
    def record_round_outcome(self, my_actual_move, payoff):
        """Record outcome and adapt parameters."""
        super().record_round_outcome(my_actual_move, payoff)
        self.recent_scores.append(payoff)
        self.adapt_parameters()


def create_enhanced_qlearning(agent_id, **kwargs):
    """Factory function for enhanced Q-learning agent."""
    return EnhancedQLearningAgent(agent_id, **kwargs)

def create_revamped_qlearning(agent_id, **kwargs):
    """Factory function for revamped Q-learning agent with best defaults."""
    return RevampedQLearningAgent(agent_id, **kwargs)

def create_adaptive_qlearning(agent_id, **kwargs):
    """Factory function for adaptive Q-learning agent."""
    return AdaptiveQLearningAgent(agent_id, **kwargs)

# Backward compatibility
def create_opponent_modeling_qlearning(agent_id, **kwargs):
    """Factory function for backward compatibility - uses revamped with opponent modeling."""
    kwargs['opponent_modeling'] = True
    return RevampedQLearningAgent(agent_id, **kwargs)