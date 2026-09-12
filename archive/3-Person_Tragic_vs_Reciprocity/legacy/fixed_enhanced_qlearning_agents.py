"""
Fixed Enhanced Q-Learning Agents without inheritance issues

This module implements enhanced Q-learning agents that do NOT inherit
from SimpleQLearningAgent, avoiding the shared Q-table bug completely.
"""

import random
import math
from collections import defaultdict, deque

from main_neighbourhood import NPERSON_COOPERATE, NPERSON_DEFECT
from main_pairwise import PAIRWISE_COOPERATE, PAIRWISE_DEFECT


class FixedEnhancedQLearningAgent:
    """
    Enhanced Q-Learning agent with proper per-opponent Q-tables.
    
    This implementation does NOT inherit from SimpleQLearningAgent,
    ensuring complete independence in pairwise learning.
    """
    
    def __init__(self, agent_id, 
                 learning_rate=0.1, 
                 discount_factor=0.9,
                 epsilon=0.3,
                 epsilon_decay=0.999,
                 epsilon_min=0.01,
                 exploration_rate=0.0,
                 state_type="basic",  # 'basic', 'fine', 'memory_enhanced'
                 memory_length=5,
                 **kwargs):
        
        self.agent_id = agent_id
        self.strategy_name = "EnhancedQLearning"
        
        # Q-learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.exploration_rate = exploration_rate
        
        # State configuration
        self.state_type = state_type
        self.memory_length = memory_length
        
        # Statistics
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0
        
        # Initialize all data structures
        self.reset_all()
    
    def reset_all(self):
        """Initialize all data structures."""
        # Pairwise mode structures
        self.pairwise_q_tables = {}  # {opponent_id: {state: {action: q_value}}}
        self.opponent_last_moves = {}  # {opponent_id: last_move}
        self.pairwise_last_states = {}
        self.pairwise_last_actions = {}
        
        # Neighborhood mode structures
        self.neighborhood_q_table = {}  # {state: {action: q_value}}
        self.last_neighborhood_state = None
        self.last_neighborhood_action = None
        self.last_coop_ratio = None
        
        # Memory structures for enhanced states
        if self.state_type == "memory_enhanced":
            self.pairwise_memories = {}  # {opponent_id: deque}
            self.neighborhood_memory = deque(maxlen=self.memory_length)
    
    def _initialize_q_dict(self):
        """Initialize Q-values for a new state."""
        return {
            'cooperate': random.uniform(-0.01, 0.01),
            'defect': random.uniform(-0.01, 0.01)
        }
    
    def _get_pairwise_state(self, opponent_id):
        """Get state representation for pairwise interaction."""
        if self.state_type == "basic":
            # Simple state based on opponent's last move
            if opponent_id not in self.opponent_last_moves:
                return 'initial'
            last_move = self.opponent_last_moves[opponent_id]
            return 'opp_coop' if last_move == PAIRWISE_COOPERATE else 'opp_defect'
        
        elif self.state_type == "memory_enhanced":
            # Include memory of recent interactions
            if opponent_id not in self.opponent_last_moves:
                base_state = 'initial'
            else:
                last_move = self.opponent_last_moves[opponent_id]
                base_state = 'opp_coop' if last_move == PAIRWISE_COOPERATE else 'opp_defect'
            
            # Add memory component
            if opponent_id not in self.pairwise_memories:
                self.pairwise_memories[opponent_id] = deque(maxlen=self.memory_length)
            
            memory = self.pairwise_memories[opponent_id]
            if not memory:
                memory_sig = "mem_none"
            else:
                coop_count = sum(1 for m in memory if m == 'cooperate')
                memory_sig = f"mem_c{coop_count}_d{len(memory)-coop_count}"
            
            return f"{base_state}_{memory_sig}"
        
        else:  # Default to basic
            return self._get_pairwise_state_basic(opponent_id)
    
    def _get_pairwise_state_basic(self, opponent_id):
        """Basic state representation."""
        if opponent_id not in self.opponent_last_moves:
            return 'initial'
        last_move = self.opponent_last_moves[opponent_id]
        return 'opp_coop' if last_move == PAIRWISE_COOPERATE else 'opp_defect'
    
    def _get_neighborhood_state(self, coop_ratio):
        """Get state representation for neighborhood game."""
        if coop_ratio is None:
            base_state = 'initial'
        elif self.state_type == "basic":
            # Basic discretization
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
            # Fine-grained discretization
            base_state = f"coop_{int(coop_ratio * 10)}"
        else:
            # Default to basic
            if coop_ratio <= 0.33:
                base_state = 'low'
            elif coop_ratio <= 0.67:
                base_state = 'medium'
            else:
                base_state = 'high'
        
        # Add memory if using memory-enhanced states
        if self.state_type == "memory_enhanced":
            if not self.neighborhood_memory:
                memory_sig = "mem_none"
            else:
                coop_count = sum(1 for a in self.neighborhood_memory if a == 'cooperate')
                memory_sig = f"mem_c{coop_count}_d{len(self.neighborhood_memory)-coop_count}"
            return f"{base_state}_{memory_sig}"
        
        return base_state
    
    def _choose_action_epsilon_greedy(self, q_values):
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(['cooperate', 'defect'])
        
        # Exploitation
        if q_values['cooperate'] > q_values['defect']:
            return 'cooperate'
        elif q_values['defect'] > q_values['cooperate']:
            return 'defect'
        else:
            # Break ties randomly
            return random.choice(['cooperate', 'defect'])
    
    # Pairwise mode methods
    def choose_action_pairwise(self, opponent_id, current_round_in_episode=None):
        """Choose action for pairwise mode with proper per-opponent Q-tables."""
        state = self._get_pairwise_state(opponent_id)
        
        # Initialize Q-table for this opponent if needed
        if opponent_id not in self.pairwise_q_tables:
            self.pairwise_q_tables[opponent_id] = {}
        
        # Initialize state if needed
        if state not in self.pairwise_q_tables[opponent_id]:
            self.pairwise_q_tables[opponent_id][state] = self._initialize_q_dict()
        
        # Get Q-values for THIS opponent
        q_values = self.pairwise_q_tables[opponent_id][state]
        
        # Choose action
        action = self._choose_action_epsilon_greedy(q_values)
        
        # Store for later update
        self.pairwise_last_states[opponent_id] = state
        self.pairwise_last_actions[opponent_id] = action
        
        # Update memory if using memory-enhanced states
        if self.state_type == "memory_enhanced":
            if opponent_id not in self.pairwise_memories:
                self.pairwise_memories[opponent_id] = deque(maxlen=self.memory_length)
            self.pairwise_memories[opponent_id].append(action)
        
        # Convert to game format
        intended_move = PAIRWISE_COOPERATE if action == 'cooperate' else PAIRWISE_DEFECT
        
        # Apply exploration noise
        actual_move = intended_move
        if random.random() < self.exploration_rate:
            actual_move = 1 - intended_move
        
        return intended_move, actual_move
    
    def record_interaction(self, opponent_id, opponent_actual_move, my_payoff,
                          my_intended_move, my_actual_move, round_num_in_episode):
        """Record interaction for pairwise mode with proper per-opponent Q-learning."""
        self.total_score += my_payoff
        
        if my_actual_move == PAIRWISE_COOPERATE:
            self.num_cooperations += 1
        else:
            self.num_defections += 1
        
        # Update opponent history BEFORE getting next state
        self.opponent_last_moves[opponent_id] = opponent_actual_move
        
        # Update Q-value if we have previous state/action
        if opponent_id in self.pairwise_last_states:
            old_state = self.pairwise_last_states[opponent_id]
            action = self.pairwise_last_actions[opponent_id]
            
            # Get new state AFTER updating opponent history
            new_state = self._get_pairwise_state(opponent_id)
            
            # Initialize new state if needed
            if new_state not in self.pairwise_q_tables[opponent_id]:
                self.pairwise_q_tables[opponent_id][new_state] = self._initialize_q_dict()
            
            # Q-learning update for THIS opponent's Q-table
            old_q = self.pairwise_q_tables[opponent_id][old_state][action]
            max_next_q = max(self.pairwise_q_tables[opponent_id][new_state].values())
            
            new_q = old_q + self.learning_rate * (
                my_payoff + self.discount_factor * max_next_q - old_q
            )
            
            self.pairwise_q_tables[opponent_id][old_state][action] = new_q
    
    def clear_opponent_history(self, opponent_id):
        """Clear history for a specific opponent (between episodes)."""
        if opponent_id in self.opponent_last_moves:
            del self.opponent_last_moves[opponent_id]
        if opponent_id in self.pairwise_last_states:
            del self.pairwise_last_states[opponent_id]
        if opponent_id in self.pairwise_last_actions:
            del self.pairwise_last_actions[opponent_id]
    
    # Neighborhood mode methods
    def choose_action(self, prev_round_overall_coop_ratio, current_round_num):
        """Choose action for neighborhood mode."""
        state = self._get_neighborhood_state(prev_round_overall_coop_ratio)
        
        # Initialize state if needed
        if state not in self.neighborhood_q_table:
            self.neighborhood_q_table[state] = self._initialize_q_dict()
        
        # Get Q-values
        q_values = self.neighborhood_q_table[state]
        
        # Choose action
        action = self._choose_action_epsilon_greedy(q_values)
        
        # Store for later update
        self.last_neighborhood_state = state
        self.last_neighborhood_action = action
        self.last_coop_ratio = prev_round_overall_coop_ratio
        
        # Update memory if using memory-enhanced states
        if self.state_type == "memory_enhanced":
            self.neighborhood_memory.append(action)
        
        # Convert to game format
        intended_move = NPERSON_COOPERATE if action == 'cooperate' else NPERSON_DEFECT
        
        # Apply exploration noise
        actual_move = intended_move
        if random.random() < self.exploration_rate:
            actual_move = 1 - intended_move
        
        return intended_move, actual_move
    
    def record_round_outcome(self, my_actual_move, payoff, new_coop_ratio=None):
        """Record outcome for neighborhood mode."""
        self.total_score += payoff
        
        if my_actual_move == NPERSON_COOPERATE:
            self.num_cooperations += 1
        else:
            self.num_defections += 1
        
        # Update Q-value if we have previous state/action
        if self.last_neighborhood_state is not None:
            # Use provided new_coop_ratio or last known ratio
            if new_coop_ratio is None:
                new_coop_ratio = self.last_coop_ratio
            
            new_state = self._get_neighborhood_state(new_coop_ratio)
            
            # Initialize new state if needed
            if new_state not in self.neighborhood_q_table:
                self.neighborhood_q_table[new_state] = self._initialize_q_dict()
            
            # Q-learning update
            old_q = self.neighborhood_q_table[self.last_neighborhood_state][self.last_neighborhood_action]
            max_next_q = max(self.neighborhood_q_table[new_state].values())
            
            new_q = old_q + self.learning_rate * (
                payoff + self.discount_factor * max_next_q - old_q
            )
            
            self.neighborhood_q_table[self.last_neighborhood_state][self.last_neighborhood_action] = new_q
    
    def get_cooperation_rate(self):
        """Get cooperation rate."""
        total_moves = self.num_cooperations + self.num_defections
        return self.num_cooperations / total_moves if total_moves > 0 else 0.0
    
    def reset(self):
        """Reset for a new run."""
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0
        
        # Clear last round info
        self.last_neighborhood_state = None
        self.last_neighborhood_action = None
        self.last_coop_ratio = None
        
        # Clear pairwise tracking but keep Q-tables
        self.pairwise_last_states = {}
        self.pairwise_last_actions = {}
        
        # Clear memories if using memory-enhanced states
        if self.state_type == "memory_enhanced":
            self.neighborhood_memory.clear()
            # Don't clear pairwise memories - they track longer-term patterns
        
        # Apply epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def reset_for_new_tournament(self):
        """Reset for pairwise tournament."""
        self.reset()
        self.opponent_last_moves = {}


# Alias for compatibility
EnhancedQLearningAgent = FixedEnhancedQLearningAgent


# Factory function
def create_fixed_enhanced_qlearning(agent_id, **kwargs):
    """Factory function for creating fixed enhanced Q-learning agents."""
    return FixedEnhancedQLearningAgent(agent_id, **kwargs)