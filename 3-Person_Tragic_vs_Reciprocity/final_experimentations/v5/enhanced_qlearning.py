#!/usr/bin/env python3
"""
Enhanced Q-Learning Agent with proper pairwise and neighborhood handling.

This implementation:
1. Properly tracks per-opponent states in pairwise mode
2. Uses epsilon decay for better exploration-exploitation balance
3. Supports memory-enhanced states for temporal patterns
4. Maintains separate Q-tables for pairwise and neighborhood modes
"""

import random
import numpy as np
from collections import deque, defaultdict
from final_agents import BaseAgent, COOPERATE, DEFECT


class EnhancedQLearningAgent(BaseAgent):
    """
    Enhanced Q-Learning agent with epsilon decay and advanced state representations.
    
    Key features:
    - Proper per-opponent tracking in pairwise mode
    - Epsilon decay for exploration-exploitation balance
    - Optional memory-enhanced states
    - Separate handling for pairwise and neighborhood games
    """
    
    def __init__(self, agent_id, 
                 learning_rate=0.1, 
                 discount_factor=0.9,
                 epsilon=0.3,  # Higher starting epsilon
                 epsilon_decay=0.999,
                 epsilon_min=0.01,
                 state_type="basic",  # 'basic', 'fine', 'memory_enhanced'
                 memory_length=5,
                 exploration_rate=0.0,  # Additional random exploration
                 **kwargs):
        
        super().__init__(agent_id, "EnhancedQLearning")
        
        # Q-learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.exploration_rate = exploration_rate
        
        # State representation
        self.state_type = state_type
        self.memory_length = memory_length
        
        # Initialize Q-tables and tracking structures
        self.reset()
    
    def reset(self):
        """Reset agent state and apply epsilon decay."""
        super().reset()
        
        # Q-tables for different modes
        self.pairwise_q_tables = {}  # {opponent_id: {state: {action: q_value}}}
        self.neighborhood_q_table = {}  # {state: {action: q_value}}
        
        # State tracking
        self.pairwise_histories = {}  # {opponent_id: deque of (my_move, opp_move)}
        self.pairwise_last_contexts = {}  # {opponent_id: {'state': state, 'action': action}}
        
        # Memory for enhanced states
        if self.state_type == "memory_enhanced":
            self.pairwise_memories = {}  # {opponent_id: deque of my actions}
            self.neighborhood_memory = deque(maxlen=self.memory_length)
        
        # Neighborhood tracking
        self.last_neighborhood_context = None
        
        # Apply epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _initialize_q_dict(self):
        """Initialize Q-values for a new state."""
        return {'cooperate': 0.0, 'defect': 0.0}
    
    def _get_pairwise_state(self, opponent_id):
        """Get state representation for pairwise interaction."""
        if opponent_id not in self.pairwise_histories:
            self.pairwise_histories[opponent_id] = deque(maxlen=2)
            
        history = self.pairwise_histories[opponent_id]
        
        if len(history) < 2:
            base_state = "start"
        else:
            base_state = str(tuple(history))
        
        # Enhance state based on type
        if self.state_type == "memory_enhanced":
            if opponent_id not in self.pairwise_memories:
                self.pairwise_memories[opponent_id] = deque(maxlen=self.memory_length)
            
            memory = self.pairwise_memories[opponent_id]
            if not memory:
                memory_sig = "mem_none"
            else:
                coop_count = sum(1 for a in memory if a == COOPERATE)
                memory_sig = f"mem_c{coop_count}_d{len(memory)-coop_count}"
            
            return f"{base_state}_{memory_sig}"
        
        return base_state
    
    def _get_neighborhood_state(self, coop_ratio):
        """Get state representation for neighborhood game."""
        if coop_ratio is None:
            base_state = "initial"
        elif self.state_type == "basic":
            # Basic discretization
            if coop_ratio <= 0.33:
                base_state = "low"
            elif coop_ratio <= 0.67:
                base_state = "medium"
            else:
                base_state = "high"
        elif self.state_type == "fine":
            # Fine-grained discretization
            base_state = f"coop_{int(coop_ratio * 10)}"
        else:
            # Default to basic
            if coop_ratio <= 0.33:
                base_state = "low"
            elif coop_ratio <= 0.67:
                base_state = "medium"
            else:
                base_state = "high"
        
        # Add memory enhancement if enabled
        if self.state_type == "memory_enhanced" and hasattr(self, 'neighborhood_memory'):
            if not self.neighborhood_memory:
                memory_sig = "mem_none"
            else:
                coop_count = sum(1 for a in self.neighborhood_memory if a == COOPERATE)
                memory_sig = f"mem_c{coop_count}_d{len(self.neighborhood_memory)-coop_count}"
            return f"{base_state}_{memory_sig}"
        
        return base_state
    
    def _choose_action_epsilon_greedy(self, q_values):
        """Choose action using epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            # Explore
            return random.choice(['cooperate', 'defect'])
        else:
            # Exploit
            if q_values['cooperate'] >= q_values['defect']:
                return 'cooperate'
            else:
                return 'defect'
    
    def choose_pairwise_action(self, opponent_id):
        """Choose action for pairwise interaction."""
        # Get state
        state = self._get_pairwise_state(opponent_id)
        
        # Initialize Q-table for this opponent if needed
        if opponent_id not in self.pairwise_q_tables:
            self.pairwise_q_tables[opponent_id] = {}
        
        # Initialize Q-values for this state if needed
        if state not in self.pairwise_q_tables[opponent_id]:
            self.pairwise_q_tables[opponent_id][state] = self._initialize_q_dict()
        
        # Get Q-values
        q_values = self.pairwise_q_tables[opponent_id][state]
        
        # Choose action
        action = self._choose_action_epsilon_greedy(q_values)
        
        # Store context for learning
        self.pairwise_last_contexts[opponent_id] = {
            'state': state,
            'action': action
        }
        
        # Update memory if using memory-enhanced states
        if self.state_type == "memory_enhanced":
            if opponent_id not in self.pairwise_memories:
                self.pairwise_memories[opponent_id] = deque(maxlen=self.memory_length)
            self.pairwise_memories[opponent_id].append(COOPERATE if action == 'cooperate' else DEFECT)
        
        # Convert to game action
        intended_move = COOPERATE if action == 'cooperate' else DEFECT
        
        # Apply exploration noise if configured
        if random.random() < self.exploration_rate:
            return 1 - intended_move
        
        return intended_move
    
    def record_pairwise_outcome(self, opponent_id, my_move, opponent_move, reward):
        """Record outcome of pairwise interaction and update Q-values."""
        self.total_score += reward
        
        # Update history
        if opponent_id not in self.pairwise_histories:
            self.pairwise_histories[opponent_id] = deque(maxlen=2)
        self.pairwise_histories[opponent_id].append((my_move, opponent_move))
        
        # Get stored context
        if opponent_id not in self.pairwise_last_contexts:
            return  # No previous action to update
        
        context = self.pairwise_last_contexts[opponent_id]
        old_state = context['state']
        action = context['action']
        
        # Get new state
        new_state = self._get_pairwise_state(opponent_id)
        
        # Initialize new state if needed
        if new_state not in self.pairwise_q_tables[opponent_id]:
            self.pairwise_q_tables[opponent_id][new_state] = self._initialize_q_dict()
        
        # Q-learning update
        old_q = self.pairwise_q_tables[opponent_id][old_state][action]
        max_next_q = max(self.pairwise_q_tables[opponent_id][new_state].values())
        
        new_q = old_q + self.learning_rate * (reward + self.discount_factor * max_next_q - old_q)
        self.pairwise_q_tables[opponent_id][old_state][action] = new_q
    
    def choose_neighborhood_action(self, coop_ratio):
        """Choose action for neighborhood game."""
        # Get state
        state = self._get_neighborhood_state(coop_ratio)
        
        # Initialize Q-values for this state if needed
        if state not in self.neighborhood_q_table:
            self.neighborhood_q_table[state] = self._initialize_q_dict()
        
        # Get Q-values
        q_values = self.neighborhood_q_table[state]
        
        # Choose action
        action = self._choose_action_epsilon_greedy(q_values)
        
        # Store context for learning
        self.last_neighborhood_context = {
            'state': state,
            'action': action
        }
        
        # Update memory if using memory-enhanced states
        if self.state_type == "memory_enhanced":
            self.neighborhood_memory.append(COOPERATE if action == 'cooperate' else DEFECT)
        
        # Convert to game action
        intended_move = COOPERATE if action == 'cooperate' else DEFECT
        
        # Apply exploration noise if configured
        if random.random() < self.exploration_rate:
            return 1 - intended_move
        
        return intended_move
    
    def record_neighborhood_outcome(self, coop_ratio, reward):
        """Record outcome of neighborhood round and update Q-values."""
        self.total_score += reward
        
        # Check if we have a previous action to update
        if self.last_neighborhood_context is None:
            return
        
        old_state = self.last_neighborhood_context['state']
        action = self.last_neighborhood_context['action']
        
        # Get new state
        new_state = self._get_neighborhood_state(coop_ratio)
        
        # Initialize new state if needed
        if new_state not in self.neighborhood_q_table:
            self.neighborhood_q_table[new_state] = self._initialize_q_dict()
        
        # Q-learning update
        old_q = self.neighborhood_q_table[old_state][action]
        max_next_q = max(self.neighborhood_q_table[new_state].values())
        
        new_q = old_q + self.learning_rate * (reward + self.discount_factor * max_next_q - old_q)
        self.neighborhood_q_table[old_state][action] = new_q


class RevampedQLearningAgent(EnhancedQLearningAgent):
    """
    Revamped Q-Learning with conservative defaults that work well.
    """
    def __init__(self, agent_id, **kwargs):
        # Set conservative defaults
        kwargs.setdefault('epsilon_decay', 0.995)  # Gentle decay
        kwargs.setdefault('state_type', 'basic')   # Use basic states
        kwargs.setdefault('exploration_rate', 0.0)  # No additional noise
        
        super().__init__(agent_id, **kwargs)


# Factory function for compatibility with the demo
def create_enhanced_qlearning(agent_id, params=None, **kwargs):
    """Factory function that accepts params dict for compatibility."""
    if params is not None:
        # Extract parameters from params dict
        kwargs.update(params)
    return EnhancedQLearningAgent(agent_id, **kwargs)


class AdaptiveEnhancedQLearningAgent(EnhancedQLearningAgent):
    """
    Adaptive Enhanced Q-Learning that adjusts parameters based on performance.
    """
    def __init__(self, agent_id, **kwargs):
        super().__init__(agent_id, **kwargs)
        
        # Performance tracking
        self.recent_rewards = deque(maxlen=20)
        self.min_lr = 0.01
        self.max_lr = 0.3
        
    def record_pairwise_outcome(self, opponent_id, my_move, opponent_move, reward):
        """Record outcome and adapt parameters."""
        super().record_pairwise_outcome(opponent_id, my_move, opponent_move, reward)
        self.recent_rewards.append(reward)
        self._adapt_parameters()
    
    def record_neighborhood_outcome(self, coop_ratio, reward):
        """Record outcome and adapt parameters."""
        super().record_neighborhood_outcome(coop_ratio, reward)
        self.recent_rewards.append(reward)
        self._adapt_parameters()
    
    def _adapt_parameters(self):
        """Adapt learning rate based on performance trend."""
        if len(self.recent_rewards) >= 10:
            # Compare recent performance
            first_half = list(self.recent_rewards)[:10]
            second_half = list(self.recent_rewards)[10:]
            
            avg_first = np.mean(first_half)
            avg_second = np.mean(second_half)
            
            if avg_second > avg_first:
                # Performance improving - reduce learning rate
                self.learning_rate = max(self.min_lr, self.learning_rate * 0.95)
            else:
                # Performance declining - increase learning rate
                self.learning_rate = min(self.max_lr, self.learning_rate * 1.05)