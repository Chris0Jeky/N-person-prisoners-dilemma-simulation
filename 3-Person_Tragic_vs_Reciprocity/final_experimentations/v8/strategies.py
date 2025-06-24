#!/usr/bin/env python3
"""
Strategy classes for modular Q-learning agents.
Implements the Strategy Pattern for flexible agent composition.
"""

import random
import numpy as np
from collections import deque
from abc import ABC, abstractmethod

# Constants
COOPERATE, DEFECT = 0, 1


# === State Strategies ===
class StateStrategy(ABC):
    """Abstract base class for state representation strategies"""
    
    @abstractmethod
    def get_state(self, agent, opponent_id):
        pass
    
    @abstractmethod
    def reset(self):
        pass


class SimpleStateStrategy(StateStrategy):
    """Original state strategy using last 2 rounds of history"""
    
    def __init__(self):
        self.histories = {}
    
    def get_state(self, agent, opponent_id):
        if opponent_id not in self.histories:
            self.histories[opponent_id] = deque(maxlen=2)
        history = self.histories[opponent_id]
        if len(history) < 2:
            return "start"
        return str(tuple(history))
    
    def update_history(self, opponent_id, my_move, opponent_move):
        if opponent_id not in self.histories:
            self.histories[opponent_id] = deque(maxlen=2)
        self.histories[opponent_id].append((my_move, opponent_move))
    
    def reset(self):
        self.histories = {}


class StatisticalSummaryStrategy(StateStrategy):
    """State strategy using opponent's overall cooperation statistics"""
    
    def __init__(self):
        self.opponent_stats = {}
    
    def get_state(self, agent, opponent_id):
        if opponent_id not in self.opponent_stats:
            return "Opponent_Disposition_Unknown"
        
        stats = self.opponent_stats[opponent_id]
        total_moves = stats['cooperated'] + stats['defected']
        
        if total_moves == 0:
            return "Opponent_Disposition_Unknown"
        
        coop_rate = stats['cooperated'] / total_moves
        
        # Discretize into categories
        if coop_rate < 0.2:
            return "Opponent_Disposition_VeryLow"
        elif coop_rate < 0.4:
            return "Opponent_Disposition_Low"
        elif coop_rate < 0.6:
            return "Opponent_Disposition_Medium"
        elif coop_rate < 0.8:
            return "Opponent_Disposition_High"
        else:
            return "Opponent_Disposition_VeryHigh"
    
    def update_stats(self, opponent_id, opponent_move):
        if opponent_id not in self.opponent_stats:
            self.opponent_stats[opponent_id] = {'cooperated': 0, 'defected': 0}
        
        if opponent_move == COOPERATE:
            self.opponent_stats[opponent_id]['cooperated'] += 1
        else:
            self.opponent_stats[opponent_id]['defected'] += 1
    
    def reset(self):
        self.opponent_stats = {}


# === Action Selection Strategies ===
class ActionStrategy(ABC):
    """Abstract base class for action selection strategies"""
    
    @abstractmethod
    def choose_action(self, q_values, **kwargs):
        """
        Choose an action based on Q-values.
        q_values: dict with 'cooperate' and 'defect' as keys
        Returns: 'cooperate' or 'defect'
        """
        pass
    
    @abstractmethod
    def reset(self):
        pass


class EpsilonGreedyStrategy(ActionStrategy):
    """Standard epsilon-greedy action selection"""
    
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
    
    def choose_action(self, q_values, **kwargs):
        if random.random() < self.epsilon:
            return random.choice(['cooperate', 'defect'])
        else:
            # Choose action with highest Q-value
            if q_values['cooperate'] >= q_values['defect']:
                return 'cooperate'
            else:
                return 'defect'
    
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
    
    def reset(self):
        pass  # Epsilon-greedy has no state to reset


class SoftmaxStrategy(ActionStrategy):
    """Softmax (Boltzmann) action selection with temperature"""
    
    def __init__(self, temperature=1.0, min_temperature=0.1, decay_rate=0.995):
        self.initial_temperature = temperature
        self.temperature = temperature
        self.min_temperature = min_temperature
        self.decay_rate = decay_rate
        self.step_count = 0
    
    def choose_action(self, q_values, **kwargs):
        # Get Q-values
        q_c = q_values['cooperate']
        q_d = q_values['defect']
        
        # Normalize Q-values to prevent overflow
        max_q = max(q_c, q_d)
        q_c_norm = q_c - max_q
        q_d_norm = q_d - max_q
        
        # Calculate softmax probabilities
        exp_c = np.exp(q_c_norm / self.temperature)
        exp_d = np.exp(q_d_norm / self.temperature)
        
        # Avoid division by zero
        total = exp_c + exp_d
        if total == 0:
            p_cooperate = 0.5
        else:
            p_cooperate = exp_c / total
        
        # Decay temperature
        self.step_count += 1
        if self.step_count % 10 == 0:  # Decay every 10 steps
            self.temperature = max(self.min_temperature, 
                                 self.temperature * self.decay_rate)
        
        # Choose action based on probability
        return 'cooperate' if random.random() < p_cooperate else 'defect'
    
    def reset(self):
        self.temperature = self.initial_temperature
        self.step_count = 0


# === Learning Strategies ===
class LearningStrategy(ABC):
    """Abstract base class for Q-value update strategies"""
    
    @abstractmethod
    def update_q_value(self, current_q, reward, next_max_q, **kwargs):
        """
        Calculate new Q-value based on current Q-value, reward, and next state's max Q.
        Returns: new Q-value
        """
        pass


class StandardQLearning(LearningStrategy):
    """Standard Q-learning update rule"""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.9):
        self.lr = learning_rate
        self.df = discount_factor
    
    def update_q_value(self, current_q, reward, next_max_q, **kwargs):
        target_q = reward + self.df * next_max_q
        return current_q + self.lr * (target_q - current_q)


class HystereticQLearning(LearningStrategy):
    """Hysteretic Q-learning with asymmetric learning rates"""
    
    def __init__(self, lr_positive=0.1, lr_negative=0.01, discount_factor=0.9):
        self.lr_positive = lr_positive
        self.lr_negative = lr_negative
        self.df = discount_factor
    
    def update_q_value(self, current_q, reward, next_max_q, **kwargs):
        target_q = reward + self.df * next_max_q
        delta = target_q - current_q
        
        if delta >= 0:
            # Good news: use positive learning rate
            return current_q + self.lr_positive * delta
        else:
            # Bad news: use negative learning rate
            return current_q + self.lr_negative * delta