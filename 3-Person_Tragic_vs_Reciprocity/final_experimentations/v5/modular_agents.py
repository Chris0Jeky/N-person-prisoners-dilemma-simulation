#!/usr/bin/env python3
"""
Modular Q-Learning agents using the Strategy Pattern.
These agents can be composed with different state, action, and learning strategies.
"""

import random
import numpy as np
from collections import deque
from final_agents import BaseAgent, COOPERATE, DEFECT
from strategies import (
    SimpleStateStrategy, StatisticalSummaryStrategy,
    EpsilonGreedyStrategy, SoftmaxStrategy,
    StandardQLearning, HystereticQLearning
)


class ModularAdaptiveQLearner(BaseAgent):
    """
    An adaptive Q-learning agent that uses pluggable strategies and adapts parameters over time.
    Combines the modular strategy pattern with adaptive learning rate and exploration.
    """
    
    def __init__(self, agent_id, state_strategy, action_strategy, learning_strategy, 
                 params=None, **kwargs):
        # Generate a descriptive name based on strategies
        strategy_name = f"Adaptive_{state_strategy.__class__.__name__}_{action_strategy.__class__.__name__}"
        super().__init__(agent_id, strategy_name)
        
        self.state_strategy = state_strategy
        self.action_strategy = action_strategy
        self.learning_strategy = learning_strategy
        self.params = params or {}
        
        self.reset()
    
    def _make_q_dict(self):
        return {'cooperate': 0.0, 'defect': 0.0}
    
    def _make_reward_window(self):
        return deque(maxlen=self.params.get('reward_window_size', 20))
    
    def _get_initial_lr(self):
        return self.params.get('initial_lr', self.params.get('lr', 0.1))
    
    def _get_initial_eps(self):
        return self.params.get('initial_eps', self.params.get('eps', 0.1))
    
    def choose_pairwise_action(self, opponent_id):
        # Get state from state strategy
        state = self.state_strategy.get_state(self, opponent_id)
        
        # Initialize Q-table entries if needed
        if opponent_id not in self.q_tables:
            self.q_tables[opponent_id] = {}
        if state not in self.q_tables[opponent_id]:
            self.q_tables[opponent_id][state] = self._make_q_dict()
        
        # Get Q-values for this state
        q_values = self.q_tables[opponent_id][state]
        
        # Update action strategy parameters if it supports adaptation
        if hasattr(self.action_strategy, 'set_epsilon'):
            # For epsilon-greedy strategy
            if opponent_id in self.epsilons:
                self.action_strategy.set_epsilon(self.epsilons[opponent_id])
        elif hasattr(self.action_strategy, 'temperature'):
            # For softmax strategy - use adaptive epsilon as temperature scaling
            if opponent_id in self.epsilons:
                # Scale temperature based on epsilon (lower epsilon = lower temperature)
                base_temp = self.action_strategy.initial_temperature
                self.action_strategy.temperature = base_temp * self.epsilons[opponent_id] / self._get_initial_eps()
        
        # Choose action using action strategy
        action = self.action_strategy.choose_action(q_values)
        
        # Store context for learning
        self.last_contexts[opponent_id] = {'state': state, 'action': action}
        
        return COOPERATE if action == 'cooperate' else DEFECT
    
    def record_pairwise_outcome(self, opponent_id, my_move, opponent_move, reward):
        self.total_score += reward
        
        context = self.last_contexts.get(opponent_id)
        if not context:
            return
        
        # Initialize tracking structures if needed
        if opponent_id not in self.learning_rates:
            self.learning_rates[opponent_id] = self._get_initial_lr()
        if opponent_id not in self.epsilons:
            self.epsilons[opponent_id] = self._get_initial_eps()
        if opponent_id not in self.reward_windows:
            self.reward_windows[opponent_id] = self._make_reward_window()
        
        # Update state strategy's memory
        if hasattr(self.state_strategy, 'update_history'):
            self.state_strategy.update_history(opponent_id, my_move, opponent_move)
        if hasattr(self.state_strategy, 'update_stats'):
            self.state_strategy.update_stats(opponent_id, opponent_move)
        
        # Get next state
        next_state = self.state_strategy.get_state(self, opponent_id)
        
        # Initialize next state Q-values if needed
        if next_state not in self.q_tables[opponent_id]:
            self.q_tables[opponent_id][next_state] = self._make_q_dict()
        
        # Update learning strategy parameters if adaptive
        if hasattr(self.learning_strategy, 'lr'):
            self.learning_strategy.lr = self.learning_rates[opponent_id]
        if hasattr(self.learning_strategy, 'lr_positive'):
            self.learning_strategy.lr_positive = self.learning_rates[opponent_id]
        
        # Update Q-value using learning strategy
        current_q = self.q_tables[opponent_id][context['state']][context['action']]
        next_max_q = max(self.q_tables[opponent_id][next_state].values())
        
        new_q = self.learning_strategy.update_q_value(current_q, reward, next_max_q)
        self.q_tables[opponent_id][context['state']][context['action']] = new_q
        
        # Track rewards and adapt parameters
        self.reward_windows[opponent_id].append(reward)
        self._adapt_parameters(opponent_id)
    
    def _adapt_parameters(self, opponent_id):
        """Adapt learning rate and exploration based on performance"""
        win_size = self.params.get('reward_window_size')
        if not win_size or len(self.reward_windows[opponent_id]) < win_size:
            return
        
        window = self.reward_windows[opponent_id]
        half = win_size // 2
        adapt_factor = self.params.get('adaptation_factor', 1.05)
        min_lr = self.params.get('min_lr', 0.05)
        max_lr = self.params.get('max_lr', 0.5)
        min_eps = self.params.get('min_eps', 0.01)
        max_eps = self.params.get('max_eps', 0.5)
        
        # Compare recent performance to earlier performance
        recent_avg = np.mean(list(window)[half:])
        early_avg = np.mean(list(window)[:half])
        
        if recent_avg > early_avg:
            # Performance improving - reduce exploration and learning rate
            self.learning_rates[opponent_id] = max(min_lr, self.learning_rates[opponent_id] / adapt_factor)
            self.epsilons[opponent_id] = max(min_eps, self.epsilons[opponent_id] / adapt_factor)
        else:
            # Performance declining - increase exploration and learning rate
            self.learning_rates[opponent_id] = min(max_lr, self.learning_rates[opponent_id] * adapt_factor)
            self.epsilons[opponent_id] = min(max_eps, self.epsilons[opponent_id] * adapt_factor)
    
    def choose_neighborhood_action(self, coop_ratio):
        """Choose action for neighborhood game"""
        state = self._get_neighborhood_state(coop_ratio)
        
        # Initialize if needed
        if state not in self.neighborhood_q_table:
            self.neighborhood_q_table[state] = self._make_q_dict()
        
        # Update action strategy parameters for neighborhood
        if hasattr(self.action_strategy, 'set_epsilon'):
            self.action_strategy.set_epsilon(self.neighborhood_epsilon)
        elif hasattr(self.action_strategy, 'temperature'):
            # Scale temperature based on neighborhood epsilon
            base_temp = self.action_strategy.initial_temperature
            self.action_strategy.temperature = base_temp * self.neighborhood_epsilon / self._get_initial_eps()
        
        # Get Q-values and choose action
        q_values = self.neighborhood_q_table[state]
        action = self.action_strategy.choose_action(q_values)
        
        self.last_neighborhood_context = {'state': state, 'action': action}
        return COOPERATE if action == 'cooperate' else DEFECT
    
    def record_neighborhood_outcome(self, coop_ratio, reward):
        """Record outcome for neighborhood game"""
        self.total_score += reward
        
        if not hasattr(self, 'last_neighborhood_context') or not self.last_neighborhood_context:
            return
        
        next_state = self._get_neighborhood_state(coop_ratio)
        
        # Initialize next state if needed
        if next_state not in self.neighborhood_q_table:
            self.neighborhood_q_table[next_state] = self._make_q_dict()
        
        # Update learning strategy parameters
        if hasattr(self.learning_strategy, 'lr'):
            self.learning_strategy.lr = self.neighborhood_lr
        if hasattr(self.learning_strategy, 'lr_positive'):
            self.learning_strategy.lr_positive = self.neighborhood_lr
        
        # Update Q-value
        current_q = self.neighborhood_q_table[self.last_neighborhood_context['state']][self.last_neighborhood_context['action']]
        next_max_q = max(self.neighborhood_q_table[next_state].values())
        
        new_q = self.learning_strategy.update_q_value(current_q, reward, next_max_q)
        self.neighborhood_q_table[self.last_neighborhood_context['state']][self.last_neighborhood_context['action']] = new_q
        
        # Adapt neighborhood parameters
        self.neighborhood_reward_window.append(reward)
        self._adapt_neighborhood_parameters()
    
    def _adapt_neighborhood_parameters(self):
        """Adapt parameters for neighborhood game"""
        win_size = self.params.get('reward_window_size')
        if not win_size or len(self.neighborhood_reward_window) < win_size:
            return
        
        window = self.neighborhood_reward_window
        half = win_size // 2
        adapt_factor = self.params.get('adaptation_factor', 1.05)
        min_lr = self.params.get('min_lr', 0.05)
        max_lr = self.params.get('max_lr', 0.5)
        min_eps = self.params.get('min_eps', 0.01)
        max_eps = self.params.get('max_eps', 0.5)
        
        if np.mean(list(window)[half:]) > np.mean(list(window)[:half]):
            self.neighborhood_lr = max(min_lr, self.neighborhood_lr / adapt_factor)
            self.neighborhood_epsilon = max(min_eps, self.neighborhood_epsilon / adapt_factor)
        else:
            self.neighborhood_lr = min(max_lr, self.neighborhood_lr * adapt_factor)
            self.neighborhood_epsilon = min(max_eps, self.neighborhood_epsilon * adapt_factor)
    
    def _get_neighborhood_state(self, coop_ratio):
        """Get state representation for neighborhood game"""
        if coop_ratio is None:
            return 'start'
        if coop_ratio <= 0.33:
            return 'low'
        elif coop_ratio <= 0.67:
            return 'medium'
        else:
            return 'high'
    
    def reset(self):
        super().reset()
        self.q_tables = {}
        self.last_contexts = {}
        self.learning_rates = {}
        self.epsilons = {}
        self.reward_windows = {}
        
        # Neighborhood attributes
        self.neighborhood_q_table = {}
        self.last_neighborhood_context = None
        self.neighborhood_lr = self._get_initial_lr()
        self.neighborhood_epsilon = self._get_initial_eps()
        self.neighborhood_reward_window = self._make_reward_window()
        
        # Reset strategies
        self.state_strategy.reset()
        self.action_strategy.reset()


class ModularQLearner(BaseAgent):
    """
    A flexible Q-learning agent that uses pluggable strategies for:
    - State representation
    - Action selection
    - Q-value updates
    """
    
    def __init__(self, agent_id, state_strategy, action_strategy, learning_strategy, **kwargs):
        # Generate a descriptive name based on strategies
        strategy_name = f"{state_strategy.__class__.__name__}_{action_strategy.__class__.__name__}"
        super().__init__(agent_id, strategy_name)
        
        self.state_strategy = state_strategy
        self.action_strategy = action_strategy
        self.learning_strategy = learning_strategy
        
        self.reset()
    
    def _make_q_dict(self):
        return {'cooperate': 0.0, 'defect': 0.0}
    
    def choose_pairwise_action(self, opponent_id):
        # Get state from state strategy
        state = self.state_strategy.get_state(self, opponent_id)
        
        # Initialize Q-table entries if needed
        if opponent_id not in self.q_tables:
            self.q_tables[opponent_id] = {}
        if state not in self.q_tables[opponent_id]:
            self.q_tables[opponent_id][state] = self._make_q_dict()
        
        # Get Q-values for this state
        q_values = self.q_tables[opponent_id][state]
        
        # Choose action using action strategy
        action = self.action_strategy.choose_action(q_values)
        
        # Store context for learning
        self.last_contexts[opponent_id] = {'state': state, 'action': action}
        
        return COOPERATE if action == 'cooperate' else DEFECT
    
    def record_pairwise_outcome(self, opponent_id, my_move, opponent_move, reward):
        self.total_score += reward
        
        context = self.last_contexts.get(opponent_id)
        if not context:
            return
        
        # Update state strategy's memory
        if hasattr(self.state_strategy, 'update_history'):
            self.state_strategy.update_history(opponent_id, my_move, opponent_move)
        if hasattr(self.state_strategy, 'update_stats'):
            self.state_strategy.update_stats(opponent_id, opponent_move)
        
        # Get next state
        next_state = self.state_strategy.get_state(self, opponent_id)
        
        # Initialize next state Q-values if needed
        if next_state not in self.q_tables[opponent_id]:
            self.q_tables[opponent_id][next_state] = self._make_q_dict()
        
        # Update Q-value using learning strategy
        current_q = self.q_tables[opponent_id][context['state']][context['action']]
        next_max_q = max(self.q_tables[opponent_id][next_state].values())
        
        new_q = self.learning_strategy.update_q_value(current_q, reward, next_max_q)
        self.q_tables[opponent_id][context['state']][context['action']] = new_q
    
    def choose_neighborhood_action(self, coop_ratio):
        """Choose action for neighborhood game"""
        state = self._get_neighborhood_state(coop_ratio)
        
        # Initialize if needed
        if state not in self.neighborhood_q_table:
            self.neighborhood_q_table[state] = self._make_q_dict()
        
        # Get Q-values and choose action
        q_values = self.neighborhood_q_table[state]
        action = self.action_strategy.choose_action(q_values)
        
        self.last_neighborhood_context = {'state': state, 'action': action}
        return COOPERATE if action == 'cooperate' else DEFECT
    
    def record_neighborhood_outcome(self, coop_ratio, reward):
        """Record outcome for neighborhood game"""
        self.total_score += reward
        
        if not hasattr(self, 'last_neighborhood_context') or not self.last_neighborhood_context:
            return
        
        next_state = self._get_neighborhood_state(coop_ratio)
        
        # Initialize next state if needed
        if next_state not in self.neighborhood_q_table:
            self.neighborhood_q_table[next_state] = self._make_q_dict()
        
        # Update Q-value
        current_q = self.neighborhood_q_table[self.last_neighborhood_context['state']][self.last_neighborhood_context['action']]
        next_max_q = max(self.neighborhood_q_table[next_state].values())
        
        new_q = self.learning_strategy.update_q_value(current_q, reward, next_max_q)
        self.neighborhood_q_table[self.last_neighborhood_context['state']][self.last_neighborhood_context['action']] = new_q
    
    def _get_neighborhood_state(self, coop_ratio):
        """Get state representation for neighborhood game"""
        if coop_ratio is None:
            return 'start'
        if coop_ratio <= 0.33:
            return 'low'
        elif coop_ratio <= 0.67:
            return 'medium'
        else:
            return 'high'
    
    def reset(self):
        super().reset()
        self.q_tables = {}
        self.last_contexts = {}
        self.neighborhood_q_table = {}
        self.last_neighborhood_context = None
        
        # Reset strategies
        self.state_strategy.reset()
        self.action_strategy.reset()


# === Factory functions for easy agent creation ===

def create_vanilla_qlearner(agent_id, **kwargs):
    """Create a vanilla Q-learner with simple state and epsilon-greedy"""
    return ModularQLearner(
        agent_id,
        SimpleStateStrategy(),
        EpsilonGreedyStrategy(epsilon=0.1),
        StandardQLearning(learning_rate=0.1, discount_factor=0.9)
    )


def create_statistical_qlearner(agent_id, **kwargs):
    """Create a Q-learner with statistical state representation"""
    return ModularQLearner(
        agent_id,
        StatisticalSummaryStrategy(),
        EpsilonGreedyStrategy(epsilon=0.1),
        StandardQLearning(learning_rate=0.1, discount_factor=0.9)
    )


def create_softmax_qlearner(agent_id, temperature=2.0, **kwargs):
    """Create a Q-learner with softmax action selection"""
    return ModularQLearner(
        agent_id,
        SimpleStateStrategy(),
        SoftmaxStrategy(temperature=temperature),
        StandardQLearning(learning_rate=0.1, discount_factor=0.9)
    )


def create_statistical_softmax_qlearner(agent_id, temperature=2.0, **kwargs):
    """Create a Q-learner with both statistical state and softmax action"""
    return ModularQLearner(
        agent_id,
        StatisticalSummaryStrategy(),
        SoftmaxStrategy(temperature=temperature),
        StandardQLearning(learning_rate=0.1, discount_factor=0.9)
    )


def create_hysteretic_statistical_qlearner(agent_id, **kwargs):
    """Create a hysteretic Q-learner with statistical state representation"""
    return ModularQLearner(
        agent_id,
        StatisticalSummaryStrategy(),
        EpsilonGreedyStrategy(epsilon=0.1),
        HystereticQLearning(lr_positive=0.1, lr_negative=0.01, discount_factor=0.9)
    )


# === Factory functions for Adaptive agents with strategies ===

def create_adaptive_baseline(agent_id, params, **kwargs):
    """Create baseline adaptive Q-learner (original implementation)"""
    from final_agents import PairwiseAdaptiveQLearner
    return PairwiseAdaptiveQLearner(agent_id, params)


def create_adaptive_statistical(agent_id, params, **kwargs):
    """Create adaptive Q-learner with statistical state representation"""
    return ModularAdaptiveQLearner(
        agent_id,
        StatisticalSummaryStrategy(),
        EpsilonGreedyStrategy(epsilon=params.get('initial_eps', 0.1)),
        StandardQLearning(learning_rate=params.get('initial_lr', 0.1), 
                         discount_factor=params.get('df', 0.9)),
        params=params
    )


def create_adaptive_softmax(agent_id, params, **kwargs):
    """Create adaptive Q-learner with softmax action selection"""
    # Convert epsilon parameters to temperature
    initial_temp = 2.0 * params.get('initial_eps', 0.1) / 0.1  # Scale based on epsilon
    return ModularAdaptiveQLearner(
        agent_id,
        SimpleStateStrategy(),
        SoftmaxStrategy(temperature=initial_temp, 
                       min_temperature=0.5,
                       decay_rate=0.998),
        StandardQLearning(learning_rate=params.get('initial_lr', 0.1),
                         discount_factor=params.get('df', 0.9)),
        params=params
    )


def create_adaptive_statistical_softmax(agent_id, params, **kwargs):
    """Create adaptive Q-learner with both statistical state and softmax action"""
    initial_temp = 2.0 * params.get('initial_eps', 0.1) / 0.1
    return ModularAdaptiveQLearner(
        agent_id,
        StatisticalSummaryStrategy(),
        SoftmaxStrategy(temperature=initial_temp,
                       min_temperature=0.5,
                       decay_rate=0.998),
        StandardQLearning(learning_rate=params.get('initial_lr', 0.1),
                         discount_factor=params.get('df', 0.9)),
        params=params
    )


def create_adaptive_hysteretic_statistical(agent_id, params, **kwargs):
    """Create adaptive Q-learner with hysteretic learning and statistical state"""
    return ModularAdaptiveQLearner(
        agent_id,
        StatisticalSummaryStrategy(),
        EpsilonGreedyStrategy(epsilon=params.get('initial_eps', 0.1)),
        HystereticQLearning(lr_positive=params.get('initial_lr', 0.1),
                           lr_negative=params.get('beta', 0.01),
                           discount_factor=params.get('df', 0.9)),
        params=params
    )