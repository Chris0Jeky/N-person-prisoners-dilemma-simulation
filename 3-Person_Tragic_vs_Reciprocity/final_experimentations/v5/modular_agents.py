#!/usr/bin/env python3
"""
Modular Q-Learning agents using the Strategy Pattern.
These agents can be composed with different state, action, and learning strategies.
"""

import random
from collections import deque
from final_agents import BaseAgent, COOPERATE, DEFECT
from strategies import (
    SimpleStateStrategy, StatisticalSummaryStrategy,
    EpsilonGreedyStrategy, SoftmaxStrategy,
    StandardQLearning, HystereticQLearning
)


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