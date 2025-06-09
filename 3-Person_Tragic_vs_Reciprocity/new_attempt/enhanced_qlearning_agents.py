"""
Enhanced Q-Learning Agents with configurable improvements

This module implements enhanced Q-learning agents that address:
1. Decaying epsilon
2. State representation excluding self
3. Longer training support
4. Opponent modeling
"""

import random
import math
from collections import defaultdict, deque

from main_neighbourhood import NPERSON_COOPERATE, NPERSON_DEFECT
from main_pairwise import PAIRWISE_COOPERATE, PAIRWISE_DEFECT


class EnhancedQLearningAgent:
    """
    Enhanced Q-Learning with multiple configurable improvements.
    """
    
    def __init__(self, agent_id, 
                 learning_rate=0.1, 
                 discount_factor=0.9,
                 epsilon=0.1,
                 epsilon_decay=1.0,  # No decay by default
                 epsilon_min=0.01,
                 exploration_rate=0.0,
                 exclude_self=False,  # Whether to exclude self from state
                 opponent_modeling=False,  # Whether to model opponents
                 state_type="basic"):
        
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
        
        # Q-table
        self.q_table = {}
        
        # Episode counter for epsilon decay
        self.episode_count = 0
        self.step_count = 0
        
        # Memory
        self.last_state = None
        self.last_action = None
        self.last_others_coop_ratio = None
        
        # Statistics
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0
        
        # For opponent modeling
        if opponent_modeling:
            self.opponent_history = defaultdict(lambda: {'C': 0, 'D': 0})
            self.opponent_models = {}  # opponent_id -> predicted cooperation rate
        
        # For pairwise mode
        self.opponent_last_moves = {}
        
    def decay_epsilon(self):
        """Apply epsilon decay."""
        if self.epsilon_decay < 1.0:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _get_state_neighborhood(self, overall_coop_ratio, my_last_action=None):
        """Get state for neighborhood mode."""
        if self.exclude_self and my_last_action is not None and overall_coop_ratio is not None:
            # Calculate cooperation ratio excluding self
            total_agents = 3  # Fixed for 3-person game
            total_coops = overall_coop_ratio * total_agents
            
            # Remove self contribution
            if my_last_action == NPERSON_COOPERATE:
                total_coops -= 1
            
            # Calculate others' cooperation ratio
            others_coop_ratio = total_coops / (total_agents - 1) if total_agents > 1 else 0
        else:
            others_coop_ratio = overall_coop_ratio if overall_coop_ratio is not None else 0.5
        
        # Store for later use
        self.last_others_coop_ratio = others_coop_ratio
        
        if self.state_type == "basic":
            # Discretize into bins
            if others_coop_ratio <= 0.2:
                base_state = 'very_low'
            elif others_coop_ratio <= 0.4:
                base_state = 'low'
            elif others_coop_ratio <= 0.6:
                base_state = 'medium'
            elif others_coop_ratio <= 0.8:
                base_state = 'high'
            else:
                base_state = 'very_high'
        elif self.state_type == "fine":
            # Finer discretization
            base_state = f"coop_{int(others_coop_ratio * 10) / 10:.1f}"
        else:
            base_state = 'default'
        
        # Add opponent model information if enabled
        if self.opponent_modeling and self.opponent_models:
            # Average predicted cooperation rates
            avg_pred = sum(self.opponent_models.values()) / len(self.opponent_models)
            model_state = f"_pred_{int(avg_pred * 10) / 10:.1f}"
            return base_state + model_state
        
        return base_state
    
    def _update_opponent_model(self, opponent_id, action):
        """Update opponent model based on observed action."""
        if not self.opponent_modeling:
            return
            
        # Update history
        if action == NPERSON_COOPERATE:
            self.opponent_history[opponent_id]['C'] += 1
        else:
            self.opponent_history[opponent_id]['D'] += 1
        
        # Update model (simple frequency-based)
        total = self.opponent_history[opponent_id]['C'] + self.opponent_history[opponent_id]['D']
        if total > 0:
            self.opponent_models[opponent_id] = self.opponent_history[opponent_id]['C'] / total
    
    def _ensure_state_exists(self, state):
        """Initialize Q-values for new states."""
        if state not in self.q_table:
            # Optimistic initialization to encourage exploration
            self.q_table[state] = {
                'cooperate': 0.1,
                'defect': 0.1
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
        # Get state (excluding self if configured)
        state = self._get_state_neighborhood(
            prev_round_overall_coop_ratio, 
            self.last_action if self.exclude_self else None
        )
        
        # Choose action
        action = self._choose_action_epsilon_greedy(state)
        
        # Store for next update
        self.last_state = state
        self.last_action = action
        
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
            # For next state, we need the cooperation ratio excluding self
            next_state = self._get_state_neighborhood(
                self.last_others_coop_ratio,
                'cooperate' if my_actual_move == NPERSON_COOPERATE else 'defect'
            )
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


class OpponentModelingQLearning(EnhancedQLearningAgent):
    """
    Q-Learning with explicit opponent modeling.
    
    This version maintains models of each opponent's behavior and uses
    this information to make better decisions.
    """
    
    def __init__(self, agent_id, **kwargs):
        # Force opponent modeling on
        kwargs['opponent_modeling'] = True
        super().__init__(agent_id, **kwargs)
        
        # More sophisticated opponent tracking
        self.opponent_action_history = defaultdict(list)  # opponent_id -> [actions]
        self.opponent_response_model = defaultdict(lambda: defaultdict(float))
        # opponent_id -> {my_action -> their_cooperation_rate}
    
    def update_opponent_response_model(self, opponent_id, my_last_action, their_action):
        """Update model of how opponent responds to my actions."""
        if my_last_action is not None:
            key = 'C' if my_last_action == NPERSON_COOPERATE else 'D'
            if their_action == NPERSON_COOPERATE:
                self.opponent_response_model[opponent_id][key] = (
                    0.9 * self.opponent_response_model[opponent_id][key] + 0.1
                )
            else:
                self.opponent_response_model[opponent_id][key] = (
                    0.9 * self.opponent_response_model[opponent_id][key]
                )
    
    def predict_opponent_response(self, opponent_id, my_action):
        """Predict how opponent will respond to my action."""
        key = 'C' if my_action == NPERSON_COOPERATE else 'D'
        if opponent_id in self.opponent_response_model:
            return self.opponent_response_model[opponent_id].get(key, 0.5)
        return 0.5  # Unknown opponent
    
    def _get_state_with_predictions(self, base_state):
        """Enhance state with opponent predictions."""
        if not self.opponent_response_model:
            return base_state
        
        # Predict responses for both possible actions
        pred_if_coop = []
        pred_if_defect = []
        
        for opp_id in self.opponent_response_model:
            pred_if_coop.append(self.predict_opponent_response(opp_id, NPERSON_COOPERATE))
            pred_if_defect.append(self.predict_opponent_response(opp_id, NPERSON_DEFECT))
        
        avg_pred_coop = sum(pred_if_coop) / len(pred_if_coop) if pred_if_coop else 0.5
        avg_pred_defect = sum(pred_if_defect) / len(pred_if_defect) if pred_if_defect else 0.5
        
        # Add predictions to state
        return (base_state, f"pred_C_{avg_pred_coop:.1f}", f"pred_D_{avg_pred_defect:.1f}")


def create_enhanced_qlearning(agent_id, **kwargs):
    """Factory function for enhanced Q-learning agent."""
    return EnhancedQLearningAgent(agent_id, **kwargs)

def create_opponent_modeling_qlearning(agent_id, **kwargs):
    """Factory function for opponent modeling Q-learning agent."""
    return OpponentModelingQLearning(agent_id, **kwargs)