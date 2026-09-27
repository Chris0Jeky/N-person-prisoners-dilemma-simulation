"""
Q-Learning Agents for 3-Person Prisoner's Dilemma

This module implements two versions of Q-learning:
1. SimpleQLearning: A basic, standard Q-learning implementation
2. NPDLQLearning: Based on the NPDL framework's Q-learning

Both work with the existing pairwise and neighborhood game structures.
"""

import random
import math

# Define constants locally to avoid external dependencies
NPERSON_COOPERATE = PAIRWISE_COOPERATE = 0
NPERSON_DEFECT = PAIRWISE_DEFECT = 1


class SimpleQLearningAgent:
    """
    Simple Q-Learning implementation for 3-person games.
    
    State representation:
    - Neighborhood mode: cooperation ratio from last round
    - Pairwise mode: each opponent's last move
    """
    
    def __init__(self, agent_id, learning_rate=0.1, discount_factor=0.9, 
                 epsilon=0.1, exploration_rate=0.0):
        self.agent_id = agent_id
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # For epsilon-greedy exploration
        self.exploration_rate = exploration_rate  # For compatibility with existing code
        
        # Q-table: state -> {action: Q-value}
        self.q_table = {}
        
        # Memory for state tracking
        self.last_state = None
        self.last_action = None
        
        # Statistics
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0
        
        # Mode-specific memory
        self.opponent_last_moves = {}  # For pairwise mode
        self.last_coop_ratio = None  # For neighborhood mode
        
    def _get_state_neighborhood(self, coop_ratio):
        """Get state representation for neighborhood mode."""
        if coop_ratio is None:
            return 'initial'
        
        # Discretize cooperation ratio into bins
        if coop_ratio <= 0.2:
            return 'very_low'
        elif coop_ratio <= 0.4:
            return 'low'
        elif coop_ratio <= 0.6:
            return 'medium'
        elif coop_ratio <= 0.8:
            return 'high'
        else:
            return 'very_high'
    
    def _get_state_pairwise(self, opponent_id):
        """Get state representation for pairwise mode."""
        if opponent_id not in self.opponent_last_moves:
            return 'initial'
        
        last_move = self.opponent_last_moves[opponent_id]
        return 'opp_coop' if last_move == PAIRWISE_COOPERATE else 'opp_defect'
    
    def _ensure_state_exists(self, state):
        """Initialize Q-values for new states."""
        if state not in self.q_table:
            # Initialize with small random values to break ties
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
            # Break ties randomly
            return random.choice(['cooperate', 'defect'])
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using standard Q-learning formula."""
        self._ensure_state_exists(state)
        self._ensure_state_exists(next_state)
        
        # Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
    
    # Neighborhood mode methods
    def choose_action(self, prev_round_overall_coop_ratio, current_round_num):
        """Choose action for neighborhood mode."""
        state = self._get_state_neighborhood(prev_round_overall_coop_ratio)
        
        # Update Q-value from previous round if we have history
        if self.last_state is not None and self.last_action is not None:
            # We don't have the reward yet, will update in record_round_outcome
            pass
        
        # Choose action
        action = self._choose_action_epsilon_greedy(state)
        
        # Store for next update
        self.last_state = state
        self.last_action = action
        self.last_coop_ratio = prev_round_overall_coop_ratio
        
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
            # Get next state from current cooperation ratio
            next_state = self._get_state_neighborhood(self.last_coop_ratio)
            self.update_q_value(self.last_state, self.last_action, payoff, next_state)
    
    # Pairwise mode methods  
    def choose_action_pairwise(self, opponent_id, current_round_in_episode):
        """Choose action for pairwise mode."""
        state = self._get_state_pairwise(opponent_id)
        
        # Choose action
        action = self._choose_action_epsilon_greedy(state)
        
        # Convert to game format
        intended_move = PAIRWISE_COOPERATE if action == 'cooperate' else PAIRWISE_DEFECT
        
        # Apply exploration rate
        actual_move = intended_move
        if random.random() < self.exploration_rate:
            actual_move = 1 - intended_move
            
        # Store state and action for this opponent
        if not hasattr(self, 'pairwise_last_states'):
            self.pairwise_last_states = {}
            self.pairwise_last_actions = {}
            
        self.pairwise_last_states[opponent_id] = state
        self.pairwise_last_actions[opponent_id] = action
        
        return intended_move, actual_move
    
    def record_interaction(self, opponent_id, opponent_actual_move, my_payoff,
                          my_intended_move, my_actual_move, round_num_in_episode):
        """Record interaction for pairwise mode."""
        self.total_score += my_payoff
        
        if my_actual_move == PAIRWISE_COOPERATE:
            self.num_cooperations += 1
        else:
            self.num_defections += 1
        
        # Update opponent history
        self.opponent_last_moves[opponent_id] = opponent_actual_move
        
        # Update Q-value
        if hasattr(self, 'pairwise_last_states') and opponent_id in self.pairwise_last_states:
            last_state = self.pairwise_last_states[opponent_id]
            last_action = self.pairwise_last_actions[opponent_id]
            next_state = self._get_state_pairwise(opponent_id)
            
            self.update_q_value(last_state, last_action, my_payoff, next_state)
    
    def clear_opponent_history(self, opponent_id):
        """Clear history for a specific opponent (between episodes)."""
        if opponent_id in self.opponent_last_moves:
            del self.opponent_last_moves[opponent_id]
    
    def get_cooperation_rate(self):
        """Get cooperation rate."""
        total_moves = self.num_cooperations + self.num_defections
        return self.num_cooperations / total_moves if total_moves > 0 else 0.0
    
    def reset(self):
        """Reset for neighborhood mode."""
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0
        self.last_state = None
        self.last_action = None
        self.last_coop_ratio = None
    
    def reset_for_new_tournament(self):
        """Reset for pairwise mode."""
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0
        self.opponent_last_moves = {}
        if hasattr(self, 'pairwise_last_states'):
            self.pairwise_last_states = {}
            self.pairwise_last_actions = {}


class NPDLQLearningAgent:
    """
    Q-Learning based on NPDL framework implementation.
    
    Features:
    - More sophisticated state representations
    - Proportional cooperation tracking
    - Compatible with both modes
    """
    
    def __init__(self, agent_id, learning_rate=0.1, discount_factor=0.9,
                 epsilon=0.1, exploration_rate=0.0, state_type="proportion_discretized"):
        self.agent_id = agent_id
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.exploration_rate = exploration_rate
        self.state_type = state_type
        
        # Q-table
        self.q_table = {}
        
        # Memory
        self.memory = []  # Simple list instead of deque for this implementation
        self.last_state = None
        self.last_action = None
        
        # Statistics
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0
        
        # Mode-specific
        self.opponent_last_moves = {}
        
    def _get_current_state(self, context=None):
        """Get state based on state_type and context."""
        if self.state_type == "basic":
            return 'standard'
        
        if context is None:
            return 'initial'
        
        # Handle neighborhood mode (cooperation ratio)
        if isinstance(context, (int, float)):
            coop_proportion = context
            
            if self.state_type == "proportion":
                return (round(coop_proportion, 2),)
            elif self.state_type == "proportion_discretized":
                # Discretize into bins
                if coop_proportion <= 0.2: 
                    return (0.2,)
                elif coop_proportion <= 0.4: 
                    return (0.4,)
                elif coop_proportion <= 0.6: 
                    return (0.6,)
                elif coop_proportion <= 0.8: 
                    return (0.8,)
                else: 
                    return (1.0,)
            elif self.state_type == "memory_enhanced":
                # Include own last move
                own_last = 1 if len(self.memory) > 0 and self.memory[-1]['my_move'] == 'cooperate' else 0
                
                # Bin the cooperation proportion
                if coop_proportion <= 0.33:
                    opp_bin = 0  # Low
                elif coop_proportion <= 0.67:
                    opp_bin = 1  # Med
                else:
                    opp_bin = 2  # High
                    
                return (own_last, opp_bin)
            else:
                return ('default', round(coop_proportion, 1))
        
        # Handle pairwise mode (opponent moves dict)
        elif isinstance(context, dict):
            # Calculate cooperation proportion from opponent moves
            total = len(context)
            if total == 0:
                coop_proportion = 0.5
            else:
                coop_count = sum(1 for move in context.values() if move == PAIRWISE_COOPERATE)
                coop_proportion = coop_count / total
            
            # Use same state representations as above
            return self._get_current_state(coop_proportion)
        
        return 'unknown'
    
    def _ensure_state_exists(self, state):
        """Initialize Q-values for new states."""
        if state not in self.q_table:
            # Optimistic initialization
            self.q_table[state] = {
                'cooperate': 0.1,
                'defect': 0.1
            }
    
    def _choose_action_epsilon_greedy(self, state):
        """Choose action using epsilon-greedy policy."""
        self._ensure_state_exists(state)
        
        # Exploration
        if random.random() < self.epsilon:
            return random.choice(['cooperate', 'defect'])
        
        # Exploitation
        q_values = self.q_table[state]
        if q_values['cooperate'] >= q_values['defect']:
            return 'cooperate'
        else:
            return 'defect'
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using NPDL-style update."""
        self._ensure_state_exists(state)
        self._ensure_state_exists(next_state)
        
        best_next_q = max(self.q_table[next_state].values())
        current_q = self.q_table[state][action]
        
        self.q_table[state][action] = (
            (1 - self.learning_rate) * current_q +
            self.learning_rate * (reward + self.discount_factor * best_next_q)
        )
    
    # Neighborhood mode
    def choose_action(self, prev_round_overall_coop_ratio, current_round_num):
        """Choose action for neighborhood mode."""
        # Get state
        state = self._get_current_state(prev_round_overall_coop_ratio)
        
        # Choose action
        action = self._choose_action_epsilon_greedy(state)
        
        # Store for update
        self.last_state = state
        self.last_action = action
        
        # Convert to game format
        intended_move = NPERSON_COOPERATE if action == 'cooperate' else NPERSON_DEFECT
        
        # Apply exploration
        actual_move = intended_move
        if random.random() < self.exploration_rate:
            actual_move = 1 - intended_move
            
        # Update memory
        self.memory.append({
            'round': current_round_num,
            'my_move': action,
            'state': state,
            'coop_ratio': prev_round_overall_coop_ratio
        })
        
        # Keep memory bounded
        if len(self.memory) > 10:
            self.memory.pop(0)
            
        return intended_move, actual_move
    
    def record_round_outcome(self, my_actual_move, payoff):
        """Record outcome for neighborhood mode."""
        self.total_score += payoff
        
        if my_actual_move == NPERSON_COOPERATE:
            self.num_cooperations += 1
        else:
            self.num_defections += 1
        
        # Update Q-value
        if self.last_state is not None and self.last_action is not None:
            # Get next state from memory
            if len(self.memory) > 0:
                next_coop_ratio = self.memory[-1].get('coop_ratio', 0.5)
                next_state = self._get_current_state(next_coop_ratio)
            else:
                next_state = 'initial'
                
            self.update_q_value(self.last_state, self.last_action, payoff, next_state)
    
    # Pairwise mode
    def choose_action_pairwise(self, opponent_id, current_round_in_episode):
        """Choose action for pairwise mode."""
        # Get state based on all opponents
        state = self._get_current_state(self.opponent_last_moves)
        
        # Choose action
        action = self._choose_action_epsilon_greedy(state)
        
        # Store for this opponent
        if not hasattr(self, 'pairwise_states'):
            self.pairwise_states = {}
            self.pairwise_actions = {}
            
        self.pairwise_states[opponent_id] = state
        self.pairwise_actions[opponent_id] = action
        
        # Convert to game format
        intended_move = PAIRWISE_COOPERATE if action == 'cooperate' else PAIRWISE_DEFECT
        
        # Apply exploration
        actual_move = intended_move
        if random.random() < self.exploration_rate:
            actual_move = 1 - intended_move
            
        return intended_move, actual_move
    
    def record_interaction(self, opponent_id, opponent_actual_move, my_payoff,
                          my_intended_move, my_actual_move, round_num_in_episode):
        """Record interaction for pairwise mode."""
        self.total_score += my_payoff
        
        if my_actual_move == PAIRWISE_COOPERATE:
            self.num_cooperations += 1
        else:
            self.num_defections += 1
        
        # Update opponent history
        self.opponent_last_moves[opponent_id] = opponent_actual_move
        
        # Update Q-value
        if hasattr(self, 'pairwise_states') and opponent_id in self.pairwise_states:
            last_state = self.pairwise_states[opponent_id]
            last_action = self.pairwise_actions[opponent_id]
            next_state = self._get_current_state(self.opponent_last_moves)
            
            self.update_q_value(last_state, last_action, my_payoff, next_state)
    
    def clear_opponent_history(self, opponent_id):
        """Clear history for opponent."""
        if opponent_id in self.opponent_last_moves:
            del self.opponent_last_moves[opponent_id]
    
    def get_cooperation_rate(self):
        """Get cooperation rate."""
        total_moves = self.num_cooperations + self.num_defections
        return self.num_cooperations / total_moves if total_moves > 0 else 0.0
    
    def reset(self):
        """Reset for neighborhood mode."""
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0
        self.last_state = None
        self.last_action = None
        self.memory = []
    
    def reset_for_new_tournament(self):
        """Reset for pairwise mode."""
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0
        self.opponent_last_moves = {}
        self.memory = []
        if hasattr(self, 'pairwise_states'):
            self.pairwise_states = {}
            self.pairwise_actions = {}


# Factory functions for easy creation
def create_simple_qlearning(agent_id, **kwargs):
    """Create a simple Q-learning agent."""
    return SimpleQLearningAgent(agent_id, **kwargs)

def create_npdl_qlearning(agent_id, **kwargs):
    """Create an NPDL-based Q-learning agent."""
    return NPDLQLearningAgent(agent_id, **kwargs)