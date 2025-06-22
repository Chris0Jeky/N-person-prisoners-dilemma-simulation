"""
Unified Agent API Implementation

This module provides all agents with a consistent API to eliminate bugs caused by
inconsistent method names across different agent types.

All agents implement:
- choose_action(**kwargs): Chooses an action based on context
- record_outcome(**kwargs): Records the result of a round
- reset(): Resets the agent for a new episode
"""

import random
from collections import deque

# --- Constants ---
COOPERATE = 0
DEFECT = 1

class StaticAgent:
    """Static strategy agent (AllC, AllD, TFT, etc.) using the Unified API."""
    def __init__(self, agent_id, strategy_name, exploration_rate=0.0):
        self.agent_id = agent_id
        self.strategy_name = strategy_name
        self.exploration_rate = exploration_rate
        self.opponent_last_moves = {}
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0

    def choose_action(self, **kwargs):
        """Unified action choice method."""
        # Determine mode based on arguments
        is_pairwise = 'opponent_id' in kwargs
        
        if is_pairwise:
            opponent_id = kwargs['opponent_id']
            if self.strategy_name in ["TFT", "TFT-E"]:
                intended_move = self.opponent_last_moves.get(opponent_id, COOPERATE)
            elif self.strategy_name == "AllC":
                intended_move = COOPERATE
            elif self.strategy_name == "AllD":
                intended_move = DEFECT
            else: # Random
                intended_move = random.choice([COOPERATE, DEFECT])
        else: # Neighborhood mode
            coop_ratio = kwargs.get('prev_round_group_coop_ratio')
            if self.strategy_name in ["TFT", "TFT-E"]:
                if coop_ratio is None: intended_move = COOPERATE
                else: intended_move = COOPERATE if random.random() < coop_ratio else DEFECT
            elif self.strategy_name == "AllC":
                intended_move = COOPERATE
            elif self.strategy_name == "AllD":
                intended_move = DEFECT
            else: # Random
                intended_move = random.choice([COOPERATE, DEFECT])
        
        # Apply exploration for TFT-E or if exploration_rate is set
        if (self.strategy_name == "TFT-E" or self.exploration_rate > 0) and random.random() < self.exploration_rate:
            return 1 - intended_move
        return intended_move

    def record_outcome(self, **kwargs):
        """Unified outcome recording method."""
        self.total_score += kwargs['payoff']
        my_move = kwargs['my_move']
        
        if my_move == COOPERATE:
            self.num_cooperations += 1
        else:
            self.num_defections += 1
            
        if 'opponent_id' in kwargs: # Pairwise
            self.opponent_last_moves[kwargs['opponent_id']] = kwargs['opponent_move']

    def reset(self):
        self.opponent_last_moves = {}
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0

    def get_cooperation_rate(self):
        total_moves = self.num_cooperations + self.num_defections
        return self.num_cooperations / total_moves if total_moves > 0 else 0.0


class BaseQLearningAgent:
    """
    A new base Q-learning class providing the fundamental structure and unified API.
    All Q-learning agents will inherit from this.
    """
    def __init__(self, agent_id, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.agent_id = agent_id
        self.strategy_name = "QLearning"
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        self.q_table = {}
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0

        # Per-opponent context for pairwise games
        self.opponent_last_moves = {}
        self.pairwise_last_states = {}
        self.pairwise_last_actions = {}

        # Context for neighborhood games
        self.last_state_nperson = None
        self.last_action_nperson = None

    def _get_state(self, **kwargs):
        """Dispatches to the correct state-finding method."""
        if 'opponent_id' in kwargs:
            return self._get_state_pairwise(**kwargs)
        else:
            return self._get_state_nperson(**kwargs)

    def _get_state_pairwise(self, **kwargs):
        """Finds state for pairwise games. MUST be implemented by child."""
        raise NotImplementedError

    def _get_state_nperson(self, **kwargs):
        """Finds state for neighborhood games. MUST be implemented by child."""
        raise NotImplementedError

    def _ensure_state_exists(self, state):
        if state not in self.q_table:
            self.q_table[state] = {'cooperate': random.uniform(-0.01, 0.01), 'defect': random.uniform(-0.01, 0.01)}

    def _choose_action_epsilon_greedy(self, state):
        self._ensure_state_exists(state)
        if random.random() < self.epsilon: return random.choice(['cooperate', 'defect'])
        q = self.q_table[state]
        if q['cooperate'] > q['defect']: return 'cooperate'
        if q['defect'] > q['cooperate']: return 'defect'
        return random.choice(['cooperate', 'defect'])
    
    def update_q_value(self, state, action, reward, next_state):
        self._ensure_state_exists(state)
        self._ensure_state_exists(next_state)
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
    def choose_action(self, **kwargs):
        """Unified action choice method for Q-learners."""
        state = self._get_state(**kwargs)
        action_str = self._choose_action_epsilon_greedy(state)
        
        if 'opponent_id' in kwargs:
            self.pairwise_last_states[kwargs['opponent_id']] = state
            self.pairwise_last_actions[kwargs['opponent_id']] = action_str
        else:
            self.last_state_nperson = state
            self.last_action_nperson = action_str
            
        return COOPERATE if action_str == 'cooperate' else DEFECT

    def record_outcome(self, **kwargs):
        """Unified outcome recording method for Q-learners."""
        payoff = kwargs['payoff']
        my_move = kwargs['my_move']
        
        self.total_score += payoff
        if my_move == COOPERATE: self.num_cooperations += 1
        else: self.num_defections += 1
        
        if 'opponent_id' in kwargs: # Pairwise update
            opponent_id = kwargs['opponent_id']
            self.opponent_last_moves[opponent_id] = kwargs['opponent_move']
            last_state = self.pairwise_last_states.get(opponent_id)
            last_action = self.pairwise_last_actions.get(opponent_id)
            if last_state and last_action:
                next_state = self._get_state(**kwargs)
                self.update_q_value(last_state, last_action, payoff, next_state)
        else: # Neighborhood update
            if self.last_state_nperson and self.last_action_nperson:
                next_state = self._get_state(**kwargs)
                self.update_q_value(self.last_state_nperson, self.last_action_nperson, payoff, next_state)

    def reset(self):
        """Must be implemented by child class to handle specific reset logic."""
        raise NotImplementedError

    def get_cooperation_rate(self):
        total_moves = self.num_cooperations + self.num_defections
        return self.num_cooperations / total_moves if total_moves > 0 else 0.0


class SimpleQLearningAgent(BaseQLearningAgent):
    """Simple Q-Learning agent with basic state representation."""
    def __init__(self, agent_id, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        super().__init__(agent_id, learning_rate, discount_factor, epsilon)
        self.strategy_name = "SimpleQLearning"

    def _get_state_pairwise(self, **kwargs):
        """Returns an opponent-specific state based on their last move."""
        opponent_id = kwargs['opponent_id']
        if opponent_id not in self.opponent_last_moves: return 'initial'
        opp_move = self.opponent_last_moves[opponent_id]
        return 'opp_coop' if opp_move == COOPERATE else 'opp_defect'

    def _get_state_nperson(self, **kwargs):
        """Returns discretized neighborhood state."""
        coop_ratio = kwargs.get('prev_round_group_coop_ratio')
        if coop_ratio is None: return 'initial'
        
        # Basic discretization
        if coop_ratio <= 0.2: return 'very_low'
        elif coop_ratio <= 0.4: return 'low'
        elif coop_ratio <= 0.6: return 'medium'
        elif coop_ratio <= 0.8: return 'high'
        else: return 'very_high'

    def reset(self):
        """Resets the agent."""
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0
        self.opponent_last_moves.clear()
        self.pairwise_last_states.clear()
        self.pairwise_last_actions.clear()
        self.last_state_nperson = None
        self.last_action_nperson = None


class EnhancedQLearningAgent(BaseQLearningAgent):
    """The new, corrected, and enhanced Q-learning agent."""
    def __init__(self, agent_id, learning_rate=0.1, discount_factor=0.9,
                 epsilon=0.3, epsilon_decay=0.999, epsilon_min=0.01,
                 state_type="basic", memory_length=5):
        super().__init__(agent_id, learning_rate, discount_factor, epsilon)
        self.strategy_name = "EnhancedQLearning"
        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.state_type = state_type
        self.memory = deque(maxlen=memory_length)

    def _get_state_pairwise(self, **kwargs):
        """Correctly returns an opponent-specific state."""
        opponent_id = kwargs['opponent_id']
        if opponent_id not in self.opponent_last_moves: return 'initial'
        opp_move = self.opponent_last_moves[opponent_id]
        return 'opp_coop' if opp_move == COOPERATE else 'opp_defect'

    def _get_state_nperson(self, **kwargs):
        """Returns neighborhood state, with optional enhancements."""
        coop_ratio = kwargs.get('prev_round_group_coop_ratio')
        if coop_ratio is None: return 'initial'
        
        if self.state_type == "basic":
            # Basic discretization (same as SimpleQLearning)
            if coop_ratio <= 0.2: base_state = 'very_low'
            elif coop_ratio <= 0.4: base_state = 'low'
            elif coop_ratio <= 0.6: base_state = 'medium'
            elif coop_ratio <= 0.8: base_state = 'high'
            else: base_state = 'very_high'
        elif self.state_type == "fine":
            # Finer discretization
            base_state = f"coop_{int(coop_ratio * 10)}"
        elif self.state_type == "memory_enhanced":
            # Basic discretization with memory
            if coop_ratio <= 0.2: base_state = 'very_low'
            elif coop_ratio <= 0.4: base_state = 'low'
            elif coop_ratio <= 0.6: base_state = 'medium'
            elif coop_ratio <= 0.8: base_state = 'high'
            else: base_state = 'very_high'
            
            coop_count = sum(1 for action in self.memory if action == 'cooperate')
            return f"{base_state}_mem{coop_count}"
        else:
            # Default to basic
            if coop_ratio <= 0.2: base_state = 'very_low'
            elif coop_ratio <= 0.4: base_state = 'low'
            elif coop_ratio <= 0.6: base_state = 'medium'
            elif coop_ratio <= 0.8: base_state = 'high'
            else: base_state = 'very_high'
        
        return base_state

    def choose_action(self, **kwargs):
        """Overrides base to add memory update."""
        action = super().choose_action(**kwargs)
        if 'opponent_id' not in kwargs and self.state_type == "memory_enhanced":
             # Only update memory for neighborhood mode for simplicity
            self.memory.append('cooperate' if action == COOPERATE else 'defect')
        return action

    def reset(self):
        """Resets the agent and applies epsilon decay."""
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0
        self.opponent_last_moves.clear()
        self.pairwise_last_states.clear()
        self.pairwise_last_actions.clear()
        self.last_state_nperson = None
        self.last_action_nperson = None
        self.memory.clear()
        # Apply epsilon decay at the end of a run
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)