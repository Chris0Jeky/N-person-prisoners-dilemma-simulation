"""
Agent implementations for the discount factor sensitivity analysis
"""

import random
from collections import defaultdict, deque
import numpy as np

# Constants
COOPERATE, DEFECT = 0, 1


# Base Agent Class
class BaseAgent:
    def __init__(self, agent_id, strategy_name):
        self.agent_id, self.strategy_name = agent_id, strategy_name
        self.total_score = 0

    def reset(self): 
        self.total_score = 0


# Static Agent (TFT, AllC, AllD, Random)
class StaticAgent(BaseAgent):
    def __init__(self, agent_id, strategy_name="TFT", error_rate=0.0, **kwargs):
        super().__init__(agent_id, strategy_name)
        self.strategy_name = strategy_name
        self.error_rate = error_rate
        self.opponent_last_moves = {}
        self.last_neighborhood_move = COOPERATE

    def _apply_error(self, intended_move):
        """Apply error rate to the intended move"""
        if random.random() < self.error_rate:
            return random.choice([COOPERATE, DEFECT])
        return intended_move

    def choose_pairwise_action(self, opponent_id):
        if self.strategy_name == "AllC":
            intended = COOPERATE
        elif self.strategy_name == "AllD":
            intended = DEFECT
        elif self.strategy_name == "Random":
            intended = random.choice([COOPERATE, DEFECT])
        elif self.strategy_name == "TFT" or self.strategy_name == "TFT-E":
            intended = self.opponent_last_moves.get(opponent_id, COOPERATE)
        else:  # Default TFT
            intended = self.opponent_last_moves.get(opponent_id, COOPERATE)
        
        return self._apply_error(intended)

    def choose_neighborhood_action(self, coop_ratio):
        if self.strategy_name == "AllC":
            intended = COOPERATE
        elif self.strategy_name == "AllD":
            intended = DEFECT
        elif self.strategy_name == "Random":
            intended = random.choice([COOPERATE, DEFECT])
        elif self.strategy_name == "TFT" or self.strategy_name == "TFT-E":
            # TFT in neighborhood: probabilistic cooperation based on cooperation ratio
            if coop_ratio is None:
                intended = COOPERATE  # Cooperate on first round
            else:
                # Cooperate with probability equal to cooperation ratio
                intended = COOPERATE if random.random() < coop_ratio else DEFECT
        else:  # Default TFT
            if coop_ratio is None:
                intended = COOPERATE
            else:
                # Probabilistic cooperation based on cooperation ratio
                intended = COOPERATE if random.random() < coop_ratio else DEFECT
        
        return self._apply_error(intended)

    def record_pairwise_outcome(self, opponent_id, my_move, opponent_move, reward):
        self.total_score += reward
        self.opponent_last_moves[opponent_id] = opponent_move

    def record_neighborhood_outcome(self, coop_ratio, reward):
        self.total_score += reward
        self.last_neighborhood_move = COOPERATE if coop_ratio and coop_ratio >= 0.5 else DEFECT

    def reset(self):
        super().reset()
        self.opponent_last_moves.clear()
        self.last_neighborhood_move = COOPERATE


# Legacy Q-Learning Agent (2-round history)
class LegacyQLearner(BaseAgent):
    """Legacy Q-Learning with 2-round history and cooperation trends"""
    
    def __init__(self, agent_id, params, **kwargs):
        super().__init__(agent_id, "LegacyQL")
        self.params = params
        self.reset()
        
    def _get_pairwise_state(self, opponent_id):
        """Get state based on last 2 rounds of history"""
        if opponent_id not in self.my_history_pairwise:
            return 'initial'
        
        my_hist = self.my_history_pairwise[opponent_id]
        opp_hist = self.opp_history_pairwise[opponent_id]
        
        if len(my_hist) < 2 or len(opp_hist) < 2:
            # Only 1 round of history
            if len(my_hist) == 1 and len(opp_hist) == 1:
                return f"1round_M{'C' if my_hist[0] == COOPERATE else 'D'}_O{'C' if opp_hist[0] == COOPERATE else 'D'}"
            return 'initial'
        
        # Full 2-round history: (My_t-2, Opp_t-2, My_t-1, Opp_t-1)
        state = f"M{self._move_to_char(my_hist[0])}{self._move_to_char(my_hist[1])}_O{self._move_to_char(opp_hist[0])}{self._move_to_char(opp_hist[1])}"
        return state
    
    def _get_neighborhood_state(self):
        """Get state based on cooperation trends and my recent behavior"""
        if len(self.coop_ratio_history) == 0:
            return 'initial'
        
        if len(self.coop_ratio_history) == 1:
            # One round of history
            ratio = self.coop_ratio_history[0]
            my_last = 'C' if len(self.my_history_nperson) > 0 and self.my_history_nperson[0] == COOPERATE else 'D' if len(self.my_history_nperson) > 0 else 'X'
            return f"1round_{self._ratio_to_category(ratio)}_M{my_last}"
        
        # Two rounds of history - look at trend
        ratio_t2 = self.coop_ratio_history[0]
        ratio_t1 = self.coop_ratio_history[1]
        trend = 'up' if ratio_t1 > ratio_t2 + 0.1 else 'down' if ratio_t1 < ratio_t2 - 0.1 else 'stable'
        
        # Include my recent behavior
        my_recent = ""
        if len(self.my_history_nperson) >= 2:
            my_recent = f"_M{self._move_to_char(self.my_history_nperson[0])}{self._move_to_char(self.my_history_nperson[1])}"
        
        return f"{self._ratio_to_category(ratio_t1)}_{trend}{my_recent}"
    
    def _move_to_char(self, move):
        """Convert move to character"""
        return 'C' if move == COOPERATE else 'D'
    
    def _ratio_to_category(self, ratio):
        """Convert ratio to category"""
        if ratio is None:
            return 'unknown'
        if ratio <= 0.33:
            return 'low'
        elif ratio <= 0.67:
            return 'medium'
        else:
            return 'high'
    
    def _ensure_state_exists(self, state, q_table):
        """Initialize Q-values for new states with optimistic values"""
        if state not in q_table:
            # Use optimistic_init parameter (default to 0.0)
            init_val = self.params.get('optimistic_init', 0.0)
            q_table[state] = {COOPERATE: init_val, DEFECT: init_val}
    
    def choose_pairwise_action(self, opponent_id):
        """Choose action for pairwise interaction"""
        state = self._get_pairwise_state(opponent_id)
        
        # Ensure Q-table exists for this opponent
        if opponent_id not in self.q_tables:
            self.q_tables[opponent_id] = {}
        
        self._ensure_state_exists(state, self.q_tables[opponent_id])
        
        # Epsilon-greedy action selection
        if random.random() < self._get_epsilon():
            return random.choice([COOPERATE, DEFECT])
        else:
            q_values = self.q_tables[opponent_id][state]
            if q_values[COOPERATE] >= q_values[DEFECT]:
                return COOPERATE
            else:
                return DEFECT
    
    def record_pairwise_outcome(self, opponent_id, my_move, opponent_move, reward):
        """Record outcome and update Q-values"""
        self.total_score += reward
        
        # Get current state
        state = self._get_pairwise_state(opponent_id)
        
        # Update histories
        if opponent_id not in self.my_history_pairwise:
            self.my_history_pairwise[opponent_id] = deque(maxlen=2)
            self.opp_history_pairwise[opponent_id] = deque(maxlen=2)
        
        self.my_history_pairwise[opponent_id].append(my_move)
        self.opp_history_pairwise[opponent_id].append(opponent_move)
        
        # Get next state
        next_state = self._get_pairwise_state(opponent_id)
        
        # Update Q-value
        self._update_q_value(opponent_id, state, my_move, reward, next_state)
        
        # Decay epsilon after each episode
        if hasattr(self, 'episode_count'):
            self.episode_count += 1
            if self.episode_count % 100 == 0:  # Decay every 100 interactions
                self._decay_epsilon()
    
    def choose_neighborhood_action(self, coop_ratio):
        """Choose action for neighborhood interaction"""
        # Update cooperation ratio history
        if coop_ratio is not None:
            self.coop_ratio_history.append(coop_ratio)
        
        state = self._get_neighborhood_state()
        self._ensure_state_exists(state, self.n_q_table)
        
        # Epsilon-greedy action selection
        if random.random() < self._get_epsilon():
            action = random.choice([COOPERATE, DEFECT])
        else:
            q_values = self.n_q_table[state]
            if q_values[COOPERATE] >= q_values[DEFECT]:
                action = COOPERATE
            else:
                action = DEFECT
        
        # Store the action for the update later
        self._last_neighborhood_action = action
        return action
    
    def record_neighborhood_outcome(self, coop_ratio, reward):
        """Record outcome and update Q-values"""
        self.total_score += reward
        
        # We need to track what action we just took
        # The simulation framework doesn't tell us, so we'll store it
        if hasattr(self, '_last_neighborhood_action'):
            action = self._last_neighborhood_action
            
            # Get current state (before updating history)
            state = self._get_neighborhood_state()
            
            # Update my history with the action we took
            self.my_history_nperson.append(action)
            
            # Get next state
            next_state = self._get_neighborhood_state()
            
            # Update Q-value
            self._update_neighborhood_q_value(state, action, reward, next_state)
    
    def _update_q_value(self, opponent_id, state, action, reward, next_state):
        """Update Q-value using standard Q-learning formula"""
        if opponent_id not in self.q_tables:
            self.q_tables[opponent_id] = {}
            
        self._ensure_state_exists(state, self.q_tables[opponent_id])
        self._ensure_state_exists(next_state, self.q_tables[opponent_id])
        
        lr = self.params.get('lr', 0.15)
        df = self.params.get('df', 0.95)
        
        current_q = self.q_tables[opponent_id][state][action]
        max_next_q = max(self.q_tables[opponent_id][next_state].values())
        
        new_q = current_q + lr * (reward + df * max_next_q - current_q)
        self.q_tables[opponent_id][state][action] = new_q
    
    def _update_neighborhood_q_value(self, state, action, reward, next_state):
        """Update Q-value for neighborhood"""
        self._ensure_state_exists(state, self.n_q_table)
        self._ensure_state_exists(next_state, self.n_q_table)
        
        lr = self.params.get('lr', 0.15)
        df = self.params.get('df', 0.95)
        
        current_q = self.n_q_table[state][action]
        max_next_q = max(self.n_q_table[next_state].values())
        
        new_q = current_q + lr * (reward + df * max_next_q - current_q)
        self.n_q_table[state][action] = new_q
    
    def _get_epsilon(self):
        """Get current epsilon value"""
        if hasattr(self, 'current_epsilon'):
            return self.current_epsilon
        return self.params.get('eps', 0.2)
    
    def _decay_epsilon(self):
        """Decay epsilon"""
        if hasattr(self, 'current_epsilon'):
            epsilon_decay = self.params.get('epsilon_decay', 0.995)
            epsilon_min = self.params.get('epsilon_min', 0.05)
            self.current_epsilon = max(epsilon_min, self.current_epsilon * epsilon_decay)
    
    def reset(self):
        """Reset agent state"""
        super().reset()
        self.q_tables = {}  # Separate Q-table per opponent
        self.n_q_table = {}  # Neighborhood Q-table
        
        # History tracking for richer state
        self.my_history_pairwise = {}  # opponent_id -> [my last 2 moves]
        self.opp_history_pairwise = {}  # opponent_id -> [their last 2 moves]
        self.my_history_nperson = deque(maxlen=2)  # my last 2 moves
        self.coop_ratio_history = deque(maxlen=2)  # last 2 cooperation ratios
        
        # Initialize epsilon
        self.current_epsilon = self.params.get('eps', 0.2)
        self.episode_count = 0


# Legacy 3-Round Q-Learning Agent (3-round history)
class Legacy3RoundQLearner(BaseAgent):
    """Legacy Q-Learning with 3-round history for better pattern detection"""
    
    def __init__(self, agent_id, params, **kwargs):
        super().__init__(agent_id, "Legacy3RoundQL")
        self.params = params
        self.history_length = params.get('history_length', 3)
        self.reset()
        
    def _get_pairwise_state(self, opponent_id):
        """Get state based on last 3 rounds of history"""
        if opponent_id not in self.my_history_pairwise:
            return 'initial'
        
        my_hist = self.my_history_pairwise[opponent_id]
        opp_hist = self.opp_history_pairwise[opponent_id]
        
        hist_len = len(my_hist)
        if hist_len == 0:
            return 'initial'
        elif hist_len == 1:
            return f"1round_M{'C' if my_hist[0] == COOPERATE else 'D'}_O{'C' if opp_hist[0] == COOPERATE else 'D'}"
        elif hist_len == 2:
            return f"2round_M{self._move_to_char(my_hist[0])}{self._move_to_char(my_hist[1])}_O{self._move_to_char(opp_hist[0])}{self._move_to_char(opp_hist[1])}"
        else:
            # Full 3-round history
            state = f"M{self._move_to_char(my_hist[0])}{self._move_to_char(my_hist[1])}{self._move_to_char(my_hist[2])}_O{self._move_to_char(opp_hist[0])}{self._move_to_char(opp_hist[1])}{self._move_to_char(opp_hist[2])}"
            return state
    
    def _get_neighborhood_state(self):
        """Get state based on cooperation trends and my recent behavior (3 rounds)"""
        if len(self.coop_ratio_history) == 0:
            return 'initial'
        
        if len(self.coop_ratio_history) == 1:
            ratio = self.coop_ratio_history[0]
            my_last = 'C' if len(self.my_history_nperson) > 0 and self.my_history_nperson[0] == COOPERATE else 'D' if len(self.my_history_nperson) > 0 else 'X'
            return f"1round_{self._ratio_to_category(ratio)}_M{my_last}"
        
        if len(self.coop_ratio_history) == 2:
            # Two rounds - simple trend
            ratio_t2 = self.coop_ratio_history[0]
            ratio_t1 = self.coop_ratio_history[1]
            trend = 'up' if ratio_t1 > ratio_t2 + 0.1 else 'down' if ratio_t1 < ratio_t2 - 0.1 else 'stable'
            my_recent = ""
            if len(self.my_history_nperson) >= 2:
                my_recent = f"_M{self._move_to_char(self.my_history_nperson[0])}{self._move_to_char(self.my_history_nperson[1])}"
            return f"2round_{self._ratio_to_category(ratio_t1)}_{trend}{my_recent}"
        
        # Three rounds - complex trend analysis
        ratio_t3 = self.coop_ratio_history[0]
        ratio_t2 = self.coop_ratio_history[1]
        ratio_t1 = self.coop_ratio_history[2]
        
        # Analyze trend over 3 rounds
        if ratio_t1 > ratio_t2 + 0.1 and ratio_t2 > ratio_t3 + 0.1:
            trend = 'strong_up'
        elif ratio_t1 < ratio_t2 - 0.1 and ratio_t2 < ratio_t3 - 0.1:
            trend = 'strong_down'
        elif ratio_t1 > ratio_t2 + 0.1:
            trend = 'up'
        elif ratio_t1 < ratio_t2 - 0.1:
            trend = 'down'
        else:
            trend = 'stable'
        
        # Include my recent 3-round behavior
        my_recent = ""
        if len(self.my_history_nperson) >= 3:
            my_recent = f"_M{self._move_to_char(self.my_history_nperson[0])}{self._move_to_char(self.my_history_nperson[1])}{self._move_to_char(self.my_history_nperson[2])}"
        
        return f"{self._ratio_to_category(ratio_t1)}_{trend}{my_recent}"
    
    def _move_to_char(self, move):
        """Convert move to character"""
        return 'C' if move == COOPERATE else 'D'
    
    def _ratio_to_category(self, ratio):
        """Convert ratio to category"""
        if ratio is None:
            return 'unknown'
        if ratio <= 0.33:
            return 'low'
        elif ratio <= 0.67:
            return 'medium'
        else:
            return 'high'
    
    def _ensure_state_exists(self, state, q_table):
        """Initialize Q-values for new states"""
        if state not in q_table:
            # Use optimistic_init parameter
            init_val = self.params.get('optimistic_init', -0.1)
            q_table[state] = {COOPERATE: init_val, DEFECT: init_val}
    
    def choose_pairwise_action(self, opponent_id):
        """Choose action for pairwise interaction"""
        state = self._get_pairwise_state(opponent_id)
        
        # Ensure Q-table exists for this opponent
        if opponent_id not in self.q_tables:
            self.q_tables[opponent_id] = {}
        
        self._ensure_state_exists(state, self.q_tables[opponent_id])
        
        # Epsilon-greedy action selection
        if random.random() < self._get_epsilon():
            return random.choice([COOPERATE, DEFECT])
        else:
            q_values = self.q_tables[opponent_id][state]
            if q_values[COOPERATE] >= q_values[DEFECT]:
                return COOPERATE
            else:
                return DEFECT
    
    def record_pairwise_outcome(self, opponent_id, my_move, opponent_move, reward):
        """Record outcome and update Q-values"""
        self.total_score += reward
        
        # Get current state
        state = self._get_pairwise_state(opponent_id)
        
        # Update histories
        if opponent_id not in self.my_history_pairwise:
            self.my_history_pairwise[opponent_id] = deque(maxlen=self.history_length)
            self.opp_history_pairwise[opponent_id] = deque(maxlen=self.history_length)
        
        self.my_history_pairwise[opponent_id].append(my_move)
        self.opp_history_pairwise[opponent_id].append(opponent_move)
        
        # Get next state
        next_state = self._get_pairwise_state(opponent_id)
        
        # Update Q-value
        self._update_q_value(opponent_id, state, my_move, reward, next_state)
        
        # Decay epsilon after each episode
        if hasattr(self, 'episode_count'):
            self.episode_count += 1
            if self.episode_count % 100 == 0:  # Decay every 100 interactions
                self._decay_epsilon()
    
    def choose_neighborhood_action(self, coop_ratio):
        """Choose action for neighborhood interaction"""
        # Update cooperation ratio history
        if coop_ratio is not None:
            self.coop_ratio_history.append(coop_ratio)
        
        state = self._get_neighborhood_state()
        self._ensure_state_exists(state, self.n_q_table)
        
        # Epsilon-greedy action selection
        if random.random() < self._get_epsilon():
            action = random.choice([COOPERATE, DEFECT])
        else:
            q_values = self.n_q_table[state]
            if q_values[COOPERATE] >= q_values[DEFECT]:
                action = COOPERATE
            else:
                action = DEFECT
        
        # Store the action for the update later
        self._last_neighborhood_action = action
        return action
    
    def record_neighborhood_outcome(self, coop_ratio, reward):
        """Record outcome and update Q-values"""
        self.total_score += reward
        
        if hasattr(self, '_last_neighborhood_action'):
            action = self._last_neighborhood_action
            
            # Get current state (before updating history)
            state = self._get_neighborhood_state()
            
            # Update my history with the action we took
            self.my_history_nperson.append(action)
            
            # Get next state
            next_state = self._get_neighborhood_state()
            
            # Update Q-value
            self._update_neighborhood_q_value(state, action, reward, next_state)
    
    def _update_q_value(self, opponent_id, state, action, reward, next_state):
        """Update Q-value using standard Q-learning formula"""
        if opponent_id not in self.q_tables:
            self.q_tables[opponent_id] = {}
            
        self._ensure_state_exists(state, self.q_tables[opponent_id])
        self._ensure_state_exists(next_state, self.q_tables[opponent_id])
        
        lr = self.params.get('lr', 0.15)
        df = self.params.get('df', 0.95)
        
        current_q = self.q_tables[opponent_id][state][action]
        max_next_q = max(self.q_tables[opponent_id][next_state].values())
        
        new_q = current_q + lr * (reward + df * max_next_q - current_q)
        self.q_tables[opponent_id][state][action] = new_q
    
    def _update_neighborhood_q_value(self, state, action, reward, next_state):
        """Update Q-value for neighborhood"""
        self._ensure_state_exists(state, self.n_q_table)
        self._ensure_state_exists(next_state, self.n_q_table)
        
        lr = self.params.get('lr', 0.15)
        df = self.params.get('df', 0.95)
        
        current_q = self.n_q_table[state][action]
        max_next_q = max(self.n_q_table[next_state].values())
        
        new_q = current_q + lr * (reward + df * max_next_q - current_q)
        self.n_q_table[state][action] = new_q
    
    def _get_epsilon(self):
        """Get current epsilon value"""
        if hasattr(self, 'current_epsilon'):
            return self.current_epsilon
        return self.params.get('eps', 0.2)
    
    def _decay_epsilon(self):
        """Decay epsilon"""
        if hasattr(self, 'current_epsilon'):
            epsilon_decay = self.params.get('epsilon_decay', 0.995)
            epsilon_min = self.params.get('epsilon_min', 0.05)
            self.current_epsilon = max(epsilon_min, self.current_epsilon * epsilon_decay)
    
    def reset(self):
        """Reset agent state"""
        super().reset()
        self.q_tables = {}  # Separate Q-table per opponent
        self.n_q_table = {}  # Neighborhood Q-table
        
        # History tracking for richer state
        self.my_history_pairwise = {}  # opponent_id -> [my last 3 moves]
        self.opp_history_pairwise = {}  # opponent_id -> [their last 3 moves]
        self.my_history_nperson = deque(maxlen=3)  # my last 3 moves
        self.coop_ratio_history = deque(maxlen=3)  # last 3 cooperation ratios
        
        # Initialize epsilon
        self.current_epsilon = self.params.get('eps', 0.2)
        self.episode_count = 0