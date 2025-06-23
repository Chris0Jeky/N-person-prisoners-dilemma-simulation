# final_agents_v2.py
import random
from collections import defaultdict, deque
import numpy as np

# --- Constants ---
COOPERATE, DEFECT = 0, 1


# --- Base Agents ---
class BaseAgent:
    def __init__(self, agent_id, strategy_name):
        self.agent_id, self.strategy_name = agent_id, strategy_name
        self.total_score = 0

    def reset(self): 
        self.total_score = 0


class StaticAgent(BaseAgent):
    def __init__(self, agent_id, strategy_name="TFT", error_rate=0.0, **kwargs):
        super().__init__(agent_id, strategy_name)
        self.strategy_name = strategy_name
        self.error_rate = error_rate
        self.initial_error_rate = error_rate  # Store initial error rate
        self.opponent_last_moves = {}
        self.last_neighborhood_move = COOPERATE
        self.round_count = 0  # Track rounds for decay
        self.decay_rate = 0.9995  # Decay factor per round (reaches ~0.01 after 10000 rounds)

    def _apply_error(self, intended_move):
        """Apply error rate to the intended move"""
        # Update error rate for TFT-E with decay
        if self.strategy_name == "TFT-E" and self.initial_error_rate > 0:
            self.round_count += 1
            # Exponential decay: error_rate = initial_rate * decay_rate^round
            self.error_rate = self.initial_error_rate * (self.decay_rate ** self.round_count)
            # Stop decaying when we get very close to 0
            if self.error_rate < 0.001:
                self.error_rate = 0.0
        
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
            if coop_ratio is None:
                intended = COOPERATE
            else:
                intended = COOPERATE if random.random() < coop_ratio else DEFECT
        else:
            if coop_ratio is None:
                intended = COOPERATE
            else:
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
        self.round_count = 0
        self.error_rate = self.initial_error_rate  # Reset error rate to initial value


# --- Vanilla Q-Learning Agent (Simple 8-state for neighborhood) ---
class VanillaQLearner(BaseAgent):
    def __init__(self, agent_id, params, **kwargs):
        super().__init__(agent_id, "VanillaQL")
        self.params = params
        self.reset()

    def _get_state(self, opponent_id):
        """Get state for pairwise interaction"""
        if opponent_id not in self.histories:
            self.histories[opponent_id] = self._make_history_deque()
        history = self.histories[opponent_id]
        if len(history) < 2: return "start"
        return str(tuple(history))

    def choose_pairwise_action(self, opponent_id):
        state = self._get_state(opponent_id)
        if opponent_id not in self.q_tables:
            self.q_tables[opponent_id] = defaultdict(lambda: defaultdict(float))
        
        if random.random() < self.params['eps']:
            return random.choice([COOPERATE, DEFECT])
        else:
            q_values = self.q_tables[opponent_id][state]
            if q_values[COOPERATE] == q_values[DEFECT]:
                return random.choice([COOPERATE, DEFECT])
            return max(q_values, key=q_values.get)

    def record_pairwise_outcome(self, opponent_id, my_move, opponent_move, reward):
        self.total_score += reward
        state = self._get_state(opponent_id)
        
        if opponent_id not in self.histories:
            self.histories[opponent_id] = self._make_history_deque()
        self.histories[opponent_id].append((my_move, opponent_move))
        
        next_state = self._get_state(opponent_id)
        self._update_q_value(opponent_id, state, my_move, reward, next_state)

    def choose_neighborhood_action(self, coop_ratio):
        """Vanilla uses simple 8-state representation"""
        state = self._get_simple_neighborhood_state()
        
        if random.random() < self.params['eps']:
            return random.choice([COOPERATE, DEFECT])
        else:
            q_values = self.n_q_table[state]
            if q_values[COOPERATE] == q_values[DEFECT]:
                return random.choice([COOPERATE, DEFECT])
            return max(q_values, key=q_values.get)

    def record_neighborhood_outcome(self, coop_ratio, reward):
        self.total_score += reward
        
        # Get current state before updating
        state = self._get_simple_neighborhood_state()
        
        # Determine my action
        if self.last_coop_ratio is None:
            my_action = COOPERATE
        else:
            my_action = COOPERATE if coop_ratio > self.last_coop_ratio else DEFECT
        
        # Update state info
        self.last_action = my_action
        self.last_coop_category = self._categorize_coop_ratio(coop_ratio) if coop_ratio is not None else None
        
        # Get next state
        next_state = self._get_simple_neighborhood_state()
        
        # Update Q-value
        self._update_neighborhood_q_value(state, my_action, reward, next_state)
        self.last_coop_ratio = coop_ratio

    def _get_simple_neighborhood_state(self):
        """Simple 8-state representation: (last_coop_category, last_action)"""
        if self.last_coop_category is None:
            return "start"
        
        # 3 categories x 2 actions = 6 states + start state
        return f"{self.last_coop_category}_{self.last_action}"

    def _categorize_coop_ratio(self, coop_ratio):
        """Convert cooperation ratio to category"""
        if coop_ratio < 0.33:
            return "low"
        elif coop_ratio < 0.67:
            return "medium"
        else:
            return "high"

    def _update_q_value(self, opponent_id, state, action, reward, next_state):
        lr = self.params['lr']
        df = self.params['df']
        
        current_q = self.q_tables[opponent_id][state][action]
        max_next_q = max(self.q_tables[opponent_id][next_state].values()) if self.q_tables[opponent_id][next_state] else 0
        new_q = current_q + lr * (reward + df * max_next_q - current_q)
        self.q_tables[opponent_id][state][action] = new_q

    def _update_neighborhood_q_value(self, state, action, reward, next_state):
        lr = self.params['lr']
        df = self.params['df']
        
        current_q = self.n_q_table[state][action]
        max_next_q = max(self.n_q_table[next_state].values()) if self.n_q_table[next_state] else 0
        new_q = current_q + lr * (reward + df * max_next_q - current_q)
        self.n_q_table[state][action] = new_q

    def _make_history_deque(self):
        return deque(maxlen=2)

    def reset(self):
        super().reset()
        self.q_tables = {}
        self.n_q_table = defaultdict(lambda: defaultdict(float))
        self.histories = {}
        self.last_coop_ratio = None
        self.last_coop_category = None
        self.last_action = COOPERATE


# --- Enhanced Q-Learning Agent (EQL with 4-round history) ---
class EnhancedQLearner(BaseAgent):
    def __init__(self, agent_id, params, **kwargs):
        super().__init__(agent_id, "EQL")
        self.params = params
        self.reset()

    def _get_state(self, opponent_id):
        """Get state for pairwise interaction"""
        if opponent_id not in self.histories:
            self.histories[opponent_id] = self._make_history_deque()
        history = self.histories[opponent_id]
        if len(history) < 2: return "start"
        return str(tuple(history))

    def choose_pairwise_action(self, opponent_id):
        state = self._get_state(opponent_id)
        if opponent_id not in self.q_tables:
            self.q_tables[opponent_id] = defaultdict(lambda: defaultdict(float))
        
        if random.random() < self._get_epsilon():
            return random.choice([COOPERATE, DEFECT])
        else:
            q_values = self.q_tables[opponent_id][state]
            if q_values[COOPERATE] == q_values[DEFECT]:
                return random.choice([COOPERATE, DEFECT])
            return max(q_values, key=q_values.get)

    def record_pairwise_outcome(self, opponent_id, my_move, opponent_move, reward):
        self.total_score += reward
        state = self._get_state(opponent_id)
        
        if opponent_id not in self.histories:
            self.histories[opponent_id] = self._make_history_deque()
        self.histories[opponent_id].append((my_move, opponent_move))
        
        next_state = self._get_state(opponent_id)
        self._update_q_value(opponent_id, state, my_move, reward, next_state)
        self._adapt_parameters(reward)

    def choose_neighborhood_action(self, coop_ratio):
        """EQL uses 4-round tuple representation"""
        state = self._get_neighborhood_state()
        
        if random.random() < self._get_epsilon():
            return random.choice([COOPERATE, DEFECT])
        else:
            q_values = self.n_q_table[state]
            if q_values[COOPERATE] == q_values[DEFECT]:
                return random.choice([COOPERATE, DEFECT])
            return max(q_values, key=q_values.get)

    def record_neighborhood_outcome(self, coop_ratio, reward):
        self.total_score += reward
        
        # Get current state before updating history
        state = self._get_neighborhood_state()
        
        # Update cooperation ratio history with category
        if coop_ratio is not None:
            category = self._categorize_coop_ratio(coop_ratio)
            self.coop_ratio_history.append(category)
        
        # Get next state after updating history
        next_state = self._get_neighborhood_state()
        
        # Determine my last move
        if self.last_coop_ratio is None:
            my_last_move = COOPERATE
        else:
            my_last_move = COOPERATE if coop_ratio > self.last_coop_ratio else DEFECT
        
        self._update_neighborhood_q_value(state, my_last_move, reward, next_state)
        self._adapt_parameters(reward)
        self.last_coop_ratio = coop_ratio

    def _categorize_coop_ratio(self, coop_ratio):
        """Convert cooperation ratio to category"""
        if coop_ratio < 0.33:
            return "low"
        elif coop_ratio < 0.67:
            return "medium"
        else:
            return "high"

    def _get_neighborhood_state(self):
        """Get state as tuple of last 2 cooperation ratio categories"""
        if len(self.coop_ratio_history) == 0:
            return "start"
        elif len(self.coop_ratio_history) < 2:
            # Pad with 'start' if we don't have 2 rounds yet
            padding = ['start'] * (2 - len(self.coop_ratio_history))
            return str(tuple(padding + list(self.coop_ratio_history)))
        else:
            # Return tuple of last 2 categories
            return str(tuple(self.coop_ratio_history))

    def _update_q_value(self, opponent_id, state, action, reward, next_state):
        lr = self._get_learning_rate()
        df = self.params['df']
        
        current_q = self.q_tables[opponent_id][state][action]
        max_next_q = max(self.q_tables[opponent_id][next_state].values()) if self.q_tables[opponent_id][next_state] else 0
        new_q = current_q + lr * (reward + df * max_next_q - current_q)
        self.q_tables[opponent_id][state][action] = new_q

    def _update_neighborhood_q_value(self, state, action, reward, next_state):
        lr = self._get_learning_rate()
        df = self.params['df']
        
        current_q = self.n_q_table[state][action]
        max_next_q = max(self.n_q_table[next_state].values()) if self.n_q_table[next_state] else 0
        new_q = current_q + lr * (reward + df * max_next_q - current_q)
        self.n_q_table[state][action] = new_q

    def _get_learning_rate(self):
        if 'lr' in self.params:
            return self.params['lr']
        else:
            return self.current_lr

    def _get_epsilon(self):
        if 'eps' in self.params:
            return self.params['eps']
        else:
            return self.current_eps

    def _adapt_parameters(self, reward):
        # Only adapt if we have adaptive parameters
        if 'initial_lr' not in self.params:
            return
            
        self.reward_history.append(reward)
        
        window_size = self.params.get('reward_window_size', 2)
        if len(self.reward_history) < window_size:
            return
        
        recent_rewards = list(self.reward_history)[-window_size:]
        avg_reward = np.mean(recent_rewards)
        
        # Check performance trend
        mid_point = window_size // 2
        first_half_avg = np.mean(recent_rewards[:mid_point])
        second_half_avg = np.mean(recent_rewards[mid_point:])
        
        factor = self.params.get('adaptation_factor', 1.08)
        
        # Adapt learning rate
        if second_half_avg > first_half_avg:
            self.current_lr = max(self.params['min_lr'], self.current_lr / factor)
        else:
            self.current_lr = min(self.params['max_lr'], self.current_lr * factor)
        
        # Adapt exploration rate
        if avg_reward > 2.5:
            self.current_eps = max(self.params['min_eps'], self.current_eps / factor)
        else:
            self.current_eps = min(self.params['max_eps'], self.current_eps * factor)

    def _make_history_deque(self):
        return deque(maxlen=2)

    def reset(self):
        super().reset()
        self.q_tables = {}
        self.n_q_table = defaultdict(lambda: defaultdict(float))
        self.histories = {}
        self.last_coop_ratio = None
        self.coop_ratio_history = deque(maxlen=2)  # Store last 2 cooperation ratio categories
        self.reward_history = deque(maxlen=1000)
        
        # Set initial parameters
        if 'initial_lr' in self.params:
            self.current_lr = self.params['initial_lr']
            self.current_eps = self.params['initial_eps']
        else:
            self.current_lr = self.params.get('lr', 0.1)
            self.current_eps = self.params.get('eps', 0.1)


# For backward compatibility, keep PairwiseAdaptiveQLearner as an alias
PairwiseAdaptiveQLearner = EnhancedQLearner


# --- Legacy Q-Learning Agent (2-round history with trends) ---
class LegacyQLearner(BaseAgent):
    """Legacy Q-Learning with 2-round history and cooperation trends from qlearning_demo_generator.py"""
    
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
            # Optimistic initialization to encourage exploration
            q_table[state] = {COOPERATE: 0.1, DEFECT: 0.1}
    
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