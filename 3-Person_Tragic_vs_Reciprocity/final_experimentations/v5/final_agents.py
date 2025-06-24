# final_agents.py
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

    def reset(self): self.total_score = 0


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


# --- Specialized Q-Learning Agents ---
class PairwiseAdaptiveQLearner(BaseAgent):
    def __init__(self, agent_id, params, **kwargs):
        super().__init__(agent_id, "PairwiseAdaptive")
        self.params = params
        self.reset()

    def _get_state(self, opponent_id):
        if opponent_id not in self.histories:
            self.histories[opponent_id] = self._make_history_deque()
        history = self.histories[opponent_id]
        if len(history) < 2: return "start"
        return str(tuple(history))

    def choose_pairwise_action(self, opponent_id):
        state = self._get_state(opponent_id)
        # Initialize if needed
        if opponent_id not in self.q_tables:
            self.q_tables[opponent_id] = {}
        if opponent_id not in self.epsilons:
            self.epsilons[opponent_id] = self._get_initial_eps()
        
        q_table = self.q_tables[opponent_id]
        epsilon = self.epsilons[opponent_id]
        
        # Initialize state if needed
        if state not in q_table:
            q_table[state] = self._make_q_dict()
        if random.random() < epsilon:
            action = random.choice([COOPERATE, DEFECT])
        else:
            action = COOPERATE if q_table[state][COOPERATE] >= q_table[state][DEFECT] else DEFECT
        self.last_contexts[opponent_id] = {'state': state, 'action': action}
        return action

    def record_pairwise_outcome(self, opponent_id, my_move, opponent_move, reward):
        self.total_score += reward
        context = self.last_contexts.get(opponent_id)
        if not context: return
        
        # Initialize if needed
        if opponent_id not in self.histories:
            self.histories[opponent_id] = self._make_history_deque()
        if opponent_id not in self.learning_rates:
            self.learning_rates[opponent_id] = self._get_initial_lr()
        if opponent_id not in self.reward_windows:
            self.reward_windows[opponent_id] = self._make_reward_window()
        
        self.histories[opponent_id].append((my_move, opponent_move))
        next_state = self._get_state(opponent_id)
        lr = self.learning_rates[opponent_id]
        q_table = self.q_tables[opponent_id]
        
        # Initialize next state if needed
        if next_state not in q_table:
            q_table[next_state] = self._make_q_dict()
        old_q = q_table[context['state']][context['action']]
        next_max_q = max(q_table[next_state].values())
        df = self.params.get('df', 0.9)
        q_table[context['state']][context['action']] = old_q + lr * (reward + df * next_max_q - old_q)
        self.reward_windows[opponent_id].append(reward)
        self._adapt_parameters(opponent_id)

    def _adapt_parameters(self, opponent_id):
        win_size = self.params.get('reward_window_size')
        if not win_size: return

        window = self.reward_windows[opponent_id]
        if len(window) < win_size: return

        half = win_size // 2
        adapt_factor = self.params.get('adaptation_factor', 1.05)
        min_lr = self.params.get('min_lr', 0.05)
        max_lr = self.params.get('max_lr', 0.5)
        min_eps = self.params.get('min_eps', 0.01)
        max_eps = self.params.get('max_eps', 0.5)

        if np.mean(list(window)[half:]) > np.mean(list(window)[:half]):
            self.learning_rates[opponent_id] = max(min_lr, self.learning_rates[opponent_id] / adapt_factor)
            self.epsilons[opponent_id] = max(min_eps, self.epsilons[opponent_id] / adapt_factor)
        else:
            self.learning_rates[opponent_id] = min(max_lr, self.learning_rates[opponent_id] * adapt_factor)
            self.epsilons[opponent_id] = min(max_eps, self.epsilons[opponent_id] * adapt_factor)

    def choose_neighborhood_action(self, coop_ratio):
        """Choose action for neighborhood game based on cooperation ratio"""
        state = self._get_neighborhood_state(coop_ratio)
        epsilon = self.params.get('initial_eps', self.params.get('eps', 0.1))
        
        # Initialize neighborhood state if needed
        if state not in self.neighborhood_q_table:
            self.neighborhood_q_table[state] = self._make_q_dict()
        
        if random.random() < self.neighborhood_epsilon:
            action = random.choice([COOPERATE, DEFECT])
        else:
            action = COOPERATE if self.neighborhood_q_table[state][COOPERATE] >= self.neighborhood_q_table[state][DEFECT] else DEFECT
        
        self.last_neighborhood_context = {'state': state, 'action': action}
        return action
    
    def record_neighborhood_outcome(self, coop_ratio, reward):
        """Record outcome for neighborhood game"""
        self.total_score += reward
        if not hasattr(self, 'last_neighborhood_context') or not self.last_neighborhood_context:
            return
            
        next_state = self._get_neighborhood_state(coop_ratio)
        
        # Initialize next state if needed
        if next_state not in self.neighborhood_q_table:
            self.neighborhood_q_table[next_state] = self._make_q_dict()
            
        old_q = self.neighborhood_q_table[self.last_neighborhood_context['state']][self.last_neighborhood_context['action']]
        next_max_q = max(self.neighborhood_q_table[next_state].values())
        df = self.params.get('df', 0.9)
        
        self.neighborhood_q_table[self.last_neighborhood_context['state']][self.last_neighborhood_context['action']] = \
            old_q + self.neighborhood_lr * (reward + df * next_max_q - old_q)
        
        self.neighborhood_reward_window.append(reward)
        self._adapt_neighborhood_parameters()
    
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
    
    def _adapt_neighborhood_parameters(self):
        """Adapt learning parameters for neighborhood game"""
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

    def _make_q_dict(self):
        return {COOPERATE: 0.0, DEFECT: 0.0}
    
    def _make_history_deque(self):
        return deque(maxlen=2)
    
    def _make_reward_window(self):
        return deque(maxlen=self.params.get('reward_window_size', 20))
    
    def _get_initial_lr(self):
        return self.params.get('initial_lr', self.params.get('lr', 0.1))
    
    def _get_initial_eps(self):
        return self.params.get('initial_eps', self.params.get('eps', 0.1))

    def reset(self):
        super().reset()
        # Use regular dicts instead of defaultdicts with lambdas
        self.q_tables = {}
        self.histories = {}
        self.reward_windows = {}
        self.learning_rates = {}
        self.epsilons = {}
        self.last_contexts = {}
        # Initialize neighborhood attributes
        self.neighborhood_q_table = {}
        self.neighborhood_lr = self._get_initial_lr()
        self.neighborhood_epsilon = self._get_initial_eps()
        self.neighborhood_reward_window = self._make_reward_window()
        self.last_neighborhood_context = None


class HystereticQLearner(BaseAgent):
    """
    Hysteretic Q-Learning agent that uses different learning rates for positive and negative updates.
    Uses higher learning rate (lr) for positive updates and lower rate (beta) for negative updates.
    This creates an optimistic bias that can lead to better cooperation.
    """
    def __init__(self, agent_id, params, **kwargs):
        super().__init__(agent_id, "Hysteretic")
        self.params = params
        self.reset()

    def _get_state(self, opponent_id):
        if opponent_id not in self.histories:
            self.histories[opponent_id] = self._make_history_deque()
        history = self.histories[opponent_id]
        if len(history) < 2: return "start"
        return str(tuple(history))

    def _make_q_dict(self):
        return {COOPERATE: 0.0, DEFECT: 0.0}
    
    def _make_history_deque(self):
        return deque(maxlen=2)

    def choose_pairwise_action(self, opponent_id):
        state = self._get_state(opponent_id)
        # Initialize if needed
        if opponent_id not in self.q_tables:
            self.q_tables[opponent_id] = {}
        
        q_table = self.q_tables[opponent_id]
        epsilon = self.params.get('eps', 0.1)
        
        # Initialize state if needed
        if state not in q_table:
            q_table[state] = self._make_q_dict()
            
        if random.random() < epsilon:
            action = random.choice(['cooperate', 'defect'])
        else:
            action = 'cooperate' if q_table[state]['cooperate'] >= q_table[state]['defect'] else 'defect'
        self.last_contexts[opponent_id] = {'state': state, 'action': action}
        return COOPERATE if action == 'cooperate' else DEFECT

    def record_pairwise_outcome(self, opponent_id, my_move, opponent_move, reward):
        self.total_score += reward
        context = self.last_contexts.get(opponent_id)
        if not context: return
        
        # Initialize if needed
        if opponent_id not in self.histories:
            self.histories[opponent_id] = self._make_history_deque()
        
        self.histories[opponent_id].append((my_move, opponent_move))
        next_state = self._get_state(opponent_id)
        q_table = self.q_tables[opponent_id]
        
        # Initialize next state if needed
        if next_state not in q_table:
            q_table[next_state] = self._make_q_dict()
            
        # Hysteretic Q-learning update
        current_q = q_table[context['state']][context['action']]
        next_max_q = max(q_table[next_state].values())
        df = self.params.get('df', 0.9)
        target_q = reward + df * next_max_q
        delta = target_q - current_q
        
        # Use different learning rates for positive and negative updates
        if delta >= 0:
            # Good news: use normal learning rate
            lr = self.params.get('lr', 0.1)
            q_table[context['state']][context['action']] = current_q + lr * delta
        else:
            # Bad news: use beta (lower learning rate)
            beta = self.params.get('beta', 0.01)
            q_table[context['state']][context['action']] = current_q + beta * delta

    def choose_neighborhood_action(self, coop_ratio):
        """Choose action for neighborhood game based on cooperation ratio"""
        state = self._get_neighborhood_state(coop_ratio)
        
        # Initialize neighborhood state if needed
        if state not in self.neighborhood_q_table:
            self.neighborhood_q_table[state] = self._make_q_dict()
        
        epsilon = self.params.get('eps', 0.1)
        if random.random() < epsilon:
            action = random.choice(['cooperate', 'defect'])
        else:
            action = 'cooperate' if self.neighborhood_q_table[state]['cooperate'] >= self.neighborhood_q_table[state]['defect'] else 'defect'
        
        self.last_neighborhood_context = {'state': state, 'action': action}
        return COOPERATE if action == 'cooperate' else DEFECT

    def record_neighborhood_outcome(self, coop_ratio, reward):
        """Record outcome for neighborhood game with hysteretic learning"""
        self.total_score += reward
        if not hasattr(self, 'last_neighborhood_context') or not self.last_neighborhood_context:
            return
            
        next_state = self._get_neighborhood_state(coop_ratio)
        
        # Initialize next state if needed
        if next_state not in self.neighborhood_q_table:
            self.neighborhood_q_table[next_state] = self._make_q_dict()
        
        # Hysteretic Q-learning update
        current_q = self.neighborhood_q_table[self.last_neighborhood_context['state']][self.last_neighborhood_context['action']]
        next_max_q = max(self.neighborhood_q_table[next_state].values())
        df = self.params.get('df', 0.9)
        target_q = reward + df * next_max_q
        delta = target_q - current_q
        
        # Use different learning rates for positive and negative updates
        if delta >= 0:
            # Good news: use normal learning rate
            lr = self.params.get('lr', 0.1)
            self.neighborhood_q_table[self.last_neighborhood_context['state']][self.last_neighborhood_context['action']] = current_q + lr * delta
        else:
            # Bad news: use beta (lower learning rate)
            beta = self.params.get('beta', 0.01)
            self.neighborhood_q_table[self.last_neighborhood_context['state']][self.last_neighborhood_context['action']] = current_q + beta * delta

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
        # Use regular dicts instead of defaultdicts for picklability
        self.q_tables = {}
        self.histories = {}
        self.last_contexts = {}
        # Initialize neighborhood attributes
        self.neighborhood_q_table = {}
        self.last_neighborhood_context = None


class NeighborhoodAdaptiveQLearner(BaseAgent):
    def __init__(self, agent_id, params, **kwargs):
        super().__init__(agent_id, "NeighborhoodAdaptive")
        self.params = params
        self.reset()

    def _get_state(self, coop_ratio):
        if coop_ratio is None: return 'start'
        if coop_ratio <= 0.33: return 'low'
        return 'medium' if coop_ratio <= 0.67 else 'high'

    def choose_neighborhood_action(self, coop_ratio):
        state = self._get_state(coop_ratio)
        
        # Initialize state if needed
        if state not in self.q_table:
            self.q_table[state] = self._make_q_dict()
            
        if random.random() < self.epsilon:
            action = random.choice(['cooperate', 'defect'])
        else:
            action = 'cooperate' if self.q_table[state]['cooperate'] >= self.q_table[state]['defect'] else 'defect'
        self.last_context = {'state': state, 'action': action}
        return COOPERATE if action == 'cooperate' else DEFECT

    def record_neighborhood_outcome(self, coop_ratio, reward):
        self.total_score += reward
        if not self.last_context: return
        next_state = self._get_state(coop_ratio)
        
        # Initialize next state if needed
        if next_state not in self.q_table:
            self.q_table[next_state] = self._make_q_dict()
            
        old_q = self.q_table[self.last_context['state']][self.last_context['action']]
        next_max_q = max(self.q_table[next_state].values())
        df = self.params.get('df', 0.9)
        self.q_table[self.last_context['state']][self.last_context['action']] = old_q + self.lr * (
                    reward + df * next_max_q - old_q)
        self.reward_window.append(reward)
        self._adapt_parameters()

    def _adapt_parameters(self):
        win_size = self.params.get('reward_window_size')
        if not win_size: return

        window = self.reward_window
        if len(window) < win_size: return

        half = win_size // 2
        adapt_factor = self.params.get('adaptation_factor', 1.05)
        min_lr = self.params.get('min_lr', 0.05)
        max_lr = self.params.get('max_lr', 0.5)
        min_eps = self.params.get('min_eps', 0.01)
        max_eps = self.params.get('max_eps', 0.5)

        if np.mean(list(window)[half:]) > np.mean(list(window)[:half]):
            self.lr = max(min_lr, self.lr / adapt_factor)
            self.epsilon = max(min_eps, self.epsilon / adapt_factor)
        else:
            self.lr = min(max_lr, self.lr * adapt_factor)
            self.epsilon = min(max_eps, self.epsilon * adapt_factor)

    def _make_q_dict(self):
        return {COOPERATE: 0.0, DEFECT: 0.0}
    
    def reset(self):
        super().reset()
        self.q_table = {}
        self.lr = self.params.get('initial_lr', self.params.get('lr', 0.1))
        self.epsilon = self.params.get('initial_eps', self.params.get('eps', 0.1))
        self.reward_window = deque(maxlen=self.params.get('reward_window_size', 20))
        self.last_context = None


# --- Legacy Q-Learning Agent (2-round history with trends) ---
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