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
            # TFT in neighborhood: cooperate if majority cooperated last round
            if coop_ratio is None:
                intended = COOPERATE
            else:
                intended = COOPERATE if coop_ratio >= 0.5 else DEFECT
        else:  # Default TFT
            if coop_ratio is None:
                intended = COOPERATE
            else:
                intended = COOPERATE if coop_ratio >= 0.5 else DEFECT
        
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
            action = random.choice(['cooperate', 'defect'])
        else:
            action = 'cooperate' if self.neighborhood_q_table[state]['cooperate'] >= self.neighborhood_q_table[state]['defect'] else 'defect'
        
        self.last_neighborhood_context = {'state': state, 'action': action}
        return COOPERATE if action == 'cooperate' else DEFECT
    
    def record_neighborhood_outcome(self, coop_ratio, reward):
        """Record outcome for neighborhood game"""
        self.total_score += reward
        if not hasattr(self, 'last_neighborhood_context') or not self.last_neighborhood_context:
            return
            
        next_state = self._get_neighborhood_state(coop_ratio)
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
        return {'cooperate': 0.0, 'defect': 0.0}
    
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
        return {'cooperate': 0.0, 'defect': 0.0}
    
    def reset(self):
        super().reset()
        self.q_table = defaultdict(self._make_q_dict)
        self.lr = self.params.get('initial_lr', self.params.get('lr', 0.1))
        self.epsilon = self.params.get('initial_eps', self.params.get('eps', 0.1))
        self.reward_window = deque(maxlen=self.params.get('reward_window_size', 20))
        self.last_context = None