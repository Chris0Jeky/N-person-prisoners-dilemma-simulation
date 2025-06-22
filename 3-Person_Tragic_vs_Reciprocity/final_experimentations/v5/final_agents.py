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
    def __init__(self, agent_id, strategy_name="TFT", **kwargs):
        super().__init__(agent_id, strategy_name)
        self.opponent_last_moves = {}

    def choose_pairwise_action(self, opponent_id):
        return self.opponent_last_moves.get(opponent_id, COOPERATE)

    def record_pairwise_outcome(self, opponent_id, my_move, opponent_move, reward):
        # Accepts my_move and reward to match the API, but only uses opponent_move.
        self.total_score += reward
        self.opponent_last_moves[opponent_id] = opponent_move

    def reset(self):
        super().reset();
        self.opponent_last_moves.clear()


# --- Specialized Q-Learning Agents ---
class PairwiseAdaptiveQLearner(BaseAgent):
    def __init__(self, agent_id, params, **kwargs):
        super().__init__(agent_id, "PairwiseAdaptive")
        self.params = params
        self.reset()

    def _get_state(self, opponent_id):
        history = self.histories[opponent_id]
        if len(history) < 2: return "start"
        return str(tuple(history))

    def choose_pairwise_action(self, opponent_id):
        state = self._get_state(opponent_id)
        q_table = self.q_tables[opponent_id]
        epsilon = self.epsilons[opponent_id]
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
        self.histories[opponent_id].append((my_move, opponent_move))
        next_state = self._get_state(opponent_id)
        lr = self.learning_rates[opponent_id]
        q_table = self.q_tables[opponent_id]
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

    def reset(self):
        super().reset()
        self.q_tables = defaultdict(lambda: defaultdict(lambda: {'cooperate': 0.0, 'defect': 0.0}))
        self.histories = defaultdict(lambda: deque(maxlen=2))
        self.reward_windows = defaultdict(lambda: deque(maxlen=self.params.get('reward_window_size', 20)))
        self.learning_rates = defaultdict(lambda: self.params.get('initial_lr', self.params.get('lr', 0.1)))
        self.epsilons = defaultdict(lambda: self.params.get('initial_eps', self.params.get('eps', 0.1)))
        self.last_contexts = {}


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

    def reset(self):
        super().reset()
        self.q_table = defaultdict(lambda: {'cooperate': 0.0, 'defect': 0.0})
        self.lr = self.params.get('initial_lr', self.params.get('lr', 0.1))
        self.epsilon = self.params.get('initial_eps', self.params.get('eps', 0.1))
        self.reward_window = deque(maxlen=self.params.get('reward_window_size', 20))
        self.last_context = None