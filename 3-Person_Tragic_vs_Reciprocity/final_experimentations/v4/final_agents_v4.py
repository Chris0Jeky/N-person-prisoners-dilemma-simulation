import random
from collections import defaultdict, deque
import numpy as np

# --- Constants ---
COOPERATE, DEFECT = 0, 1


# --- Base Classes (Simplified for clarity) ---
class BaseAgent:
    def __init__(self, agent_id, strategy_name):
        self.agent_id, self.strategy_name = agent_id, strategy_name
        self.total_score, self.num_cooperations, self.num_defections = 0, 0, 0

    def choose_action(self, context):
        raise NotImplementedError

    def record_outcome(self, context):
        self.total_score += context['reward']
        if context['my_move'] == COOPERATE:
            self.num_cooperations += 1
        else:
            self.num_defections += 1

    def reset(self):
        self.total_score, self.num_cooperations, self.num_defections = 0, 0, 0


class StaticAgent(BaseAgent):
    def __init__(self, agent_id, strategy_name, **kwargs):
        super().__init__(agent_id, strategy_name)
        self.opponent_last_moves = {}

    def choose_action(self, context):
        if context['mode'] == 'pairwise':
            return self.opponent_last_moves.get(context['opponent_id'], COOPERATE)
        coop_ratio = context.get('coop_ratio')
        return COOPERATE if coop_ratio is None or random.random() < coop_ratio else DEFECT

    def record_outcome(self, context):
        super().record_outcome(context)
        if context['mode'] == 'pairwise': self.opponent_last_moves[context['opponent_id']] = context['opponent_move']

    def reset(self):
        super().reset();
        self.opponent_last_moves.clear()


# --- Q-Learning Implementations ---
class VanillaQLearningAgent(BaseAgent):
    def __init__(self, agent_id, lr=0.1, df=0.9, eps=0.1, **kwargs):
        super().__init__(agent_id, "VanillaQLearning")
        self.lr, self.df, self.epsilon = lr, df, eps
        self.q_table = defaultdict(lambda: {'cooperate': 0.0, 'defect': 0.0})
        self.last_contexts = {}

    def _get_state(self, context):
        if context['mode'] == 'pairwise':
            move = context.get('last_opponent_move')
            return 'start' if move is None else 'opp_coop' if move == COOPERATE else 'opp_defect'
        ratio = context.get('coop_ratio')
        if ratio is None: return 'start'
        if ratio <= 0.33: return 'low'
        return 'medium' if ratio <= 0.67 else 'high'

    def _get_action(self, state, epsilon):
        if random.random() < epsilon: return random.choice(['cooperate', 'defect'])
        q_vals = self.q_table[state]
        return 'cooperate' if q_vals['cooperate'] >= q_vals['defect'] else 'defect'

    def _update_q(self, state, action, reward, next_state, lr, df):
        old_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state].values())
        self.q_table[state][action] = old_q + lr * (reward + df * next_max_q - old_q)

    def choose_action(self, context):
        state = self._get_state(context)
        action = self._get_action(state, self.epsilon)
        key = context.get('opponent_id', 'n_person')
        self.last_contexts[key] = {'state': state, 'action': action}
        return COOPERATE if action == 'cooperate' else DEFECT

    def record_outcome(self, context):
        super().record_outcome(context)
        key = context.get('opponent_id', 'n_person')
        if key in self.last_contexts:
            last_ctx = self.last_contexts[key]
            next_state = self._get_state(context)
            self._update_q(last_ctx['state'], last_ctx['action'], context['reward'], next_state, self.lr, self.df)

    def reset(self):
        super().reset();
        self.q_table.clear();
        self.last_contexts.clear()


class TrulyAdaptiveAgent(VanillaQLearningAgent):
    """
    This agent maintains separate adaptive parameters for each opponent,
    allowing it to learn distinct, optimal strategies for each interaction.
    """

    def __init__(self, agent_id, **kwargs):
        super().__init__(agent_id)
        self.strategy_name = "TrulyAdaptive"

        # --- Adaptive parameters stored per-opponent ---
        self.initial_lr, self.min_lr, self.max_lr = 0.1, 0.05, 0.5
        self.initial_eps, self.min_eps, self.max_eps = 0.1, 0.01, 0.4
        self.adapt_factor = 1.1

        self.learning_rates = defaultdict(lambda: self.initial_lr)
        self.epsilons = defaultdict(lambda: self.initial_eps)
        self.reward_windows = defaultdict(lambda: deque(maxlen=20))

    def _adapt(self, key):
        """Adapt parameters for a specific key (opponent_id or 'n_person')."""
        window = self.reward_windows[key]
        if len(window) < window.maxlen: return

        first_half = np.mean(list(window)[:10])
        second_half = np.mean(list(window)[10:])

        if second_half > first_half:  # Performance improving, be conservative
            self.learning_rates[key] = max(self.min_lr, self.learning_rates[key] / self.adapt_factor)
            self.epsilons[key] = max(self.min_eps, self.epsilons[key] / self.adapt_factor)
        else:  # Performance declining, be aggressive
            self.learning_rates[key] = min(self.max_lr, self.learning_rates[key] * self.adapt_factor)
            self.epsilons[key] = min(self.max_eps, self.epsilons[key] * self.adapt_factor)

    def choose_action(self, context):
        key = context.get('opponent_id', 'n_person')
        state = self._get_state(context)
        # Use the specific epsilon for this opponent
        action = self._get_action(state, self.epsilons[key])
        self.last_contexts[key] = {'state': state, 'action': action}
        return COOPERATE if action == 'cooperate' else DEFECT

    def record_outcome(self, context):
        super(VanillaQLearningAgent, self).record_outcome(context)
        key = context.get('opponent_id', 'n_person')

        # Record reward for this specific opponent
        self.reward_windows[key].append(context['reward'])

        if key in self.last_contexts:
            last_ctx = self.last_contexts[key]
            next_state = self._get_state(context)
            # Use the specific learning rate for this opponent
            self._update_q(last_ctx['state'], last_ctx['action'], context['reward'], next_state,
                           self.learning_rates[key], self.df)

        # Adapt parameters for this specific opponent
        self._adapt(key)

    def reset(self):
        super().reset()
        self.learning_rates.clear()
        self.epsilons.clear()
        self.reward_windows.clear()