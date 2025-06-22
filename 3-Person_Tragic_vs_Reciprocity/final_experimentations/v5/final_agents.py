# final_agents.py
import random
from collections import defaultdict, deque

# --- Constants ---
COOPERATE, DEFECT = 0, 1


# --- Base and Static Agents ---
class BaseAgent:
    """Base class for all agents."""

    def __init__(self, agent_id, strategy_name):
        self.agent_id, self.strategy_name = agent_id, strategy_name
        self.total_score = 0

    def reset(self): self.total_score = 0


class StaticAgent(BaseAgent):
    """Simple agent with a fixed strategy like Tit-for-Tat."""

    def __init__(self, agent_id, strategy_name="TFT", **kwargs):
        super().__init__(agent_id, strategy_name)
        self.opponent_last_moves = {}

    def choose_pairwise_action(self, opponent_id):
        return self.opponent_last_moves.get(opponent_id, COOPERATE)

    def record_pairwise_outcome(self, opponent_id, opponent_move):
        self.opponent_last_moves[opponent_id] = opponent_move

    def reset(self):
        super().reset();
        self.opponent_last_moves.clear()


# --- Specialized Q-Learning Agents ---

class PairwiseAdaptiveQLearner(BaseAgent):
    """
    A Q-learning agent designed exclusively for pairwise interactions.
    It maintains a separate 'brain' (Q-table, adaptive parameters, memory)
    for each opponent it encounters.
    """

    def __init__(self, agent_id, params):
        super().__init__(agent_id, "PairwiseAdaptive")
        self.params = params
        self.reset()

    def _get_state(self, opponent_id):
        """State is based on the last two moves in this specific interaction."""
        history = self.histories[opponent_id]
        if len(history) < 2: return "start"
        # State is a tuple of the last two outcomes: ((my_move, opp_move), (my_move, opp_move))
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

        # Update opponent-specific history and Q-table
        self.histories[opponent_id].append((my_move, opponent_move))
        next_state = self._get_state(opponent_id)

        # Q-learning update
        lr = self.learning_rates[opponent_id]
        q_table = self.q_tables[opponent_id]
        old_q = q_table[context['state']][context['action']]
        next_max_q = max(q_table[next_state].values())
        q_table[context['state']][context['action']] = old_q + lr * (reward + self.params['df'] * next_max_q - old_q)

        # Adaptive learning logic for this specific opponent
        self.reward_windows[opponent_id].append(reward)
        self._adapt_parameters(opponent_id)

    def _adapt_parameters(self, opponent_id):
        """Adapt LR and Epsilon for a single opponent based on performance."""
        window = self.reward_windows[opponent_id]
        if len(window) < self.params['reward_window_size']: return

        half_point = self.params['reward_window_size'] // 2
        first_half_avg = sum(list(window)[:half_point]) / half_point
        second_half_avg = sum(list(window)[half_point:]) / half_point

        # If performance is improving, become more conservative (exploit)
        if second_half_avg > first_half_avg:
            self.learning_rates[opponent_id] = max(self.params['min_lr'],
                                                   self.learning_rates[opponent_id] / self.params['adaptation_factor'])
            self.epsilons[opponent_id] = max(self.params['min_eps'],
                                             self.epsilons[opponent_id] / self.params['adaptation_factor'])
        # If performance is declining, become more aggressive (explore)
        else:
            self.learning_rates[opponent_id] = min(self.params['max_lr'],
                                                   self.learning_rates[opponent_id] * self.params['adaptation_factor'])
            self.epsilons[opponent_id] = min(self.params['max_eps'],
                                             self.epsilons[opponent_id] * self.params['adaptation_factor'])

    def reset(self):
        super().reset()
        self.q_tables = defaultdict(lambda: defaultdict(lambda: {'cooperate': 0.0, 'defect': 0.0}))
        self.histories = defaultdict(lambda: deque(maxlen=2))
        self.reward_windows = defaultdict(lambda: deque(maxlen=self.params.get('reward_window_size', 20)))
        self.learning_rates = defaultdict(lambda: self.params.get('initial_lr', 0.1))
        self.epsilons = defaultdict(lambda: self.params.get('initial_eps', 0.1))
        self.last_contexts = {}


class NeighborhoodAdaptiveQLearner(BaseAgent):
    """A Q-learning agent designed exclusively for neighborhood interactions."""

    def __init__(self, agent_id, params):
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
        self.q_table[self.last_context['state']][self.last_context['action']] = old_q + self.lr * (
                    reward + self.params['df'] * next_max_q - old_q)

        # Adaptive learning logic
        self.reward_window.append(reward)
        self._adapt_parameters()

    def _adapt_parameters(self):
        window = self.reward_window
        if len(window) < self.params['reward_window_size']: return
        half = self.params['reward_window_size'] // 2
        if sum(list(window)[half:]) / half > sum(list(window)[:half]) / half:
            self.lr = max(self.params['min_lr'], self.lr / self.params['adaptation_factor'])
            self.epsilon = max(self.params['min_eps'], self.epsilon / self.params['adaptation_factor'])
        else:
            self.lr = min(self.params['max_lr'], self.lr * self.params['adaptation_factor'])
            self.epsilon = min(self.params['max_eps'], self.epsilon * self.params['adaptation_factor'])

    def reset(self):
        super().reset()
        self.q_table = defaultdict(lambda: {'cooperate': 0.0, 'defect': 0.0})
        self.lr = self.params.get('initial_lr', 0.1)
        self.epsilon = self.params.get('initial_eps', 0.1)
        self.reward_window = deque(maxlen=self.params.get('reward_window_size', 20))
        self.last_context = None