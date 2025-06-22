import random
from collections import defaultdict, deque
import numpy as np

# --- Constants ---
COOPERATE = 0
DEFECT = 1


# --- Base Classes ---
class BaseAgent:
    """Base class defining the unified API for all agents."""

    def __init__(self, agent_id, strategy_name="Base"):
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


class QLearningAgentBase(BaseAgent):
    """Base for Q-learners, handling Q-table and basic update logic."""

    def __init__(self, agent_id, strategy_name, lr, df, eps):
        super().__init__(agent_id, strategy_name)
        self.lr, self.df, self.epsilon = lr, df, eps
        self.q_table = defaultdict(lambda: {'cooperate': 0.0, 'defect': 0.0})

    def _get_action_from_q(self, state):
        if random.random() < self.epsilon: return random.choice(['cooperate', 'defect'])
        q = self.q_table[state]
        return 'cooperate' if q['cooperate'] >= q['defect'] else 'defect'

    def _update_q_value(self, state, action, reward, next_state):
        old_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state].values())
        self.q_table[state][action] = old_q + self.lr * (reward + self.df * next_max_q - old_q)


# --- Agent Implementations ---
class StaticAgent(BaseAgent):
    """Static strategy agent (AllC, AllD, TFT, etc.)."""

    def __init__(self, agent_id, strategy_name, exploration_rate=0.0, **kwargs):
        super().__init__(agent_id, strategy_name)
        self.exploration_rate = exploration_rate
        self.opponent_last_moves = {}

    def choose_action(self, context):
        mode = context['mode']
        if mode == 'pairwise':
            opponent_id = context['opponent_id']
            if self.strategy_name == "TFT":
                move = self.opponent_last_moves.get(opponent_id, COOPERATE)
            elif self.strategy_name == "AllC":
                move = COOPERATE
            elif self.strategy_name == "AllD":
                move = DEFECT
            else:
                move = random.choice([COOPERATE, DEFECT])
        else:
            coop_ratio = context.get('coop_ratio')
            if coop_ratio is None:
                move = COOPERATE
            else:
                move = COOPERATE if random.random() < coop_ratio else DEFECT
        if random.random() < self.exploration_rate: return 1 - move
        return move

    def record_outcome(self, context):
        super().record_outcome(context)
        if context['mode'] == 'pairwise': self.opponent_last_moves[context['opponent_id']] = context['opponent_move']

    def reset(self):
        super().reset()
        self.opponent_last_moves.clear()


class VanillaQLearningAgent(QLearningAgentBase):
    """The baseline Q-learner with simple, effective state management."""

    def __init__(self, agent_id, lr=0.1, df=0.9, eps=0.1, **kwargs):
        super().__init__(agent_id, "VanillaQLearning", lr, df, eps)
        self.last_contexts = {}

    def _get_state(self, context):
        if context['mode'] == 'pairwise':
            last_opp_move = context.get('last_opponent_move')
            return 'start' if last_opp_move is None else ('opp_coop' if last_opp_move == COOPERATE else 'opp_defect')
        else:  # Neighborhood
            coop_ratio = context.get('coop_ratio')
            if coop_ratio is None: return 'start'
            if coop_ratio <= 0.33: return 'low'
            if coop_ratio <= 0.67: return 'medium'
            return 'high'

    def choose_action(self, context):
        state = self._get_state(context)
        action = self._get_action_from_q(state)
        context_key = context['opponent_id'] if context['mode'] == 'pairwise' else 'n_person'
        self.last_contexts[context_key] = {'state': state, 'action': action}
        return COOPERATE if action == 'cooperate' else DEFECT

    def record_outcome(self, context):
        super().record_outcome(context)
        context_key = context['opponent_id'] if context['mode'] == 'pairwise' else 'n_person'
        last_ctx = self.last_contexts.get(context_key)
        if last_ctx:
            next_state = self._get_state(context)
            self._update_q_value(last_ctx['state'], last_ctx['action'], context['reward'], next_state)

    def reset(self):
        super().reset()
        self.q_table.clear()
        self.last_contexts.clear()


class AdaptiveAgent(VanillaQLearningAgent):
    """
    An agent that adaptively adjusts its learning rate and epsilon
    based on performance to escape bad cycles and exploit good ones.
    """

    def __init__(self, agent_id, **kwargs):
        super().__init__(agent_id)  # Start with simple QL defaults
        self.strategy_name = "AdaptiveAgent"

        # Adaptive parameters
        self.min_lr, self.max_lr = 0.05, 0.15
        self.min_eps, self.max_eps = 0.01, 0.1
        self.reward_window = deque(maxlen=20)
        self.lr_adaptation_factor = 1.05  # How quickly to change lr/eps
        self.eps_adaptation_factor = 1.02

    def adapt_parameters(self):
        if len(self.reward_window) < self.reward_window.maxlen:
            return  # Don't adapt until the window is full

        # Check for performance trend
        first_half_avg = np.mean(list(self.reward_window)[:10])
        second_half_avg = np.mean(list(self.reward_window)[10:])

        if second_half_avg > first_half_avg + 0.05:  # Performance improving
            # Become more conservative
            self.lr = max(self.min_lr, self.lr / self.lr_adaptation_factor)
            self.epsilon = max(self.min_eps, self.epsilon / self.eps_adaptation_factor)
        else:  # Performance stagnant or declining
            # Get more aggressive
            self.lr = min(self.max_lr, self.lr * self.lr_adaptation_factor)
            self.epsilon = min(self.max_eps, self.epsilon * self.eps_adaptation_factor)

    def record_outcome(self, context):
        super().record_outcome(context)
        # Record reward and adapt
        self.reward_window.append(context['reward'])
        self.adapt_parameters()

    def reset(self):
        super().reset()
        self.reward_window.clear()
        # Reset to default parameters for the next run
        self.lr = 0.1
        self.epsilon = 0.1