import random
from collections import defaultdict, deque
import numpy as np  

# --- Constants ---
COOPERATE = 0
DEFECT = 1


# --- Base Classes ---
class BaseAgent:
    """A base class defining the unified API for all agents."""

    def __init__(self, agent_id, strategy_name="Base"):
        self.agent_id = agent_id
        self.strategy_name = strategy_name
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0

    def choose_action(self, context):
        raise NotImplementedError

    def record_outcome(self, context):
        self.total_score += context['reward']
        if context['my_move'] == COOPERATE:
            self.num_cooperations += 1
        else:
            self.num_defections += 1

    def reset(self):
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0


class QLearningAgentBase(BaseAgent):
    """Base for Q-learners, handling Q-table logic."""

    def __init__(self, agent_id, strategy_name, lr=0.1, df=0.9, eps=0.1):
        super().__init__(agent_id, strategy_name)
        self.lr, self.df, self.epsilon = lr, df, eps
        self.q_table = defaultdict(lambda: {'cooperate': 0.0, 'defect': 0.0})

    def _get_action_from_q(self, state):
        if random.random() < self.epsilon: return random.choice(['cooperate', 'defect'])
        q_vals = self.q_table[state]
        return 'cooperate' if q_vals['cooperate'] >= q_vals['defect'] else 'defect'

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
            if self.strategy_name in ["TFT", "TFT-E"]:
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


class SimpleQLearningAgent(QLearningAgentBase):
    """The basic Q-learner with simple, effective state management."""

    def __init__(self, agent_id, lr=0.1, df=0.9, eps=0.1, **kwargs):
        super().__init__(agent_id, "SimpleQLearning", lr, df, eps)
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


class VanillaEnhancedAgent(SimpleQLearningAgent):
    """
    This agent uses the exact same logic and parameters as SimpleQLearningAgent.
    It serves as a control group to prove the architecture is sound.
    Its performance should be identical to SimpleQLearningAgent.
    """

    def __init__(self, agent_id, **kwargs):
        # Force identical parameters
        super().__init__(agent_id, lr=0.1, df=0.9, eps=0.1)
        self.strategy_name = "VanillaEnhanced"


class BetterEnhancedAgent(SimpleQLearningAgent):
    """
    An agent with carefully selected, effective enhancements.
    - Epsilon decay allows for exploration then exploitation.
    - State representation is slightly more informative but not overly complex.
    """

    def __init__(self, agent_id, lr=0.1, df=0.95, eps_start=0.3, eps_end=0.01, eps_decay_rate=500, **kwargs):
        super().__init__(agent_id, lr, df, eps_start)
        self.strategy_name = "BetterEnhanced"
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_rate = eps_decay_rate
        self.step = 0

    def _get_state(self, context):
        # Pairwise state remains simple and effective
        if context['mode'] == 'pairwise':
            return super()._get_state(context)

        # Neighborhood state adds one piece of information: my last move
        coop_ratio = context.get('coop_ratio')
        last_move = context.get('my_last_move')  # Need to pass this in context

        if coop_ratio is None:
            base_state = 'start'
        elif coop_ratio <= 0.33:
            base_state = 'low'
        elif coop_ratio <= 0.67:
            base_state = 'medium'
        else:
            base_state = 'high'

        move_state = 'start' if last_move is None else ('C' if last_move == COOPERATE else 'D')
        return f"{base_state}_My_{move_state}"

    def choose_action(self, context):
        # Epsilon decay based on steps taken
        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * \
                       np.exp(-1. * self.step / self.eps_decay_rate)
        self.step += 1
        return super().choose_action(context)

    def reset(self):
        super().reset()
        self.step = 0
        self.epsilon = self.eps_start