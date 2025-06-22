import random
from collections import defaultdict, deque

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
    """Base class for all Q-learning agents, handling Q-table logic."""

    def __init__(self, agent_id, strategy_name, lr=0.1, df=0.9, eps=0.1):
        super().__init__(agent_id, strategy_name)
        self.lr = lr
        self.df = df
        self.epsilon = eps
        self.q_table = defaultdict(lambda: defaultdict(float))

    def _get_action_from_q(self, state):
        if random.random() < self.epsilon:
            return random.choice(['cooperate', 'defect'])
        # Use .get() to avoid KeyError on first encounter
        coop_q = self.q_table[state].get('cooperate', 0.0)
        defect_q = self.q_table[state].get('defect', 0.0)
        return 'cooperate' if coop_q >= defect_q else 'defect'

    def _update_q_value(self, state, action, reward, next_state):
        old_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0
        new_q = old_q + self.lr * (reward + self.df * next_max_q - old_q)
        self.q_table[state][action] = new_q


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
        else:  # Neighborhood
            coop_ratio = context.get('coop_ratio')
            if coop_ratio is None:
                move = COOPERATE
            else:
                move = COOPERATE if random.random() < coop_ratio else DEFECT
        if random.random() < self.exploration_rate: return 1 - move
        return move

    def record_outcome(self, context):
        super().record_outcome(context)
        if context['mode'] == 'pairwise':
            self.opponent_last_moves[context['opponent_id']] = context['opponent_move']

    def reset(self):
        super().reset()
        self.opponent_last_moves.clear()


class SimpleQLearningAgent(QLearningAgentBase):
    """The basic Q-learner with simple state management."""

    def __init__(self, agent_id, lr=0.1, df=0.9, eps=0.1, **kwargs):
        super().__init__(agent_id, "SimpleQLearning", lr, df, eps)
        self.last_contexts = {}  # opponent_id -> context or 'n_person' -> context

    def _get_state(self, context):
        if context['mode'] == 'pairwise':
            last_opp_move = context.get('last_opponent_move')
            return 'start' if last_opp_move is None else ('opp_coop' if last_opp_move == COOPERATE else 'opp_defect')
        else:  # Neighborhood
            coop_ratio = context.get('coop_ratio')
            if coop_ratio is None: return 'start'
            if coop_ratio <= 0.2: return 'very_low'
            if coop_ratio <= 0.4: return 'low'
            if coop_ratio <= 0.6: return 'medium'
            if coop_ratio <= 0.8: return 'high'
            return 'very_high'

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
        self.last_contexts.clear()


class EnhancedQLearningAgent(SimpleQLearningAgent):
    """Enhanced agent inheriting from SimpleQLearningAgent for robustness."""

    def __init__(self, agent_id, lr=0.1, df=0.9, eps_start=0.5, eps_end=0.01, eps_decay=0.995, state_type="enhanced",
                 **kwargs):
        super().__init__(agent_id, lr, df, eps_start)
        self.strategy_name = "EnhancedQLearning"
        self.initial_epsilon = self.epsilon
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.state_type = state_type
        self.memory = deque(maxlen=2)

    def _get_state(self, context):
        # Let parent handle pairwise state, as it is already optimal.
        if context['mode'] == 'pairwise':
            return super()._get_state(context)

        # Override for enhanced neighborhood state.
        coop_ratio = context.get('coop_ratio')
        if coop_ratio is None: return "start"

        my_hist_str = "".join(['C' if m == COOPERATE else 'D' for m in self.memory]) if self.memory else "Start"
        ratio_str = f"{coop_ratio:.1f}"
        return f"MyHist_{my_hist_str}_Ratio_{ratio_str}"

    def record_outcome(self, context):
        # Let parent handle Q-update logic.
        super().record_outcome(context)
        # Add own move to memory for next state calculation.
        if context['mode'] == 'neighborhood':
            self.memory.append(context['my_move'])

    def reset(self):
        super().reset()
        self.memory.clear()
        # Apply epsilon decay at the end of a run.
        self.epsilon = max(self.eps_end, self.initial_epsilon * (self.eps_decay ** self.num_cooperations))