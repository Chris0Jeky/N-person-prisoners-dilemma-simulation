import random
from collections import deque

# --- Constants ---
COOPERATE = 0
DEFECT = 1

# --- Payoff Matrices ---
PAYOFFS_2P = {
    (COOPERATE, COOPERATE): (3, 3),
    (COOPERATE, DEFECT): (0, 5),
    (DEFECT, COOPERATE): (5, 0),
    (DEFECT, DEFECT): (1, 1),
}

# N-Person Payoff Constants
T, R, P, S = 5, 3, 1, 0

def nperson_payoff(my_move, num_other_cooperators, total_agents):
    """Calculate N-Person payoff based on the linear formula."""
    if total_agents <= 1:
        return R if my_move == COOPERATE else P
    if my_move == COOPERATE:
        return S + (R - S) * (num_other_cooperators / (total_agents - 1))
    else:  # Defect
        return P + (T - P) * (num_other_cooperators / (total_agents - 1))


class BaseAgent:
    """A base class defining the unified API for all agents."""

    def __init__(self, agent_id, strategy_name="Base"):
        self.agent_id = agent_id
        self.strategy_name = strategy_name
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0

    def choose_action(self, **kwargs):
        """Must be implemented by subclasses."""
        raise NotImplementedError

    def record_outcome(self, **kwargs):
        """Must be implemented by subclasses."""
        raise NotImplementedError

    def reset(self):
        """Resets the agent's stats for a new run."""
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0

    def get_cooperation_rate(self):
        total_moves = self.num_cooperations + self.num_defections
        return self.num_cooperations / total_moves if total_moves > 0 else 0.0


class StaticAgent(BaseAgent):
    """Static strategy agent using the unified API."""

    def __init__(self, agent_id, strategy_name, exploration_rate=0.0):
        super().__init__(agent_id, strategy_name)
        self.exploration_rate = exploration_rate if strategy_name == "TFT-E" else 0.0
        self.opponent_last_moves = {}

    def choose_action(self, **kwargs):
        is_pairwise = 'opponent_id' in kwargs
        if is_pairwise:
            opponent_id = kwargs['opponent_id']
            if self.strategy_name in ["TFT", "TFT-E"]:
                intended_move = self.opponent_last_moves.get(opponent_id, COOPERATE)
            elif self.strategy_name == "AllC":
                intended_move = COOPERATE
            elif self.strategy_name == "AllD":
                intended_move = DEFECT
            else:
                intended_move = random.choice([COOPERATE, DEFECT])
        else:  # Neighborhood
            coop_ratio = kwargs.get('prev_round_group_coop_ratio')
            if self.strategy_name in ["TFT", "TFT-E"]:
                if coop_ratio is None:
                    intended_move = COOPERATE
                else:
                    intended_move = COOPERATE if random.random() < coop_ratio else DEFECT
            elif self.strategy_name == "AllC":
                intended_move = COOPERATE
            elif self.strategy_name == "AllD":
                intended_move = DEFECT
            else:  # Random
                intended_move = random.choice([COOPERATE, DEFECT])

        # Apply exploration for TFT-E
        if self.strategy_name == "TFT-E" and random.random() < self.exploration_rate:
            return 1 - intended_move
        return intended_move

    def record_outcome(self, **kwargs):
        self.total_score += kwargs['payoff']
        if kwargs['my_move'] == COOPERATE:
            self.num_cooperations += 1
        else:
            self.num_defections += 1
        if 'opponent_id' in kwargs:
            self.opponent_last_moves[kwargs['opponent_id']] = kwargs['opponent_move']

    def reset(self):
        super().reset()
        self.opponent_last_moves.clear()


class QLearningAgent(BaseAgent):
    """Base class for all Q-learning agents."""

    def __init__(self, agent_id, strategy_name, lr=0.1, df=0.9, eps=0.1):
        super().__init__(agent_id, strategy_name)
        self.lr = lr
        self.df = df
        self.epsilon = eps
        self.q_table = {}

    def _ensure_state_exists(self, state):
        if state not in self.q_table:
            self.q_table[state] = {'cooperate': 0.0, 'defect': 0.0}

    def _get_action(self, state):
        self._ensure_state_exists(state)
        if random.random() < self.epsilon: return random.choice(['cooperate', 'defect'])
        q_vals = self.q_table[state]
        return 'cooperate' if q_vals['cooperate'] >= q_vals['defect'] else 'defect'

    def _update_q(self, state, action, reward, next_state):
        self._ensure_state_exists(state)
        self._ensure_state_exists(next_state)
        old_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state].values())
        new_q = old_q + self.lr * (reward + self.df * next_max_q - old_q)
        self.q_table[state][action] = new_q


class SimpleQLearningAgent(QLearningAgent):
    """The basic Q-learner with simple, correct state management."""

    def __init__(self, agent_id, lr=0.1, df=0.9, eps=0.1):
        super().__init__(agent_id, "SimpleQLearning", lr, df, eps)
        self.opponent_last_moves = {}
        self.last_pairwise_context = {}
        self.last_nperson_context = None

    def choose_action(self, **kwargs):
        is_pairwise = 'opponent_id' in kwargs
        if is_pairwise:
            opponent_id = kwargs['opponent_id']
            last_opp_move = self.opponent_last_moves.get(opponent_id)
            state = 'initial' if last_opp_move is None else ('opp_coop' if last_opp_move == COOPERATE else 'opp_defect')
            action = self._get_action(state)
            self.last_pairwise_context[opponent_id] = {'state': state, 'action': action}
            return COOPERATE if action == 'cooperate' else DEFECT
        else:  # Neighborhood
            coop_ratio = kwargs.get('prev_round_group_coop_ratio')
            if coop_ratio is None:
                state = 'initial'
            elif coop_ratio <= 0.2:
                state = 'very_low'
            elif coop_ratio <= 0.4:
                state = 'low'
            elif coop_ratio <= 0.6:
                state = 'medium'
            elif coop_ratio <= 0.8:
                state = 'high'
            else:
                state = 'very_high'
            action = self._get_action(state)
            self.last_nperson_context = {'state': state, 'action': action}
            return COOPERATE if action == 'cooperate' else DEFECT

    def record_outcome(self, **kwargs):
        self.total_score += kwargs['payoff']
        if kwargs['my_move'] == COOPERATE:
            self.num_cooperations += 1
        else:
            self.num_defections += 1

        is_pairwise = 'opponent_id' in kwargs
        if is_pairwise:
            opponent_id = kwargs['opponent_id']
            context = self.last_pairwise_context.get(opponent_id)
            if context:
                self.opponent_last_moves[opponent_id] = kwargs['opponent_move']
                last_opp_move = self.opponent_last_moves[opponent_id]
                next_state = 'opp_coop' if last_opp_move == COOPERATE else 'opp_defect'
                self._update_q(context['state'], context['action'], kwargs['payoff'], next_state)
        else:  # Neighborhood
            context = self.last_nperson_context
            if context:
                coop_ratio = kwargs.get('prev_round_group_coop_ratio')
                if coop_ratio is None:
                    next_state = 'initial'
                elif coop_ratio <= 0.2:
                    next_state = 'very_low'
                elif coop_ratio <= 0.4:
                    next_state = 'low'
                elif coop_ratio <= 0.6:
                    next_state = 'medium'
                elif coop_ratio <= 0.8:
                    next_state = 'high'
                else:
                    next_state = 'very_high'
                self._update_q(context['state'], context['action'], kwargs['payoff'], next_state)

    def reset(self):
        super().reset()
        self.opponent_last_moves.clear()
        self.last_pairwise_context.clear()
        self.last_nperson_context = None


class EnhancedQLearningAgent(SimpleQLearningAgent):
    """Enhanced Q-Learner inheriting from the simple one to ensure correctness."""

    def __init__(self, agent_id, lr=0.1, df=0.9, eps=0.3, eps_decay=0.999, eps_min=0.01, state_type="basic", mem_len=5):
        super().__init__(agent_id, lr, df, eps)
        self.strategy_name = "EnhancedQLearning"
        self.initial_epsilon = eps
        self.epsilon_decay = eps_decay
        self.epsilon_min = eps_min
        self.state_type = state_type
        self.memory = deque(maxlen=mem_len)

    def choose_action(self, **kwargs):
        # Pairwise logic is inherited directly from SimpleQLearningAgent and is NOT changed.
        if 'opponent_id' in kwargs:
            return super().choose_action(**kwargs)

        # Neighborhood logic is enhanced here.
        coop_ratio = kwargs.get('prev_round_group_coop_ratio')

        # --- State Representation ---
        if coop_ratio is None:
            base_state = 'initial'
        elif self.state_type == 'fine':
            base_state = f"coop_{int(coop_ratio * 10)}"
        else:  # basic or memory_enhanced
            if coop_ratio <= 0.2:
                base_state = 'very_low'
            elif coop_ratio <= 0.4:
                base_state = 'low'
            elif coop_ratio <= 0.6:
                base_state = 'medium'
            elif coop_ratio <= 0.8:
                base_state = 'high'
            else:
                base_state = 'very_high'

        state = base_state
        if self.state_type == 'memory_enhanced':
            coop_count = sum(1 for action in self.memory if action == 'cooperate')
            state = f"{base_state}_mem{coop_count}"

        action = self._get_action(state)
        self.last_nperson_context = {'state': state, 'action': action}
        self.memory.append(action)  # Update memory
        return COOPERATE if action == 'cooperate' else DEFECT

    def reset(self):
        super().reset()
        self.memory.clear()
        # Epsilon decay is applied at the end of a run.
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)