import random
from collections import defaultdict, deque

# --- Constants ---
COOPERATE = 0
DEFECT = 1

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
        raise NotImplementedError

    def reset(self):
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0

class StaticAgent(BaseAgent):
    """Static strategy agent using the new, robust API."""
    def __init__(self, agent_id, strategy_name, exploration_rate=0.0, **kwargs):
        super().__init__(agent_id, strategy_name)
        self.exploration_rate = exploration_rate
        self.opponent_last_moves = {}

    def choose_action(self, context):
        mode = context['mode']
        if mode == 'pairwise':
            opponent_id = context['opponent_id']
            if self.strategy_name in ["TFT", "TFT-E"]: move = self.opponent_last_moves.get(opponent_id, COOPERATE)
            elif self.strategy_name == "AllC": move = COOPERATE
            elif self.strategy_name == "AllD": move = DEFECT
            else: move = random.choice([COOPERATE, DEFECT])
        else: # Neighborhood
            coop_ratio = context.get('coop_ratio')
            if coop_ratio is None: move = COOPERATE
            else: move = COOPERATE if random.random() < coop_ratio else DEFECT

        if random.random() < self.exploration_rate: return 1 - move
        return move

    def record_outcome(self, context):
        self.total_score += context['reward']
        if context['my_move'] == COOPERATE: self.num_cooperations += 1
        else: self.num_defections += 1
        if context['mode'] == 'pairwise':
            self.opponent_last_moves[context['opponent_id']] = context['opponent_move']

    def reset(self):
        super().reset()
        self.opponent_last_moves.clear()

class EnhancedQLearningAgent(BaseAgent):
    """
    A from-scratch, robust Enhanced Q-Learning agent.
    - Handles pairwise and neighborhood modes with completely separate internal logic.
    - Uses opponent-specific Q-tables and history for pairwise mode.
    - Features epsilon decay and an enhanced state representation.
    """
    def __init__(self, agent_id, lr=0.1, df=0.9, eps_start=0.5, eps_end=0.01, eps_decay=0.995, **kwargs):
        super().__init__(agent_id, "EnhancedQLearning")
        # Learning parameters
        self.lr = lr
        self.df = df
        self.epsilon = eps_start
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        # --- Pairwise-specific state ---
        self.pairwise_q_tables = defaultdict(lambda: defaultdict(lambda: {'cooperate': 0.1, 'defect': 0.1}))
        self.pairwise_history = defaultdict(lambda: deque(maxlen=2)) # Stores (my_move, opp_move) tuples

        # --- Neighborhood-specific state ---
        self.nperson_q_table = defaultdict(lambda: defaultdict(lambda: {'cooperate': 0.1, 'defect': 0.1}))
        self.nperson_history = deque(maxlen=2) # Stores my own moves
        self.nperson_coop_ratio_history = deque(maxlen=2)

    def _get_pairwise_state(self, opponent_id):
        history = self.pairwise_history[opponent_id]
        if len(history) < 1:
            return "start"
        # State is the last round's outcome: (my_move, opponent_move)
        return str(history[-1])

    def _get_nperson_state(self):
        if not self.nperson_coop_ratio_history:
            return "start"

        # State includes my last move and the recent cooperation trend
        my_last_move = 'C' if self.nperson_history[-1] == COOPERATE else 'D'

        trend = 'stable'
        if len(self.nperson_coop_ratio_history) > 1:
            if self.nperson_coop_ratio_history[-1] > self.nperson_coop_ratio_history[-2]: trend = 'up'
            elif self.nperson_coop_ratio_history[-1] < self.nperson_coop_ratio_history[-2]: trend = 'down'

        return f"MyMove_{my_last}_Trend_{trend}"

    def choose_action(self, context):
        mode = context['mode']
        if mode == 'pairwise':
            opponent_id = context['opponent_id']
            q_table = self.pairwise_q_tables[opponent_id]
            state = self._get_pairwise_state(opponent_id)
        else: # Neighborhood
            q_table = self.nperson_q_table
            state = self._get_nperson_state()

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            action = random.choice(['cooperate', 'defect'])
        else:
            action = max(q_table[state], key=q_table[state].get)

        # Store context for learning
        self.last_context = {'mode': mode, 'state': state, 'action': action}
        if mode == 'pairwise':
            self.last_context['opponent_id'] = opponent_id

        return COOPERATE if action == 'cooperate' else DEFECT

    def record_outcome(self, context):
        self.total_score += context['reward']
        my_move = context['my_move']
        if my_move == COOPERATE: self.num_cooperations += 1
        else: self.num_defections += 1

        # Retrieve context from the action choice step
        last_ctx = getattr(self, 'last_context', None)
        if not last_ctx or last_ctx.get('mode') != context['mode']:
            return # Skip learning if context is mismatched

        if context['mode'] == 'pairwise':
            opponent_id = context['opponent_id']
            if last_ctx.get('opponent_id') != opponent_id: return # Ensure we're updating for the correct opponent

            self.pairwise_history[opponent_id].append((my_move, context['opponent_move']))
            next_state = self._get_pairwise_state(opponent_id)
            q_table = self.pairwise_q_tables[opponent_id]

        else: # Neighborhood
            self.nperson_history.append(my_move)
            self.nperson_coop_ratio_history.append(context['coop_ratio'])
            next_state = self._get_nperson_state()
            q_table = self.nperson_q_table

        # Q-learning update rule
        old_q = q_table[last_ctx['state']][last_ctx['action']]
        next_max_q = max(q_table[next_state].values()) if q_table[next_state] else 0
        new_q = old_q + self.lr * (context['reward'] + self.df * next_max_q - old_q)
        q_table[last_ctx['state']][last_ctx['action']] = new_q

        # Clear last context to prevent accidental reuse
        del self.last_context

    def reset(self):
        super().reset()
        self.pairwise_q_tables.clear()
        self.pairwise_history.clear()
        self.nperson_q_table.clear()
        self.nperson_history.clear()
        self.nperson_coop_ratio_history.clear()
        # Apply epsilon decay for the next run
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)