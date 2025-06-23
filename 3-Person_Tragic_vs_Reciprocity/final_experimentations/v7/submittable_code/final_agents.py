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
            self.q_tables[opponent_id] = defaultdict(lambda: defaultdict(float))
        
        # Epsilon-greedy
        if random.random() < self._get_epsilon():
            return random.choice([COOPERATE, DEFECT])
        else:
            q_values = self.q_tables[opponent_id][state]
            if q_values[COOPERATE] == q_values[DEFECT]:
                return random.choice([COOPERATE, DEFECT])
            return max(q_values, key=q_values.get)

    def record_pairwise_outcome(self, opponent_id, my_move, opponent_move, reward):
        self.total_score += reward
        state = self._get_state(opponent_id)
        
        if opponent_id not in self.histories:
            self.histories[opponent_id] = self._make_history_deque()
        self.histories[opponent_id].append((my_move, opponent_move))
        
        next_state = self._get_state(opponent_id)
        self._update_q_value(opponent_id, state, my_move, reward, next_state)
        self._adapt_parameters(reward)

    def choose_neighborhood_action(self, coop_ratio):
        # First, we need to get the current state (before this round's ratio is added)
        state = self._get_neighborhood_state()
        
        # Epsilon-greedy
        if random.random() < self._get_epsilon():
            return random.choice([COOPERATE, DEFECT])
        else:
            q_values = self.n_q_table[state]
            if q_values[COOPERATE] == q_values[DEFECT]:
                return random.choice([COOPERATE, DEFECT])
            return max(q_values, key=q_values.get)

    def record_neighborhood_outcome(self, coop_ratio, reward):
        self.total_score += reward
        
        # Get current state before updating history
        state = self._get_neighborhood_state()
        
        # Update cooperation ratio history with category
        if coop_ratio is not None:
            category = self._categorize_coop_ratio(coop_ratio)
            self.coop_ratio_history.append(category)
        
        # Get next state after updating history
        next_state = self._get_neighborhood_state()
        
        # Determine my last move based on current cooperation ratio vs previous
        if self.last_coop_ratio is None:
            my_last_move = COOPERATE  # Assume cooperation on first round
        else:
            # Infer action from change in cooperation ratio
            my_last_move = COOPERATE if coop_ratio > self.last_coop_ratio else DEFECT
        
        self._update_neighborhood_q_value(state, my_last_move, reward, next_state)
        self._adapt_parameters(reward)
        self.last_coop_ratio = coop_ratio

    def _categorize_coop_ratio(self, coop_ratio):
        """Convert cooperation ratio to category"""
        if coop_ratio < 0.33:
            return "low"
        elif coop_ratio < 0.67:
            return "medium"
        else:
            return "high"

    def _get_neighborhood_state(self):
        """Get state as tuple of last 4 cooperation ratio categories"""
        if len(self.coop_ratio_history) == 0:
            return "start"
        elif len(self.coop_ratio_history) < 4:
            # Pad with 'start' if we don't have 4 rounds yet
            padding = ['start'] * (4 - len(self.coop_ratio_history))
            return str(tuple(padding + list(self.coop_ratio_history)))
        else:
            # Return tuple of last 4 categories
            return str(tuple(self.coop_ratio_history))

    def _update_q_value(self, opponent_id, state, action, reward, next_state):
        lr = self._get_learning_rate()
        df = self.params['df']
        
        current_q = self.q_tables[opponent_id][state][action]
        max_next_q = max(self.q_tables[opponent_id][next_state].values()) if self.q_tables[opponent_id][next_state] else 0
        new_q = current_q + lr * (reward + df * max_next_q - current_q)
        self.q_tables[opponent_id][state][action] = new_q

    def _update_neighborhood_q_value(self, state, action, reward, next_state):
        lr = self._get_learning_rate()
        df = self.params['df']
        
        current_q = self.n_q_table[state][action]
        max_next_q = max(self.n_q_table[next_state].values()) if self.n_q_table[next_state] else 0
        new_q = current_q + lr * (reward + df * max_next_q - current_q)
        self.n_q_table[state][action] = new_q

    def _get_learning_rate(self):
        if 'lr' in self.params:
            return self.params['lr']
        else:
            return self.current_lr

    def _get_epsilon(self):
        if 'eps' in self.params:
            return self.params['eps']
        else:
            return self.current_eps

    def _adapt_parameters(self, reward):
        # Only adapt if we have adaptive parameters
        if 'initial_lr' not in self.params:
            return
            
        self.reward_history.append(reward)
        
        window_size = self.params.get('reward_window_size', 75)
        if len(self.reward_history) < window_size:
            return
        
        recent_rewards = list(self.reward_history)[-window_size:]
        avg_reward = np.mean(recent_rewards)
        
        # Check performance trend
        mid_point = window_size // 2
        first_half_avg = np.mean(recent_rewards[:mid_point])
        second_half_avg = np.mean(recent_rewards[mid_point:])
        
        factor = self.params.get('adaptation_factor', 1.08)
        
        # Adapt learning rate
        if second_half_avg > first_half_avg:
            # Performance improving, can reduce learning rate
            self.current_lr = max(self.params['min_lr'], self.current_lr / factor)
        else:
            # Performance declining, increase learning rate
            self.current_lr = min(self.params['max_lr'], self.current_lr * factor)
        
        # Adapt exploration rate
        if avg_reward > 2.5:  # Good performance
            self.current_eps = max(self.params['min_eps'], self.current_eps / factor)
        else:
            self.current_eps = min(self.params['max_eps'], self.current_eps * factor)

    def _make_history_deque(self):
        return deque(maxlen=2)

    def reset(self):
        super().reset()
        self.q_tables = {}
        self.n_q_table = defaultdict(lambda: defaultdict(float))
        self.histories = {}
        self.last_coop_ratio = None
        self.coop_ratio_history = deque(maxlen=4)  # Store last 4 cooperation ratio categories
        self.reward_history = deque(maxlen=1000)
        
        # Set initial parameters
        if 'initial_lr' in self.params:
            self.current_lr = self.params['initial_lr']
            self.current_eps = self.params['initial_eps']
        else:
            self.current_lr = self.params.get('lr', 0.1)
            self.current_eps = self.params.get('eps', 0.1)