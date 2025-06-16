"""
Enhanced Q-Learning Agent with Advanced Features

Implements improvements to address Q-learning exploitation issues:
- State representation excluding self
- Decaying epsilon
- Opponent modeling
- Extended training support
"""

from typing import Tuple, Optional, Dict, Any
import random
from collections import defaultdict, deque
from ..base.agent import NPDAgent, PairwiseAgent


class EnhancedQLearningAgent(NPDAgent, PairwiseAgent):
    """
    Enhanced Q-Learning with configurable improvements.
    
    Key improvements:
    1. Exclude self from state calculation (solves state aliasing)
    2. Epsilon decay for better exploitation
    3. Opponent modeling for adaptive play
    4. Support for extended training periods
    """
    
    def __init__(self,
                 agent_id: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 1.0,
                 epsilon_min: float = 0.01,
                 exploration_rate: float = 0.0,
                 exclude_self: bool = False,
                 opponent_modeling: bool = False,
                 state_type: str = "basic",
                 optimistic_init: float = 0.1):
        """
        Initialize enhanced Q-Learning agent.
        
        Args:
            agent_id: Unique identifier
            learning_rate: Q-learning alpha parameter
            discount_factor: Q-learning gamma parameter
            epsilon: Initial epsilon for epsilon-greedy
            epsilon_decay: Multiplicative decay factor per episode
            epsilon_min: Minimum epsilon value
            exploration_rate: Additional random exploration
            exclude_self: Whether to exclude self from state representation
            opponent_modeling: Whether to model opponent behavior
            state_type: State discretization ("basic", "fine", "coarse")
            optimistic_init: Initial Q-value for optimistic initialization
        """
        # Initialize both parent classes
        NPDAgent.__init__(self, agent_id, exploration_rate)
        PairwiseAgent.__init__(self, agent_id, exploration_rate)
        
        self.name = "Enhanced Q-Learning"
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.exclude_self = exclude_self
        self.opponent_modeling = opponent_modeling
        self.state_type = state_type
        self.optimistic_init = optimistic_init
        
        # Q-table with optimistic initialization
        self.q_table = defaultdict(lambda: {
            'cooperate': optimistic_init,
            'defect': optimistic_init
        })
        
        # Episode and step counters
        self.episode_count = 0
        self.step_count = 0
        
        # Memory for learning
        self.last_state = None
        self.last_action = None
        self.last_my_action = None  # For exclude_self feature
        
        # Opponent modeling
        if opponent_modeling:
            self.opponent_history = defaultdict(lambda: {'cooperate': 0, 'defect': 0})
            self.opponent_models = {}  # opponent_id -> predicted cooperation rate
        
        # Adaptive learning rate
        self.use_adaptive_lr = True
        self.state_visits = defaultdict(int)
        
    def start_new_episode(self):
        """Called at the start of a new episode to apply decay."""
        self.episode_count += 1
        self.decay_epsilon()
    
    def decay_epsilon(self):
        """Apply epsilon decay."""
        if self.epsilon_decay < 1.0:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _get_npd_state(self, 
                      cooperation_ratio: Optional[float],
                      my_last_action: Optional[int] = None) -> str:
        """
        Get state for N-Person game with optional self-exclusion.
        """
        if cooperation_ratio is None:
            return 'initial'
        
        # Calculate cooperation ratio excluding self if enabled
        if self.exclude_self and my_last_action is not None:
            # Assuming we know total number of agents (will be passed from game)
            num_agents = getattr(self, '_num_agents', 3)  # Default to 3
            total_coops = cooperation_ratio * num_agents
            
            # Remove self contribution
            if my_last_action == 0:  # I cooperated
                total_coops -= 1
            
            # Calculate others' cooperation ratio
            others_coop_ratio = total_coops / (num_agents - 1) if num_agents > 1 else 0
            others_coop_ratio = max(0, min(1, others_coop_ratio))  # Clamp to [0, 1]
        else:
            others_coop_ratio = cooperation_ratio
        
        # Discretize based on state_type
        if self.state_type == "basic":
            if others_coop_ratio <= 0.2:
                base_state = 'very_low'
            elif others_coop_ratio <= 0.4:
                base_state = 'low'
            elif others_coop_ratio <= 0.6:
                base_state = 'medium'
            elif others_coop_ratio <= 0.8:
                base_state = 'high'
            else:
                base_state = 'very_high'
        elif self.state_type == "fine":
            # 10% increments
            base_state = f"coop_{int(others_coop_ratio * 10) / 10:.1f}"
        elif self.state_type == "coarse":
            # Three levels
            if others_coop_ratio <= 0.33:
                base_state = 'low'
            elif others_coop_ratio <= 0.67:
                base_state = 'medium'
            else:
                base_state = 'high'
        else:
            base_state = f"coop_{others_coop_ratio:.2f}"
        
        # Add opponent model information if enabled
        if self.opponent_modeling and hasattr(self, 'opponent_models') and self.opponent_models:
            avg_pred = sum(self.opponent_models.values()) / len(self.opponent_models)
            model_suffix = f"_pred_{int(avg_pred * 10) / 10:.1f}"
            return base_state + model_suffix
        
        return base_state
    
    def _update_opponent_model(self, agent_actions: Dict[int, int]):
        """Update opponent models based on observed actions."""
        if not self.opponent_modeling:
            return
        
        for agent_id, action in agent_actions.items():
            if agent_id == self.agent_id:
                continue
                
            # Update history
            if action == 0:  # Cooperate
                self.opponent_history[agent_id]['cooperate'] += 1
            else:
                self.opponent_history[agent_id]['defect'] += 1
            
            # Update model (simple frequency-based)
            total = (self.opponent_history[agent_id]['cooperate'] + 
                    self.opponent_history[agent_id]['defect'])
            if total > 0:
                self.opponent_models[agent_id] = (
                    self.opponent_history[agent_id]['cooperate'] / total
                )
    
    def _get_adaptive_learning_rate(self, state: str) -> float:
        """Get adaptive learning rate based on state visits."""
        if not self.use_adaptive_lr:
            return self.learning_rate
        
        # Decay learning rate with visits
        visits = self.state_visits[state]
        return self.learning_rate / (1 + visits * 0.01)
    
    def _epsilon_greedy_action(self, state: str) -> str:
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(['cooperate', 'defect'])
        
        # Choose best action
        q_values = self.q_table[state]
        if q_values['cooperate'] > q_values['defect']:
            return 'cooperate'
        elif q_values['defect'] > q_values['cooperate']:
            return 'defect'
        else:
            return random.choice(['cooperate', 'defect'])
    
    def _update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-value with adaptive learning rate."""
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        
        # Get adaptive learning rate
        lr = self._get_adaptive_learning_rate(state)
        
        # Q-learning update
        new_q = current_q + lr * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state][action] = new_q
        
        # Update visit count
        self.state_visits[state] += 1
        self.step_count += 1
    
    def set_num_agents(self, num_agents: int):
        """Set the number of agents for state calculation."""
        self._num_agents = num_agents
    
    def choose_action(self, *args, **kwargs) -> Tuple[int, int]:
        """Choose action with game mode detection."""
        # Detect game mode
        if len(args) >= 2 and isinstance(args[1], int):
            # Pairwise mode
            return self._choose_action_pairwise(args[0], args[1])
        else:
            # N-Person mode
            cooperation_ratio = args[0] if args else kwargs.get('cooperation_ratio')
            round_number = args[1] if len(args) > 1 else kwargs.get('round_number', 0)
            return self._choose_action_npd(cooperation_ratio, round_number)
    
    def _choose_action_npd(self,
                          cooperation_ratio: Optional[float],
                          round_number: int) -> Tuple[int, int]:
        """Choose action for N-Person game."""
        # Get current state
        state = self._get_npd_state(cooperation_ratio, self.last_my_action)
        
        # Update Q-value from last round
        if self.last_state is not None and self.last_action is not None:
            # Calculate reward from score change
            last_payoff = self.total_score - getattr(self, '_last_total_score', 0)
            self._update_q_value(self.last_state, self.last_action, last_payoff, state)
        
        # Choose action
        action_str = self._epsilon_greedy_action(state)
        intended_action = 0 if action_str == 'cooperate' else 1
        
        # Store for next update
        self.last_state = state
        self.last_action = action_str
        self.last_my_action = intended_action
        self._last_total_score = self.total_score
        
        # Apply exploration
        actual_action = self.apply_exploration(intended_action)
        return intended_action, actual_action
    
    def _choose_action_pairwise(self,
                               opponent_id: int,
                               round_in_episode: int) -> Tuple[int, int]:
        """Choose action for pairwise game."""
        # Get state based on opponent's last move
        if opponent_id not in self.opponent_last_moves:
            state = 'initial'
        else:
            last_move = self.opponent_last_moves[opponent_id]
            state = 'opp_cooperated' if last_move == 0 else 'opp_defected'
        
        # Add opponent model if enabled
        if self.opponent_modeling and opponent_id in self.opponent_models:
            pred_coop = self.opponent_models[opponent_id]
            state += f"_pred_{int(pred_coop * 10) / 10:.1f}"
        
        # Choose action
        action_str = self._epsilon_greedy_action(state)
        intended_action = 0 if action_str == 'cooperate' else 1
        
        # Store for update
        if not hasattr(self, '_pairwise_memory'):
            self._pairwise_memory = {}
        self._pairwise_memory[opponent_id] = (state, action_str)
        
        # Apply exploration
        actual_action = self.apply_exploration(intended_action)
        return intended_action, actual_action
    
    def update_from_game_state(self, agent_actions: Dict[int, int]):
        """Update opponent models from observed actions."""
        if self.opponent_modeling:
            self._update_opponent_model(agent_actions)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get extended statistics including Q-learning info."""
        stats = super().get_stats()
        stats.update({
            'epsilon': self.epsilon,
            'episodes': self.episode_count,
            'steps': self.step_count,
            'q_table_size': len(self.q_table),
            'exclude_self': self.exclude_self,
            'opponent_modeling': self.opponent_modeling
        })
        return stats
    
    def get_q_table_summary(self) -> Dict:
        """Get Q-table summary for analysis."""
        summary = {}
        for state, actions in self.q_table.items():
            summary[state] = {
                'cooperate_q': actions['cooperate'],
                'defect_q': actions['defect'],
                'best_action': 'cooperate' if actions['cooperate'] >= actions['defect'] else 'defect',
                'visits': self.state_visits.get(state, 0)
            }
        return summary