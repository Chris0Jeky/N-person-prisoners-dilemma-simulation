"""
Q-Learning Agent for N-Person Prisoner's Dilemma

Supports both N-Person and Pairwise game modes.
"""

from typing import Tuple, Optional, Dict
import random
from collections import defaultdict
from ..base.agent import NPDAgent, PairwiseAgent


class QLearningAgent(NPDAgent, PairwiseAgent):
    """
    Q-Learning agent that can play in both N-Person and Pairwise modes.
    
    Features:
    - Epsilon-greedy exploration
    - Configurable state representations
    - Support for both game modes
    """
    
    def __init__(self,
                 agent_id: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9,
                 epsilon: float = 0.1,
                 exploration_rate: float = 0.0,
                 state_type: str = "basic"):
        """
        Initialize Q-Learning agent.
        
        Args:
            agent_id: Unique identifier
            learning_rate: Q-learning alpha parameter
            discount_factor: Q-learning gamma parameter
            epsilon: Epsilon for epsilon-greedy exploration
            exploration_rate: Additional random exploration (legacy compatibility)
            state_type: Type of state representation ("basic", "detailed")
        """
        # Initialize both parent classes
        NPDAgent.__init__(self, agent_id, exploration_rate)
        PairwiseAgent.__init__(self, agent_id, exploration_rate)
        
        self.name = "Q-Learning"
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.state_type = state_type
        
        # Q-table
        self.q_table = defaultdict(lambda: {
            'cooperate': random.uniform(-0.01, 0.01),
            'defect': random.uniform(-0.01, 0.01)
        })
        
        # Memory for learning
        self.last_state = None
        self.last_action = None
        
        # Game mode flag
        self.is_pairwise_mode = False
        
    def _get_npd_state(self, cooperation_ratio: Optional[float]) -> str:
        """Get state for N-Person game."""
        if cooperation_ratio is None:
            return 'initial'
        
        if self.state_type == "basic":
            # Simple discretization
            if cooperation_ratio <= 0.33:
                return 'low_coop'
            elif cooperation_ratio <= 0.67:
                return 'med_coop'
            else:
                return 'high_coop'
        else:  # detailed
            # Finer discretization
            bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            labels = ['very_low', 'low', 'medium', 'high', 'very_high']
            for i in range(len(bins) - 1):
                if cooperation_ratio <= bins[i + 1]:
                    return labels[i]
            return labels[-1]
    
    def _get_pairwise_state(self, opponent_id: int) -> str:
        """Get state for pairwise game."""
        if opponent_id not in self.opponent_last_moves:
            return 'initial'
        
        last_move = self.opponent_last_moves[opponent_id]
        return 'opp_cooperated' if last_move == 0 else 'opp_defected'
    
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
        """Update Q-value using Q-learning update rule."""
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state][action] = new_q
    
    def choose_action(self, *args, **kwargs) -> Tuple[int, int]:
        """
        Choose action based on Q-learning policy.
        
        Detects game mode based on arguments.
        """
        # Detect game mode based on arguments
        if len(args) >= 2 and isinstance(args[1], int):
            # Pairwise mode: (opponent_id, round_in_episode)
            self.is_pairwise_mode = True
            return self._choose_action_pairwise(args[0], args[1])
        else:
            # N-Person mode: (cooperation_ratio, round_number)
            self.is_pairwise_mode = False
            cooperation_ratio = args[0] if args else kwargs.get('cooperation_ratio')
            round_number = args[1] if len(args) > 1 else kwargs.get('round_number', 0)
            return self._choose_action_npd(cooperation_ratio, round_number)
    
    def _choose_action_npd(self, 
                          cooperation_ratio: Optional[float],
                          round_number: int) -> Tuple[int, int]:
        """Choose action for N-Person game."""
        # Get current state
        state = self._get_npd_state(cooperation_ratio)
        
        # Update Q-value from last round
        if self.last_state is not None and self.last_action is not None:
            # Use last payoff as reward
            last_payoff = self.total_score - getattr(self, '_last_total_score', 0)
            self._update_q_value(self.last_state, self.last_action, last_payoff, state)
        
        # Choose action
        action_str = self._epsilon_greedy_action(state)
        intended_action = 0 if action_str == 'cooperate' else 1
        
        # Store for next update
        self.last_state = state
        self.last_action = action_str
        self._last_total_score = self.total_score
        
        # Apply exploration
        actual_action = self.apply_exploration(intended_action)
        return intended_action, actual_action
    
    def _choose_action_pairwise(self,
                               opponent_id: int,
                               round_in_episode: int) -> Tuple[int, int]:
        """Choose action for pairwise game."""
        # Get current state
        state = self._get_pairwise_state(opponent_id)
        
        # Choose action
        action_str = self._epsilon_greedy_action(state)
        intended_action = 0 if action_str == 'cooperate' else 1
        
        # Store state and action for this opponent
        if not hasattr(self, '_pairwise_memory'):
            self._pairwise_memory = {}
        self._pairwise_memory[opponent_id] = (state, action_str)
        
        # Apply exploration
        actual_action = self.apply_exploration(intended_action)
        return intended_action, actual_action
    
    def record_interaction(self,
                          opponent_id: int,
                          opponent_action: int,
                          my_action: int,
                          my_payoff: float):
        """Override to handle Q-learning updates in pairwise mode."""
        super().record_interaction(opponent_id, opponent_action, my_action, my_payoff)
        
        # Update Q-value if we have memory for this opponent
        if hasattr(self, '_pairwise_memory') and opponent_id in self._pairwise_memory:
            last_state, last_action = self._pairwise_memory[opponent_id]
            next_state = self._get_pairwise_state(opponent_id)
            self._update_q_value(last_state, last_action, my_payoff, next_state)
    
    def get_q_table_summary(self) -> Dict:
        """Get a summary of the Q-table for analysis."""
        summary = {}
        for state, actions in self.q_table.items():
            summary[state] = {
                'cooperate_q': actions['cooperate'],
                'defect_q': actions['defect'],
                'best_action': 'cooperate' if actions['cooperate'] > actions['defect'] else 'defect'
            }
        return summary
    
    def reset(self):
        """Reset agent state but preserve learned Q-values."""
        super().reset()
        self.last_state = None
        self.last_action = None
        if hasattr(self, '_pairwise_memory'):
            self._pairwise_memory.clear()