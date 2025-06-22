"""
Corrected Enhanced Q-Learning Agents with proper inheritance

This module implements enhanced Q-learning agents that correctly inherit
from SimpleQLearningAgent to avoid the pairwise state representation bug.

Key improvements:
1. Proper inheritance from SimpleQLearningAgent
2. Epsilon decay for better exploration-exploitation balance
3. Optional memory enhancement for temporal patterns
4. Lightweight opponent modeling when beneficial
5. Conservative parameter defaults
"""

import random
import math
from collections import defaultdict, deque

from main_neighbourhood import NPERSON_COOPERATE, NPERSON_DEFECT
from main_pairwise import PAIRWISE_COOPERATE, PAIRWISE_DEFECT

# Import the SimpleQLearningAgent from the legacy module
from qlearning_agents import SimpleQLearningAgent


class CorrectedEnhancedQLearningAgent(SimpleQLearningAgent):
    """
    Enhanced Q-Learning agent that inherits from SimpleQLearningAgent.
    
    This agent builds upon the solid foundation of the simple agent,
    adding features like epsilon decay and more advanced state representations
    in a modular and correct way.
    
    Enhancements:
    - Epsilon Decay: Epsilon is no longer fixed, allowing for more exploration at the start.
    - Advanced State Representation (for neighborhood mode): Optional, more granular states.
    - Memory Buffer: Can use a memory of past actions to inform the state.
    """
    def __init__(self, agent_id, 
                 learning_rate=0.1, 
                 discount_factor=0.9,
                 epsilon=0.3,  # Higher starting epsilon for exploration
                 epsilon_decay=0.999,
                 epsilon_min=0.01,
                 exploration_rate=0.0,
                 state_type="basic",  # 'basic', 'fine', 'memory_enhanced'
                 memory_length=5,
                 **kwargs):  # Absorb other potential arguments
        
        # Initialize the parent class (SimpleQLearningAgent)
        super().__init__(agent_id, learning_rate, discount_factor, epsilon, exploration_rate)
        
        # Strategy name for compatibility
        self.strategy_name = "EnhancedQLearning"
        
        # Enhanced-specific attributes
        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.state_type = state_type
        
        # Memory buffer for 'memory_enhanced' state type
        self.memory_length = memory_length
        if self.state_type == "memory_enhanced":
            self.memory = deque(maxlen=memory_length)
            
        # For tracking pairwise states per opponent
        self.pairwise_last_states = {}
        self.pairwise_last_actions = {}
            
    def _get_state_neighborhood(self, coop_ratio):
        """
        Overrides the parent method to provide more sophisticated state options.
        The pairwise state logic is inherited and remains unchanged.
        """
        if coop_ratio is None:
            return 'initial'

        # Basic state representation (inherited logic)
        if self.state_type == "basic":
            return super()._get_state_neighborhood(coop_ratio)
        
        # Fine-grained state representation
        elif self.state_type == "fine":
            # Discretize into 10 bins
            return f"coop_{int(coop_ratio * 10)}"
            
        # State enhanced with agent's own action history
        elif self.state_type == "memory_enhanced":
            base_state = super()._get_state_neighborhood(coop_ratio)
            # Create a summary of own recent actions
            if not self.memory:
                memory_signature = "mem_none"
            else:
                coop_count = sum(1 for action in self.memory if action == 'cooperate')
                defect_count = len(self.memory) - coop_count
                memory_signature = f"mem_c{coop_count}_d{defect_count}"
            
            return f"{base_state}_{memory_signature}"
        
        else:
            # Default to basic if state_type is unknown
            return super()._get_state_neighborhood(coop_ratio)

    def choose_action(self, prev_round_overall_coop_ratio, current_round_num):
        """Choose action for neighborhood mode."""
        state = self._get_state_neighborhood(prev_round_overall_coop_ratio)
        action = self._choose_action_epsilon_greedy(state)
        
        # Store state and action for the next Q-update
        self.last_state = state
        self.last_action = action
        self.last_coop_ratio = prev_round_overall_coop_ratio
        
        # Update memory buffer if using it
        if self.state_type == "memory_enhanced":
            self.memory.append(action)
        
        intended_move = NPERSON_COOPERATE if action == 'cooperate' else NPERSON_DEFECT
        actual_move = intended_move
        if random.random() < self.exploration_rate:
            actual_move = 1 - intended_move
            
        return intended_move, actual_move

    def record_round_outcome(self, my_actual_move, payoff, new_coop_ratio=None):
        """
        Record outcome for neighborhood mode with correct Q-update.
        """
        self.total_score += payoff
        action_str = 'cooperate' if my_actual_move == NPERSON_COOPERATE else 'defect'
        if action_str == 'cooperate':
            self.num_cooperations += 1
        else:
            self.num_defections += 1
            
        # Update Q-value using the reward for the last action
        if self.last_state is not None and self.last_action is not None:
            # Use the stored coop ratio if new one not provided
            if new_coop_ratio is None:
                new_coop_ratio = self.last_coop_ratio
            next_state = self._get_state_neighborhood(new_coop_ratio)
            self.update_q_value(self.last_state, self.last_action, payoff, next_state)

    def choose_action_pairwise(self, opponent_id, current_round_in_episode):
        """
        Choose action for pairwise mode - uses inherited logic from SimpleQLearningAgent.
        """
        state = self._get_state_pairwise(opponent_id)
        action = self._choose_action_epsilon_greedy(state)
        
        # Store per-opponent state/action
        self.pairwise_last_states[opponent_id] = state
        self.pairwise_last_actions[opponent_id] = action
        
        intended_move = PAIRWISE_COOPERATE if action == 'cooperate' else PAIRWISE_DEFECT
        
        actual_move = intended_move
        if random.random() < self.exploration_rate:
            actual_move = 1 - intended_move
        
        return intended_move, actual_move

    def record_interaction(self, opponent_id, opponent_actual_move, my_payoff,
                          my_intended_move, my_actual_move, round_num_in_episode):
        """Record interaction for pairwise mode - extends parent functionality."""
        self.total_score += my_payoff
        
        if my_actual_move == PAIRWISE_COOPERATE:
            self.num_cooperations += 1
        else:
            self.num_defections += 1
        
        # Update opponent history
        self.opponent_last_moves[opponent_id] = opponent_actual_move
        
        # Update Q-value
        if opponent_id in self.pairwise_last_states:
            last_state = self.pairwise_last_states[opponent_id]
            last_action = self.pairwise_last_actions[opponent_id]
            next_state = self._get_state_pairwise(opponent_id)
            self.update_q_value(last_state, last_action, my_payoff, next_state)

    def decay_epsilon(self):
        """Applies decay to epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset(self):
        """Resets agent for a new run and decays epsilon."""
        super().reset()  # Call the parent's reset method
        self.decay_epsilon()  # Decay epsilon after each full run/episode
        
        # Clear memory buffer if it exists
        if hasattr(self, 'memory'):
            self.memory.clear()
            
        # Clear pairwise tracking
        self.pairwise_last_states = {}
        self.pairwise_last_actions = {}


class EnhancedQLearningAgent(CorrectedEnhancedQLearningAgent):
    """
    Alias for backward compatibility.
    This is the corrected enhanced Q-learning agent.
    """
    pass


class RevampedQLearningAgent(CorrectedEnhancedQLearningAgent):
    """
    Revamped Q-Learning that combines the best of both implementations.
    
    Key features:
    - Based on SimpleQLearning's successful approach
    - Adds epsilon decay for better exploration-exploitation balance
    - Optional memory enhancement for temporal patterns
    - Conservative parameter defaults
    """
    
    def __init__(self, agent_id, **kwargs):
        # Set conservative defaults that performed well
        kwargs.setdefault('epsilon_decay', 0.995)  # Gentle decay
        kwargs.setdefault('state_type', 'basic')   # Use basic states by default
        kwargs.setdefault('exploration_rate', 0.0)  # No additional exploration
        
        super().__init__(agent_id, **kwargs)


class AdaptiveQLearningAgent(CorrectedEnhancedQLearningAgent):
    """
    Adaptive Q-Learning that adjusts its parameters based on performance.
    
    This agent monitors its performance and adjusts exploration and
    learning parameters dynamically.
    """
    
    def __init__(self, agent_id, **kwargs):
        super().__init__(agent_id, **kwargs)
        
        # Performance tracking
        self.recent_scores = deque(maxlen=10)
        self.performance_trend = 0
        
        # Adaptive parameters
        self.min_learning_rate = 0.01
        self.max_learning_rate = 0.3
        
    def adapt_parameters(self):
        """Adapt learning parameters based on recent performance."""
        if len(self.recent_scores) >= 5:
            # Calculate trend
            first_half = sum(list(self.recent_scores)[:5]) / 5
            second_half = sum(list(self.recent_scores)[5:]) / 5
            self.performance_trend = second_half - first_half
            
            # Adjust learning rate based on trend
            if self.performance_trend > 0:
                # Performance improving, reduce learning rate
                self.learning_rate = max(self.min_learning_rate, 
                                       self.learning_rate * 0.95)
            else:
                # Performance declining, increase learning rate
                self.learning_rate = min(self.max_learning_rate,
                                       self.learning_rate * 1.05)
    
    def record_round_outcome(self, my_actual_move, payoff, new_coop_ratio=None):
        """Record outcome and adapt parameters."""
        super().record_round_outcome(my_actual_move, payoff, new_coop_ratio)
        self.recent_scores.append(payoff)
        self.adapt_parameters()


# Factory functions for easy creation in simulation scripts
def create_corrected_enhanced_qlearning(agent_id, **kwargs):
    """Factory function for the corrected enhanced Q-learning agent."""
    return CorrectedEnhancedQLearningAgent(agent_id, **kwargs)

def create_enhanced_qlearning(agent_id, **kwargs):
    """Factory function for enhanced Q-learning agent."""
    return EnhancedQLearningAgent(agent_id, **kwargs)

def create_revamped_qlearning(agent_id, **kwargs):
    """Factory function for revamped Q-learning agent with best defaults."""
    return RevampedQLearningAgent(agent_id, **kwargs)

def create_adaptive_qlearning(agent_id, **kwargs):
    """Factory function for adaptive Q-learning agent."""
    return AdaptiveQLearningAgent(agent_id, **kwargs)


# Test the implementation
if __name__ == "__main__":
    print("CorrectedEnhancedQLearningAgent has been defined.")
    print("It inherits from SimpleQLearningAgent to ensure correct pairwise logic.")
    print("Enhancements like epsilon decay and advanced states are added on top.")
    
    # Create an instance to show it works
    enhanced_agent = create_corrected_enhanced_qlearning(
        agent_id="EQL-1", 
        state_type="memory_enhanced"
    )
    print(f"\nCreated agent '{enhanced_agent.agent_id}' with:")
    print(f" - Epsilon: {enhanced_agent.epsilon} (will decay by {enhanced_agent.epsilon_decay})")
    print(f" - State Type: {enhanced_agent.state_type}")
    print(f" - Inherits pairwise logic from SimpleQLearningAgent: {hasattr(enhanced_agent, '_get_state_pairwise')}")