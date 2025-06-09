"""
N-Person Reinforcement Learning Strategies

This module implements RL strategies specifically designed for N-person games,
addressing the unique challenges of multi-agent learning in groups.

Key improvements over standard RL:
1. Group-aware state representations
2. Scaled learning parameters based on group size
3. Reward shaping for group cooperation
4. Non-stationarity handling for multiple learners
"""

import math
import random
from collections import deque
from typing import Dict, Any, Hashable, Optional, List

from .agents import (
    QLearningStrategy,
    HystereticQLearningStrategy,
    WolfPHCStrategy,
    LRAQLearningStrategy,
    UCB1QLearningStrategy
)


class NPersonStateMixin:
    """Mixin class providing N-person specific state representations."""
    
    def _extract_group_features(self, agent, interaction_context, memory_window=5):
        """Extract rich features for N-person games."""
        features = {}
        
        # Basic cooperation statistics
        if isinstance(interaction_context, dict):
            if 'opponent_coop_proportion' in interaction_context:
                # Pairwise mode
                coop_rate = interaction_context['opponent_coop_proportion']
                features['mode'] = 'pairwise'
            else:
                # Neighborhood mode
                num_neighbors = len(interaction_context)
                if num_neighbors > 0:
                    num_coop = sum(1 for move in interaction_context.values() if move == "cooperate")
                    coop_rate = num_coop / num_neighbors
                else:
                    coop_rate = 0.0
                features['mode'] = 'neighborhood'
                features['group_size'] = num_neighbors
        else:
            coop_rate = 0.0
            features['mode'] = 'unknown'
        
        features['cooperation_rate'] = round(coop_rate, 2)
        
        # Cooperation trend over recent rounds
        if len(agent.memory) >= 2:
            recent_coop_rates = []
            for i in range(min(memory_window, len(agent.memory))):
                round_info = agent.memory[-(i+1)]
                context = round_info.get('neighbor_moves', {})
                
                if isinstance(context, dict) and 'opponent_coop_proportion' in context:
                    recent_coop_rates.append(context['opponent_coop_proportion'])
                elif isinstance(context, dict) and context:
                    n = len(context)
                    if n > 0:
                        c = sum(1 for m in context.values() if m == "cooperate")
                        recent_coop_rates.append(c / n)
            
            if len(recent_coop_rates) >= 2:
                # Calculate trend: positive, negative, or stable
                trend = recent_coop_rates[0] - recent_coop_rates[-1]
                if trend > 0.1:
                    features['cooperation_trend'] = 1  # Increasing
                elif trend < -0.1:
                    features['cooperation_trend'] = -1  # Decreasing
                else:
                    features['cooperation_trend'] = 0  # Stable
                    
                # Calculate volatility
                if len(recent_coop_rates) > 2:
                    diffs = [abs(recent_coop_rates[i] - recent_coop_rates[i+1]) 
                            for i in range(len(recent_coop_rates)-1)]
                    features['cooperation_volatility'] = round(sum(diffs) / len(diffs), 2)
                else:
                    features['cooperation_volatility'] = 0.0
            else:
                features['cooperation_trend'] = 0
                features['cooperation_volatility'] = 0.0
        else:
            features['cooperation_trend'] = 0
            features['cooperation_volatility'] = 0.0
        
        # My recent behavior
        if agent.memory:
            recent_moves = [m['my_move'] for m in list(agent.memory)[-memory_window:]]
            my_coop_rate = sum(1 for m in recent_moves if m == 'cooperate') / len(recent_moves)
            features['my_cooperation_rate'] = round(my_coop_rate, 2)
            
            # Cooperation streak
            streak = 0
            for move in reversed(recent_moves):
                if move == 'cooperate':
                    streak += 1
                else:
                    break
            features['my_cooperation_streak'] = streak
        else:
            features['my_cooperation_rate'] = 0.5
            features['my_cooperation_streak'] = 0
        
        # Critical mass indicators (cooperation > 50% is often a threshold)
        features['above_threshold'] = coop_rate > 0.5
        features['distance_to_threshold'] = round(coop_rate - 0.5, 2)
        
        return features
    
    def _get_n_person_state(self, agent, N: Optional[int] = None) -> Hashable:
        """Get state representation for N-person games."""
        if not agent.memory:
            return ('initial', N if N else 'unknown')
        
        last_round = agent.memory[-1]
        context = last_round.get('neighbor_moves', {})
        
        features = self._extract_group_features(agent, context)
        
        # Determine effective group size
        if N is not None:
            group_size = N
        elif features.get('group_size'):
            group_size = features['group_size']
        else:
            # Estimate from context
            if features['mode'] == 'neighborhood':
                group_size = len(context) if isinstance(context, dict) else 5
            else:
                group_size = 5  # Default assumption
        
        # Create state tuple based on state_type
        if self.state_type == "n_person_basic":
            # Minimal state: cooperation level and trend
            coop_level = 'low' if features['cooperation_rate'] < 0.33 else \
                        'med' if features['cooperation_rate'] < 0.67 else 'high'
            return (coop_level, features['cooperation_trend'], group_size)
            
        elif self.state_type == "n_person_rich":
            # Rich state with multiple features
            return (
                round(features['cooperation_rate'], 1),
                features['cooperation_trend'],
                features['above_threshold'],
                round(features['cooperation_volatility'], 1),
                features['my_cooperation_streak'] > 2,
                group_size
            )
            
        elif self.state_type == "n_person_adaptive":
            # Adaptive state that changes with group size
            if group_size <= 5:
                # Small group: track individual patterns
                return (
                    round(features['cooperation_rate'], 2),
                    features['my_cooperation_rate'],
                    features['cooperation_trend']
                )
            else:
                # Large group: focus on aggregate statistics
                return (
                    features['above_threshold'],
                    features['cooperation_trend'],
                    features['cooperation_volatility'] > 0.2
                )
        else:
            # Fallback to enhanced standard state
            return (
                round(features['cooperation_rate'], 1),
                features['cooperation_trend'],
                group_size
            )


class NPersonQLearning(QLearningStrategy, NPersonStateMixin):
    """Q-Learning adapted for N-person games."""
    
    def __init__(self, N: Optional[int] = None, 
                 state_type: str = "n_person_basic",
                 scale_learning: bool = True,
                 **kwargs):
        # Override state type to ensure N-person state
        kwargs['state_type'] = state_type
        super().__init__(**kwargs)
        self.N = N
        self.scale_learning = scale_learning
        self.group_baseline = 0.0
        self.baseline_alpha = 0.01
        
    def _get_current_state(self, agent) -> Hashable:
        """Override to use N-person state representation."""
        return self._get_n_person_state(agent, self.N)
    
    def _shape_reward(self, reward: float, action: str, 
                     agent_memory: deque, next_context: Dict) -> float:
        """Shape reward to encourage group cooperation."""
        shaped_reward = reward
        
        # Extract cooperation rate from context
        if isinstance(next_context, dict) and 'opponent_coop_proportion' in next_context:
            coop_rate = next_context['opponent_coop_proportion']
        elif isinstance(next_context, dict) and next_context:
            n = len(next_context)
            coop_rate = sum(1 for m in next_context.values() if m == "cooperate") / n if n > 0 else 0
        else:
            coop_rate = 0
        
        # Bonus for cooperating when group is cooperative
        if action == "cooperate" and coop_rate > 0.5:
            group_size = self.N if self.N else len(next_context) if isinstance(next_context, dict) else 5
            cooperation_bonus = 1.0 / math.sqrt(group_size)
            shaped_reward += cooperation_bonus
        
        # Penalty for defecting in cooperative groups
        if action == "defect" and coop_rate > 0.7:
            shaped_reward -= 0.5
        
        return shaped_reward
    
    def update(self, agent, action, reward, neighbor_moves):
        """Update with N-person aware modifications."""
        # Update group baseline
        self.group_baseline = (1 - self.baseline_alpha) * self.group_baseline + \
                             self.baseline_alpha * reward
        
        # Shape reward
        shaped_reward = self._shape_reward(reward, action, agent.memory, neighbor_moves)
        
        # Scale learning rate based on group size if enabled
        if self.scale_learning and self.N:
            original_lr = self.learning_rate
            self.learning_rate = original_lr / math.sqrt(self.N)
        
        # Standard Q-update with shaped reward
        state_executed = agent.last_state_representation
        if state_executed is None:
            return
            
        next_state = self._get_current_state(agent)
        self._ensure_state_exists(agent, next_state)
        
        best_next_q = max(agent.q_values[next_state].values())
        current_q = agent.q_values[state_executed][action]
        
        agent.q_values[state_executed][action] = (
            (1 - self.learning_rate) * current_q +
            self.learning_rate * (shaped_reward + self.discount_factor * best_next_q)
        )
        
        # Restore original learning rate
        if self.scale_learning and self.N:
            self.learning_rate = original_lr


class NPersonHystereticQ(HystereticQLearningStrategy, NPersonStateMixin):
    """Hysteretic Q-Learning with N-person adaptations."""
    
    def __init__(self, N: Optional[int] = None,
                 state_type: str = "n_person_basic",
                 scale_optimism: bool = True,
                 **kwargs):
        kwargs['state_type'] = state_type
        super().__init__(**kwargs)
        self.N = N
        self.scale_optimism = scale_optimism
        
        # Scale beta (negative learning rate) with group size
        if self.scale_optimism and self.N and self.N > 2:
            self.beta = self.beta * math.sqrt(self.N / 2)
    
    def _get_current_state(self, agent) -> Hashable:
        """Override to use N-person state representation."""
        return self._get_n_person_state(agent, self.N)
    
    def update(self, agent, action, reward, neighbor_moves):
        """Update with extra optimism for group cooperation."""
        state_executed = agent.last_state_representation
        if state_executed is None:
            return
        
        next_state = self._get_current_state(agent)
        self._ensure_state_exists(agent, next_state)
        
        # Check if group is cooperating
        if isinstance(neighbor_moves, dict) and 'opponent_coop_proportion' in neighbor_moves:
            group_coop_rate = neighbor_moves['opponent_coop_proportion']
        elif isinstance(neighbor_moves, dict) and neighbor_moves:
            n = len(neighbor_moves)
            group_coop_rate = sum(1 for m in neighbor_moves.values() if m == "cooperate") / n if n > 0 else 0
        else:
            group_coop_rate = 0
        
        # Calculate target Q-value
        best_next_q = max(agent.q_values[next_state].values())
        target = reward + self.discount_factor * best_next_q
        current = agent.q_values[state_executed][action]
        
        # Use different learning rates with group-aware adjustments
        if target > current:  # Positive experience
            effective_lr = self.learning_rate
            # Extra optimism for cooperation in cooperative groups
            if action == "cooperate" and group_coop_rate > 0.6:
                effective_lr = min(self.learning_rate * 1.5, 0.9)
            
            agent.q_values[state_executed][action] = (
                (1 - effective_lr) * current + effective_lr * target
            )
        else:  # Negative experience
            effective_beta = self.beta
            # Be even more forgiving of cooperation failures in large groups
            if action == "cooperate" and self.N and self.N > 5:
                effective_beta = self.beta / 2
            
            agent.q_values[state_executed][action] = (
                (1 - effective_beta) * current + effective_beta * target
            )


class NPersonWolfPHC(WolfPHCStrategy, NPersonStateMixin):
    """Wolf-PHC adapted for N-person games."""
    
    def __init__(self, N: Optional[int] = None,
                 state_type: str = "n_person_basic",
                 use_nash_baseline: bool = True,
                 **kwargs):
        kwargs['state_type'] = state_type
        super().__init__(**kwargs)
        self.N = N
        self.use_nash_baseline = use_nash_baseline
        self.group_performance_history = deque(maxlen=100)
        
    def _get_current_state(self, agent) -> Hashable:
        """Override to use N-person state representation."""
        return self._get_n_person_state(agent, self.N)
    
    def _compute_nash_value(self, N: int) -> float:
        """Estimate Nash equilibrium value for N-person PD.
        
        In N-person PD, the Nash equilibrium is typically all-defect,
        but the value depends on the payoff structure.
        """
        # Assuming standard PD payoffs: R=3, S=0, T=5, P=1
        # Nash equilibrium (all defect) gives each player P=1
        # But in practice, mixed strategies might emerge
        
        # Simple approximation: Nash value decreases with group size
        # due to increased temptation to defect
        base_nash = 1.0  # All-defect payoff
        
        # Adjust based on group size (larger groups -> lower cooperation)
        if N <= 2:
            return base_nash + 0.5  # Some cooperation possible
        elif N <= 5:
            return base_nash + 0.2
        elif N <= 10:
            return base_nash + 0.1
        else:
            return base_nash
    
    def update(self, agent, action, reward, neighbor_moves):
        """Update with N-person aware winning criteria."""
        # Track group performance
        self.group_performance_history.append(reward)
        
        # Get state information
        state_executed = agent.last_state_representation
        if state_executed is None:
            return
        
        # Update average payoff and policy
        self.average_payoff = (
            (1 - self.alpha_avg) * self.average_payoff + self.alpha_avg * reward
        )
        self._update_average_policy(state_executed)
        
        # Calculate V_pi and V_pi_avg
        q_coop = agent.q_values[state_executed]["cooperate"]
        q_def = agent.q_values[state_executed]["defect"]
        
        # Current policy value
        if q_coop >= q_def:
            pi_coop = 1.0 - self.epsilon + self.epsilon / 2
            pi_def = self.epsilon / 2
        else:
            pi_def = 1.0 - self.epsilon + self.epsilon / 2
            pi_coop = self.epsilon / 2
        
        V_pi = pi_coop * q_coop + pi_def * q_def
        
        # Average policy value
        if state_executed in self.average_policy:
            avg_pi_coop = self.average_policy[state_executed]["cooperate"]
            avg_pi_def = self.average_policy[state_executed]["defect"]
            V_pi_avg = avg_pi_coop * q_coop + avg_pi_def * q_def
        else:
            V_pi_avg = V_pi
        
        # Determine if winning using N-person criteria
        winning = False
        
        if self.use_nash_baseline and self.N:
            nash_value = self._compute_nash_value(self.N)
            # Also consider group average
            group_avg = sum(self.group_performance_history) / len(self.group_performance_history) \
                       if self.group_performance_history else 0
            
            # Win if above both Nash and recent group average
            winning = V_pi > max(nash_value, group_avg, V_pi_avg)
        else:
            # Standard Wolf-PHC criterion
            winning = V_pi >= V_pi_avg
        
        # Set learning rate based on winning status
        current_alpha = self.alpha_win if winning else self.alpha_lose
        
        # Scale by group size for stability
        if self.N and self.N > 2:
            current_alpha = current_alpha / math.sqrt(self.N / 2)
        
        # Update Q-values
        next_state = self._get_current_state(agent)
        self._ensure_state_exists(agent, next_state)
        
        best_next_q = max(agent.q_values[next_state].values())
        current_q = agent.q_values[state_executed][action]
        
        agent.q_values[state_executed][action] = (
            (1 - current_alpha) * current_q + 
            current_alpha * (reward + self.discount_factor * best_next_q)
        )


# Factory function extension
def create_n_person_strategy(strategy_type: str, N: int, **kwargs):
    """Create N-person aware RL strategies."""
    strategies = {
        "n_person_q_learning": NPersonQLearning,
        "n_person_hysteretic_q": NPersonHystereticQ,
        "n_person_wolf_phc": NPersonWolfPHC,
    }
    
    if strategy_type not in strategies:
        raise ValueError(f"Unknown N-person strategy type: {strategy_type}")
    
    strategy_class = strategies[strategy_type]
    return strategy_class(N=N, **kwargs)