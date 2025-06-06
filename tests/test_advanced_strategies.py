"""
Tests for advanced agent strategies.

This module provides dedicated tests for advanced reinforcement learning strategies:
- LRA-Q (Learning Rate Adjusting Q-Learning)
- UCB1 (Upper Confidence Bound)
- Wolf-PHC (Win or Learn Fast - Policy Hill Climbing)
- Hysteretic Q-Learning
"""

import pytest
import random
import numpy as np
from unittest.mock import Mock, patch
from collections import deque

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from npdl.core.agents import Agent, create_strategy
from npdl.core.environment import Environment
from npdl.core.utils import create_payoff_matrix


class TestLRAQLearning:
    """Test suite for Learning Rate Adjusting Q-Learning strategy."""
    
    def test_lra_q_initialization(self):
        """Test LRA-Q agent initialization."""
        agent = Agent(
            agent_id=0,
            strategy="lra_q",
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon=0.1
        )
        
        assert agent.strategy_type == "lra_q"
        assert hasattr(agent.strategy, 'base_learning_rate')
        assert hasattr(agent.strategy, 'state_learning_rates')
        assert agent.strategy.base_learning_rate == 0.1
        
    def test_lra_q_learning_rate_adjustment(self):
        """Test that LRA-Q adjusts learning rates based on state visits."""
        agent = Agent(agent_id=0, strategy="lra_q", learning_rate=0.5)
        
        # Create a simple environment
        agents = [agent, Agent(agent_id=1, strategy="always_cooperate")]
        env = Environment(agents, create_payoff_matrix(2), network_type="fully_connected")
        
        # Run multiple rounds to accumulate state visits
        for _ in range(10):
            moves, payoffs = env.run_round()
            
        # Check that learning rates have been adjusted
        assert len(agent.strategy.state_learning_rates) > 0
        
        # Check that frequently visited states have lower learning rates
        for state, lr in agent.strategy.state_learning_rates.items():
            assert lr <= agent.strategy.base_learning_rate
            
    def test_lra_q_state_specific_learning(self):
        """Test that LRA-Q uses state-specific learning rates."""
        agent = Agent(agent_id=0, strategy="lra_q", learning_rate=0.5, epsilon=0.0)
        
        # Manually set different learning rates for different states
        agent.strategy.state_learning_rates["state1"] = 0.1
        agent.strategy.state_learning_rates["state2"] = 0.4
        
        # Test update for state1
        agent.q_values = {"state1": {"cooperate": 0.0, "defect": 0.0}}
        agent.strategy.update(agent, "cooperate", 10.0, {"neighbor": "cooperate"})
        
        # The Q-value should be updated with the state-specific learning rate
        # New Q = Old Q + lr * (reward - old Q) = 0 + 0.1 * (10 - 0) = 1.0
        # (Note: actual implementation may vary based on next state value)
        
    def test_lra_q_convergence_behavior(self):
        """Test that LRA-Q converges to stable Q-values."""
        agent = Agent(agent_id=0, strategy="lra_q", learning_rate=0.3, epsilon=0.1)
        
        # Create consistent environment
        agents = [agent, Agent(agent_id=1, strategy="always_cooperate")]
        env = Environment(agents, create_payoff_matrix(2), network_type="fully_connected")
        
        # Track Q-value changes
        q_history = []
        
        # Run many rounds
        for i in range(100):
            moves, payoffs = env.run_round()
            if agent.q_values:
                # Get average Q-value
                avg_q = np.mean([
                    q_val 
                    for state_q in agent.q_values.values() 
                    for q_val in state_q.values()
                ])
                q_history.append(avg_q)
                
        # Check that Q-values stabilize (variance decreases over time)
        if len(q_history) > 20:
            early_variance = np.var(q_history[:20])
            late_variance = np.var(q_history[-20:])
            assert late_variance <= early_variance or late_variance < 0.5


class TestUCB1QLearning:
    """Test suite for UCB1 Q-Learning strategy."""
    
    def test_ucb1_initialization(self):
        """Test UCB1 agent initialization."""
        agent = Agent(
            agent_id=0,
            strategy="ucb1_q",
            learning_rate=0.1,
            discount_factor=0.9,
            exploration_constant=2.0
        )
        
        assert agent.strategy_type == "ucb1_q"
        assert hasattr(agent.strategy, 'exploration_constant')
        assert hasattr(agent.strategy, 'action_counts')
        assert hasattr(agent.strategy, 'total_count')
        assert agent.strategy.exploration_constant == 2.0
        
    def test_ucb1_exploration_bonus(self):
        """Test that UCB1 adds exploration bonus to rarely chosen actions."""
        agent = Agent(agent_id=0, strategy="ucb1_q", exploration_constant=2.0)
        
        # Initialize Q-values
        state = "test_state"
        agent.q_values[state] = {"cooperate": 1.0, "defect": 1.0}
        
        # Set up action counts - defect chosen less frequently
        agent.strategy.action_counts[state] = {"cooperate": 10, "defect": 1}
        agent.strategy.total_count = 11
        
        # With exploration bonus, defect should be chosen despite equal Q-values
        # because it has been explored less
        moves_chosen = {"cooperate": 0, "defect": 0}
        for _ in range(20):
            agent.memory = [{"neighbor_moves": {"neighbor": "cooperate"}}]
            move = agent.choose_move([])
            moves_chosen[move] += 1
            
        # Defect should be chosen more often due to exploration bonus
        assert moves_chosen["defect"] > 0
        
    def test_ucb1_action_counting(self):
        """Test that UCB1 correctly tracks action counts."""
        agent = Agent(agent_id=0, strategy="ucb1_q")
        
        # Create environment
        agents = [agent, Agent(agent_id=1, strategy="always_defect")]
        env = Environment(agents, create_payoff_matrix(2), network_type="fully_connected")
        
        # Run several rounds
        for _ in range(10):
            moves, payoffs = env.run_round()
            
        # Check that action counts are tracked
        assert agent.strategy.total_count > 0
        
        # Check that action counts exist for visited states
        for state in agent.strategy.action_counts:
            action_counts = agent.strategy.action_counts[state]
            assert sum(action_counts.values()) > 0
            
    def test_ucb1_balanced_exploration(self):
        """Test that UCB1 balances exploration and exploitation."""
        agent = Agent(agent_id=0, strategy="ucb1_q", exploration_constant=1.0, epsilon=0.0)
        
        # Set up a state with clear best action but unequal exploration
        state = "test_state"
        agent.q_values[state] = {"cooperate": 5.0, "defect": 2.0}
        agent.strategy.action_counts[state] = {"cooperate": 100, "defect": 5}
        agent.strategy.total_count = 105
        
        # Despite cooperate having higher Q-value, defect should sometimes be chosen
        # due to low exploration count
        defect_chosen = False
        for _ in range(50):
            agent.memory = [{"neighbor_moves": {"neighbor": "cooperate"}}]
            if agent.choose_move([]) == "defect":
                defect_chosen = True
                break
                
        assert defect_chosen, "UCB1 should explore under-sampled actions"


class TestWolfPHC:
    """Test suite for Wolf-PHC (Win or Learn Fast - Policy Hill Climbing) strategy."""
    
    def test_wolf_phc_initialization(self):
        """Test Wolf-PHC agent initialization."""
        agent = Agent(
            agent_id=0,
            strategy="wolf_phc",
            learning_rate=0.1,
            discount_factor=0.9,
            win_learning_rate=0.01,
            lose_learning_rate=0.1
        )
        
        assert agent.strategy_type == "wolf_phc"
        assert hasattr(agent.strategy, 'win_learning_rate')
        assert hasattr(agent.strategy, 'lose_learning_rate')
        assert hasattr(agent.strategy, 'policy')
        assert hasattr(agent.strategy, 'average_policy')
        assert agent.strategy.win_learning_rate == 0.01
        assert agent.strategy.lose_learning_rate == 0.1
        
    def test_wolf_phc_policy_initialization(self):
        """Test that Wolf-PHC initializes policies correctly."""
        agent = Agent(agent_id=0, strategy="wolf_phc")
        
        # Run one round to initialize state
        agent.update_memory("cooperate", {"neighbor": "cooperate"}, 3.0)
        agent.choose_move([])
        
        # Check that policies are initialized
        assert len(agent.strategy.policy) > 0
        assert len(agent.strategy.average_policy) > 0
        
        # Check that policies sum to 1
        for state in agent.strategy.policy:
            policy_sum = sum(agent.strategy.policy[state].values())
            assert abs(policy_sum - 1.0) < 0.01
            
    def test_wolf_phc_win_vs_lose_learning(self):
        """Test that Wolf-PHC uses different learning rates for winning vs losing."""
        agent = Agent(
            agent_id=0,
            strategy="wolf_phc",
            win_learning_rate=0.01,
            lose_learning_rate=0.2
        )
        
        # Set up initial state
        state = "test_state"
        agent.q_values[state] = {"cooperate": 3.0, "defect": 2.0}
        agent.strategy.policy[state] = {"cooperate": 0.6, "defect": 0.4}
        agent.strategy.average_policy[state] = {"cooperate": 0.5, "defect": 0.5}
        agent.strategy.policy_counts[state] = 10
        
        # Simulate an update where agent is "winning" (current > average)
        initial_policy = agent.strategy.policy[state]["cooperate"]
        agent.strategy.update(agent, "cooperate", 5.0, {"neighbor": "cooperate"})
        
        # Policy should change slowly when winning
        policy_change = abs(agent.strategy.policy[state]["cooperate"] - initial_policy)
        assert policy_change < 0.05  # Small change due to win_learning_rate
        
    def test_wolf_phc_policy_improvement(self):
        """Test that Wolf-PHC improves policy toward better actions."""
        agent = Agent(agent_id=0, strategy="wolf_phc", epsilon=0.0)
        
        # Set up state with clear best action
        state = "test_state"
        agent.q_values[state] = {"cooperate": 5.0, "defect": 1.0}
        agent.strategy.policy[state] = {"cooperate": 0.3, "defect": 0.7}  # Bad initial policy
        agent.strategy.average_policy[state] = {"cooperate": 0.3, "defect": 0.7}
        agent.strategy.policy_counts[state] = 5
        
        # Run multiple updates
        for _ in range(20):
            agent.memory = [{"neighbor_moves": {"neighbor": "cooperate"}}]
            move = agent.choose_move([])
            agent.strategy.update(agent, move, 3.0, {"neighbor": "cooperate"})
            
        # Policy should shift toward cooperate (higher Q-value)
        assert agent.strategy.policy[state]["cooperate"] > 0.5
        
    def test_wolf_phc_stochastic_action_selection(self):
        """Test that Wolf-PHC selects actions stochastically according to policy."""
        agent = Agent(agent_id=0, strategy="wolf_phc", epsilon=0.0)
        
        # Set up policy with specific probabilities
        state = "test_state"
        agent.strategy.policy[state] = {"cooperate": 0.7, "defect": 0.3}
        agent.memory = [{"neighbor_moves": {"neighbor": "cooperate"}}]
        
        # Sample many actions
        action_counts = {"cooperate": 0, "defect": 0}
        for _ in range(1000):
            # Need to ensure we're in the right state
            agent.strategy._get_state({"neighbor": "cooperate"})
            action = agent.choose_move([])
            action_counts[action] += 1
            
        # Check that actions follow policy distribution (with some tolerance)
        coop_rate = action_counts["cooperate"] / 1000
        assert 0.6 < coop_rate < 0.8  # Should be close to 0.7


class TestHystereticQLearning:
    """Test suite for Hysteretic Q-Learning strategy."""
    
    def test_hysteretic_q_initialization(self):
        """Test Hysteretic Q-Learning initialization."""
        agent = Agent(
            agent_id=0,
            strategy="hysteretic_q",
            learning_rate=0.1,
            discount_factor=0.9,
            optimistic_learning_rate=0.2,
            pessimistic_learning_rate=0.05
        )
        
        assert agent.strategy_type == "hysteretic_q"
        assert hasattr(agent.strategy, 'optimistic_learning_rate')
        assert hasattr(agent.strategy, 'pessimistic_learning_rate')
        assert agent.strategy.optimistic_learning_rate == 0.2
        assert agent.strategy.pessimistic_learning_rate == 0.05
        
    def test_hysteretic_q_asymmetric_learning(self):
        """Test that Hysteretic Q uses different rates for positive/negative updates."""
        agent = Agent(
            agent_id=0,
            strategy="hysteretic_q",
            optimistic_learning_rate=0.5,
            pessimistic_learning_rate=0.1,
            epsilon=0.0
        )
        
        # Initialize Q-values
        state = "test_state"
        agent.q_values[state] = {"cooperate": 2.0, "defect": 2.0}
        
        # Test positive update (reward > current Q)
        agent.memory = [{"neighbor_moves": {"neighbor": "cooperate"}}]
        initial_q = agent.q_values[state]["cooperate"]
        agent.strategy.update(agent, "cooperate", 5.0, {"neighbor": "cooperate"})
        
        # Should use optimistic learning rate
        q_increase = agent.q_values[state]["cooperate"] - initial_q
        assert q_increase > 0
        
        # Test negative update (reward < current Q)
        agent.q_values[state]["defect"] = 5.0
        initial_q = agent.q_values[state]["defect"]
        agent.strategy.update(agent, "defect", 1.0, {"neighbor": "cooperate"})
        
        # Should use pessimistic learning rate (smaller change)
        q_decrease = initial_q - agent.q_values[state]["defect"]
        assert q_decrease > 0
        assert q_decrease < q_increase  # Pessimistic update should be smaller
        
    def test_hysteretic_q_optimistic_bias(self):
        """Test that Hysteretic Q-learning develops optimistic bias."""
        agent = Agent(
            agent_id=0,
            strategy="hysteretic_q",
            optimistic_learning_rate=0.3,
            pessimistic_learning_rate=0.05,
            epsilon=0.1
        )
        
        # Create environment with mixed outcomes
        agents = [
            agent,
            Agent(agent_id=1, strategy="tit_for_tat"),
            Agent(agent_id=2, strategy="random")
        ]
        env = Environment(agents, create_payoff_matrix(3), network_type="fully_connected")
        
        # Run many rounds
        for _ in range(100):
            moves, payoffs = env.run_round()
            
        # Check that Q-values show optimistic bias
        # Average Q-value should be relatively high due to asymmetric learning
        if agent.q_values:
            avg_q = np.mean([
                q_val 
                for state_q in agent.q_values.values() 
                for q_val in state_q.values()
            ])
            # Optimistic bias should lead to higher Q-values
            assert avg_q > 0  # Should be positive in mixed environment
            
    def test_hysteretic_q_cooperation_promotion(self):
        """Test that Hysteretic Q-learning promotes cooperation."""
        # Create two hysteretic Q-learners
        agents = [
            Agent(agent_id=0, strategy="hysteretic_q", 
                  optimistic_learning_rate=0.3, pessimistic_learning_rate=0.05),
            Agent(agent_id=1, strategy="hysteretic_q",
                  optimistic_learning_rate=0.3, pessimistic_learning_rate=0.05)
        ]
        
        env = Environment(agents, create_payoff_matrix(2), network_type="fully_connected")
        
        # Track cooperation over time
        cooperation_rates = []
        
        for i in range(50):
            moves, payoffs = env.run_round()
            coop_rate = sum(1 for move in moves.values() if move == "cooperate") / len(moves)
            if i >= 10:  # Skip initial exploration phase
                cooperation_rates.append(coop_rate)
                
        # Hysteretic Q-learning should maintain relatively high cooperation
        avg_cooperation = np.mean(cooperation_rates) if cooperation_rates else 0
        assert avg_cooperation > 0.3  # Should achieve reasonable cooperation


class TestStrategyComparison:
    """Compare performance of different advanced strategies."""
    
    def test_advanced_strategies_convergence(self):
        """Test that all advanced strategies converge to stable behavior."""
        strategies = ["lra_q", "ucb1_q", "wolf_phc", "hysteretic_q"]
        
        for strategy in strategies:
            agent = Agent(agent_id=0, strategy=strategy, epsilon=0.1)
            opponent = Agent(agent_id=1, strategy="tit_for_tat")
            
            env = Environment([agent, opponent], create_payoff_matrix(2), 
                            network_type="fully_connected")
            
            # Track score progression
            score_history = []
            
            for i in range(50):
                moves, payoffs = env.run_round()
                if i % 5 == 0:
                    score_history.append(agent.score)
                    
            # Check that score increases over time (learning is happening)
            if len(score_history) > 2:
                assert score_history[-1] >= score_history[0]
                
    def test_advanced_strategies_against_defector(self):
        """Test how advanced strategies handle always-defect opponent."""
        strategies = ["lra_q", "ucb1_q", "wolf_phc", "hysteretic_q"]
        
        for strategy in strategies:
            agent = Agent(agent_id=0, strategy=strategy, epsilon=0.05)
            defector = Agent(agent_id=1, strategy="always_defect")
            
            env = Environment([agent, defector], create_payoff_matrix(2),
                            network_type="fully_connected")
            
            # Run many rounds
            defection_count = 0
            for i in range(50):
                moves, payoffs = env.run_round()
                if i >= 20 and moves[0] == "defect":  # After learning phase
                    defection_count += 1
                    
            # Should learn to defect against always-defect
            assert defection_count > 15  # Should defect most of the time


if __name__ == '__main__':
    pytest.main([__file__, '-v'])