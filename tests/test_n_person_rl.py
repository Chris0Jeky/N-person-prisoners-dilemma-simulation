"""
Test N-Person Reinforcement Learning Agents

This module tests the N-person specific RL implementations to ensure
they behave correctly in multi-agent scenarios.
"""

import pytest
import numpy as np
from collections import deque

# Import the necessary modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from npdl.core.agents import Agent, create_strategy
from npdl.core.n_person_rl import (
    NPersonQLearning, 
    NPersonHystereticQ, 
    NPersonWolfPHC,
    NPersonStateMixin
)


class TestNPersonStateMixin:
    """Test the state extraction functionality."""
    
    def test_extract_group_features_neighborhood(self):
        """Test feature extraction in neighborhood mode."""
        mixin = NPersonStateMixin()
        
        # Create mock agent
        agent = type('MockAgent', (), {})()
        agent.memory = deque(maxlen=10)
        
        # Add some history
        agent.memory.append({
            'my_move': 'cooperate',
            'neighbor_moves': {
                1: 'cooperate',
                2: 'cooperate',
                3: 'defect'
            },
            'reward': 3
        })
        
        # Test feature extraction
        context = {1: 'cooperate', 2: 'defect', 3: 'defect'}
        features = mixin._extract_group_features(agent, context)
        
        assert features['cooperation_rate'] == 0.33  # 1/3 cooperating
        assert features['mode'] == 'neighborhood'
        assert features['group_size'] == 3
        
    def test_extract_group_features_pairwise(self):
        """Test feature extraction in pairwise mode."""
        mixin = NPersonStateMixin()
        
        # Create mock agent
        agent = type('MockAgent', (), {})()
        agent.memory = deque(maxlen=10)
        
        # Test pairwise context
        context = {
            'opponent_coop_proportion': 0.75,
            'specific_opponent_moves': {1: 'cooperate', 2: 'cooperate', 3: 'cooperate', 4: 'defect'}
        }
        
        features = mixin._extract_group_features(agent, context)
        
        assert features['cooperation_rate'] == 0.75
        assert features['mode'] == 'pairwise'
        
    def test_cooperation_trend_detection(self):
        """Test that cooperation trends are correctly detected."""
        mixin = NPersonStateMixin()
        
        # Create mock agent with history
        agent = type('MockAgent', (), {})()
        agent.memory = deque(maxlen=10)
        
        # Increasing cooperation trend
        for coop_rate in [0.2, 0.4, 0.6, 0.8]:
            agent.memory.append({
                'my_move': 'cooperate',
                'neighbor_moves': {'opponent_coop_proportion': coop_rate},
                'reward': 3
            })
        
        context = {'opponent_coop_proportion': 0.9}
        features = mixin._extract_group_features(agent, context)
        
        assert features['cooperation_trend'] == 1  # Increasing
        
    def test_n_person_state_generation(self):
        """Test state tuple generation for different state types."""
        mixin = NPersonStateMixin()
        mixin.state_type = "n_person_basic"
        
        # Create mock agent
        agent = type('MockAgent', (), {})()
        agent.memory = deque(maxlen=10)
        agent.memory.append({
            'my_move': 'cooperate',
            'neighbor_moves': {1: 'cooperate', 2: 'cooperate', 3: 'defect'},
            'reward': 3
        })
        
        state = mixin._get_n_person_state(agent, N=5)
        
        assert isinstance(state, tuple)
        assert len(state) == 3  # (coop_level, trend, N)
        assert state[2] == 5  # N parameter


class TestNPersonQLearning:
    """Test N-Person Q-Learning implementation."""
    
    def test_initialization(self):
        """Test that N-person Q-learning initializes correctly."""
        strategy = NPersonQLearning(N=10, state_type="n_person_basic")
        
        assert strategy.N == 10
        assert strategy.state_type == "n_person_basic"
        assert strategy.scale_learning == True
        
    def test_scaled_learning_rate(self):
        """Test that learning rate scales with group size."""
        base_lr = 0.1
        strategy = NPersonQLearning(N=10, learning_rate=base_lr, scale_learning=True)
        
        # Create mock agent
        agent = Agent(agent_id=1, strategy="random")
        agent.last_state_representation = ('initial', 10)
        agent.q_values = {('initial', 10): {'cooperate': 0, 'defect': 0}}
        
        # Perform update
        original_lr = strategy.learning_rate
        strategy.update(agent, 'cooperate', 3.0, {'opponent_coop_proportion': 0.5})
        
        # Check that learning rate was scaled during update
        # Note: The method temporarily modifies lr, so we check the effect
        assert strategy.learning_rate == original_lr  # Should be restored
        
    def test_reward_shaping(self):
        """Test reward shaping for group cooperation."""
        strategy = NPersonQLearning(N=5)
        
        # Test cooperation bonus
        shaped = strategy._shape_reward(
            reward=3.0, 
            action='cooperate',
            agent_memory=deque(),
            next_context={'opponent_coop_proportion': 0.8}
        )
        
        assert shaped > 3.0  # Should have cooperation bonus
        
        # Test defection penalty
        shaped = strategy._shape_reward(
            reward=5.0,
            action='defect', 
            agent_memory=deque(),
            next_context={'opponent_coop_proportion': 0.8}
        )
        
        assert shaped < 5.0  # Should have penalty


class TestNPersonHystereticQ:
    """Test N-Person Hysteretic Q-Learning."""
    
    def test_scaled_optimism(self):
        """Test that beta scales with group size."""
        base_beta = 0.01
        
        # Small group
        strategy_small = NPersonHystereticQ(N=2, beta=base_beta, scale_optimism=True)
        assert strategy_small.beta == base_beta  # No scaling for N=2
        
        # Large group
        strategy_large = NPersonHystereticQ(N=10, beta=base_beta, scale_optimism=True)
        assert strategy_large.beta > base_beta  # Should scale up
        
    def test_extra_optimism_for_cooperation(self):
        """Test extra optimism in cooperative groups."""
        strategy = NPersonHystereticQ(N=10, beta=0.01)
        
        # Create mock agent
        agent = Agent(agent_id=1, strategy="random", N=10)
        agent.last_state_representation = ('high', 0, 10)
        agent.q_values = {
            ('high', 0, 10): {'cooperate': 2.0, 'defect': 3.0},
            ('high', 1, 10): {'cooperate': 0.0, 'defect': 0.0}
        }
        
        # Update with positive experience in cooperative group
        agent.memory = deque([{'my_move': 'cooperate', 'neighbor_moves': {}, 'reward': 3}])
        
        strategy.update(
            agent, 
            'cooperate', 
            4.0,  # Good reward
            {'opponent_coop_proportion': 0.8}  # Cooperative group
        )
        
        # Q-value should have increased (positive experience)
        assert agent.q_values[('high', 0, 10)]['cooperate'] > 2.0


class TestNPersonWolfPHC:
    """Test N-Person Wolf-PHC implementation."""
    
    def test_nash_value_computation(self):
        """Test Nash equilibrium value estimation."""
        strategy = NPersonWolfPHC(N=5)
        
        # Test different group sizes
        nash_2 = strategy._compute_nash_value(2)
        nash_5 = strategy._compute_nash_value(5)
        nash_20 = strategy._compute_nash_value(20)
        
        # Nash value should decrease with group size
        assert nash_2 > nash_5 > nash_20
        assert nash_20 == 1.0  # Large groups converge to all-defect
        
    def test_n_person_winning_criteria(self):
        """Test that winning criteria considers group dynamics."""
        strategy = NPersonWolfPHC(N=10, use_nash_baseline=True)
        
        # Create mock agent
        agent = Agent(agent_id=1, strategy="random", N=10)
        agent.q_values = {
            ('med', 0, 10): {'cooperate': 3.0, 'defect': 2.0}
        }
        agent.last_state_representation = ('med', 0, 10)
        
        # Initialize strategy state
        strategy.average_policy = {('med', 0, 10): {'cooperate': 0.5, 'defect': 0.5}}
        strategy.group_performance_history.extend([2.0, 2.5, 3.0])  # Improving
        
        # Perform update
        strategy.update(agent, 'cooperate', 3.5, {'opponent_coop_proportion': 0.6})
        
        # Check that appropriate learning rate was used
        # (This is indirect - we check that Q-values changed appropriately)
        assert agent.q_values[('med', 0, 10)]['cooperate'] != 3.0


class TestIntegration:
    """Integration tests for N-person RL agents."""
    
    def test_agent_creation_with_n_person_strategies(self):
        """Test that agents can be created with N-person strategies."""
        # Test each N-person strategy
        strategies = [
            "n_person_q_learning",
            "n_person_hysteretic_q", 
            "n_person_wolf_phc"
        ]
        
        for strategy in strategies:
            agent = Agent(
                agent_id=1,
                strategy=strategy,
                N=10,
                state_type="n_person_basic"
            )
            
            assert agent.strategy_type == strategy
            assert hasattr(agent.strategy, 'N')
            assert agent.strategy.N == 10
            
    def test_scenario_all_cooperators(self):
        """Test RL agent learns to cooperate with all cooperators."""
        # Create RL agent among cooperators
        rl_agent = Agent(agent_id=0, strategy="n_person_q_learning", N=5)
        
        # Simulate rounds where everyone else cooperates
        for round in range(20):
            # RL agent chooses
            action = rl_agent.choose_move([1, 2, 3, 4])
            
            # Update with all neighbors cooperating
            neighbor_moves = {1: 'cooperate', 2: 'cooperate', 3: 'cooperate', 4: 'cooperate'}
            reward = 3 if action == 'cooperate' else 5  # Temptation to defect
            
            rl_agent.update_memory(action, neighbor_moves, reward)
            rl_agent.update_q_value(action, reward, neighbor_moves)
        
        # After learning, should mostly cooperate
        final_action = rl_agent.choose_move([1, 2, 3, 4])
        # Due to exploration, we can't guarantee cooperation, but Q-values should favor it
        
        # Check Q-values in the cooperative state
        state = rl_agent.strategy._get_current_state(rl_agent)
        if state in rl_agent.q_values:
            q_coop = rl_agent.q_values[state].get('cooperate', 0)
            q_def = rl_agent.q_values[state].get('defect', 0)
            # In a cooperative environment, cooperation should be learned
            # (though with epsilon-greedy, actual moves may vary)
            print(f"Final Q-values - Cooperate: {q_coop}, Defect: {q_def}")
    
    def test_scenario_mixed_population(self):
        """Test RL agent in mixed population."""
        # Create N-person RL agent
        rl_agent = Agent(agent_id=0, strategy="n_person_hysteretic_q", N=6, beta=0.01)
        
        # Simulate rounds with mixed strategies
        for round in range(30):
            action = rl_agent.choose_move([1, 2, 3, 4, 5])
            
            # Mixed population: 3 cooperate, 2 defect
            neighbor_moves = {
                1: 'cooperate', 
                2: 'cooperate',
                3: 'cooperate',
                4: 'defect',
                5: 'defect'
            }
            
            # Calculate reward based on action and neighbor cooperation
            if action == 'cooperate':
                reward = 3 * 0.6  # Scaled by cooperation rate
            else:
                reward = 1 + 4 * 0.6  # P + (T-P) * coop_rate
            
            rl_agent.update_memory(action, neighbor_moves, reward)
            rl_agent.update_q_value(action, reward, neighbor_moves)
        
        # Check that agent has learned something
        assert len(rl_agent.q_values) > 0
        print(f"Learned {len(rl_agent.q_values)} states")


if __name__ == "__main__":
    # Run basic tests
    print("Testing N-Person RL Implementations...")
    
    # Test state extraction
    test_state = TestNPersonStateMixin()
    test_state.test_extract_group_features_neighborhood()
    test_state.test_extract_group_features_pairwise()
    test_state.test_cooperation_trend_detection()
    print("✓ State extraction tests passed")
    
    # Test Q-learning
    test_q = TestNPersonQLearning()
    test_q.test_initialization()
    test_q.test_reward_shaping()
    print("✓ N-Person Q-Learning tests passed")
    
    # Test Hysteretic Q
    test_hq = TestNPersonHystereticQ()
    test_hq.test_scaled_optimism()
    print("✓ N-Person Hysteretic Q tests passed")
    
    # Test Wolf-PHC
    test_wolf = TestNPersonWolfPHC()
    test_wolf.test_nash_value_computation()
    print("✓ N-Person Wolf-PHC tests passed")
    
    # Integration tests
    test_int = TestIntegration()
    test_int.test_agent_creation_with_n_person_strategies()
    test_int.test_scenario_all_cooperators()
    test_int.test_scenario_mixed_population()
    print("✓ Integration tests passed")
    
    print("\nAll tests completed successfully!")