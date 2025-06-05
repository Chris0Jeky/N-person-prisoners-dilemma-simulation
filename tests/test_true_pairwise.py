"""
Comprehensive tests for the true pairwise implementation.

These tests verify that agents can make individual decisions for each opponent
and maintain separate relationships.
"""

import pytest
import numpy as np
from typing import List, Dict, Any

from npdl.core.true_pairwise import (
    OpponentSpecificMemory,
    TruePairwiseTFT,
    TruePairwiseGTFT,
    TruePairwisePavlov,
    TruePairwiseQLearning,
    TruePairwiseAdaptive,
    TruePairwiseEnvironment
)
from npdl.core.true_pairwise_adapter import (
    TruePairwiseAgentAdapter,
    create_true_pairwise_agent,
    TruePairwiseSimulationAdapter
)


class TestOpponentSpecificMemory:
    """Test the opponent-specific memory functionality."""
    
    def test_memory_initialization(self):
        """Test memory is properly initialized."""
        memory = OpponentSpecificMemory("opponent_1", memory_length=5)
        assert memory.opponent_id == "opponent_1"
        assert memory.memory_length == 5
        assert memory.get_cooperation_rate() == 0.5  # Default for unknown
        assert memory.get_last_move() is None
        
    def test_memory_tracking(self):
        """Test memory correctly tracks interactions."""
        memory = OpponentSpecificMemory("opponent_1")
        
        # Add some interactions
        memory.add_interaction("cooperate", "cooperate", 3)
        memory.add_interaction("cooperate", "defect", 0)
        memory.add_interaction("defect", "cooperate", 5)
        
        assert memory.total_interactions == 3
        assert memory.cooperation_count == 2
        assert memory.defection_count == 1
        assert memory.get_cooperation_rate() == 2/3
        assert memory.get_last_move() == "cooperate"
        
    def test_memory_length_limit(self):
        """Test memory respects length limit."""
        memory = OpponentSpecificMemory("opponent_1", memory_length=3)
        
        # Add more interactions than memory length
        for i in range(5):
            memory.add_interaction("cooperate", "cooperate", 3)
            
        assert len(memory.interaction_history) == 3
        assert memory.total_interactions == 5  # Total count still tracked
        
    def test_reciprocity_score(self):
        """Test reciprocity calculation."""
        memory = OpponentSpecificMemory("opponent_1")
        
        # Perfect reciprocity
        memory.add_interaction("cooperate", "defect", 0)  # We cooperated
        memory.add_interaction("defect", "cooperate", 5)  # Opponent reciprocated our cooperation
        memory.add_interaction("cooperate", "defect", 0)  # We cooperated again
        
        score = memory.get_reciprocity_score()
        assert score == 1.0  # Perfect reciprocity
        
        # No reciprocity
        memory.reset()
        memory.add_interaction("cooperate", "defect", 0)
        memory.add_interaction("defect", "defect", 1)  # Opponent didn't reciprocate
        memory.add_interaction("cooperate", "cooperate", 3)  # Opponent didn't reciprocate
        
        score = memory.get_reciprocity_score()
        assert score == 0.0


class TestTruePairwiseAgents:
    """Test individual agent strategies in true pairwise mode."""
    
    def test_tft_opponent_specific(self):
        """Test TFT responds differently to different opponents."""
        tft = TruePairwiseTFT("tft_agent")
        
        # First interaction with both opponents
        assert tft.choose_action_for_opponent("nice_opponent", 0) == "cooperate"
        assert tft.choose_action_for_opponent("mean_opponent", 0) == "cooperate"
        
        # Update memories differently
        tft.update_memory("nice_opponent", "cooperate", "cooperate", 3)
        tft.update_memory("mean_opponent", "cooperate", "defect", 0)
        
        # Second round - should respond differently
        assert tft.choose_action_for_opponent("nice_opponent", 1) == "cooperate"
        assert tft.choose_action_for_opponent("mean_opponent", 1) == "defect"
        
    def test_gtft_generosity(self):
        """Test GTFT is sometimes generous to defectors."""
        np.random.seed(42)
        gtft = TruePairwiseGTFT("gtft_agent", generosity=0.5)
        
        # Opponent defected
        gtft.update_memory("opponent", "cooperate", "defect", 0)
        
        # Count generous responses
        generous_count = 0
        for _ in range(100):
            if gtft.choose_action_for_opponent("opponent", 1) == "cooperate":
                generous_count += 1
                
        # Should be generous roughly 50% of the time
        assert 40 <= generous_count <= 60
        
    def test_pavlov_win_stay_lose_shift(self):
        """Test Pavlov follows win-stay/lose-shift."""
        pavlov = TruePairwisePavlov("pavlov_agent")
        
        # Good outcome (mutual cooperation)
        pavlov.update_memory("opponent", "cooperate", "cooperate", 3)
        assert pavlov.choose_action_for_opponent("opponent", 1) == "cooperate"  # Stay
        
        # Bad outcome (got exploited)
        pavlov.update_memory("opponent", "cooperate", "defect", 0)
        assert pavlov.choose_action_for_opponent("opponent", 2) == "defect"  # Shift
        
        # Good outcome (successful exploitation)
        pavlov.update_memory("opponent", "defect", "cooperate", 5)
        assert pavlov.choose_action_for_opponent("opponent", 3) == "defect"  # Stay
        
    def test_qlearning_separate_qtables(self):
        """Test Q-learning maintains separate Q-tables per opponent."""
        ql = TruePairwiseQLearning("ql_agent", epsilon=0)  # No exploration
        
        # Train against cooperator
        for _ in range(10):
            state = ql.get_state_for_opponent("cooperator")
            action = ql.choose_action_for_opponent("cooperator", 0)
            ql.update_memory("cooperator", action, "cooperate", 3)
            next_state = ql.get_state_for_opponent("cooperator")
            ql.update_q_value("cooperator", state, action, 3, next_state)
            
        # Train against defector
        for _ in range(10):
            state = ql.get_state_for_opponent("defector")
            action = ql.choose_action_for_opponent("defector", 0)
            ql.update_memory("defector", action, "defect", 1)
            next_state = ql.get_state_for_opponent("defector")
            ql.update_q_value("defector", state, action, 1, next_state)
            
        # Check Q-tables are different
        assert "cooperator" in ql.q_tables
        assert "defector" in ql.q_tables
        assert ql.q_tables["cooperator"] != ql.q_tables["defector"]
        
    def test_adaptive_strategy_assessment(self):
        """Test adaptive agent correctly identifies opponent strategies."""
        adaptive = TruePairwiseAdaptive("adaptive_agent", assessment_period=5)
        
        # Simulate always cooperate opponent
        for i in range(10):
            adaptive.update_memory("coop_opponent", "cooperate", "cooperate", 3)
            
        strategy = adaptive.assess_opponent_strategy("coop_opponent")
        assert strategy == "always_cooperate"
        
        # Simulate TFT opponent
        adaptive.reset_episode_memory("tft_opponent")
        moves = ["cooperate", "defect", "cooperate", "defect", "cooperate"]
        for i, my_move in enumerate(moves):
            opponent_move = moves[i-1] if i > 0 else "cooperate"
            adaptive.update_memory("tft_opponent", my_move, opponent_move, 3)
            
        strategy = adaptive.assess_opponent_strategy("tft_opponent")
        assert strategy == "tit_for_tat"


class TestTruePairwiseEnvironment:
    """Test the true pairwise environment functionality."""
    
    def test_environment_setup(self):
        """Test environment is properly set up."""
        agents = [
            TruePairwiseTFT("agent1"),
            TruePairwiseTFT("agent2"),
            TruePairwiseTFT("agent3")
        ]
        
        env = TruePairwiseEnvironment(agents, episodes=1, rounds_per_episode=10)
        assert len(env.agents) == 3
        assert env.episodes == 1
        assert env.rounds_per_episode == 10
        
    def test_single_game(self):
        """Test a single game between two agents."""
        agent1 = TruePairwiseTFT("agent1")
        agent2 = TruePairwiseTFT("agent2")
        
        env = TruePairwiseEnvironment([agent1, agent2])
        result = env.play_single_game("agent1", "agent2", 0)
        
        assert result['agent1_id'] == "agent1"
        assert result['agent2_id'] == "agent2"
        assert result['action1'] == "cooperate"  # TFT starts nice
        assert result['action2'] == "cooperate"
        assert result['payoff1'] == 3
        assert result['payoff2'] == 3
        
    def test_noise_application(self):
        """Test noise is correctly applied."""
        agents = [TruePairwiseTFT("agent1"), TruePairwiseTFT("agent2")]
        env = TruePairwiseEnvironment(agents, noise_level=1.0)  # Always flip
        
        result = env.play_single_game("agent1", "agent2", 0)
        
        # With 100% noise, cooperate should become defect
        assert result['intended_action1'] == "cooperate"
        assert result['action1'] == "defect"
        assert result['noise_applied'] == True
        
    def test_different_responses_to_different_opponents(self):
        """Test agents can cooperate with one opponent while defecting against another."""
        tft = TruePairwiseTFT("tft")
        always_coop = TruePairwiseTFT("cooperator")  # Will always cooperate
        always_defect = TruePairwiseAdaptive("defector")  # We'll make this defect
        
        # Manually set up defector to always defect
        always_defect.choose_action_for_opponent = lambda opp_id, round: "defect"
        
        agents = [tft, always_coop, always_defect]
        env = TruePairwiseEnvironment(agents, rounds_per_episode=3)
        
        # Run a few rounds
        for round_num in range(3):
            round_results = env.run_round(round_num)
            
        # Check TFT's memories
        tft_memory_coop = tft.get_opponent_memory("cooperator")
        tft_memory_defect = tft.get_opponent_memory("defector")
        
        # TFT should have different relationships
        assert tft_memory_coop.get_cooperation_rate() == 1.0  # Cooperator always cooperated
        assert tft_memory_defect.get_cooperation_rate() == 0.0  # Defector always defected
        
        # TFT should respond differently in next round
        assert tft.choose_action_for_opponent("cooperator", 3) == "cooperate"
        assert tft.choose_action_for_opponent("defector", 3) == "defect"


class TestTruePairwiseIntegration:
    """Test integration with existing framework."""
    
    def test_adapter_creation(self):
        """Test creating agents through the adapter."""
        config = {
            'type': 'tit_for_tat',
            'id': 'test_agent'
        }
        
        agent = create_true_pairwise_agent(config)
        assert isinstance(agent, TruePairwiseAgentAdapter)
        assert agent.agent_id == 'test_agent'
        
    def test_true_pairwise_agent_creation(self):
        """Test creating native true pairwise agents."""
        configs = [
            {'type': 'true_pairwise_tft', 'id': 'tft1'},
            {'type': 'true_pairwise_gtft', 'id': 'gtft1', 'generosity': 0.2},
            {'type': 'true_pairwise_pavlov', 'id': 'pavlov1'},
            {'type': 'true_pairwise_q_learning', 'id': 'ql1'},
            {'type': 'true_pairwise_adaptive', 'id': 'adaptive1'}
        ]
        
        for config in configs:
            agent = create_true_pairwise_agent(config)
            assert agent.agent_id == config['id']
            
    def test_simulation_adapter(self):
        """Test the simulation adapter."""
        config = {
            'agents': [
                {'type': 'true_pairwise_tft', 'id': 'agent1'},
                {'type': 'true_pairwise_tft', 'id': 'agent2'},
                {'type': 'true_pairwise_pavlov', 'id': 'agent3'}
            ],
            'rounds': 10,
            'episodes': 2,
            'reset_between_episodes': True
        }
        
        adapter = TruePairwiseSimulationAdapter(config)
        adapter.setup()
        
        assert len(adapter.agents) == 3
        assert adapter.environment is not None
        
        # Run simulation
        results = adapter.run()
        assert 'episodes' in results
        assert 'final_statistics' in results
        assert len(results['episodes']) == 2
        
    def test_results_conversion(self):
        """Test converting results to analysis format."""
        config = {
            'agents': [
                {'type': 'true_pairwise_tft', 'id': 'agent1'},
                {'type': 'true_pairwise_tft', 'id': 'agent2'}
            ],
            'rounds': 5
        }
        
        adapter = TruePairwiseSimulationAdapter(config)
        adapter.run()
        
        converted = adapter.get_results_for_analysis()
        assert 'agents' in converted
        assert 'rounds' in converted
        assert 'network' in converted
        assert converted['network']['interaction_mode'] == 'true_pairwise'


class TestScenarioComparison:
    """Test comparing aggregate vs individual pairwise modes."""
    
    def test_exploitation_difference(self):
        """Test that individual mode allows selective exploitation."""
        # In aggregate mode, TFT must choose one action for all
        # In individual mode, TFT can cooperate with cooperators and defect against defectors
        
        # This would require running both modes and comparing results
        # The individual mode should achieve higher scores when facing mixed opponents
        pass  # Placeholder for integration test
        

if __name__ == "__main__":
    pytest.main([__file__, "-v"])