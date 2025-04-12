"""
Tests for the pairwise interaction mode.

This file contains tests that verify the correct functionality of the
pairwise interaction mode in the environment and agent strategies.
"""

import unittest
import random
import numpy as np
from npdl.core.agents import Agent, TitForTatStrategy, GenerousTitForTatStrategy
from npdl.core.environment import Environment
from npdl.core.utils import create_payoff_matrix, get_pairwise_payoffs


class TestPairwiseInteraction(unittest.TestCase):
    
    def setUp(self):
        """Set up agents and environment for testing."""
        # Fix random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Create different types of agents
        self.agents = [
            Agent(agent_id=0, strategy="tit_for_tat"),
            Agent(agent_id=1, strategy="always_cooperate"),
            Agent(agent_id=2, strategy="always_defect"),
            Agent(agent_id=3, strategy="q_learning", epsilon=0.0),  # Deterministic
            Agent(agent_id=4, strategy="lra_q", epsilon=0.0)        # Deterministic
        ]
        
        # Create payoff matrix
        self.payoff_matrix = create_payoff_matrix(
            len(self.agents), 
            payoff_type="linear"
        )
        
        # Create environment with pairwise interaction
        self.env = Environment(
            self.agents,
            self.payoff_matrix,
            network_type="fully_connected",
            interaction_mode="pairwise",
            R=3, S=0, T=5, P=1
        )
    
    def test_tit_for_tat_behavior(self):
        """Test that TitForTat properly responds in pairwise mode."""
        # Run one round to establish first moves
        moves, _ = self.env.run_round()
        
        # TFT should have cooperated on first round
        self.assertEqual(moves[0], "cooperate", "TFT should cooperate on first round")
        
        # Modify TFT's memory to simulate all defection from opponents
        tft_agent = self.agents[0]
        tft_agent.memory.clear()
        tft_agent.memory.append({
            'my_move': 'cooperate',
            'neighbor_moves': {'opponent_coop_proportion': 0.0},  # All defected
            'reward': 0.0
        })
        
        # TFT should now defect
        strategy = TitForTatStrategy()
        move = strategy.choose_move(tft_agent, [])
        self.assertEqual(move, "defect", "TFT should defect after all opponents defected")
        
        # Modify TFT's memory to simulate mixed cooperation
        tft_agent.memory.clear()
        tft_agent.memory.append({
            'my_move': 'cooperate',
            'neighbor_moves': {'opponent_coop_proportion': 0.6},  # Most cooperated
            'reward': 2.0
        })
        
        # TFT should now cooperate
        move = strategy.choose_move(tft_agent, [])
        self.assertEqual(move, "cooperate", "TFT should cooperate when most opponents cooperated")
    
    def test_generous_tit_for_tat_behavior(self):
        """Test that GenerousTitForTat properly responds in pairwise mode."""
        # Create a GTFT agent and strategy
        gtft_agent = Agent(agent_id=5, strategy="generous_tit_for_tat", generosity=0.5)
        strategy = GenerousTitForTatStrategy(generosity=0.5)
        
        # Simulate memory with all defections
        gtft_agent.memory.append({
            'my_move': 'cooperate',
            'neighbor_moves': {'opponent_coop_proportion': 0.0},  # All defected
            'reward': 0.0
        })
        
        # With generosity=0.5, GTFT should forgive ~50% of the time
        # Run multiple times to check probabilistic behavior
        defect_count = 0
        coop_count = 0
        for _ in range(100):
            move = strategy.choose_move(gtft_agent, [])
            if move == "defect":
                defect_count += 1
            else:
                coop_count += 1
        
        # We expect around 50% cooperation due to generosity
        self.assertTrue(0.3 <= coop_count/100 <= 0.7, 
                        "GTFT should cooperate ~50% of the time with generosity=0.5")
    
    def test_q_learning_state_representation(self):
        """Test that Q-learning correctly handles state representations in pairwise mode."""
        q_agent = self.agents[3]  # Q-learning agent
        
        # Simulate memory with mixed cooperation
        q_agent.memory.append({
            'my_move': 'cooperate',
            'neighbor_moves': {'opponent_coop_proportion': 0.6},
            'reward': 2.0
        })
        
        # Get state representation 
        state = q_agent.strategy._get_current_state(q_agent)
        
        # The state should be a tuple containing the discretized proportion
        self.assertIsInstance(state, tuple, "State should be a tuple")
        
        # Run a full round and check the Q-values are updated
        old_q_values = q_agent.q_values.copy()
        self.env.run_round()
        
        # Q-values should have been updated
        self.assertNotEqual(old_q_values, q_agent.q_values, 
                           "Q-values should be updated after a round")
    
    def test_run_pairwise_round(self):
        """Test that _run_pairwise_round produces valid results."""
        # Run a pairwise round
        moves, payoffs = self.env._run_pairwise_round()
        
        # Check that all agents made a move
        self.assertEqual(len(moves), len(self.agents), 
                        "All agents should have made moves")
        
        # Check that all agents received payoffs
        self.assertEqual(len(payoffs), len(self.agents), 
                        "All agents should have received payoffs")
        
        # Check that always_cooperate agent made its expected move
        self.assertEqual(moves[1], "cooperate", 
                        "Always Cooperate agent should have cooperated")
        
        # Check that always_defect agent made its expected move
        self.assertEqual(moves[2], "defect", 
                        "Always Defect agent should have defected")
        
        # Payoffs should be valid
        for agent_id, payoff in payoffs.items():
            self.assertGreaterEqual(payoff, 0, "Payoffs should be non-negative")
    
    def test_get_pairwise_payoffs(self):
        """Test that get_pairwise_payoffs returns the correct values."""
        # Test all four possible combinations
        cc_payoffs = get_pairwise_payoffs("cooperate", "cooperate", 3, 0, 5, 1)
        cd_payoffs = get_pairwise_payoffs("cooperate", "defect", 3, 0, 5, 1)
        dc_payoffs = get_pairwise_payoffs("defect", "cooperate", 3, 0, 5, 1)
        dd_payoffs = get_pairwise_payoffs("defect", "defect", 3, 0, 5, 1)
        
        # Check correct values
        self.assertEqual(cc_payoffs, (3, 3), "C-C should give (R,R) payoffs")
        self.assertEqual(cd_payoffs, (0, 5), "C-D should give (S,T) payoffs")
        self.assertEqual(dc_payoffs, (5, 0), "D-C should give (T,S) payoffs")
        self.assertEqual(dd_payoffs, (1, 1), "D-D should give (P,P) payoffs")


if __name__ == "__main__":
    unittest.main()
