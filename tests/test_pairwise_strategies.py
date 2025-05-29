"""
Tests for pairwise interaction mode and strategy behaviors.

This file contains unit tests specifically for verifying that reactive strategies
(TFT variants) correctly use specific opponent history in pairwise mode.
"""
import pytest
import random
from collections import deque

from npdl.core.agents import Agent
from npdl.core.agents import (
    TitForTatStrategy, GenerousTitForTatStrategy, 
    SuspiciousTitForTatStrategy, TitForTwoTatsStrategy
)


class TestPairwiseTitForTatStrategies:
    """Test TFT strategy variants in pairwise interaction mode."""

    def test_tft_cooperates_first_round(self):
        """Test that TFT cooperates on the first round."""
        agent = Agent(agent_id=0, strategy="tit_for_tat")
        assert agent.choose_move([]) == "cooperate"

    def test_tft_defects_if_any_opponent_defected(self):
        """Test that TFT defects if ANY opponent defected in pairwise mode."""
        agent = Agent(agent_id=0, strategy="tit_for_tat")
        
        # Set up memory with specific opponent moves
        neighbor_moves = {
            "opponent_coop_proportion": 0.67,  # 2/3 cooperated
            "specific_opponent_moves": {
                1: "cooperate",
                2: "defect",    # One defector
                3: "cooperate"
            }
        }
        agent.update_memory("cooperate", neighbor_moves, 2.0)
        
        # TFT should defect because opponent 2 defected
        assert agent.choose_move([]) == "defect"

    def test_tft_defects_if_any_opponent_defected(self):
        """Test that TFT defects if ANY opponent defected in pairwise mode."""
        agent = Agent(agent_id=0, strategy="tit_for_tat")
        
        # Set up memory with specific opponent moves
        neighbor_moves = {
            "opponent_coop_proportion": 0.67,  # 2/3 cooperated
            "specific_opponent_moves": {
                1: "cooperate",
                2: "defect",    # One defector
                3: "cooperate"
            }
        }
        agent.update_memory("cooperate", neighbor_moves, 2.0)
        
        # TFT should defect because opponent 2 defected
        assert agent.choose_move([]) == "defect"

    def test_tft_cooperates_if_all_opponents_cooperated(self):
        """Test that TFT cooperates only if ALL opponents cooperated."""
        agent = Agent(agent_id=0, strategy="tit_for_tat")
        
        # Set up memory with all cooperators
        neighbor_moves = {
            "opponent_coop_proportion": 1.0,
            "specific_opponent_moves": {
                1: "cooperate",
                2: "cooperate",
                3: "cooperate"
            }
        }
        agent.update_memory("cooperate", neighbor_moves, 3.0)
        
        # TFT should cooperate because all opponents cooperated
        assert agent.choose_move([]) == "cooperate"

    def test_tft_fallback_to_proportion(self):
        """Test TFT fallback behavior when specific moves not available."""
        agent = Agent(agent_id=0, strategy="tit_for_tat")
        
        # Set up memory with only proportion (backward compatibility)
        neighbor_moves = {
            "opponent_coop_proportion": 0.99  # Almost all cooperated
        }
        agent.update_memory("cooperate", neighbor_moves, 3.0)
        
        # Should cooperate when proportion >= 0.99
        assert agent.choose_move([]) == "cooperate"
        
        # Now test with lower proportion
        neighbor_moves = {"opponent_coop_proportion": 0.8}
        agent.update_memory("cooperate", neighbor_moves, 2.5)
        
        # Should defect when proportion < 0.99
        assert agent.choose_move([]) == "defect"

    def test_generous_tft_sometimes_cooperates_after_defection(self):
        """Test that Generous TFT sometimes forgives defection."""
        agent = Agent(agent_id=0, strategy="generous_tit_for_tat", generosity=0.5)
        
        # Set up memory with one defector
        neighbor_moves = {
            "opponent_coop_proportion": 0.67,
            "specific_opponent_moves": {
                1: "cooperate",
                2: "defect",
                3: "cooperate"
            }
        }
        agent.update_memory("cooperate", neighbor_moves, 2.0)
        
        # With generosity=0.5, should cooperate about 50% of the time
        cooperate_count = 0
        for _ in range(100):
            if agent.strategy.choose_move(agent, []) == "cooperate":
                cooperate_count += 1
        
        # Should be roughly 50% cooperation (allow for randomness)
        assert 30 <= cooperate_count <= 70

    def test_suspicious_tft_starts_with_defection(self):
        """Test that Suspicious TFT defects on first round."""
        agent = Agent(agent_id=0, strategy="suspicious_tit_for_tat")
        assert agent.choose_move([]) == "defect"

    def test_suspicious_tft_requires_all_cooperation(self):
        """Test that Suspicious TFT only cooperates if ALL cooperated."""
        agent = Agent(agent_id=0, strategy="suspicious_tit_for_tat")
        
        # Test with all cooperators
        neighbor_moves = {
            "opponent_coop_proportion": 1.0,
            "specific_opponent_moves": {
                1: "cooperate",
                2: "cooperate"
            }
        }
        agent.update_memory("defect", neighbor_moves, 3.0)
        assert agent.choose_move([]) == "cooperate"
        
        # Test with one defector
        neighbor_moves = {
            "opponent_coop_proportion": 0.5,
            "specific_opponent_moves": {
                1: "cooperate",
                2: "defect"
            }
        }
        agent.update_memory("cooperate", neighbor_moves, 1.5)
        assert agent.choose_move([]) == "defect"

    def test_tit_for_two_tats_requires_two_defections(self):
        """Test TF2T requires opponent to defect twice in a row."""
        agent = Agent(agent_id=0, strategy="tit_for_two_tats")
        
        # First two rounds - cooperate by default
        assert agent.choose_move([]) == "cooperate"
        
        # Round 1: Opponent 1 defects, opponent 2 cooperates
        neighbor_moves1 = {
            "opponent_coop_proportion": 0.5,
            "specific_opponent_moves": {
                1: "defect",
                2: "cooperate"
            }
        }
        agent.update_memory("cooperate", neighbor_moves1, 1.5)
        
        # Round 2: Opponent 1 cooperates, opponent 2 defects
        neighbor_moves2 = {
            "opponent_coop_proportion": 0.5,
            "specific_opponent_moves": {
                1: "cooperate",
                2: "defect"
            }
        }
        agent.update_memory("cooperate", neighbor_moves2, 1.5)
        
        # No opponent defected twice in a row - should cooperate
        assert agent.choose_move([]) == "cooperate"
        
        # Round 3: Opponent 1 defects again (now twice in a row)
        neighbor_moves3 = {
            "opponent_coop_proportion": 0.5,
            "specific_opponent_moves": {
                1: "defect",
                2: "cooperate"
            }
        }
        agent.update_memory("cooperate", neighbor_moves3, 1.5)
        
        # Opponent 1 defected in rounds 1 and 3 (not consecutive from agent's perspective)
        # Actually, we need to check the last TWO rounds from agent's current memory
        # Let me fix this test...
        
        # Clear and rebuild memory for a cleaner test
        agent.memory.clear()
        
        # Round 1: Opponent 1 defects
        neighbor_moves1 = {
            "opponent_coop_proportion": 0.5,
            "specific_opponent_moves": {1: "defect", 2: "cooperate"}
        }
        agent.update_memory("cooperate", neighbor_moves1, 1.5)
        
        # Round 2: Opponent 1 defects again
        neighbor_moves2 = {
            "opponent_coop_proportion": 0.5,
            "specific_opponent_moves": {1: "defect", 2: "cooperate"}
        }
        agent.update_memory("cooperate", neighbor_moves2, 1.5)
        
        # Now opponent 1 has defected twice in a row - should defect
        assert agent.choose_move([]) == "defect"

    def test_tft_with_new_opponent(self):
        """Test TFT behavior when opponents change between rounds."""
        agent = Agent(agent_id=0, strategy="tit_for_tat")
        
        # Round 1: Play against opponents 1 and 2
        neighbor_moves1 = {
            "opponent_coop_proportion": 0.5,
            "specific_opponent_moves": {
                1: "defect",
                2: "cooperate"
            }
        }
        agent.update_memory("cooperate", neighbor_moves1, 1.5)
        
        # Should defect because opponent 1 defected
        assert agent.choose_move([]) == "defect"
        
        # Round 2: Play against opponents 3 and 4 (different opponents)
        neighbor_moves2 = {
            "opponent_coop_proportion": 1.0,
            "specific_opponent_moves": {
                3: "cooperate",
                4: "cooperate"
            }
        }
        agent.update_memory("defect", neighbor_moves2, 3.0)
        
        # Should cooperate because current opponents all cooperated
        assert agent.choose_move([]) == "cooperate"

    def test_strategies_with_empty_opponent_list(self):
        """Test strategy behavior when no opponents (edge case)."""
        strategies = [
            ("tit_for_tat", "cooperate"),
            ("generous_tit_for_tat", "cooperate"),
            ("suspicious_tit_for_tat", "defect"),
            ("tit_for_two_tats", "cooperate")
        ]
        
        for strategy_name, expected_first_move in strategies:
            agent = Agent(agent_id=0, strategy=strategy_name)
            
            # First move with no memory
            assert agent.choose_move([]) == expected_first_move
            
            # After a round with no opponents
            neighbor_moves = {
                "opponent_coop_proportion": 0.5,  # Default
                "specific_opponent_moves": {}      # Empty
            }
            agent.update_memory(expected_first_move, neighbor_moves, 0)
            
            # Check behavior with empty opponent list
            if strategy_name == "tit_for_tat":
                assert agent.choose_move([]) == "cooperate"
            elif strategy_name == "generous_tit_for_tat":
                assert agent.choose_move([]) == "cooperate"
            elif strategy_name == "suspicious_tit_for_tat":
                assert agent.choose_move([]) == "defect"
            elif strategy_name == "tit_for_two_tats":
                assert agent.choose_move([]) == "cooperate"
