"""
Integration tests for pairwise interaction mode.

This file tests the complete pairwise interaction workflow including
environment setup, round execution, and strategy behaviors.
"""
import pytest
import numpy as np
from npdl.core.agents import Agent
from npdl.core.environment import Environment
from npdl.core.utils import create_payoff_matrix


class TestPairwiseIntegration:
    """Test pairwise interaction mode end-to-end."""

    def test_pairwise_round_stores_specific_moves(self):
        """Test that pairwise round correctly stores specific opponent moves."""
        # Create agents
        agents = [
            Agent(agent_id=0, strategy="always_cooperate"),
            Agent(agent_id=1, strategy="always_defect"),
            Agent(agent_id=2, strategy="tit_for_tat")
        ]
        
        # Create environment in pairwise mode
        payoff_matrix = create_payoff_matrix(len(agents))
        env = Environment(
            agents,
            payoff_matrix,
            network_type="fully_connected",
            interaction_mode="pairwise",
            R=3, S=0, T=5, P=1
        )
        
        # Run a round
        moves, payoffs = env.run_round()
        
        # Check that moves are as expected
        assert moves[0] == "cooperate"  # Always cooperate
        assert moves[1] == "defect"      # Always defect
        assert moves[2] == "cooperate"   # TFT cooperates first round
        
        # Check that memory was updated correctly
        for agent in agents:
            assert len(agent.memory) == 1
            memory_entry = agent.memory[0]
            
            # Check that both aggregate and specific data are stored
            assert "opponent_coop_proportion" in memory_entry["neighbor_moves"]
            assert "specific_opponent_moves" in memory_entry["neighbor_moves"]
            
            # Verify specific opponent moves
            specific_moves = memory_entry["neighbor_moves"]["specific_opponent_moves"]
            if agent.agent_id == 0:  # Always cooperate agent
                assert specific_moves[1] == "defect"
                assert specific_moves[2] == "cooperate"
            elif agent.agent_id == 1:  # Always defect agent
                assert specific_moves[0] == "cooperate"
                assert specific_moves[2] == "cooperate"
            elif agent.agent_id == 2:  # TFT agent
                assert specific_moves[0] == "cooperate"
                assert specific_moves[1] == "defect"

    def test_tft_responds_correctly_in_pairwise(self):
        """Test that TFT responds correctly to specific opponents in pairwise mode."""
        # Create agents
        agents = [
            Agent(agent_id=0, strategy="always_cooperate"),
            Agent(agent_id=1, strategy="always_defect"),
            Agent(agent_id=2, strategy="tit_for_tat")
        ]
        
        # Create environment
        payoff_matrix = create_payoff_matrix(len(agents))
        env = Environment(
            agents,
            payoff_matrix,
            network_type="fully_connected",
            interaction_mode="pairwise",
            R=3, S=0, T=5, P=1
        )
        
        # Round 1
        moves1, _ = env.run_round()
        assert moves1[2] == "cooperate"  # TFT cooperates first
        
        # Round 2 - TFT should defect because agent 1 defected
        moves2, _ = env.run_round()
        assert moves2[2] == "defect"  # TFT defects after seeing defection
        
        # Let's add more agents to test more complex scenarios
        agents.extend([
            Agent(agent_id=3, strategy="always_cooperate"),
            Agent(agent_id=4, strategy="always_cooperate")
        ])
        
        # Recreate environment with more agents
        payoff_matrix = create_payoff_matrix(len(agents))
        env = Environment(
            agents,
            payoff_matrix,
            network_type="fully_connected",
            interaction_mode="pairwise",
            R=3, S=0, T=5, P=1
        )
        
        # Reset TFT agent
        agents[2].reset()
        
        # Round 1 with 5 agents
        moves1, _ = env.run_round()
        assert moves1[2] == "cooperate"  # TFT cooperates first
        
        # Check TFT's view of opponents
        tft_memory = agents[2].memory[0]["neighbor_moves"]
        assert tft_memory["opponent_coop_proportion"] == 0.75  # 3/4 opponents cooperated
        assert len(tft_memory["specific_opponent_moves"]) == 4
        
        # Round 2 - TFT should still defect because one opponent defected
        moves2, _ = env.run_round()
        assert moves2[2] == "defect"

    def test_payoff_calculation_in_pairwise(self):
        """Test that payoffs are calculated correctly in pairwise mode."""
        # Create 3 agents
        agents = [
            Agent(agent_id=0, strategy="always_cooperate"),
            Agent(agent_id=1, strategy="always_cooperate"),
            Agent(agent_id=2, strategy="always_defect")
        ]
        
        # Create environment
        payoff_matrix = create_payoff_matrix(len(agents))
        env = Environment(
            agents,
            payoff_matrix,
            network_type="fully_connected",
            interaction_mode="pairwise",
            R=3, S=0, T=5, P=1
        )
        
        # Run round
        moves, payoffs = env.run_round()
        
        # Agent 0 (cooperator) plays:
        # - vs Agent 1 (cooperator): gets R=3
        # - vs Agent 2 (defector): gets S=0
        # Total: 3 + 0 = 3
        assert payoffs[0] == 3
        
        # Agent 1 (cooperator) plays:
        # - vs Agent 0 (cooperator): gets R=3
        # - vs Agent 2 (defector): gets S=0
        # Total: 3 + 0 = 3
        assert payoffs[1] == 3
        
        # Agent 2 (defector) plays:
        # - vs Agent 0 (cooperator): gets T=5
        # - vs Agent 1 (cooperator): gets T=5
        # Total: 5 + 5 = 10
        assert payoffs[2] == 10

    def test_neighborhood_vs_pairwise_comparison(self):
        """Test differences between neighborhood and pairwise modes."""
        # Create agents
        agents_neighborhood = [
            Agent(agent_id=0, strategy="always_cooperate"),
            Agent(agent_id=1, strategy="always_defect"),
            Agent(agent_id=2, strategy="tit_for_tat"),
            Agent(agent_id=3, strategy="tit_for_tat")
        ]
        
        agents_pairwise = [
            Agent(agent_id=0, strategy="always_cooperate"),
            Agent(agent_id=1, strategy="always_defect"),
            Agent(agent_id=2, strategy="tit_for_tat"),
            Agent(agent_id=3, strategy="tit_for_tat")
        ]
        
        # Create environments
        payoff_matrix = create_payoff_matrix(4)
        
        env_neighborhood = Environment(
            agents_neighborhood,
            payoff_matrix,
            network_type="fully_connected",
            interaction_mode="neighborhood"
        )
        
        env_pairwise = Environment(
            agents_pairwise,
            payoff_matrix,
            network_type="fully_connected",
            interaction_mode="pairwise",
            R=3, S=0, T=5, P=1
        )
        
        # Run simulations
        num_rounds = 10
        for _ in range(num_rounds):
            env_neighborhood.run_round()
            env_pairwise.run_round()
        
        # Compare final states
        # In pairwise mode, TFT agents should defect because of the one defector
        # In neighborhood mode, TFT might cooperate or defect depending on which neighbor they copy
        
        # Check TFT behavior in last round
        last_moves_neighborhood = []
        last_moves_pairwise = []
        
        # Run one more round to see current behavior
        moves_n, _ = env_neighborhood.run_round()
        moves_p, _ = env_pairwise.run_round()
        
        # In pairwise, both TFT agents should defect
        assert moves_p[2] == "defect"
        assert moves_p[3] == "defect"
        
        # In neighborhood, TFT behavior is more variable
        # (they might copy from cooperator or defector)
        print(f"Neighborhood TFT moves: {moves_n[2]}, {moves_n[3]}")
        print(f"Pairwise TFT moves: {moves_p[2]}, {moves_p[3]}")
        
        # The key difference is that pairwise TFT is deterministic
        # while neighborhood TFT has randomness in neighbor selection
