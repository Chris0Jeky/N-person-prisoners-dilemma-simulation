"""
Test suite for TFT ecosystem-aware behavior.

These tests verify that TFT agents respond to the proportion of cooperation
in their ecosystem rather than just mimicking random neighbors.
"""

import pytest
from typing import List, Dict
from npdl.core.agents import Agent
from npdl.core.environment import Environment
import networkx as nx


class TestTFTEcosystemBehavior:
    """Test TFT behavior based on ecosystem cooperation proportion."""
    
    def test_tft_cooperates_in_cooperative_neighborhood(self):
        """TFT should cooperate when majority of neighbors cooperate."""
        # Create a small fully connected network
        agents = [
            Agent(agent_id="tft", strategy="tit_for_tat", cooperation_threshold=0.5),
            Agent(agent_id="coop1", strategy="always_cooperate"),
            Agent(agent_id="coop2", strategy="always_cooperate"),
            Agent(agent_id="defect1", strategy="always_defect"),
        ]
        
        # Create fully connected network
        network = nx.complete_graph(4)
        env = Environment(agents, network, interaction_mode="neighborhood")
        
        # Run one round to establish history
        env.run(rounds=1)
        
        # Get TFT agent's last round info
        tft_agent = agents[0]
        last_round = tft_agent.memory[-1]
        neighbor_moves = last_round['neighbor_moves']
        
        # Verify TFT saw 2 cooperators and 1 defector (66.7% cooperation)
        coop_count = sum(1 for move in neighbor_moves.values() if move == "cooperate")
        assert coop_count == 2
        assert len(neighbor_moves) == 3
        
        # Run another round - TFT should cooperate with high cooperation proportion
        env.run(rounds=1)
        tft_second_round = tft_agent.memory[-1]
        assert tft_second_round['my_move'] == "cooperate"
    
    def test_tft_defects_in_defective_neighborhood(self):
        """TFT should defect when majority of neighbors defect."""
        agents = [
            Agent(agent_id="tft", strategy="tit_for_tat", cooperation_threshold=0.5),
            Agent(agent_id="defect1", strategy="always_defect"),
            Agent(agent_id="defect2", strategy="always_defect"),
            Agent(agent_id="coop1", strategy="always_cooperate"),
        ]
        
        network = nx.complete_graph(4)
        env = Environment(agents, network, interaction_mode="neighborhood")
        
        # Run one round to establish history
        env.run(rounds=1)
        
        # Get TFT agent's info
        tft_agent = agents[0]
        last_round = tft_agent.memory[-1]
        neighbor_moves = last_round['neighbor_moves']
        
        # Verify TFT saw 1 cooperator and 2 defectors (33.3% cooperation)
        coop_count = sum(1 for move in neighbor_moves.values() if move == "cooperate")
        assert coop_count == 1
        assert len(neighbor_moves) == 3
        
        # Run another round - TFT should defect with low cooperation proportion
        env.run(rounds=1)
        tft_second_round = tft_agent.memory[-1]
        assert tft_second_round['my_move'] == "defect"
    
    def test_tft_threshold_behavior(self):
        """Test TFT behavior at different cooperation thresholds."""
        # Test with 50% cooperation - should cooperate with default 0.5 threshold
        agents = [
            Agent(agent_id="tft", strategy="tit_for_tat", cooperation_threshold=0.5),
            Agent(agent_id="coop1", strategy="always_cooperate"),
            Agent(agent_id="defect1", strategy="always_defect"),
        ]
        
        network = nx.complete_graph(3)
        env = Environment(agents, network, interaction_mode="neighborhood")
        
        # Run two rounds
        env.run(rounds=2)
        
        tft_agent = agents[0]
        # At exactly 50% cooperation, TFT should cooperate (threshold is inclusive)
        assert tft_agent.memory[-1]['my_move'] == "cooperate"
        
        # Test with higher threshold - should defect at 50% cooperation
        agents_strict = [
            Agent(agent_id="tft_strict", strategy="tit_for_tat", cooperation_threshold=0.6),
            Agent(agent_id="coop1", strategy="always_cooperate"),
            Agent(agent_id="defect1", strategy="always_defect"),
        ]
        
        env_strict = Environment(agents_strict, nx.complete_graph(3), interaction_mode="neighborhood")
        env_strict.run(rounds=2)
        
        tft_strict = agents_strict[0]
        # With 50% cooperation and 60% threshold, TFT should defect
        assert tft_strict.memory[-1]['my_move'] == "defect"
    
    def test_tft_small_world_vs_fully_connected(self):
        """Test TFT behavior differs based on network topology."""
        # Create 6 agents: 1 TFT, 3 cooperators, 2 defectors
        agents_template = [
            Agent(agent_id="tft", strategy="tit_for_tat", cooperation_threshold=0.5),
            Agent(agent_id="coop1", strategy="always_cooperate"),
            Agent(agent_id="coop2", strategy="always_cooperate"),
            Agent(agent_id="coop3", strategy="always_cooperate"),
            Agent(agent_id="defect1", strategy="always_defect"),
            Agent(agent_id="defect2", strategy="always_defect"),
        ]
        
        # Test 1: Fully connected - TFT sees all agents
        agents_fc = [Agent(a.agent_id, a.strategy_type, cooperation_threshold=0.5) 
                     for a in agents_template]
        network_fc = nx.complete_graph(6)
        env_fc = Environment(agents_fc, network_fc, interaction_mode="neighborhood")
        env_fc.run(rounds=2)
        
        tft_fc = agents_fc[0]
        fc_neighbors = tft_fc.memory[-2]['neighbor_moves']  # Second-to-last round
        assert len(fc_neighbors) == 5  # TFT sees all 5 other agents
        
        # In fully connected, TFT sees 3 cooperators and 2 defectors (60% cooperation)
        # So it should cooperate
        assert tft_fc.memory[-1]['my_move'] == "cooperate"
        
        # Test 2: Small world where TFT only connects to defectors
        agents_sw = [Agent(a.agent_id, a.strategy_type, cooperation_threshold=0.5) 
                     for a in agents_template]
        network_sw = nx.Graph()
        network_sw.add_edges_from([
            (0, 4),  # TFT to defect1
            (0, 5),  # TFT to defect2
            (1, 2),  # coop1 to coop2
            (2, 3),  # coop2 to coop3
            (3, 1),  # coop3 to coop1
            (4, 5),  # defect1 to defect2
        ])
        
        env_sw = Environment(agents_sw, network_sw, interaction_mode="neighborhood")
        env_sw.run(rounds=2)
        
        tft_sw = agents_sw[0]
        sw_neighbors = tft_sw.memory[-2]['neighbor_moves']
        assert len(sw_neighbors) == 2  # TFT only sees 2 neighbors
        
        # In small world, TFT only sees defectors (0% cooperation)
        # So it should defect
        assert tft_sw.memory[-1]['my_move'] == "defect"
    
    def test_proportional_tft_behavior(self):
        """Test ProportionalTitForTat probabilistic behavior."""
        # Run multiple trials to test probabilistic behavior
        cooperation_results = []
        
        for _ in range(100):  # Run 100 trials
            agents = [
                Agent(agent_id="ptft", strategy="proportional_tit_for_tat"),
                Agent(agent_id="coop1", strategy="always_cooperate"),
                Agent(agent_id="coop2", strategy="always_cooperate"),
                Agent(agent_id="defect1", strategy="always_defect"),
            ]
            
            network = nx.complete_graph(4)
            env = Environment(agents, network, interaction_mode="neighborhood")
            env.run(rounds=2)
            
            ptft_agent = agents[0]
            # PTFT saw 2/3 cooperation (66.7%)
            # So it should cooperate ~66.7% of the time
            cooperation_results.append(ptft_agent.memory[-1]['my_move'] == "cooperate")
        
        # Check that cooperation rate is approximately 66.7% (allow some variance)
        cooperation_rate = sum(cooperation_results) / len(cooperation_results)
        assert 0.57 < cooperation_rate < 0.77  # Within 10% of expected 0.667
    
    def test_tft_pairwise_mode_compatibility(self):
        """Verify TFT still works correctly in pairwise mode."""
        agents = [
            Agent(agent_id="tft", strategy="tit_for_tat"),
            Agent(agent_id="coop", strategy="always_cooperate"),
            Agent(agent_id="defect", strategy="always_defect"),
        ]
        
        network = nx.complete_graph(3)
        env = Environment(agents, network, interaction_mode="pairwise")
        env.run(rounds=2)
        
        tft_agent = agents[0]
        last_round = tft_agent.memory[-1]
        
        # In pairwise mode, TFT should use aggregate cooperation proportion
        assert 'neighbor_moves' in last_round
        neighbor_info = last_round['neighbor_moves']
        
        # Should have opponent_coop_proportion in pairwise mode
        assert 'opponent_coop_proportion' in neighbor_info
        
        # With 1 cooperator and 1 defector, proportion should be 0.5
        # TFT should cooperate at exactly 50% with default threshold
        assert tft_agent.memory[-1]['my_move'] == "cooperate"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])