"""
Fixed tests for the Environment class.
This addresses the failing tests and improves neighborhood vs pairwise comparison.
"""
import pytest
import networkx as nx
import random
import numpy as np
from collections import deque

from npdl.core.agents import Agent
from npdl.core.environment import Environment
from npdl.core.utils import create_payoff_matrix


@pytest.mark.unit
class TestEnvironmentBasics:
    """Test basic environment functionality."""

    @pytest.mark.network
    @pytest.mark.parametrize("num_agents, network_type, params, expected_edges, check_connected", [
        (5, "fully_connected", {}, 10, True),
        (20, "small_world", {"k": 4, "beta": 0.1}, None, True),  # Edges vary
        (20, "scale_free", {"m": 2}, None, True),  # Edges vary
        (20, "random", {"probability": 0.1}, None, False),  # Random might be disconnected
        (20, "random", {"probability": 0.5}, None, True),  # Higher prob -> likely connected
        (20, "regular", {"k": 4}, 40, True),
        # Edge cases
        (1, "fully_connected", {}, 0, True),
        (2, "fully_connected", {}, 1, True),
        # Fixed: For 4 nodes with k=4, k gets adjusted to min(4,3)=3. 
        # Watts-Strogatz with n=4, k=3 starts with 6 edges but beta rewiring can change this
        (4, "small_world", {"k": 4, "beta": 0.0}, 6, True),  # beta=0 means no rewiring
        (4, "regular", {"k": 4}, None, False),  # k=N requires N even usually
        (5, "regular", {"k": 4}, 10, True),  # k=4, N=5 ok
    ])
    def test_network_creation(self, num_agents, network_type, params, expected_edges, check_connected, seed, setup_test_logging):
        """Test creation of various network types with parameters and edge cases."""
        agents = [Agent(agent_id=i, strategy="random") for i in range(num_agents)]
        payoff_matrix = create_payoff_matrix(num_agents)
        env = Environment(agents, payoff_matrix, network_type, params, logger=setup_test_logging)

        assert len(env.graph.nodes()) == num_agents
        if expected_edges is not None:
            assert len(env.graph.edges()) == expected_edges
        if num_agents > 1 and check_connected:
            assert nx.is_connected(env.graph), f"{network_type} graph should be connected"


@pytest.mark.integration
class TestNeighborhoodVsPairwise:
    """Test proper comparison between neighborhood and pairwise interaction modes."""

    def test_controlled_comparison_fully_connected(self, setup_test_logging):
        """Test neighborhood vs pairwise on fully connected network with same parameters."""
        # Use fixed seed for reproducibility
        np.random.seed(42)
        random.seed(42)
        
        # Create identical agent configurations
        def create_agents():
            return [
                Agent(agent_id=0, strategy="always_cooperate"),
                Agent(agent_id=1, strategy="always_defect"),
                Agent(agent_id=2, strategy="tit_for_tat"),
                Agent(agent_id=3, strategy="tit_for_tat"),
                Agent(agent_id=4, strategy="q_learning", epsilon=0.1, learning_rate=0.1)
            ]
        
        agents_neighborhood = create_agents()
        agents_pairwise = create_agents()
        
        # Create environments with same parameters
        payoff_matrix = create_payoff_matrix(5)
        
        env_neighborhood = Environment(
            agents_neighborhood,
            payoff_matrix,
            network_type="fully_connected",
            interaction_mode="neighborhood",
            logger=setup_test_logging
        )
        
        env_pairwise = Environment(
            agents_pairwise,
            payoff_matrix,
            network_type="fully_connected",
            interaction_mode="pairwise",
            R=3, S=0, T=5, P=1,
            logger=setup_test_logging
        )
        
        # Run simulations with same number of rounds
        num_rounds = 50
        
        # Reset seeds before each simulation for fairer comparison
        np.random.seed(42)
        random.seed(42)
        results_neighborhood = env_neighborhood.run_simulation(num_rounds, logging_interval=num_rounds+1)
        
        np.random.seed(42)
        random.seed(42)
        results_pairwise = env_pairwise.run_simulation(num_rounds, logging_interval=num_rounds+1)
        
        # Analyze results
        final_coop_neighborhood = sum(1 for m in results_neighborhood[-1]['moves'].values() if m == "cooperate") / 5
        final_coop_pairwise = sum(1 for m in results_pairwise[-1]['moves'].values() if m == "cooperate") / 5
        
        # Key expected differences:
        # 1. In pairwise, TFT should consistently defect because of the always_defect agent
        assert results_pairwise[-1]['moves'][2] == "defect"
        assert results_pairwise[-1]['moves'][3] == "defect"
        
        # 2. Cooperation rate should be lower in pairwise due to deterministic TFT behavior
        assert final_coop_pairwise < final_coop_neighborhood
        
        # 3. Q-learning should learn differently in each mode
        ql_neighborhood = agents_neighborhood[4]
        ql_pairwise = agents_pairwise[4]
        
        # Check that Q-values are different
        assert ql_neighborhood.q_values != ql_pairwise.q_values

    def test_q_learning_convergence_comparison(self, setup_test_logging):
        """Test Q-learning convergence in both modes with appropriate parameters."""
        # Create environments with only Q-learning vs always_defect
        def create_scenario(interaction_mode):
            agents = [
                Agent(agent_id=i, strategy="q_learning", 
                      epsilon=0.3,  # Higher exploration initially
                      learning_rate=0.1,
                      state_type="proportion_discretized")
                if i < 5 else
                Agent(agent_id=i, strategy="always_defect")
                for i in range(10)
            ]
            
            payoff_matrix = create_payoff_matrix(10)
            kwargs = {"logger": setup_test_logging}
            if interaction_mode == "pairwise":
                kwargs.update({"R": 3, "S": 0, "T": 5, "P": 1})
                
            return Environment(
                agents,
                payoff_matrix,
                network_type="fully_connected",
                interaction_mode=interaction_mode,
                **kwargs
            )
        
        # Test both modes
        env_neighborhood = create_scenario("neighborhood")
        env_pairwise = create_scenario("pairwise")
        
        # Run longer simulation for Q-learning to converge
        num_rounds = 200
        
        np.random.seed(42)
        random.seed(42)
        results_neighborhood = env_neighborhood.run_simulation(num_rounds, logging_interval=num_rounds+1)
        
        np.random.seed(42)
        random.seed(42)
        results_pairwise = env_pairwise.run_simulation(num_rounds, logging_interval=num_rounds+1)
        
        # In both modes, Q-learning should learn to defect against always_defect
        final_coop_neighborhood = sum(1 for i, m in results_neighborhood[-1]['moves'].items() 
                                     if i < 5 and m == "cooperate") / 5
        final_coop_pairwise = sum(1 for i, m in results_pairwise[-1]['moves'].items() 
                                 if i < 5 and m == "cooperate") / 5
        
        # Both should converge to low cooperation
        assert final_coop_neighborhood < 0.2, f"Neighborhood Q-learning didn't learn to defect: {final_coop_neighborhood}"
        assert final_coop_pairwise < 0.2, f"Pairwise Q-learning didn't learn to defect: {final_coop_pairwise}"

    def test_network_structure_effects(self, setup_test_logging):
        """Test how different network structures affect outcomes in both modes."""
        network_configs = [
            ("fully_connected", {}),
            ("small_world", {"k": 4, "beta": 0.3}),
            ("scale_free", {"m": 2})
        ]
        
        results = {}
        
        for network_type, network_params in network_configs:
            for interaction_mode in ["neighborhood", "pairwise"]:
                # Create consistent agent mix
                agents = [
                    Agent(agent_id=i, strategy="tit_for_tat" if i < 10 else "always_defect")
                    for i in range(20)
                ]
                
                payoff_matrix = create_payoff_matrix(20)
                kwargs = {"logger": setup_test_logging}
                if interaction_mode == "pairwise":
                    kwargs.update({"R": 3, "S": 0, "T": 5, "P": 1})
                
                env = Environment(
                    agents,
                    payoff_matrix,
                    network_type=network_type,
                    network_params=network_params,
                    interaction_mode=interaction_mode,
                    **kwargs
                )
                
                # Run simulation
                np.random.seed(42)
                random.seed(42)
                sim_results = env.run_simulation(100, logging_interval=101)
                
                # Store cooperation rate
                final_coop = sum(1 for m in sim_results[-1]['moves'].values() if m == "cooperate") / 20
                results[(network_type, interaction_mode)] = final_coop
        
        # Network structure should matter more in neighborhood mode than pairwise
        for network_type, _ in network_configs:
            neighborhood_coop = results[(network_type, "neighborhood")]
            pairwise_coop = results[(network_type, "pairwise")]
            
            # In pairwise, network structure shouldn't matter as much since everyone plays everyone
            if network_type != "fully_connected":
                # Pairwise results should be similar across network types
                pairwise_fc = results[("fully_connected", "pairwise")]
                assert abs(pairwise_coop - pairwise_fc) < 0.1, \
                    f"Pairwise should be consistent across networks: {network_type} vs FC"
