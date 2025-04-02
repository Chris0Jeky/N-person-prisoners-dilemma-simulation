"""
Tests for the Environment class and its functionality.
"""
import pytest
import networkx as nx
import random
import numpy as np

from npdl.core.agents import Agent
from npdl.core.environment import Environment
from npdl.core.utils import create_payoff_matrix


class TestEnvironmentBasics:
    """Test basic environment functionality."""

    def test_environment_initialization(self, simple_agents):
        """Test that environment initializes correctly."""
        payoff_matrix = create_payoff_matrix(len(simple_agents))
        env = Environment(simple_agents, payoff_matrix, "fully_connected", {})
        
        assert len(env.agents) == len(simple_agents)
        assert env.payoff_matrix == payoff_matrix
        assert env.network_type == "fully_connected"
        assert isinstance(env.graph, nx.Graph)
        assert len(env.history) == 0

    def test_network_creation_fully_connected(self):
        """Test creation of fully connected network."""
        num_agents = 5
        agents = [Agent(agent_id=i, strategy="random") for i in range(num_agents)]
        payoff_matrix = create_payoff_matrix(num_agents)
        
        env = Environment(agents, payoff_matrix, "fully_connected", {})
        
        # Check graph properties
        assert len(env.graph.nodes()) == num_agents
        assert len(env.graph.edges()) == (num_agents * (num_agents - 1)) // 2  # Complete graph edges
        
        # Check network dictionary
        for agent_id, neighbors in env.network.items():
            assert len(neighbors) == num_agents - 1  # All agents except self
            assert agent_id not in neighbors  # No self-loops

    def test_network_creation_small_world(self, seed):
        """Test creation of small world network."""
        num_agents = 10
        agents = [Agent(agent_id=i, strategy="random") for i in range(num_agents)]
        payoff_matrix = create_payoff_matrix(num_agents)
        
        env = Environment(agents, payoff_matrix, "small_world", {"k": 4, "beta": 0.1})
        
        # Check graph properties
        assert len(env.graph.nodes()) == num_agents
        assert nx.is_connected(env.graph)  # Graph should be connected
        
        # Check average clustering and path length (small-world properties)
        avg_clustering = nx.average_clustering(env.graph)
        avg_path_length = nx.average_shortest_path_length(env.graph)
        
        # Small world networks typically have high clustering and low path length
        assert avg_clustering > 0.1
        assert avg_path_length < num_agents / 2

    def test_network_creation_scale_free(self, seed):
        """Test creation of scale-free network."""
        num_agents = 20
        agents = [Agent(agent_id=i, strategy="random") for i in range(num_agents)]
        payoff_matrix = create_payoff_matrix(num_agents)
        
        env = Environment(agents, payoff_matrix, "scale_free", {"m": 2})
        
        # Check graph properties
        assert len(env.graph.nodes()) == num_agents
        assert nx.is_connected(env.graph)  # Graph should be connected
        
        # Check degree distribution (scale-free networks have power-law degree distribution)
        degrees = [d for _, d in env.graph.degree()]
        
        # There should be some high-degree nodes (hubs)
        assert max(degrees) > 5
        
        # And many low-degree nodes
        assert degrees.count(2) + degrees.count(3) > num_agents / 3

    def test_get_neighbors(self, simple_environment, simple_agents):
        """Test getting neighbors for agents."""
        for agent in simple_agents:
            neighbors = simple_environment.get_neighbors(agent.agent_id)
            
            # In a fully connected network, each agent should be connected to all others
            assert len(neighbors) == len(simple_agents) - 1
            assert agent.agent_id not in neighbors  # No self-loops


class TestPayoffCalculation:
    """Test payoff calculation mechanisms."""

    def test_linear_payoff_calculation(self):
        """Test payoff calculation with linear payoff function."""
        # Create environment with known payoff values
        num_agents = 4
        agents = [Agent(agent_id=i, strategy="random") for i in range(num_agents)]
        payoff_matrix = create_payoff_matrix(
            num_agents, 
            payoff_type="linear",
            params={"R": 3, "S": 0, "T": 5, "P": 1}
        )
        
        env = Environment(agents, payoff_matrix, "fully_connected", {})
        
        # Test with all agents cooperating
        all_cooperate = {i: "cooperate" for i in range(num_agents)}
        payoffs = env.calculate_payoffs(all_cooperate, include_global_bonus=False)
        
        # When all cooperate, each agent gets R=3
        for agent_id, payoff in payoffs.items():
            assert payoff == 3.0
        
        # Test with all agents defecting
        all_defect = {i: "defect" for i in range(num_agents)}
        payoffs = env.calculate_payoffs(all_defect, include_global_bonus=False)
        
        # When all defect, each agent gets P=1
        for agent_id, payoff in payoffs.items():
            assert payoff == 1.0
        
        # Test with mixed strategies (2 cooperate, 2 defect)
        mixed_moves = {0: "cooperate", 1: "cooperate", 2: "defect", 3: "defect"}
        payoffs = env.calculate_payoffs(mixed_moves, include_global_bonus=False)
        
        # Cooperating agents get S + (R-S) * (1/3) = 0 + (3-0) * (1/3) = 1.0
        # Defecting agents get P + (T-P) * (2/3) = 1 + (5-1) * (2/3) = 3.67
        assert payoffs[0] == pytest.approx(1.0)
        assert payoffs[1] == pytest.approx(1.0)
        assert payoffs[2] == pytest.approx(3.67, abs=0.01)
        assert payoffs[3] == pytest.approx(3.67, abs=0.01)

    def test_global_bonus(self):
        """Test payoff calculation with global cooperation bonus."""
        num_agents = 5
        agents = [Agent(agent_id=i, strategy="random") for i in range(num_agents)]
        payoff_matrix = create_payoff_matrix(num_agents)
        
        env = Environment(agents, payoff_matrix, "fully_connected", {})
        
        # Test with all agents cooperating
        all_cooperate = {i: "cooperate" for i in range(num_agents)}
        
        # Without global bonus
        payoffs_no_bonus = env.calculate_payoffs(all_cooperate, include_global_bonus=False)
        
        # With global bonus
        payoffs_with_bonus = env.calculate_payoffs(all_cooperate, include_global_bonus=True)
        
        # Global bonus should be coop_rate^2 * 2 = 1.0^2 * 2 = 2.0
        for agent_id in range(num_agents):
            assert payoffs_with_bonus[agent_id] == payoffs_no_bonus[agent_id] + 2.0
        
        # Test with mixed cooperation (60% cooperation)
        mixed_moves = {0: "cooperate", 1: "cooperate", 2: "cooperate", 
                       3: "defect", 4: "defect"}
        
        # Without global bonus
        payoffs_no_bonus = env.calculate_payoffs(mixed_moves, include_global_bonus=False)
        
        # With global bonus
        payoffs_with_bonus = env.calculate_payoffs(mixed_moves, include_global_bonus=True)
        
        # Global bonus should be coop_rate^2 * 2 = 0.6^2 * 2 = 0.72
        # Only cooperating agents get the bonus
        for agent_id in range(3):  # Cooperating agents
            assert payoffs_with_bonus[agent_id] == pytest.approx(payoffs_no_bonus[agent_id] + 0.72, abs=0.01)
        
        for agent_id in range(3, 5):  # Defecting agents
            assert payoffs_with_bonus[agent_id] == payoffs_no_bonus[agent_id]


class TestSimulationRun:
    """Test running simulations."""

    def test_run_round(self, simple_environment):
        """Test running a single round."""
        # Run one round
        moves, payoffs = simple_environment.run_round()
        
        # Check that we get moves and payoffs for all agents
        assert len(moves) == len(simple_environment.agents)
        assert len(payoffs) == len(simple_environment.agents)
        
        # Verify agent scores and memory were updated
        for agent in simple_environment.agents:
            assert agent.score == payoffs[agent.agent_id]
            assert len(agent.memory) == 1
            assert agent.memory[0]['my_move'] == moves[agent.agent_id]
            assert agent.memory[0]['reward'] == payoffs[agent.agent_id]

    def test_run_simulation(self, simple_environment):
        """Test running a full simulation."""
        num_rounds = 5
        results = simple_environment.run_simulation(num_rounds)
        
        # Check that we get results for all rounds
        assert len(results) == num_rounds
        
        # Check that results contain expected data
        for i, round_result in enumerate(results):
            assert round_result['round'] == i
            assert len(round_result['moves']) == len(simple_environment.agents)
            assert len(round_result['payoffs']) == len(simple_environment.agents)
        
        # Verify agent scores are cumulative
        for agent in simple_environment.agents:
            expected_score = sum(results[i]['payoffs'][agent.agent_id] for i in range(num_rounds))
            assert agent.score == pytest.approx(expected_score)
            
            # Verify agent memory (should have most recent rounds)
            memory_length = agent.memory.maxlen
            expected_memory_rounds = min(num_rounds, memory_length)
            assert len(agent.memory) == expected_memory_rounds

    def test_network_rewiring(self, seed):
        """Test dynamic network rewiring."""
        num_agents = 10
        agents = [
            Agent(agent_id=i, strategy="always_cooperate" if i < 5 else "always_defect") 
            for i in range(num_agents)
        ]
        payoff_matrix = create_payoff_matrix(num_agents)
        
        env = Environment(agents, payoff_matrix, "small_world", {"k": 4, "beta": 0.1})
        
        # Store original network structure
        original_edges = set(env.graph.edges())
        
        # Run simulation with network rewiring
        rewiring_prob = 1.0  # 100% rewiring probability for testing
        env.run_simulation(
            5,  # 5 rounds
            rewiring_interval=1,  # Rewire every round
            rewiring_prob=rewiring_prob
        )
        
        # Check that network has changed
        new_edges = set(env.graph.edges())
        assert new_edges != original_edges
        
        # Network should still be connected
        assert nx.is_connected(env.graph)


class TestExportFunctionality:
    """Test exporting environment data."""

    def test_get_network_metrics(self, simple_environment):
        """Test getting network metrics."""
        metrics = simple_environment.get_network_metrics()
        
        # Check that we get the expected metrics
        assert "num_nodes" in metrics
        assert "num_edges" in metrics
        assert "avg_degree" in metrics
        assert "density" in metrics
        
        # For fully connected network
        assert metrics["num_nodes"] == len(simple_environment.agents)
        expected_edges = (metrics["num_nodes"] * (metrics["num_nodes"] - 1)) // 2
        assert metrics["num_edges"] == expected_edges
        assert metrics["density"] == 1.0  # Fully connected

    def test_export_network_structure(self, simple_environment):
        """Test exporting network structure."""
        export_data = simple_environment.export_network_structure()
        
        # Check that we get the expected data
        assert "nodes" in export_data
        assert "edges" in export_data
        assert "network_type" in export_data
        assert "network_params" in export_data
        assert "metrics" in export_data
        
        # Verify data
        assert len(export_data["nodes"]) == len(simple_environment.agents)
        assert export_data["network_type"] == "fully_connected"
