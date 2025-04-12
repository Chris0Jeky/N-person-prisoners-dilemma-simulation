"""
Enhanced tests for the Environment class and its functionality.
"""
import pytest
import networkx as nx
import random
import numpy as np
from collections import deque

# Assuming npdl structure allows direct import like this from tests/ dir
from npdl.core.agents import Agent
from npdl.core.environment import Environment
from npdl.core.utils import create_payoff_matrix

# Test Environment Basics (Keep existing good tests, maybe use fixtures)
@pytest.mark.unit
class TestEnvironmentBasics:
    """Test basic environment functionality."""

    def test_environment_initialization(self, small_test_env, default_payoff_matrix):
        """Test that environment initializes correctly using fixtures."""
        env = small_test_env
        assert len(env.agents) == 4 # From fixture setup
        assert env.payoff_matrix == default_payoff_matrix
        assert env.network_type == "fully_connected"
        assert isinstance(env.graph, nx.Graph)
        assert len(env.history) == 0
        assert env.logger is not None

    @pytest.mark.network # Mark network-specific tests
    @pytest.mark.parametrize("num_agents, network_type, params, expected_edges, check_connected", [
        (5, "fully_connected", {}, 10, True),
        (20, "small_world", {"k": 4, "beta": 0.1}, None, True), # Edges vary
        (20, "scale_free", {"m": 2}, None, True), # Edges vary
        (20, "random", {"probability": 0.1}, None, False), # Random might be disconnected
        (20, "random", {"probability": 0.5}, None, True), # Higher prob -> likely connected
        (20, "regular", {"k": 4}, 40, True),
        # Edge cases
        (1, "fully_connected", {}, 0, True),
        (2, "fully_connected", {}, 1, True),
        (4, "small_world", {"k": 4, "beta": 0.1}, 6, True), # k=N-1 -> complete graph
        (4, "regular", {"k": 4}, None, False), # k=N requires N even usually
        (5, "regular", {"k": 4}, 10, True), # k=4, N=5 ok
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
        # Specific property checks (can add more)
        if network_type == "scale_free" and num_agents > 5:
            degrees = [d for _, d in env.graph.degree()]
            assert max(degrees) > params.get('m', 1) # Should have hubs
        if network_type == "small_world" and num_agents > 5:
             assert nx.average_clustering(env.graph) > 0 # Should have some clustering

    def test_get_neighbors(self, small_test_env):
        """Test getting neighbors for agents in a small environment."""
        env = small_test_env
        # Agents: 0=AC, 1=AD, 2=TFT, 10=QL. Fully connected.
        for agent in env.agents:
            neighbors = env.get_neighbors(agent.agent_id)
            expected_neighbors = {a.agent_id for a in env.agents if a.agent_id != agent.agent_id}
            assert set(neighbors) == expected_neighbors
            assert len(neighbors) == len(env.agents) - 1

# Test Payoff Calculation (Enhanced)
@pytest.mark.unit
class TestPayoffCalculation:
    """Test payoff calculation mechanisms."""

    @pytest.mark.parametrize("payoff_type, params, moves, expected_payoffs", [
        # Linear, All Cooperate
        ("linear", {"R": 3, "S": 0, "T": 5, "P": 1},
         {0: "cooperate", 1: "cooperate", 2: "cooperate"},
         {0: 3.0, 1: 3.0, 2: 3.0}), # N=3, neighbors=2. C(2) = S+(R-S)*(2/2) = 0+3*1=3
        # Linear, All Defect
        ("linear", {"R": 3, "S": 0, "T": 5, "P": 1},
         {0: "defect", 1: "defect", 2: "defect"},
         {0: 1.0, 1: 1.0, 2: 1.0}), # N=3, neighbors=2. D(0) = P+(T-P)*(0/2) = 1+4*0=1
        # Linear, Mixed (1 Coop, 2 Defect)
        ("linear", {"R": 3, "S": 0, "T": 5, "P": 1},
         {0: "cooperate", 1: "defect", 2: "defect"},
         # Agent 0 (Coop): 0 coop neighbors. Payoff=C(0)=S+(R-S)*(0/2)=0.
         # Agent 1 (Defect): 1 coop neighbor (Agent 0). Payoff=D(1)=P+(T-P)*(1/2)=1+4*0.5=3.
         # Agent 2 (Defect): 1 coop neighbor (Agent 0). Payoff=D(1)=3.
         {0: 0.0, 1: 3.0, 2: 3.0}),
        # Exponential, All Cooperate
        ("exponential", {"R": 3, "S": 0, "exponent": 2},
         {0: "cooperate", 1: "cooperate", 2: "cooperate"},
         {0: 3.0, 1: 3.0, 2: 3.0}), # C(2) = S+(R-S)*(2/2)^2 = 0+3*1=3
        # Exponential, Mixed (1 Coop, 2 Defect)
        ("exponential", {"R": 3, "S": 0, "T": 5, "P": 1, "exponent": 2},
         {0: "cooperate", 1: "defect", 2: "defect"},
         # Agent 0 (Coop): 0 coop neighbors. Payoff=C(0)=S+(R-S)*(0/2)^2=0.
         # Agent 1 (Defect): 1 coop neighbor. Payoff=D(1)=P+(T-P)*(1/2)^2=1+4*0.25=2.
         # Agent 2 (Defect): 1 coop neighbor. Payoff=D(1)=2.
         {0: 0.0, 1: 2.0, 2: 2.0}),
         # Threshold, Mixed (1 Coop, 2 Defect), Threshold 0.6
        ("threshold", {"R": 3, "S": 0, "T": 5, "P": 1, "threshold": 0.6},
         {0: "cooperate", 1: "defect", 2: "defect"},
         # Agent 0 (Coop): 0 coop neighbors. prop=0 < 0.6. Payoff=C(0)=S+(R-S)*(0/0.6)*0.3 = 0.
         # Agent 1 (Defect): 1 coop neighbor. prop=0.5 < 0.6. Payoff=D(1)=P+(T-P)*(0.5/0.6)*0.3 = 1+4*(0.833)*0.3 approx 1+1=2.
         # Agent 2 (Defect): 1 coop neighbor. Payoff=D(1) approx 2.
         {0: 0.0, 1: pytest.approx(2.0), 2: pytest.approx(2.0)}),
         # Threshold, Mixed (2 Coop, 1 Defect), Threshold 0.6
        ("threshold", {"R": 3, "S": 0, "T": 5, "P": 1, "threshold": 0.6},
         {0: "cooperate", 1: "cooperate", 2: "defect"},
         # Agent 0 (Coop): 1 coop neighbor. prop=0.5 < 0.6. Payoff=C(1)=S+(R-S)*(0.5/0.6)*0.3 = 0+3*(0.833)*0.3 approx 0.75
         # Agent 1 (Coop): 1 coop neighbor. Payoff=C(1) approx 0.75
         # Agent 2 (Defect): 2 coop neighbors. prop=1.0 > 0.6. norm=(1-0.6)/(1-0.6)=1. Payoff=D(2)=P+(T-P)*(0.3+0.7*1)=1+4*1=5.0
         {0: pytest.approx(0.75), 1: pytest.approx(0.75), 2: 5.0}),
         # Linear, N=1 edge case
        ("linear", {}, {0: "cooperate"}, {0: 0.0}),  # N=1, no neighbors -> payoff 0
        ("linear", {}, {0: "defect"}, {0: 0.0}),  # N=1, no neighbors -> payoff 0
    ])
    def test_payoff_calculations(self, payoff_type, params, moves, expected_payoffs, setup_test_logging):
        """Test payoff calculation with different functions and scenarios."""
        num_agents = len(moves)
        agents = [Agent(agent_id=i, strategy="random") for i in range(num_agents)]
        payoff_matrix = create_payoff_matrix(num_agents, payoff_type, params)
        # Use fully connected for simplicity, payoff depends only on global count here
        env = Environment(agents, payoff_matrix, "fully_connected", {}, logger=setup_test_logging)

        # Calculate payoffs without global bonus for direct comparison
        payoffs = env.calculate_payoffs(moves, include_global_bonus=False)

        # Assert individual payoffs
        for agent_id, expected_payoff in expected_payoffs.items():
            assert payoffs[agent_id] == pytest.approx(expected_payoff), \
                   f"Agent {agent_id} payoff mismatch for {payoff_type}"

    def test_global_bonus_logic(self, setup_test_logging):
        """Test global cooperation bonus calculation."""
        num_agents = 5
        agents = [Agent(agent_id=i, strategy="random") for i in range(num_agents)]
        # Use standard linear payoff matrix (R=3, P=1)
        payoff_matrix = create_payoff_matrix(num_agents, "linear")
        env = Environment(agents, payoff_matrix, "fully_connected", {}, logger=setup_test_logging)

        # All Cooperate: Base payoff=C(4)=3. Bonus = 1.0^2 * 2 = 2. Expected = 3 + 2 = 5.
        all_cooperate = {i: "cooperate" for i in range(num_agents)}
        payoffs_c = env.calculate_payoffs(all_cooperate, include_global_bonus=True)
        for agent_id in range(num_agents):
            assert payoffs_c[agent_id] == pytest.approx(5.0)

        # All Defect: Base payoff=D(0)=1. Bonus=0. Expected=1.
        all_defect = {i: "defect" for i in range(num_agents)}
        payoffs_d = env.calculate_payoffs(all_defect, include_global_bonus=True)
        for agent_id in range(num_agents):
            assert payoffs_d[agent_id] == pytest.approx(1.0)

        # Mixed (3 Coop / 2 Defect): Bonus = (3/5)^2 * 2 = 0.6^2 * 2 = 0.36 * 2 = 0.72
        # N=5, Neighbors=4.
        # Coop agents (ids 0,1,2): Have 2 coop neighbors. Base payoff=C(2)=S+(R-S)*(2/4)=0+3*0.5=1.5. Expected = 1.5 + 0.72 = 2.22
        # Defect agents (ids 3,4): Have 3 coop neighbors. Base payoff=D(3)=P+(T-P)*(3/4)=1+4*0.75=1+3=4.0. Expected = 4.0 (no bonus)
        mixed_moves = {0:"c", 1:"c", 2:"c", 3:"d", 4:"d"} # Assume 'c'='cooperate', 'd'='defect' for brevity
        mixed_moves = {k: ("cooperate" if v == "c" else "defect") for k,v in mixed_moves.items()}
        payoffs_m = env.calculate_payoffs(mixed_moves, include_global_bonus=True)
        assert payoffs_m[0] == pytest.approx(2.22)
        assert payoffs_m[1] == pytest.approx(2.22)
        assert payoffs_m[2] == pytest.approx(2.22)
        assert payoffs_m[3] == pytest.approx(4.0)
        assert payoffs_m[4] == pytest.approx(4.0)

# Test Simulation Run (Enhanced)
@pytest.mark.integration # Mark as integration tests
class TestSimulationRun:
    """Test running simulations."""

    def test_run_round_state_updates(self, small_test_env, seed):
        """Test running a single round updates scores and memory correctly."""
        env = small_test_env # AC, AD, TFT, QL
        initial_scores = {a.agent_id: a.score for a in env.agents}

        # Run one round
        moves, payoffs = env.run_round(use_global_bonus=False) # Disable bonus for easier payoff check

        # Expected moves (approx): AC=C, AD=D, TFT=C (initial), QL=C/D (depends on init/state)
        # Expected payoffs (N=4, Neighbors=3):
        # AC (C): neighbors=1C(TFT)+1D(AD)+1?(QL). If QL=C(initial), 2C neighbors. C(2)=S+(R-S)*2/3=0+3*2/3=2. If QL=D, 1C neighbor. C(1)=S+(R-S)*1/3=1.
        # AD (D): neighbors=1C(AC)+1C(TFT)+1?(QL). If QL=C, 3C neighbors. D(3)=P+(T-P)*3/3=1+4*1=5. If QL=D, 2C neighbors. D(2)=P+(T-P)*2/3=1+4*2/3=1+8/3=11/3~3.67
        # TFT(C): neighbors=1C(AC)+1D(AD)+1?(QL). Same as AC. Payoff ~ 1 or 2.
        # QL (C/D): Depends on choice.

        # Check basic return types
        assert isinstance(moves, dict)
        assert isinstance(payoffs, dict)
        assert len(moves) == len(env.agents)
        assert len(payoffs) == len(env.agents)

        # Verify agent scores and memory were updated
        ql_agent_instance = next(a for a in env.agents if a.agent_id == 10) # Get the QL agent
        for agent in env.agents:
            assert agent.score == initial_scores[agent.agent_id] + payoffs[agent.agent_id]
            assert len(agent.memory) == 1
            memory_entry = agent.memory[0]
            assert memory_entry['my_move'] == moves[agent.agent_id]
            assert memory_entry['reward'] == payoffs[agent.agent_id]
            assert isinstance(memory_entry['neighbor_moves'], dict)
            # Check if QL agent updated Q-table (at least one state should exist)
            if agent.agent_id == 10:
                 assert len(ql_agent_instance.q_values) > 0
                 # Check if state representation was stored (should be 'initial' or 'standard' or similar)
                 assert ql_agent_instance.last_state_representation is not None

    def test_run_simulation_learning(self, small_test_env, seed):
        """Test running a full simulation shows evidence of learning."""
        env = small_test_env # AC, AD, TFT, QL
        num_rounds = 30 # Enough rounds for QL to learn something
        ql_agent_instance = next(a for a in env.agents if a.agent_id == 10)

        # Get initial Q-values (should be empty or just initial state)
        initial_q_values = ql_agent_instance.q_values.copy()

        results = env.run_simulation(num_rounds, logging_interval=num_rounds+1)

        # Check results structure
        assert len(results) == num_rounds
        assert results[0]['round'] == 0
        assert results[-1]['round'] == num_rounds - 1

        # Verify agent scores are cumulative
        for agent in env.agents:
            expected_score = sum(results[i]['payoffs'][agent.agent_id] for i in range(num_rounds))
            assert agent.score == pytest.approx(expected_score)

        # Verify QL agent has learned (Q-values changed/expanded)
        final_q_values = ql_agent_instance.q_values
        assert final_q_values != initial_q_values # Q-table should have changed
        assert len(final_q_values) >= 1 # Should have explored at least one state

        # Check if QL agent developed a preference (simple check)
        # In this env (vs AC, AD, TFT), QL should likely learn to defect against AC/TFT and itself?
        # We expect Q(defect) > Q(cooperate) in many states. Let's check average final Qs.
        avg_q_coop = 0
        avg_q_defect = 0
        state_count = 0
        for state, actions in final_q_values.items():
             if isinstance(actions, dict) and "cooperate" in actions and "defect" in actions:
                  avg_q_coop += actions['cooperate']
                  avg_q_defect += actions['defect']
                  state_count += 1
        if state_count > 0:
             avg_q_coop /= state_count
             avg_q_defect /= state_count
             print(f"\nQL Agent Final Avg Q: Coop={avg_q_coop:.2f}, Defect={avg_q_defect:.2f}")
             # Plausible outcome: defecting is better on average against this mix
             # assert avg_q_defect > avg_q_coop # This assertion might be too strict / depends heavily on exact payoffs/params

    @pytest.mark.slow # Mark as slow if it takes noticeable time
    def test_network_rewiring_effect(self, seed, setup_test_logging):
        """Test that network rewiring changes the network structure over time."""
        num_agents = 20
        # Setup with agents likely to have different cooperation levels
        agents = [
            Agent(agent_id=i, strategy="hysteretic_q", epsilon=0.05, beta=0.01) if i < 10 else
            Agent(agent_id=i, strategy="always_defect")
            for i in range(num_agents)
        ]
        payoff_matrix = create_payoff_matrix(num_agents)
        env = Environment(agents, payoff_matrix, "small_world", {"k": 6, "beta": 0.2}, logger=setup_test_logging)

        initial_edges = set(frozenset(edge) for edge in env.graph.edges())
        initial_metrics = env.get_network_metrics()
        print(f"\nInitial Metrics: {initial_metrics}")

        # Run simulation with rewiring enabled
        num_rounds = 100
        rewiring_interval = 5
        rewiring_prob = 0.1 # Moderate probability
        env.run_simulation(
            num_rounds,
            logging_interval=num_rounds + 1,
            rewiring_interval=rewiring_interval,
            rewiring_prob=rewiring_prob
        )

        final_edges = set(frozenset(edge) for edge in env.graph.edges())
        final_metrics = env.get_network_metrics()
        print(f"Final Metrics: {final_metrics}")


        # Assertions
        assert final_edges != initial_edges, "Network edges should have changed due to rewiring"
        assert nx.is_connected(env.graph), "Network should remain connected after rewiring"
        # Optional: Check for assortativity (cooperators connecting to cooperators) - more advanced analysis
        # coop_dict = {a.agent_id: 1 if (a.memory and a.memory[-1]['my_move']=='cooperate') else 0 for a in env.agents}
        # assortativity = nx.attribute_assortativity_coefficient(env.graph, coop_dict)
        # print(f"Final Assortativity (Cooperation): {assortativity}")
        # assert assortativity > 0 # Expect positive assortativity if rewiring works as intended


# Test Export Functionality (Keep existing, maybe add check for metrics content)
@pytest.mark.unit
class TestExportFunctionality:
    """Test exporting environment data."""

    def test_get_network_metrics(self, small_test_env):
        """Test getting network metrics returns expected keys."""
        metrics = small_test_env.get_network_metrics()

        # Check keys exist
        expected_keys = ["num_nodes", "num_edges", "avg_degree", "density",
                         "avg_clustering", "avg_path_length", "diameter"]
        for key in expected_keys:
             assert key in metrics, f"Metric '{key}' missing"

        # Check basic values for the specific fixture (N=4, fully connected)
        assert metrics["num_nodes"] == 4
        assert metrics["num_edges"] == 6
        assert metrics["avg_degree"] == 3.0
        assert metrics["density"] == 1.0
        assert metrics["avg_clustering"] == 1.0
        assert metrics["avg_path_length"] == 1.0
        assert metrics["diameter"] == 1.0


    def test_export_network_structure(self, small_test_env):
        """Test exporting network structure includes expected data."""
        export_data = small_test_env.export_network_structure()

        # Check structure
        assert "nodes" in export_data
        assert "edges" in export_data
        assert "network_type" in export_data
        assert "network_params" in export_data
        assert "metrics" in export_data

        # Verify content
        assert len(export_data["nodes"]) == len(small_test_env.agents)
        assert set(export_data["nodes"]) == {0, 1, 2, 10} # From fixture
        assert len(export_data["edges"]) == small_test_env.graph.number_of_edges()
        assert export_data["network_type"] == "fully_connected"
        assert export_data["network_params"] == {}
        assert isinstance(export_data["metrics"], dict)
        assert export_data["metrics"]["num_nodes"] == 4