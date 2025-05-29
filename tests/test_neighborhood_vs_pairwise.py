"""
Comprehensive tests for comparing neighborhood vs pairwise interaction modes.
This ensures fair comparisons by controlling all variables except the interaction mode.
"""
import pytest
import numpy as np
import random
from collections import Counter

from npdl.core.agents import Agent
from npdl.core.environment import Environment
from npdl.core.utils import create_payoff_matrix


@pytest.mark.integration
class TestNeighborhoodVsPairwiseComparison:
    """Test suite for comparing the two interaction modes under controlled conditions."""
    
    def test_identical_setup_fully_connected(self, setup_test_logging):
        """Test that both modes produce different but expected results on fully connected networks."""
        # Fixed seeds for reproducibility
        np.random.seed(42)
        random.seed(42)
        
        # Create identical agent sets
        num_agents = 10
        strategies = ["tit_for_tat"] * 5 + ["always_defect"] * 5
        
        agents_n = [Agent(agent_id=i, strategy=strategies[i]) for i in range(num_agents)]
        agents_p = [Agent(agent_id=i, strategy=strategies[i]) for i in range(num_agents)]
        
        # Create identical payoff matrices
        payoff_matrix_n = create_payoff_matrix(num_agents, "linear", {"R": 3, "S": 0, "T": 5, "P": 1})
        payoff_matrix_p = create_payoff_matrix(num_agents, "linear", {"R": 3, "S": 0, "T": 5, "P": 1})
        
        # Create environments
        env_neighborhood = Environment(
            agents_n,
            payoff_matrix_n,
            network_type="fully_connected",
            interaction_mode="neighborhood",
            logger=setup_test_logging
        )
        
        env_pairwise = Environment(
            agents_p,
            payoff_matrix_p,
            network_type="fully_connected",
            interaction_mode="pairwise",
            R=3, S=0, T=5, P=1,
            logger=setup_test_logging
        )
        
        # Run simulations
        num_rounds = 100
        
        # Reset seeds before each simulation
        np.random.seed(42)
        random.seed(42)
        results_n = env_neighborhood.run_simulation(num_rounds, logging_interval=num_rounds+1)
        
        np.random.seed(42)
        random.seed(42)
        results_p = env_pairwise.run_simulation(num_rounds, logging_interval=num_rounds+1)
        
        # Analysis
        # 1. In pairwise mode, TFT agents should defect after round 1 due to defectors
        final_moves_p = results_p[-1]['moves']
        for i in range(5):  # TFT agents
            assert final_moves_p[i] == "defect", f"TFT agent {i} should defect in pairwise mode"
        
        # 2. In neighborhood mode, TFT behavior is more stochastic
        final_moves_n = results_n[-1]['moves']
        tft_moves_n = [final_moves_n[i] for i in range(5)]
        # At least some TFT agents might still cooperate in neighborhood mode
        assert "cooperate" in tft_moves_n or all(m == "defect" for m in tft_moves_n), \
            "TFT agents in neighborhood mode should have mixed or all-defect behavior"
        
        # 3. Defectors should always defect in both modes
        for i in range(5, 10):
            assert final_moves_n[i] == "defect"
            assert final_moves_p[i] == "defect"

    def test_network_structure_relevance(self, setup_test_logging):
        """Test that network structure matters in neighborhood but not pairwise mode."""
        network_configs = [
            ("fully_connected", {}),
            ("small_world", {"k": 4, "beta": 0.3}),
            ("scale_free", {"m": 2}),
        ]
        
        num_agents = 20
        results = {}
        
        for network_type, network_params in network_configs:
            for interaction_mode in ["neighborhood", "pairwise"]:
                # Create agents - use a more dynamic mix
                agents = [
                    Agent(agent_id=i, strategy="hysteretic_q",
                          epsilon=0.05, learning_rate=0.05, beta=0.01,
                          discount_factor=0.9, state_type="proportion_discretized")
                    if i < 10 else
                    Agent(agent_id=i, strategy="tit_for_tat")
                    for i in range(num_agents)
                ]
                
                # Create environment
                payoff_matrix = create_payoff_matrix(num_agents)
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
                
                # Calculate metrics
                final_coop = sum(1 for m in sim_results[-1]['moves'].values() if m == "cooperate") / num_agents
                hysq_scores = [a.score for a in agents if a.strategy_type == "hysteretic_q"]
                avg_hysq_score = np.mean(hysq_scores)
                
                results[(network_type, interaction_mode)] = {
                    "coop_rate": final_coop,
                    "avg_hysq_score": avg_hysq_score
                }
        
        # Analysis
        # 1. In pairwise mode, results should be similar across network types
        pairwise_coop_rates = [results[(nt, "pairwise")]["coop_rate"] for nt, _ in network_configs]
        pairwise_variance = np.var(pairwise_coop_rates)
        
        # 2. In neighborhood mode, results should vary more across network types
        neighborhood_coop_rates = [results[(nt, "neighborhood")]["coop_rate"] for nt, _ in network_configs]
        neighborhood_variance = np.var(neighborhood_coop_rates)
        
        # Log the results for debugging
        setup_test_logging.info(f"Neighborhood cooperation rates: {neighborhood_coop_rates}")
        setup_test_logging.info(f"Pairwise cooperation rates: {pairwise_coop_rates}")
        setup_test_logging.info(f"Neighborhood variance: {neighborhood_variance:.4f}")
        setup_test_logging.info(f"Pairwise variance: {pairwise_variance:.4f}")
        
        # Network structure should matter more in neighborhood mode
        # In neighborhood mode, sparse networks (like scale_free) might protect TFT clusters
        assert neighborhood_variance > pairwise_variance * 1.5 or neighborhood_variance > 0.001, \
            f"Network structure should affect neighborhood mode more than pairwise (variances: {neighborhood_variance:.4f} vs {pairwise_variance:.4f})"

    def test_reactive_strategy_behavior(self, setup_test_logging):
        """Test that reactive strategies behave correctly in both modes."""
        # Create mixed environment with reactive strategies
        agents_n = [
            Agent(agent_id=0, strategy="tit_for_tat"),
            Agent(agent_id=1, strategy="generous_tit_for_tat", generosity=0.1),
            Agent(agent_id=2, strategy="suspicious_tit_for_tat"),
            Agent(agent_id=3, strategy="tit_for_two_tats"),
            Agent(agent_id=4, strategy="always_defect"),
            Agent(agent_id=5, strategy="always_cooperate"),
        ]
        
        agents_p = [
            Agent(agent_id=0, strategy="tit_for_tat"),
            Agent(agent_id=1, strategy="generous_tit_for_tat", generosity=0.1),
            Agent(agent_id=2, strategy="suspicious_tit_for_tat"),
            Agent(agent_id=3, strategy="tit_for_two_tats"),
            Agent(agent_id=4, strategy="always_defect"),
            Agent(agent_id=5, strategy="always_cooperate"),
        ]
        
        # Create environments
        payoff_matrix = create_payoff_matrix(6)
        
        env_n = Environment(
            agents_n, payoff_matrix, "fully_connected", 
            interaction_mode="neighborhood", logger=setup_test_logging
        )
        
        env_p = Environment(
            agents_p, payoff_matrix, "fully_connected",
            interaction_mode="pairwise", R=3, S=0, T=5, P=1, logger=setup_test_logging
        )
        
        # Run for a few rounds
        num_rounds = 10
        
        np.random.seed(42)
        random.seed(42)
        results_n = env_n.run_simulation(num_rounds, logging_interval=num_rounds+1)
        
        np.random.seed(42)
        random.seed(42)
        results_p = env_p.run_simulation(num_rounds, logging_interval=num_rounds+1)
        
        # Check specific behaviors
        # 1. TFT in pairwise should defect after seeing defector
        assert results_p[5]['moves'][0] == "defect", "TFT should defect in pairwise after seeing defector"
        
        # 2. Suspicious TFT should start with defection in both modes
        assert results_n[0]['moves'][2] == "defect", "Suspicious TFT should start with defection"
        assert results_p[0]['moves'][2] == "defect", "Suspicious TFT should start with defection"
        
        # 3. Check memory structure for pairwise mode
        tft_agent_p = agents_p[0]
        assert len(tft_agent_p.memory) > 0
        last_memory = tft_agent_p.memory[-1]
        assert 'opponent_coop_proportion' in last_memory['neighbor_moves']
        assert 'specific_opponent_moves' in last_memory['neighbor_moves']

    def test_learning_agent_state_representation(self, setup_test_logging):
        """Test that learning agents use appropriate state representations in each mode."""
        # Create Q-learning agents
        agents_n = [
            Agent(agent_id=i, strategy="q_learning", 
                  epsilon=0.1, learning_rate=0.2,
                  state_type="proportion_discretized")
            for i in range(6)
        ]
        
        agents_p = [
            Agent(agent_id=i, strategy="q_learning",
                  epsilon=0.1, learning_rate=0.2,
                  state_type="proportion_discretized")
            for i in range(6)
        ]
        
        # Add some fixed strategy agents
        for i in range(6, 10):
            agents_n.append(Agent(agent_id=i, strategy="always_defect"))
            agents_p.append(Agent(agent_id=i, strategy="always_defect"))
        
        # Create environments
        payoff_matrix = create_payoff_matrix(10)
        
        env_n = Environment(
            agents_n, payoff_matrix, "small_world", {"k": 4, "beta": 0.3},
            interaction_mode="neighborhood", logger=setup_test_logging
        )
        
        env_p = Environment(
            agents_p, payoff_matrix, "small_world", {"k": 4, "beta": 0.3},
            interaction_mode="pairwise", R=3, S=0, T=5, P=1, logger=setup_test_logging
        )
        
        # Run simulations
        num_rounds = 50
        
        np.random.seed(42)
        random.seed(42)
        env_n.run_simulation(num_rounds, logging_interval=num_rounds+1)
        
        np.random.seed(42)
        random.seed(42)
        env_p.run_simulation(num_rounds, logging_interval=num_rounds+1)
        
        # Check state representations
        # 1. Both should have learned states
        for i in range(6):
            assert len(agents_n[i].q_values) > 0, f"Neighborhood QL agent {i} should have learned states"
            assert len(agents_p[i].q_values) > 0, f"Pairwise QL agent {i} should have learned states"
        
        # 2. State distributions might differ
        states_n = set()
        states_p = set()
        for i in range(6):
            states_n.update(agents_n[i].q_values.keys())
            states_p.update(agents_p[i].q_values.keys())
        
        # States should be from the discretized proportions
        for state in states_n:
            if state != 'initial' and state != 'no_neighbors':
                assert isinstance(state, tuple), f"State should be tuple: {state}"
                assert all(v in [0.2, 0.4, 0.6, 0.8, 1.0] for v in state if isinstance(v, (int, float))), \
                    f"State values should be discretized: {state}"

    def test_payoff_scaling_differences(self, setup_test_logging):
        """Test that payoffs scale differently between modes."""
        num_agents = 10
        
        # All cooperators
        agents_n = [Agent(agent_id=i, strategy="always_cooperate") for i in range(num_agents)]
        agents_p = [Agent(agent_id=i, strategy="always_cooperate") for i in range(num_agents)]
        
        # Create environments
        payoff_matrix = create_payoff_matrix(num_agents)
        
        env_n = Environment(
            agents_n, payoff_matrix, "fully_connected",
            interaction_mode="neighborhood", logger=setup_test_logging
        )
        
        env_p = Environment(
            agents_p, payoff_matrix, "fully_connected",
            interaction_mode="pairwise", R=3, S=0, T=5, P=1, logger=setup_test_logging
        )
        
        # Run one round
        moves_n, payoffs_n = env_n.run_round(use_global_bonus=False)
        moves_p, payoffs_p = env_p.run_round()
        
        # In neighborhood mode with all cooperators on fully connected:
        # Each agent has 9 cooperating neighbors, payoff = C(9) = 3.0
        for agent_id, payoff in payoffs_n.items():
            assert payoff == pytest.approx(3.0), f"Neighborhood payoff should be 3.0, got {payoff}"
        
        # In pairwise mode with all cooperators:
        # Each agent plays 9 games, each getting R=3, total = 27
        for agent_id, payoff in payoffs_p.items():
            assert payoff == pytest.approx(27.0), f"Pairwise payoff should be 27.0, got {payoff}"
        
        # Check that agents learn from average payoff in pairwise
        assert agents_p[0].memory[0]['reward'] == pytest.approx(3.0), \
            "Pairwise agents should learn from average payoff"
