"""
Enhanced integration tests for the N-Person Prisoner's Dilemma simulation.
Focuses on workflows, interactions, and scenario variations.
"""
from random import random

import pytest
import os
import json
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Assuming npdl structure allows direct import like this from tests/ dir
from npdl.core.agents import Agent
from npdl.core.environment import Environment
from npdl.core.utils import create_payoff_matrix

# Import main functions (adjust if structure changed)
from main import setup_experiment, save_results, load_scenarios

# Test Simulation Setup (Enhanced)
@pytest.mark.integration
class TestSimulationSetup:
    """Test setting up simulations from scenarios."""

    def test_setup_experiment_basic(self, base_scenario_dict, setup_test_logging):
        """Test setting up a basic experiment from a scenario dictionary."""
        logger = setup_test_logging
        scenario = base_scenario_dict

        env, theoretical_scores = setup_experiment(scenario, logger)

        # Check environment setup
        assert isinstance(env, Environment)
        assert len(env.agents) == scenario["num_agents"]
        assert env.network_type == scenario["network_type"]
        assert env.network_params == scenario["network_params"]
        assert env.payoff_matrix is not None

        # Check agent distribution
        agent_counts = {}
        for agent in env.agents:
            agent_counts[agent.strategy_type] = agent_counts.get(agent.strategy_type, 0) + 1
        assert agent_counts == scenario["agent_strategies"]

        # Check agent parameters (spot check QL agent)
        ql_agent = next((a for a in env.agents if a.strategy_type == 'q_learning'), None)
        assert ql_agent is not None
        assert ql_agent.strategy.learning_rate == scenario['learning_rate']
        assert ql_agent.strategy.discount_factor == scenario['discount_factor']
        assert ql_agent.strategy.epsilon == scenario['epsilon']
        assert ql_agent.strategy.state_type == scenario['state_type']
        assert ql_agent.q_init_type == scenario['q_init_type']
        assert ql_agent.memory.maxlen == scenario['memory_length']


        # Check theoretical scores calculation
        assert "max_cooperation" in theoretical_scores
        assert "max_defection" in theoretical_scores
        assert "half_half" in theoretical_scores
        assert isinstance(theoretical_scores["max_cooperation"], (int, float))

    def test_setup_experiment_missing_optional_params(self, base_scenario_dict, setup_test_logging):
        """Test setup with a scenario missing optional parameters uses defaults."""
        logger = setup_test_logging
        scenario = base_scenario_dict.copy()
        # Remove some optional parameters
        del scenario["memory_length"]
        del scenario["q_init_type"]
        del scenario["payoff_type"] # Should default to linear

        env, _ = setup_experiment(scenario, logger)

        # Check defaults were applied (using Agent's defaults)
        ql_agent = next((a for a in env.agents if a.strategy_type == 'q_learning'), None)
        assert ql_agent.memory.maxlen == 10 # Default in Agent.__init__
        assert ql_agent.q_init_type == "zero" # Default in Agent.__init__
        # Payoff matrix should still be created using linear defaults
        assert env.payoff_matrix["C"][scenario['num_agents']-1] == pytest.approx(3.0) # Default R

    def test_setup_experiment_invalid_scenario(self, base_scenario_dict, setup_test_logging):
        """Test setup raises errors for invalid scenarios."""
        logger = setup_test_logging

        # Invalid strategy
        scenario_bad_strat = base_scenario_dict.copy()
        scenario_bad_strat["agent_strategies"] = {"unknown_strategy": 10}
        with pytest.raises(ValueError, match="Unknown strategy type"):
            setup_experiment(scenario_bad_strat, logger)

        # Invalid network type
        scenario_bad_net = base_scenario_dict.copy()
        scenario_bad_net["network_type"] = "nonexistent_network"
        with pytest.raises(ValueError, match="Unknown network type"):
            setup_experiment(scenario_bad_net, logger)

        # Missing required key
        scenario_missing_key = base_scenario_dict.copy()
        del scenario_missing_key["num_agents"]
        with pytest.raises(KeyError): # Or ValueError depending on implementation
            setup_experiment(scenario_missing_key, logger)


    def test_load_scenarios_valid_invalid(self, tmp_path: Path):
        """Test loading scenarios from valid and invalid JSON files."""
        valid_scenarios = [
            {"scenario_name": "V1", "num_agents": 5, "num_rounds": 10, "network_type": "fc", "agent_strategies": {"ac": 5}},
            {"scenario_name": "V2", "num_agents": 10, "num_rounds": 20, "network_type": "sw", "agent_strategies": {"ad": 10}}
        ]
        valid_file = tmp_path / "valid.json"
        with open(valid_file, 'w') as f:
            # Need to map simple names back for loading if setup_experiment expects full names
             for s in valid_scenarios:
                 s["network_type"] = {"fc": "fully_connected", "sw": "small_world"}.get(s["network_type"], s["network_type"])
                 s["agent_strategies"] = {{"ac": "always_cooperate", "ad": "always_defect"}.get(k,k): v for k,v in s["agent_strategies"].items()}
             json.dump(valid_scenarios, f)


        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("this is not json")

        nonexistent_file = tmp_path / "nonexistent.json"

        # Test valid loading
        loaded = load_scenarios(str(valid_file))
        assert len(loaded) == 2
        assert loaded[0]['scenario_name'] == "V1"
        assert loaded[0]['agent_strategies']['always_cooperate'] == 5

        # Test invalid loading
        with pytest.raises(json.JSONDecodeError):
            load_scenarios(str(invalid_file))

        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            load_scenarios(str(nonexistent_file))


# Test Result Saving (Enhanced)
@pytest.mark.integration
class TestResultSaving:
    """Test saving and loading simulation results."""

    def test_save_and_load_results_structure(self, base_scenario_dict, tmp_results_dir: Path, setup_test_logging):
        """Test saving results creates correct file structure and basic content."""
        logger = setup_test_logging
        scenario = base_scenario_dict
        scenario['agent_strategies'] = {'hysteretic_q': 5, 'always_defect': 5} # Ensure QL agent is present

        # Set up and run a short simulation
        env, _ = setup_experiment(scenario, logger)
        round_results = env.run_simulation(5) # Just 5 rounds for structure check

        # Save results
        scenario_name = scenario["scenario_name"]
        run_number = 0
        save_results(
            scenario_name, run_number, env.agents, round_results,
            results_dir=str(tmp_results_dir), logger=logger, env=env # Pass env for network saving
        )

        # Check file structure and existence
        run_dir = tmp_results_dir / scenario_name / f"run_{run_number:02d}"
        assert run_dir.is_dir()
        agents_file = run_dir / "experiment_results_agents.csv"
        rounds_file = run_dir / "experiment_results_rounds.csv"
        network_file = run_dir / "experiment_results_network.json"
        assert agents_file.is_file()
        assert rounds_file.is_file()
        assert network_file.is_file(), "Network file should have been saved"

        # Basic content check (more detailed checks in analysis tests)
        agents_df = pd.read_csv(agents_file)
        rounds_df = pd.read_csv(rounds_file)
        with open(network_file, 'r') as f:
            network_data = json.load(f)

        assert len(agents_df) == scenario["num_agents"]
        assert len(rounds_df) == scenario["num_agents"] * 5 # 5 rounds
        assert "agent_id" in agents_df.columns
        assert "strategy" in agents_df.columns
        assert "final_score" in agents_df.columns
        assert "final_q_values_avg" in agents_df.columns # Check QL output col
        assert "full_q_values" in agents_df.columns
        assert "round" in rounds_df.columns
        assert "payoff" in rounds_df.columns
        assert "nodes" in network_data
        assert "edges" in network_data
        assert "network_type" in network_data
        assert "metrics" in network_data

        # Check Q value saving (crude check for non-empty string/dict)
        hysq_agent_row = agents_df[agents_df['strategy'] == 'hysteretic_q'].iloc[0]
        assert pd.notna(hysq_agent_row['final_q_values_avg'])
        assert pd.notna(hysq_agent_row['full_q_values'])
        assert hysq_agent_row['full_q_values'] != '{}' # Should have learned something


# Test End-to-End Simulations (Enhanced)
@pytest.mark.integration
class TestEndToEndSimulation:
    """Test running complete simulations with different scenarios and verifying outcomes."""

    @pytest.mark.parametrize("scenario_name, agent_mix, network_type, expected_final_coop_range", [
        ("AllCooperate", {"always_cooperate": 10}, "fully_connected", (1.0, 1.0)),
        ("AllDefect", {"always_defect": 10}, "fully_connected", (0.0, 0.0)),
        ("TFT_vs_AllC", {"tit_for_tat": 5, "always_cooperate": 5}, "fully_connected", (1.0, 1.0)),
        ("TFT_vs_AllD", {"tit_for_tat": 5, "always_defect": 5}, "fully_connected", (0.0, 0.05)), # TFT defects after round 1
        ("HysQOpt_vs_TFT_FC", {"hysteretic_q": 5, "tit_for_tat": 5}, "fully_connected", (0.8, 1.0)), # Expect high coop based on sweeps
        ("WolfOpt_vs_TFT_SW", {"wolf_phc": 5, "tit_for_tat": 5}, "small_world", (0.7, 1.0)), # Expect high coop
        ("BasicQL_vs_AllD_SW", {"q_learning": 5, "always_defect": 5}, "small_world", (0.0, 0.1)), # QL should learn to defect
    ])
    @pytest.mark.slow # Mark longer simulations as slow
    def test_simulation_outcomes(self, scenario_name, agent_mix, network_type, expected_final_coop_range,
                                 base_scenario_dict, tmp_results_dir, setup_test_logging, seed):
        """Test specific simulation scenarios for expected cooperation levels."""
        logger = setup_test_logging
        scenario = base_scenario_dict.copy()

        # Update scenario based on parameters
        scenario["scenario_name"] = scenario_name
        scenario["num_agents"] = sum(agent_mix.values())
        scenario["agent_strategies"] = agent_mix
        scenario["network_type"] = network_type
        scenario["num_rounds"] = 100 # Use moderate number of rounds

        # Add optimized params if testing optimized agents
        if "HysQOpt" in scenario_name:
             scenario.update({"learning_rate": 0.05, "beta": 0.01, "epsilon": 0.05, "discount_factor": 0.9})
        if "WolfOpt" in scenario_name:
             scenario.update({"learning_rate": 0.1, "alpha_win": 0.01, "alpha_lose": 0.1, "epsilon": 0.1, "discount_factor": 0.9})

        # Run simulation (only 1 run for this focused test)
        env, _ = setup_experiment(scenario, logger)
        round_results = env.run_simulation(scenario["num_rounds"], logging_interval=scenario["num_rounds"]+1)

        # Calculate final cooperation rate
        final_round = round_results[-1]
        final_coop_rate = sum(1 for move in final_round['moves'].values() if move == "cooperate") / len(final_round['moves'])

        logger.info(f"Scenario '{scenario_name}' final coop rate: {final_coop_rate:.3f}")

        # Assert cooperation rate is within the expected range
        min_expected, max_expected = expected_final_coop_range
        assert min_expected <= final_coop_rate <= max_expected, \
            f"Final coop rate {final_coop_rate:.3f} out of expected range ({min_expected:.2f}, {max_expected:.2f})"

        # Optional: Assert relative strategy scores if meaningful
        if "HysQOpt_vs_TFT" in scenario_name or "WolfOpt_vs_TFT" in scenario_name:
            hysq_score = np.mean([a.score for a in env.agents if a.strategy_type == 'hysteretic_q']) if 'hysteretic_q' in agent_mix else 0
            wolf_score = np.mean([a.score for a in env.agents if a.strategy_type == 'wolf_phc']) if 'wolf_phc' in agent_mix else 0
            tft_score = np.mean([a.score for a in env.agents if a.strategy_type == 'tit_for_tat'])
            # In high coop, scores should be similar (mostly R payoffs)
            assert abs(max(hysq_score, wolf_score) - tft_score) < scenario["num_rounds"] * 0.5 # Allow small difference

        if "BasicQL_vs_AllD" in scenario_name:
            ql_score = np.mean([a.score for a in env.agents if a.strategy_type == 'q_learning'])
            alld_score = np.mean([a.score for a in env.agents if a.strategy_type == 'always_defect'])
            # Both should get approx P payoff per round
            assert abs(ql_score - alld_score) < scenario["num_rounds"] * 0.5


    def test_simulation_with_global_bonus(self, hysteretic_scenario_dict, setup_test_logging, seed):
        """Test that enabling global bonus increases cooperation vs baseline."""
        logger = setup_test_logging

        # Scenario WITHOUT bonus
        scenario_no_bonus = hysteretic_scenario_dict.copy()
        scenario_no_bonus["scenario_name"] = "HysQ_vs_AllD_NoBonus"
        scenario_no_bonus["agent_strategies"] = {"hysteretic_q": 5, "always_defect": 5}
        scenario_no_bonus["network_type"] = "small_world" # Bonus might be more effective on SW/FC
        scenario_no_bonus["num_rounds"] = 100
        scenario_no_bonus["use_global_bonus"] = False
        # Use optimized params found previously
        scenario_no_bonus.update({"learning_rate": 0.05, "beta": 0.01, "epsilon": 0.05, "discount_factor": 0.9})

        env_no_bonus, _ = setup_experiment(scenario_no_bonus, logger)
        results_no_bonus = env_no_bonus.run_simulation(scenario_no_bonus["num_rounds"], logging_interval=101)
        final_coop_no_bonus = sum(1 for m in results_no_bonus[-1]['moves'].values() if m=='cooperate') / scenario_no_bonus['num_agents']

        # Scenario WITH bonus
        scenario_bonus = scenario_no_bonus.copy()
        scenario_bonus["scenario_name"] = "HysQ_vs_AllD_WithBonus"
        scenario_bonus["use_global_bonus"] = True

        # Use same seed for closer comparison
        random.seed(seed)
        np.random.seed(seed)
        env_bonus, _ = setup_experiment(scenario_bonus, logger)
        results_bonus = env_bonus.run_simulation(scenario_bonus["num_rounds"], logging_interval=101)
        final_coop_bonus = sum(1 for m in results_bonus[-1]['moves'].values() if m=='cooperate') / scenario_bonus['num_agents']

        logger.info(f"Coop Rate WITHOUT Bonus: {final_coop_no_bonus:.3f}")
        logger.info(f"Coop Rate WITH Bonus: {final_coop_bonus:.3f}")

        # Expect bonus to potentially help HysQ maintain some cooperation against AllD
        assert final_coop_bonus >= final_coop_no_bonus, "Global bonus should generally not decrease cooperation"
        # It might not guarantee high cooperation here, just test if it has *some* positive effect