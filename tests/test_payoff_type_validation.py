"""Unsupported families fail before a run; supported payoff numbers stay fixed."""

import json
import logging
import sys

import pandas as pd
import pytest

from npdl import cli
from npdl.core.utils import create_payoff_matrix
from npdl.experiments import ValidationError
from npdl.simulation import experiments, runner


def scenario_config(**overrides):
    scenario = {
        "scenario_name": "Small_World_QL",
        "num_agents": 3,
        "num_rounds": 1,
        "network_type": "fully_connected",
        "agent_strategies": {"always_cooperate": 3},
    }
    scenario.update(overrides)
    return scenario


@pytest.mark.parametrize("payoff_type", ["exponental", "quadratic", "", None])
def test_unsupported_payoff_types_are_rejected(payoff_type):
    with pytest.raises(ValueError, match="payoff_type"):
        create_payoff_matrix(3, payoff_type)


@pytest.mark.parametrize("payoff_type", ["exponental", "quadratic", "", None])
def test_invalid_payoff_rejected_before_output(tmp_path, monkeypatch, payoff_type):
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps([scenario_config(payoff_type=payoff_type)]))

    def unexpected_setup(*args):
        pytest.fail("invalid payoff reached simulation setup")

    monkeypatch.setattr(runner, "setup_experiment", unexpected_setup)
    assert (
        runner.run_simulation(
            scenario_file=str(path),
            results_dir=str(tmp_path / "results"),
            log_dir=str(tmp_path / "logs"),
            num_runs=1,
        )
        == 1
    )
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("loader", [runner.load_scenarios, experiments.load_scenarios])
def test_runner_loaders_enforce_scenario_schema(tmp_path, loader):
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps([scenario_config(num_agents=0)]))
    with pytest.raises(ValidationError):
        loader(str(path))


def test_cli_refuses_invalid_payoff(tmp_path, monkeypatch):
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps([scenario_config(payoff_type="exponental")]))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "npdl",
            "simulate",
            "--scenario_file",
            str(path),
            "--results_dir",
            str(tmp_path / "results"),
            "--log_dir",
            str(tmp_path / "logs"),
        ],
    )
    assert cli.main() == 1
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    "payoff_type", ["linear", "threshold", "exponential", "missing"]
)
def test_supported_payoffs_keep_values_and_record_effective_type(tmp_path, payoff_type):
    scenario = scenario_config()
    if payoff_type != "missing":
        scenario["payoff_type"] = payoff_type
    selected = "linear" if payoff_type == "missing" else payoff_type
    env, _ = runner.setup_experiment(scenario, logging.getLogger(__name__))
    expected = {
        "linear": {"C": [0.0, 1.5, 3.0], "D": [1.0, 3.0, 5.0]},
        "threshold": {"C": [0.0, 0.9, 3.0], "D": [1.0, 2.2, 5.0]},
        "exponential": {"C": [0.0, 0.75, 3.0], "D": [1.0, 2.0, 5.0]},
    }[selected]
    for action in ("C", "D"):
        assert env.payoff_matrix[action] == pytest.approx(expected[action])
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps([scenario]))
    assert (
        runner.run_simulation(
            scenario_file=str(path),
            results_dir=str(tmp_path / "results"),
            log_dir=str(tmp_path / "logs"),
            num_runs=1,
        )
        == 0
    )
    run_dir = tmp_path / "results" / "Small_World_QL" / "run_00"
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["effective_payoff_type"] == selected
    assert (
        pd.read_csv(run_dir / "experiment_results_rounds.csv")["payoff"].tolist()
        == [3.0] * 3
    )


@pytest.mark.parametrize("writer", [runner.save_results, experiments.save_results])
def test_both_result_writers_record_effective_type(tmp_path, writer):
    writer(
        "Small_World_QL",
        0,
        [],
        [],
        results_dir=str(tmp_path),
        effective_payoff_type="threshold",
    )
    metadata = json.loads(
        (tmp_path / "Small_World_QL" / "run_00" / "run_metadata.json").read_text()
    )
    assert metadata == {
        "scenario_name": "Small_World_QL",
        "effective_payoff_type": "threshold",
    }
