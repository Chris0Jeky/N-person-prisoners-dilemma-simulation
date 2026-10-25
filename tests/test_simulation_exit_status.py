"""Simulation and requested analysis failures must reach the CLI exit status."""

import json
import logging
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from npdl.analysis import analysis
from npdl.simulation import runner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scenario_file = tmp_path / "scenario.json"
    scenario_file.write_text(
        json.dumps(
            [
                {
                    "scenario_name": "tiny",
                    "num_agents": 2,
                    "num_rounds": 2,
                    "network_type": "fully_connected",
                    "agent_strategies": {"always_cooperate": 2},
                    "logging_interval": 1,
                }
            ]
        ),
        encoding="utf-8",
    )
    logger = Mock(level=logging.INFO)
    monkeypatch.setattr(runner, "setup_logging", Mock(return_value=logger))
    return {
        "scenario_file": str(scenario_file),
        "results_dir": str(tmp_path / "results"),
        "log_dir": str(tmp_path / "logs"),
        "num_runs": 1,
    }, logger


@pytest.mark.parametrize(
    "failure", ["unavailable", "io_error", "runtime_error", "no_results"]
)
def test_requested_analysis_failure_returns_nonzero(tiny_run, monkeypatch, failure):
    config, logger = tiny_run
    if failure == "unavailable":
        analyzer = None
        diagnostic = "Analysis module not available"
    elif failure == "no_results":
        monkeypatch.setattr(
            analysis, "create_analysis_report", Mock(side_effect=OSError("unreadable"))
        )
        analyzer = Mock(wraps=analysis.analyze_multiple_scenarios)
        diagnostic = "No scenarios were successfully analyzed"
    else:
        error = (
            OSError("disk full") if failure == "io_error" else RuntimeError("bad data")
        )
        analyzer = Mock(side_effect=error)
        diagnostic = f"Error running analysis: {error}"
    monkeypatch.setattr(runner, "analyze_multiple_scenarios", analyzer)

    status = runner.run_simulation(**config, analyze=True)

    assert status == 1
    assert any(diagnostic in call.args[0] for call in logger.error.call_args_list)
    assert not any(
        "Analysis complete" in call.args[0] for call in logger.info.call_args_list
    )
    assert Path(
        config["results_dir"], "tiny", "run_00", "experiment_results_rounds.csv"
    ).is_file()
    if analyzer is not None:
        analyzer.assert_called_once_with(
            ["tiny"], config["results_dir"], "analysis_results"
        )


@pytest.mark.parametrize("unavailable", [False, True])
def test_unrequested_analysis_does_not_fail(tiny_run, monkeypatch, unavailable):
    config, logger = tiny_run
    analyzer = None if unavailable else Mock(side_effect=RuntimeError("must not run"))
    monkeypatch.setattr(runner, "analyze_multiple_scenarios", analyzer)

    assert runner.run_simulation(**config, analyze=False) == 0

    logger.error.assert_not_called()
    if analyzer is not None:
        analyzer.assert_not_called()


def test_successful_analysis_returns_zero(tiny_run, monkeypatch):
    config, logger = tiny_run
    analyzer = Mock(return_value=pd.DataFrame({"scenario_name": ["tiny"]}))
    monkeypatch.setattr(runner, "analyze_multiple_scenarios", analyzer)

    assert runner.run_simulation(**config, analyze=True) == 0

    analyzer.assert_called_once_with(
        ["tiny"], config["results_dir"], "analysis_results"
    )
    logger.error.assert_not_called()
    assert any(
        "Analysis complete" in call.args[0] for call in logger.info.call_args_list
    )


@pytest.mark.parametrize("analyze", [False, True])
def test_cli_exit_status_when_analysis_output_is_blocked(tiny_run, tmp_path, analyze):
    config, _ = tiny_run
    blocked_output = tmp_path / "analysis_results"
    blocked_output.write_text("preserve this file", encoding="utf-8")
    command = [
        sys.executable,
        str(ROOT / "run.py"),
        "simulate",
        "--scenario_file",
        config["scenario_file"],
        "--results_dir",
        config["results_dir"],
        "--log_dir",
        config["log_dir"],
    ]
    if analyze:
        command.append("--analyze")

    result = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, timeout=60
    )

    assert result.returncode == (1 if analyze else 0), result.stdout + result.stderr
    assert blocked_output.read_text(encoding="utf-8") == "preserve this file"
    assert Path(
        config["results_dir"], "tiny", "run_00", "experiment_results_rounds.csv"
    ).is_file()
    if analyze:
        assert "No scenarios were successfully analyzed" in result.stderr
        assert "Analysis complete" not in result.stderr


def write_scenarios(tmp_path, names):
    path = tmp_path / "scenarios.json"
    path.write_text(
        json.dumps(
            [
                {
                    "scenario_name": name,
                    "num_agents": 2,
                    "num_rounds": 1,
                    "network_type": "fully_connected",
                    "agent_strategies": {
                        "typo_strategy" if name == "Bad" else "always_cooperate": 2
                    },
                }
                for name in names
            ]
        )
    )
    return path


def test_cli_reports_failed_scenario(tmp_path):
    path = write_scenarios(tmp_path, ["Bad"])
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "run.py"),
            "simulate",
            "--scenario_file",
            str(path),
            "--results_dir",
            str(tmp_path / "results"),
            "--log_dir",
            str(tmp_path / "logs"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "Unknown strategy type: typo_strategy" in proc.stderr
    assert proc.returncode == 1
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("analyze", [False, True])
@pytest.mark.parametrize("names", [["Bad", "Good"], ["Good", "Bad"], ["Good"]])
def test_batch_keeps_valid_results_and_reports_any_failure(
    tmp_path, monkeypatch, names, analyze
):
    path = write_scenarios(tmp_path, names)
    analyzer = Mock(return_value=pd.DataFrame({"scenario_name": ["Good"]}))
    monkeypatch.setattr(runner, "analyze_multiple_scenarios", analyzer)
    status = runner.run_simulation(
        scenario_file=str(path),
        results_dir=str(tmp_path / "results"),
        log_dir=str(tmp_path / "logs"),
        num_runs=1,
        analyze=analyze,
    )
    assert status == (1 if "Bad" in names else 0)
    if analyze:
        analyzer.assert_called_once_with(
            ["Good"], str(tmp_path / "results"), "analysis_results"
        )
    else:
        analyzer.assert_not_called()
    saved = tmp_path / "results" / "Good" / "run_00"
    assert (saved / "experiment_results_agents.csv").is_file()
    assert (saved / "experiment_results_rounds.csv").is_file()
    assert not (tmp_path / "results" / "Bad").exists()


@pytest.mark.parametrize("stage", ["setup", "simulation"])
def test_failed_run_is_reported_even_when_later_run_succeeds(
    tmp_path, monkeypatch, stage
):
    path = write_scenarios(tmp_path, ["Good"])
    calls = 0
    original = (
        runner.setup_experiment
        if stage == "setup"
        else runner.Environment.run_simulation
    )

    def fail_one_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        # Setup is also called once before the per-run loop.
        if calls == (2 if stage == "setup" else 1):
            raise ValueError("injected run failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        runner if stage == "setup" else runner.Environment,
        "setup_experiment" if stage == "setup" else "run_simulation",
        fail_one_run,
    )
    status = runner.run_simulation(
        scenario_file=str(path),
        results_dir=str(tmp_path / "results"),
        log_dir=str(tmp_path / "logs"),
        num_runs=2,
    )
    assert status == 1
    saved = tmp_path / "results" / "Good"
    assert not (saved / "run_00").exists()
    assert (saved / "run_01" / "experiment_results_rounds.csv").is_file()
    assert "injected run failure" in (tmp_path / "logs" / "experiment.log").read_text()
