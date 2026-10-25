"""Failed scenarios and runs must remain visible in the command's exit status."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from npdl.simulation import runner


@pytest.fixture
def analysis_backend(monkeypatch):
    from npdl.analysis import analysis

    analysis.plt.switch_backend("Agg")
    monkeypatch.setenv("MPLBACKEND", "Agg")


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


@pytest.mark.parametrize("names", [["Bad", "Good"], ["Good", "Bad"], ["Good"]])
def test_batch_keeps_valid_results_and_reports_any_failure(tmp_path, names):
    path = write_scenarios(tmp_path, names)
    status = runner.run_simulation(
        scenario_file=str(path),
        results_dir=str(tmp_path / "results"),
        log_dir=str(tmp_path / "logs"),
        num_runs=1,
    )
    assert status == (1 if "Bad" in names else 0)
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


@pytest.mark.parametrize("failed_scenario", ["First", "Second", None])
def test_analysis_reports_missing_scenario_and_preserves_results(
    tmp_path, monkeypatch, failed_scenario, analysis_backend
):
    monkeypatch.chdir(tmp_path)
    names = ["First", "Second"]
    path = write_scenarios(tmp_path, names)
    output = tmp_path / "analysis_results"
    if failed_scenario:
        # A directory at the report's file path forces a real write failure.
        (output / f"{failed_scenario}_metrics.json").mkdir(parents=True)

    status = runner.run_simulation(
        scenario_file=str(path),
        results_dir=str(tmp_path / "results"),
        log_dir=str(tmp_path / "logs"),
        num_runs=1,
        analyze=True,
    )

    expected = [name for name in names if name != failed_scenario]
    comparison = pd.read_csv(output / "comparative_analysis.csv")
    assert comparison["scenario_name"].tolist() == expected
    for name in names:
        saved = tmp_path / "results" / name / "run_00"
        assert (saved / "experiment_results_agents.csv").is_file()
        assert (saved / "experiment_results_rounds.csv").is_file()
        assert (output / f"{name}_metrics.json").is_file() == (name in expected)

    assert status == (1 if failed_scenario else 0)
    log = (tmp_path / "logs" / "experiment.log").read_text()
    if failed_scenario:
        assert f"Analysis incomplete. No report generated for: {failed_scenario}" in log
        assert "Analysis complete." not in log
    else:
        assert "Analysis complete." in log


@pytest.mark.parametrize("failed_scenario", ["First", None])
def test_cli_exit_status_reflects_partial_analysis(
    tmp_path, failed_scenario, analysis_backend
):
    path = write_scenarios(tmp_path, ["First", "Second"])
    output = tmp_path / "analysis_results"
    if failed_scenario:
        (output / f"{failed_scenario}_metrics.json").mkdir(parents=True)

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
            "--analyze",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )

    comparison = pd.read_csv(output / "comparative_analysis.csv")
    assert comparison["scenario_name"].tolist() == (
        ["Second"] if failed_scenario else ["First", "Second"]
    )
    assert (output / "Second_metrics.json").is_file()
    assert proc.returncode == (1 if failed_scenario else 0)
    if failed_scenario:
        assert f"No report generated for: {failed_scenario}" in proc.stderr
        assert "Analysis complete." not in proc.stderr
