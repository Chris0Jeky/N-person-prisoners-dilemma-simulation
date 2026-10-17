"""Both result writers keep scenario-controlled paths inside the output root."""

import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from npdl.core.agents import Agent
from npdl.simulation import experiments, runner


@pytest.mark.parametrize("writer", [runner.save_results, experiments.save_results])
@pytest.mark.parametrize(
    "name",
    [
        "../../outside",
        "/tmp/outside",
        "C:\\outside",
        "C:outside",
        "\\outside",
        "\\\\host\\share",
        ".",
        "..",
        "",
        "CON",
        "nul.txt",
        "COM1",
        "LPT9",
        "trailing.",
        "trailing ",
        "name:stream",
        "NUL .csv",
        "COM\u00b9.txt",
    ],
)
def test_result_writer_rejects_unsafe_names(tmp_path, writer, name):
    sentinel = tmp_path / "outside" / "run_00" / "experiment_results_agents.csv"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"collected research")
    root = tmp_path / "nested" / "results"
    with pytest.raises(ValueError):
        writer(name, 0, [], [], results_dir=str(root))
    assert sentinel.read_bytes() == b"collected research"
    assert not root.exists()


@pytest.mark.parametrize("writer", [runner.save_results, experiments.save_results])
def test_normal_name_retains_directory_and_label(tmp_path, writer):
    agent = Agent(0, "always_cooperate")
    writer("Small_World_QL", 0, [agent], [], results_dir=str(tmp_path))
    path = tmp_path / "Small_World_QL" / "run_00" / "experiment_results_agents.csv"
    assert pd.read_csv(path)["scenario_name"].tolist() == ["Small_World_QL"]


@pytest.mark.parametrize("writer", [runner.save_results, experiments.save_results])
@pytest.mark.parametrize(
    "existing, requested",
    [
        ("Small_World_QL", "small_world_ql"),
        ("Caf\u00e9", "Cafe\u0301"),
    ],
)
def test_portable_sibling_collisions_are_rejected(
    tmp_path, writer, existing, requested
):
    writer(existing, 0, [], [], results_dir=str(tmp_path))
    sentinel = tmp_path / existing / "run_00" / "experiment_results_agents.csv"
    original = sentinel.read_bytes()
    with pytest.raises(ValueError, match="colli"):
        writer(requested, 0, [], [], results_dir=str(tmp_path))
    assert sentinel.read_bytes() == original


@pytest.mark.parametrize("writer", [runner.save_results, experiments.save_results])
@pytest.mark.parametrize("level", ["scenario", "run", "file"])
def test_existing_link_targets_are_refused(tmp_path, writer, level):
    root = tmp_path / "results"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "experiment_results_agents.csv"
    sentinel.write_bytes(b"collected research")
    target = root / "Small_World_QL"
    if level != "scenario":
        target = target / "run_00"
    if level == "file":
        target = target / "experiment_results_agents.csv"
    target.parent.mkdir(parents=True)
    if os.name == "nt" and level != "file":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(target), str(outside)],
            check=True,
            capture_output=True,
        )
    else:
        try:
            target.symlink_to(
                sentinel if level == "file" else outside,
                target_is_directory=level != "file",
            )
        except OSError as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="link|reparse"):
        writer("Small_World_QL", 0, [], [], results_dir=str(root))
    assert sentinel.read_bytes() == b"collected research"


def test_scenario_pack_rejects_names_before_running(tmp_path, monkeypatch):
    scenario = {
        "scenario_name": "../../outside",
        "num_agents": 2,
        "num_rounds": 1,
        "network_type": "fully_connected",
        "agent_strategies": {"always_cooperate": 2},
    }
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps([scenario]))

    def unexpected_setup(*args):
        pytest.fail("invalid scenario reached simulation setup")

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
@pytest.mark.parametrize(
    "names",
    [
        ("Small_World_QL", "small_world_ql"),
        ("Caf\u00e9", "Cafe\u0301"),
    ],
)
def test_scenario_pack_rejects_portable_collisions(tmp_path, loader, names):
    path = tmp_path / "scenarios.json"
    path.write_text(
        json.dumps(
            [
                {
                    "scenario_name": name,
                    "num_agents": 2,
                    "num_rounds": 1,
                    "network_type": "fully_connected",
                    "agent_strategies": {"always_cooperate": 2},
                }
                for name in names
            ]
        )
    )
    with pytest.raises(ValueError, match="colli"):
        loader(str(path))


def test_runner_reports_blocked_destination(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = root / "Small_World_QL"
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(target), str(outside)],
            check=True,
            capture_output=True,
        )
    else:
        target.symlink_to(outside, target_is_directory=True)
    path = tmp_path / "scenarios.json"
    path.write_text(
        json.dumps(
            [
                {
                    "scenario_name": "Small_World_QL",
                    "num_agents": 2,
                    "num_rounds": 1,
                    "network_type": "fully_connected",
                    "agent_strategies": {"always_cooperate": 2},
                }
            ]
        )
    )
    assert (
        runner.run_simulation(
            scenario_file=str(path),
            results_dir=str(root),
            log_dir=str(tmp_path / "logs"),
            num_runs=1,
        )
        == 1
    )
    assert list(outside.iterdir()) == []
