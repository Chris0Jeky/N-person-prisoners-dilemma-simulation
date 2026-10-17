"""Portable scenario labels and output containment shared by both runners."""

import json
import os
import stat
import unicodedata
from pathlib import Path


def _portable_key(name):
    return unicodedata.normalize("NFKC", name).casefold()


def validate_scenario_name(name):
    """Keep existing safe names; reject path syntax and Windows device names."""
    if not isinstance(name, str) or not name:
        raise ValueError("scenario_name must be a nonempty portable directory name")
    normalized = _portable_key(name)
    device = normalized.split(".")[0].rstrip(" ")
    if (
        normalized in (".", "..")
        or normalized.endswith((".", " "))
        or any(ord(char) < 32 or char in '<>:"/\\|?*' for char in normalized)
        or device in {"con", "prn", "aux", "nul", "conin$", "conout$"}
        or device
        in {f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10)}
    ):
        raise ValueError("scenario_name contains unsafe path syntax or a reserved name")


def validate_scenario_names(scenarios):
    """Reject unsafe or colliding labels before running any scenario in a pack."""
    if not isinstance(scenarios, list):
        raise ValueError("The simulation runner requires a flat scenario list")
    seen = set()
    for scenario in scenarios:
        name = scenario["scenario_name"]
        validate_scenario_name(name)
        key = _portable_key(name)
        if key in seen:
            raise ValueError("scenario_name collision in scenario file")
        seen.add(key)


def _reject_link(path):
    if os.path.lexists(path):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(
            info, "st_file_attributes", 0
        ) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise ValueError("Result destination contains a symlink or reparse point")


def prepare_run_directory(results_dir, scenario_name, run_number, base_filename):
    """Check all destinations before creating directories or overwriting files."""
    validate_scenario_name(scenario_name)
    validate_scenario_name(base_filename)
    root = Path(results_dir).resolve()
    scenario_dir = root / scenario_name
    run_dir = scenario_dir / f"run_{run_number:02d}"
    if root.exists():
        for sibling in root.iterdir():
            if sibling.name != scenario_name and _portable_key(
                sibling.name
            ) == _portable_key(scenario_name):
                raise ValueError(
                    "scenario_name collision with existing output directory"
                )
    destinations = [scenario_dir, run_dir] + [
        run_dir / f"{base_filename}_{suffix}"
        for suffix in ("agents.csv", "rounds.csv", "network.json")
    ]
    destinations.append(run_dir / "run_metadata.json")
    for destination in destinations:
        _reject_link(destination)
        if root not in destination.resolve().parents:
            raise ValueError("Result destination escapes results_dir")
    # shortcut: preflight assumes no concurrent directory replacement; use directory handles in hostile shared workspaces.
    run_dir.mkdir(parents=True, exist_ok=True)
    return os.fspath(run_dir)


def write_run_metadata(run_dir, scenario_name, effective_payoff_type):
    """Record the validated family without changing existing result CSVs."""
    with open(
        os.path.join(run_dir, "run_metadata.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "scenario_name": scenario_name,
                "effective_payoff_type": effective_payoff_type,
            },
            handle,
            indent=2,
        )
