"""Smoke test for the RL strategy comparison demo.

Imports ``scripts/demos/compare_rl_strategies.py`` by path and runs every
scenario for a few rounds so the Environment wiring is exercised without a
full 500-round comparison.
"""

import importlib.util
import json
import math
import os

import matplotlib

matplotlib.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(ROOT, "scripts", "demos", "compare_rl_strategies.py")


def load_demo():
    """Load the demo module without requiring it to be a package."""
    spec = importlib.util.spec_from_file_location(
        "compare_rl_strategies_demo", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_rl_strategies_smoke(tmp_path):
    """A short run writes both plots and a finite per-scenario summary."""
    demo = load_demo()
    scenarios = demo.create_comparison_scenarios()

    demo.main(num_rounds=3, output_dir=os.fspath(tmp_path))

    for name in (
        "rl_comparison_results.png",
        "rl_learning_curves.png",
        "rl_comparison_summary.json",
    ):
        assert (tmp_path / name).is_file()

    with open(tmp_path / "rl_comparison_summary.json", encoding="utf-8") as handle:
        summary = json.load(handle)

    assert set(summary) == {scenario["name"] for scenario in scenarios}
    assert len(summary) == len(scenarios)
    for entry in summary.values():
        for key in ("final_cooperation", "avg_cooperation"):
            value = entry[key]
            assert isinstance(value, (int, float))
            assert math.isfinite(value)
            assert 0.0 <= value <= 1.0


def test_default_round_count_is_unchanged():
    """Calling the scenario builder still requests 500 rounds."""
    demo = load_demo()
    for scenario in demo.create_comparison_scenarios():
        assert scenario["num_rounds"] == 500
