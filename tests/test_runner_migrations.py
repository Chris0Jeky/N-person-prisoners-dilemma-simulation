"""W5 runner-migration tests: registry wiring for ``scripts/runners/*``.

Live simulation tests use the smallest viable configs and run in tmp dirs;
the two live tests complete in a few seconds combined.
"""

import inspect
import json
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "runners"))

import run_evolutionary_generator as evo_mod  # noqa: E402
import run_parameter_sweep as psweep_mod  # noqa: E402
import run_scenario_generator as gen_mod  # noqa: E402
import run_sweep_analysis as sweep_mod  # noqa: E402

from npdl.experiments import verify_manifest  # noqa: E402


def _seed_default(func):
    params = inspect.signature(func).parameters
    assert "seed" in params, f"{func.__name__} has no seed parameter"
    return params["seed"].default


class TestMigrationWiring:
    def test_all_runners_take_seed_default_zero(self):
        assert _seed_default(gen_mod.run_scenario_generation) == 0
        assert _seed_default(sweep_mod.run_sweep_and_analysis) == 0
        assert _seed_default(evo_mod.run_evolutionary_scenario_generation) == 0
        assert _seed_default(psweep_mod.run_single_strategy_sweep) == 0

    def test_metadata_timestamp_is_overridable(self):
        params = inspect.signature(gen_mod.save_scenario_metadata).parameters
        assert params["timestamp"].default is None

    def test_sweep_analysis_imports_without_optional_helper(self):
        helper = sweep_mod.create_scenario_comparison_report
        assert helper is None or callable(helper)

    def test_evolutionary_imports_without_scipy(self):
        assert hasattr(evo_mod, "stats")  # None when scipy is unavailable

    def test_offspring_names_are_counter_based(self):
        parent = {
            "num_agents": 4,
            "num_rounds": 10,
            "network_type": "fully_connected",
            "network_params": {},
            "interaction_mode": "neighborhood",
            "agent_strategies": {"tit_for_tat": 2, "q_learning": 2},
        }
        first = evo_mod.crossover(parent, parent)["scenario_name"]
        second = evo_mod.crossover(parent, parent)["scenario_name"]
        assert re.fullmatch(r"Evo_\d{6}_\d{4}", first)
        assert re.fullmatch(r"Evo_\d{6}_\d{4}", second)
        assert first != second

    def test_crossover_is_independent_of_hash_seed(self):
        # Set iteration order depends on PYTHONHASHSEED, which differs per
        # process; a seeded crossover must not, or --seed cannot reproduce.
        script = (
            "import json, random, sys\n"
            f"sys.path.insert(0, {ROOT!r})\n"
            f"sys.path.insert(0, {os.path.join(ROOT, 'scripts', 'runners')!r})\n"
            "import run_evolutionary_generator as evo\n"
            "strats = ['tit_for_tat', 'q_learning', 'always_defect',\n"
            "          'always_cooperate', 'pavlov', 'generous_tit_for_tat']\n"
            "p1 = {'num_agents': 30, 'num_rounds': 10,\n"
            "      'network_type': 'fully_connected', 'network_params': {},\n"
            "      'interaction_mode': 'neighborhood',\n"
            "      'agent_strategies': {s: 5 for s in strats},\n"
            "      'payoff_params': {'R': 3, 'S': 0, 'T': 5, 'P': 1}}\n"
            "p2 = dict(p1, agent_strategies={s: 5 for s in reversed(strats)},\n"
            "          payoff_params={'R': 4, 'S': 1, 'T': 6, 'P': 2})\n"
            "random.seed(0)\n"
            "out = [evo.crossover(p1, p2) for _ in range(5)]\n"
            "for c in out:\n"
            "    c.pop('scenario_name')\n"
            "print(json.dumps(out, sort_keys=True))\n"
        )
        outputs = set()
        for hash_seed in ("1", "2", "3"):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed, MPLBACKEND="Agg")
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
            outputs.add(result.stdout)
        assert len(outputs) == 1


class TestTinyLiveRuns:
    def test_scenario_generation_registers_run(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # keeps scenario_generator.log out of the repo
        results_dir = str(tmp_path / "gen")
        gen_mod.run_scenario_generation(
            num_scenarios_to_generate=1,
            num_eval_runs=1,
            num_save_runs=1,
            top_n_to_save=1,
            results_dir=results_dir,
            log_level_str="ERROR",
            seed=7,
        )
        with open(os.path.join(results_dir, "run_info.json")) as f:
            info = json.load(f)
        assert info["seed"] == 7
        assert re.fullmatch(r"[0-9a-f]{64}", info["config_hash"])
        assert os.path.isfile(os.path.join(results_dir, "manifest.json"))
        assert verify_manifest(results_dir) == []

    def test_single_strategy_sweep_writes_csv(self, tmp_path):
        output_dir = str(tmp_path / "sweep")
        base_scenario = {
            "scenario_name_prefix": "Tiny",
            "num_rounds": 10,
            "network_type": "fully_connected",
            "network_params": {},
            "fixed_opponents": {"tit_for_tat": 2},
            "payoff_type": "linear",
            "state_type": "proportion_discretized",
            "q_init_type": "zero",
            "memory_length": 2,
            "logging_interval": 11,
        }
        strategy_config = {
            "target_agent_count": 2,
            "parameter_grid": {"epsilon": [0.1]},
        }
        global_settings = {
            "num_runs_per_combo": 1,
            "output_base_dir": output_dir,
            "log_level": "ERROR",
        }
        psweep_mod.run_single_strategy_sweep(
            {"global_settings": global_settings},
            "tit_for_tat",
            strategy_config,
            base_scenario,
            global_settings,
            seed=0,
        )
        csv_path = os.path.join(output_dir, "sweep_results_tit_for_tat.csv")
        assert os.path.isfile(csv_path)
        with open(csv_path) as f:
            header, row = f.read().strip().split("\n")
        assert "epsilon" in header.split(",")
        assert row.split(",")[0] == "0.1"

    def test_sweep_csv_is_independent_of_hash_seed(self, tmp_path):
        # Metric columns used to follow set iteration order, which depends
        # on PYTHONHASHSEED, so the same --seed gave different CSV bytes.
        script = (
            "import sys\n"
            f"sys.path.insert(0, {ROOT!r})\n"
            f"sys.path.insert(0, {os.path.join(ROOT, 'scripts', 'runners')!r})\n"
            "import run_parameter_sweep as ps\n"
            "out = sys.argv[1]\n"
            "base = {'scenario_name_prefix': 'Tiny', 'num_rounds': 10,\n"
            "        'network_type': 'fully_connected', 'network_params': {},\n"
            "        'fixed_opponents': {'tit_for_tat': 2},\n"
            "        'payoff_type': 'linear',\n"
            "        'state_type': 'proportion_discretized',\n"
            "        'q_init_type': 'zero', 'memory_length': 2,\n"
            "        'logging_interval': 11}\n"
            "gs = {'num_runs_per_combo': 1, 'output_base_dir': out,\n"
            "      'log_level': 'ERROR'}\n"
            "ps.run_single_strategy_sweep({'global_settings': gs}, 'q_learning',\n"
            "    {'target_agent_count': 2, 'parameter_grid': {'epsilon': [0.1]}},\n"
            "    base, gs, seed=0)\n"
        )
        contents = set()
        for hash_seed in ("1", "2", "3", "4"):
            out = str(tmp_path / hash_seed)
            env = dict(os.environ, PYTHONHASHSEED=hash_seed, MPLBACKEND="Agg")
            subprocess.run(
                [sys.executable, "-c", script, out],
                capture_output=True,
                env=env,
                check=True,
            )
            with open(os.path.join(out, "sweep_results_q_learning.csv")) as f:
                contents.add(f.read())
        assert len(contents) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
