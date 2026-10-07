"""W5 schema tests: scenarios/*.json and configs/*.json validate.

All 8 checked-in data files must pass their schema; crafted invalid files
must fail with errors that name the offending path.
"""

import glob
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, ROOT)

from npdl.experiments import (  # noqa: E402
    ValidationError,
    validate_config_file,
    validate_scenario_file,
)
from npdl.experiments.validate import (  # noqa: E402
    CONFIG_SCHEMA_PATH,
    SCENARIO_SCHEMA_PATH,
)

EXPECTED_SCENARIO_FILES = {
    "chris_testing_scenarios.json",
    "enhanced_scenarios.json",
    "express_scenarios.json",
    "pairwise_scenarios.json",
    "scenarios.json",
    "true_pairwise_scenarios.json",
}
EXPECTED_CONFIG_FILES = {"multi_sweep_config.json", "sweep_config.json"}

GOOD_FLAT_SCENARIO = {
    "scenario_name": "TinyTest",
    "num_agents": 4,
    "num_rounds": 10,
    "network_type": "fully_connected",
    "agent_strategies": {"tit_for_tat": 4},
}


def _data_files(directory):
    return sorted(
        path
        for path in glob.glob(os.path.join(ROOT, directory, "*.json"))
        if os.path.basename(path) != "schema.json"
    )


def _write_tmp(tmp_path, name, payload):
    path = str(tmp_path / name)
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


class TestCheckedInFiles:
    def test_schemas_are_valid_json(self):
        for schema_path in ("scenarios/schema.json", "configs/schema.json"):
            with open(os.path.join(ROOT, schema_path)) as f:
                schema = json.load(f)
            assert schema["$schema"].endswith("draft/2020-12/schema")

    def test_all_six_scenario_files_pass(self):
        paths = _data_files("scenarios")
        assert {os.path.basename(p) for p in paths} == EXPECTED_SCENARIO_FILES
        for path in paths:
            validate_scenario_file(path)

    def test_both_config_files_pass(self):
        paths = _data_files("configs")
        assert {os.path.basename(p) for p in paths} == EXPECTED_CONFIG_FILES
        for path in paths:
            validate_config_file(path)


class TestInvalidScenarios:
    @pytest.mark.parametrize(
        "mutate,expected_path",
        [
            (lambda s: s.pop("scenario_name"), "[0].scenario_name"),
            (lambda s: s.update(num_agents="4"), "[0].num_agents"),
            (lambda s: s.update(num_agents=True), "[0].num_agents"),
            (lambda s: s.update(network_type="hyperbolic"), "[0].network_type"),
            (
                lambda s: s.update(agent_strategies={"tit_for_tat": -1}),
                "[0].agent_strategies.tit_for_tat",
            ),
            (lambda s: s.update(agent_strategies={}), "[0].agent_strategies"),
            (lambda s: s.update(num_rounds=0), "[0].num_rounds"),
        ],
    )
    def test_invalid_flat_scenario_fails(self, tmp_path, mutate, expected_path):
        scenario = dict(GOOD_FLAT_SCENARIO)
        mutate(scenario)
        path = _write_tmp(tmp_path, "bad.json", [scenario])
        with pytest.raises(ValidationError) as excinfo:
            validate_scenario_file(path)
        assert expected_path in str(excinfo.value)

    def test_wrong_top_level_shape_fails(self, tmp_path):
        path = _write_tmp(tmp_path, "bad.json", {"not_scenarios": []})
        with pytest.raises(ValidationError):
            validate_scenario_file(path)

    def test_grouped_scenario_missing_strategies_fails(self, tmp_path):
        payload = {"scenarios": [{"name": "group", "scenarios": [{"name": "inner"}]}]}
        path = _write_tmp(tmp_path, "bad.json", payload)
        with pytest.raises(ValidationError) as excinfo:
            validate_scenario_file(path)
        assert "agent_strategies" in str(excinfo.value)

    def test_grouped_scenario_bad_pairwise_mode_fails(self, tmp_path):
        payload = {
            "scenarios": [
                {
                    "name": "group",
                    "scenarios": [
                        {
                            "name": "inner",
                            "agent_strategies": [{"type": "tit_for_tat", "count": 2}],
                            "num_rounds": 5,
                            "pairwise_mode": "telepathic",
                        }
                    ],
                }
            ]
        }
        path = _write_tmp(tmp_path, "bad.json", payload)
        with pytest.raises(ValidationError) as excinfo:
            validate_scenario_file(path)
        assert "pairwise_mode" in str(excinfo.value)


class TestInvalidConfigs:
    def test_empty_object_fails(self, tmp_path):
        path = _write_tmp(tmp_path, "bad.json", {})
        with pytest.raises(ValidationError):
            validate_config_file(path)

    def test_grid_values_must_be_arrays(self, tmp_path):
        payload = {
            "global_settings": {"num_runs_per_combo": 1},
            "base_scenario": {
                "scenario_name_prefix": "p",
                "num_rounds": 5,
                "network_type": "random",
            },
            "strategy_sweeps": {
                "tit_for_tat": {
                    "target_agent_count": 2,
                    "parameter_grid": {"epsilon": 0.1},
                }
            },
        }
        path = _write_tmp(tmp_path, "bad.json", payload)
        with pytest.raises(ValidationError) as excinfo:
            validate_config_file(path)
        assert "strategy_sweeps.tit_for_tat.parameter_grid.epsilon" in str(
            excinfo.value
        )

    def test_zero_runs_per_combo_fails(self, tmp_path):
        payload = {
            "global_settings": {"num_runs_per_combo": 0},
            "base_scenario": {
                "scenario_name_prefix": "p",
                "num_rounds": 5,
                "network_type": "random",
            },
            "strategy_sweeps": {
                "tit_for_tat": {
                    "target_agent_count": 2,
                    "parameter_grid": {"epsilon": [0.1]},
                }
            },
        }
        path = _write_tmp(tmp_path, "bad.json", payload)
        with pytest.raises(ValidationError) as excinfo:
            validate_config_file(path)
        assert "global_settings.num_runs_per_combo" in str(excinfo.value)

    def test_missing_sweeps_key_fails(self, tmp_path):
        payload = {
            "global_settings": {"num_runs_per_combo": 1},
            "base_scenario": {
                "scenario_name_prefix": "p",
                "num_rounds": 5,
                "network_type": "random",
            },
        }
        path = _write_tmp(tmp_path, "bad.json", payload)
        with pytest.raises(ValidationError):
            validate_config_file(path)


class TestPackagedSchemas:
    def test_packaged_copies_match_repo_copies(self):
        pairs = [
            (
                os.path.join(ROOT, "scenarios", "schema.json"),
                SCENARIO_SCHEMA_PATH,
            ),
            (
                os.path.join(ROOT, "configs", "schema.json"),
                CONFIG_SCHEMA_PATH,
            ),
        ]
        for repo_path, packaged_path in pairs:
            with open(repo_path, "rb") as f:
                repo_bytes = f.read()
            with open(packaged_path, "rb") as f:
                packaged_bytes = f.read()
            assert repo_bytes == packaged_bytes

    def test_schema_paths_live_inside_package_and_validators_use_them(self, tmp_path):
        npdl_dir = os.path.join(ROOT, "npdl")
        for schema_path in (SCENARIO_SCHEMA_PATH, CONFIG_SCHEMA_PATH):
            assert os.path.isfile(schema_path)
            assert os.path.abspath(schema_path).startswith(
                os.path.abspath(npdl_dir) + os.sep
            )
            assert os.path.abspath(schema_path) not in (
                os.path.abspath(os.path.join(ROOT, "scenarios", "schema.json")),
                os.path.abspath(os.path.join(ROOT, "configs", "schema.json")),
            )
        path = _write_tmp(tmp_path, "good.json", [dict(GOOD_FLAT_SCENARIO)])
        validate_scenario_file(path)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
