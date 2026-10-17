"""Untrusted strategy-rate cells are data, including at the script entry point."""

import csv
import json
import os
import runpy
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import numpy as np
import pytest

from scripts.analysis import analyze_cooperation_patterns as analysis

SCRIPT = Path(analysis.__file__).resolve()


def write_csv(tmp_path, cell):
    directory = tmp_path / "analysis_results"
    directory.mkdir(exist_ok=True)
    with (directory / "comparative_analysis.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scenario_name",
                "final_cooperation_rate",
                "total_agents",
                "strategy_cooperation_rates",
            ]
        )
        writer.writerow(["Small_World_QL", 0.5, 3, cell])


def test_csv_expression_is_rejected_without_execution(tmp_path, monkeypatch):
    write_csv(tmp_path, '(__import__("os").system("untrusted"), {"TFT": 0.5})[1]')
    monkeypatch.chdir(tmp_path)
    command = Mock(return_value=0)
    monkeypatch.setattr(os, "system", command)
    with pytest.raises(ValueError, match="strategy_cooperation_rates"):
        analysis.identify_key_factors(analysis.load_experiment_data())
    command.assert_not_called()


@pytest.mark.parametrize("cell", ['{"TFT": 0.5}', "{'TFT': 0.5}"])
def test_json_and_historical_rates_are_unchanged(tmp_path, monkeypatch, cell):
    write_csv(tmp_path, cell)
    monkeypatch.chdir(tmp_path)
    result = analysis.identify_key_factors(analysis.load_experiment_data())
    assert result.iloc[0].to_dict() == {
        "strategy": "TFT",
        "avg_cooperation": 0.5,
        "instances": 1,
        "std_dev": 0.0,
    }


@pytest.mark.parametrize(
    "cell",
    [
        "{'TFT':",
        "[]",
        '{"unknown": 0.5}',
        '{"TFT": true}',
        '{"TFT": "0.5"}',
        '{"TFT": NaN}',
        '{"TFT": 1e999}',
        '{"TFT": -0.1}',
        '{"TFT": 1.1}',
        " " * 16385,
    ],
)
def test_invalid_rates_fail_explicitly(cell):
    with pytest.raises(ValueError, match="strategy_cooperation_rates"):
        analysis.identify_key_factors(
            pd.DataFrame({"strategy_cooperation_rates": [cell]})
        )


def test_analysis_cli_rejects_code(tmp_path, monkeypatch):
    write_csv(tmp_path, '(__import__("os").system("untrusted"), {"TFT": 0.5})[1]')
    monkeypatch.chdir(tmp_path)
    command = Mock(return_value=0)
    monkeypatch.setattr(os, "system", command)
    with pytest.raises(ValueError, match="strategy_cooperation_rates"):
        runpy.run_path(str(SCRIPT), run_name="__main__")
    command.assert_not_called()
    assert not (tmp_path / "cooperation_analysis_summary.png").exists()


def test_csv_load_bounds_records_and_cells(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(analysis, "MAX_ANALYSIS_ROWS", 1, raising=False)
    write_csv(tmp_path, "{'TFT': 0.5}")
    path = tmp_path / "analysis_results" / "comparative_analysis.csv"
    with path.open("a", newline="") as handle:
        csv.writer(handle).writerow(["second", 0.5, 3, "{'TFT': 0.5}"])
    with pytest.raises(ValueError, match="rows"):
        analysis.load_experiment_data()
    write_csv(tmp_path, " " * 16385)
    with pytest.raises(ValueError, match="field|cell"):
        analysis.load_experiment_data()


def test_csv_load_bounds_total_file_and_row_size(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_csv(tmp_path, "{'TFT': 0.5}")
    monkeypatch.setattr(analysis, "MAX_ANALYSIS_BYTES", 1)
    with pytest.raises(ValueError, match="file size"):
        analysis.load_experiment_data()
    monkeypatch.setattr(analysis, "MAX_ANALYSIS_BYTES", 16 * 1024 * 1024)
    path = tmp_path / "analysis_results" / "comparative_analysis.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["column"])
        writer.writerow(["a" * 10000] * 7)
    with pytest.raises(ValueError, match="row"):
        analysis.load_experiment_data()


def test_comparative_export_uses_json_without_changing_metrics(tmp_path, monkeypatch):
    from npdl.analysis import analysis as reports

    rates = {"tit_for_tat": np.float64(0.5)}
    monkeypatch.setattr(
        reports,
        "create_analysis_report",
        lambda *args: {
            "final_cooperation_rate": 0.5,
            "strategy_cooperation_rates": rates,
        },
    )
    monkeypatch.setattr(reports.plt, "savefig", lambda *args, **kwargs: None)
    output = tmp_path / "analysis_results"
    result = reports.analyze_multiple_scenarios(
        ["Small_World_QL"], output_dir=str(output)
    )
    assert result.iloc[0]["strategy_cooperation_rates"] == rates
    cell = pd.read_csv(output / "comparative_analysis.csv").iloc[0][
        "strategy_cooperation_rates"
    ]
    assert json.loads(cell) == {"tit_for_tat": 0.5}
    monkeypatch.chdir(tmp_path)
    stats = analysis.identify_key_factors(analysis.load_experiment_data())
    assert stats.iloc[0]["avg_cooperation"] == 0.5
