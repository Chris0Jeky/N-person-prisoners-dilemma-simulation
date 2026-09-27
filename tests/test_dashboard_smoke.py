"""W8 dashboard smoke test: canonical Dash figures from small fixture data.

Renders every dashboard tab's figure by calling the Dash callback functions
directly against a small on-disk ``results/`` fixture -- no browser window
or live server is started. Skipped when the optional dashboard stack is
absent (minimal envs install only the simulation/test dependencies; W9 CI
installs the full dependencies).
"""

import json

import pytest

pytest.importorskip("dash", reason="dashboard stack not installed in minimal env")
pytest.importorskip(
    "dash_bootstrap_components",
    reason="dashboard stack not installed in minimal env",
)
pytest.importorskip("plotly", reason="dashboard stack not installed in minimal env")
pytest.importorskip("flask", reason="dashboard stack not installed in minimal env")

import pandas as pd
import plotly.graph_objects as go
from flask import Flask

from npdl.visualization.dashboard import (
    app,
    populate_scenarios,
    server,
    update_cooperation_graph,
    update_network_graph,
    update_payoff_graph,
    update_run_dropdown,
    update_score_graph,
)

SCENARIO = "SmokeScenario"
STRATEGIES = ["tit_for_tat", "always_defect"]


@pytest.fixture
def smoke_results_dir(tmp_path, monkeypatch):
    """Write a small ``results/`` tree and run the test from its parent."""
    run_dir = tmp_path / "results" / SCENARIO / "run_00"
    run_dir.mkdir(parents=True)

    agents_df = pd.DataFrame(
        {
            "scenario_name": ["smoke"] * 4,
            "run_number": [0] * 4,
            "agent_id": [0, 1, 2, 3],
            "strategy": [
                "tit_for_tat",
                "tit_for_tat",
                "always_defect",
                "always_defect",
            ],
            "final_score": [45.0, 42.0, 85.0, 62.0],
        }
    )
    rounds_df = pd.DataFrame(
        {
            "round": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
            "run_number": [0] * 12,
            "agent_id": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            "move": [
                "cooperate",
                "cooperate",
                "defect",
                "defect",
                "cooperate",
                "defect",
                "defect",
                "cooperate",
                "cooperate",
                "cooperate",
                "defect",
                "cooperate",
            ],
            "payoff": [1.0, 1.0, 3.0, 3.0, 1.5, 2.5, 2.5, 1.5, 2.0, 2.0, 3.5, 2.0],
            "strategy": [
                "tit_for_tat",
                "tit_for_tat",
                "always_defect",
                "always_defect",
            ]
            * 3,
        }
    )
    network_data = {
        "nodes": [0, 1, 2, 3],
        "edges": [[0, 1], [0, 2], [1, 2], [2, 3]],
        "network_type": "sample",
        "network_params": {"info": "smoke"},
    }
    agents_df.to_csv(run_dir / "experiment_results_agents.csv", index=False)
    rounds_df.to_csv(run_dir / "experiment_results_rounds.csv", index=False)
    with open(run_dir / "experiment_results_network.json", "w") as f:
        json.dump(network_data, f)

    monkeypatch.chdir(tmp_path)
    return tmp_path / "results"


@pytest.mark.visualization
class TestDashboardSmoke:
    """Smoke-render every dashboard figure from the fixture tree."""

    def test_app_builds_with_registered_callbacks(self):
        """App layout exists and tab callbacks are registered."""
        assert app.layout is not None
        assert isinstance(server, Flask)
        assert len(app.callback_map) >= 5

    def test_populate_scenarios_finds_fixture(self, smoke_results_dir):
        """Scenario dropdown lists the fixture scenario."""
        assert smoke_results_dir.is_dir()
        options = populate_scenarios(None)
        assert {"label": SCENARIO, "value": SCENARIO} in options

    def test_update_run_dropdown_single_run(self, smoke_results_dir):
        """Run dropdown offers the single fixture run."""
        options, value = update_run_dropdown(SCENARIO)
        assert options == [{"label": "Run 0", "value": 0}]
        assert value == 0

    def test_cooperation_figure_has_strategy_and_overall_traces(
        self, smoke_results_dir
    ):
        """Cooperation tab renders one trace per strategy plus Overall."""
        fig = update_cooperation_graph(SCENARIO, 0, STRATEGIES, [0, 2], 1)
        assert isinstance(fig, go.Figure)
        assert {trace.name for trace in fig.data} == set(STRATEGIES) | {"Overall"}
        assert "Run 0" in fig.layout.title.text

    def test_payoff_figure_has_strategy_traces(self, smoke_results_dir):
        """Payoffs tab renders one trace per selected strategy."""
        fig = update_payoff_graph(SCENARIO, 0, STRATEGIES, [0, 2], 1)
        assert isinstance(fig, go.Figure)
        assert {trace.name for trace in fig.data} == set(STRATEGIES)
        assert "Average Payoff" in fig.layout.title.text

    def test_score_figure_has_boxes(self, smoke_results_dir):
        """Final-scores tab renders one box per selected strategy."""
        fig = update_score_graph(SCENARIO, 0, STRATEGIES, 1)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == len(STRATEGIES)
        assert all(trace.type == "box" for trace in fig.data)

    def test_network_figure_has_edge_and_node_traces(self, smoke_results_dir):
        """Network tab renders edges plus one marker per agent."""
        fig = update_network_graph(SCENARIO, 0, 1, 1)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # edge trace + node trace
        assert len(fig.data[1].x) == 4

    def test_empty_selection_returns_placeholder_figures(self, smoke_results_dir):
        """Missing scenario/strategies yield placeholders, not errors."""
        fig = update_cooperation_graph(None, 0, STRATEGIES, [0, 2], 1)
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.annotations) > 0

        fig = update_cooperation_graph(SCENARIO, 0, [], [0, 2], 1)
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.annotations) > 0

        fig = update_network_graph(None, 0, 1, 1)
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.annotations) > 0
