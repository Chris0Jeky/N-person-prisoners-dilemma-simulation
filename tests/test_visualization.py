"""
Enhanced tests for data processing and visualization components.
"""
import pytest
import pandas as pd
import networkx as nx
import numpy as np
import tempfile
import os
import json
from pathlib import Path
import plotly.graph_objects as go # For inspecting figure data

# Import visualization components
from npdl.visualization.data_processor import (
    get_payoffs_by_strategy,
    get_strategy_scores,
    get_strategy_colors,
    prepare_network_data
)
from npdl.visualization.data_loader import (
    get_available_scenarios,
    load_scenario_results,
    get_cooperation_rates,
    get_strategy_cooperation_rates,
    load_network_structure,
    get_available_runs
)
from npdl.visualization.network_viz import (
    generate_network_positions,
    create_network_figure
)

# --- Fixtures (keep existing ones) ---
@pytest.fixture
def sample_rounds_df():
    # ... (keep existing fixture) ...
    return pd.DataFrame({
        'round': [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
        'run_number': [0]*12, # Add run number
        'agent_id': [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        'move': ['cooperate', 'cooperate', 'defect', 'defect',
                 'cooperate', 'defect', 'defect', 'cooperate',
                 'cooperate', 'cooperate', 'defect', 'cooperate'],
        'payoff': [1.0, 1.0, 3.0, 3.0, 1.5, 2.5, 2.5, 1.5, 2.0, 2.0, 3.5, 2.0],
        'strategy': ['tit_for_tat', 'tit_for_tat', 'always_defect', 'q_learning',
                     'tit_for_tat', 'tit_for_tat', 'always_defect', 'q_learning',
                     'tit_for_tat', 'tit_for_tat', 'always_defect', 'q_learning']
    })


@pytest.fixture
def sample_agents_df():
    # ... (keep existing fixture) ...
     return pd.DataFrame({
        'scenario_name': ['test'] * 4,
        'run_number': [0] * 4,
        'agent_id': [0, 1, 2, 3],
        'strategy': ['tit_for_tat', 'tit_for_tat', 'always_defect', 'q_learning'],
        'final_score': [45.0, 42.0, 85.0, 62.0],
        # Add mock Q-value columns for testing processing
        'final_q_values_avg': ['{"avg_cooperate": 2.5, "avg_defect": 1.5}',
                               '{"avg_cooperate": 2.8, "avg_defect": 1.1}',
                               None, # Non-QL agent
                               '{"avg_cooperate": 1.0, "avg_defect": 4.5}' ],
        'full_q_values': ['{...}', '{...}', None, '{...}']
    })


@pytest.fixture
def sample_network():
    # ... (keep existing fixture) ...
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3)])
    return G

@pytest.fixture
def mock_results_dir(tmp_path: Path, sample_agents_df, sample_rounds_df, sample_network):
    """Creates a mock results directory structure for data loader tests."""
    scenario_name = "MockScenario"
    run_num = 0
    base_dir = tmp_path / "mock_results"
    run_dir = base_dir / scenario_name / f"run_{run_num:02d}"
    run_dir.mkdir(parents=True)

    # Save sample data
    agents_file = run_dir / "experiment_results_agents.csv"
    rounds_file = run_dir / "experiment_results_rounds.csv"
    network_file = run_dir / "experiment_results_network.json"

    sample_agents_df.to_csv(agents_file, index=False)
    sample_rounds_df.to_csv(rounds_file, index=False)

    # Save sample network
    network_data = {
        "nodes": list(sample_network.nodes()),
        "edges": [list(e) for e in sample_network.edges()], # Ensure lists for JSON
        "network_type": "sample",
        "network_params": {"info": "mock"},
        "metrics": {"num_nodes": sample_network.number_of_nodes()}
    }
    with open(network_file, 'w') as f:
        json.dump(network_data, f)

    # Create another scenario dir for get_available_scenarios test
    (base_dir / "AnotherScenario" / "run_00").mkdir(parents=True)
    # Create an empty dir to be ignored
    (base_dir / "_hidden_dir").mkdir(parents=True)


    return base_dir # Return the base results path


# --- Tests (marked with @pytest.mark.visualization) ---

@pytest.mark.visualization
@pytest.mark.unit
class TestDataProcessing:
    """Test data processing functions."""

    def test_get_payoffs_by_strategy_valid(self, sample_rounds_df):
        # ... (Keep existing test - seems okay) ...
        payoffs = get_payoffs_by_strategy(sample_rounds_df)
        assert not payoffs.empty
        assert "round" in payoffs.columns and "strategy" in payoffs.columns and "payoff" in payoffs.columns
        round0 = payoffs[payoffs["round"] == 0]
        assert round0[round0["strategy"] == "tit_for_tat"]["payoff"].iloc[0] == pytest.approx(1.0)

    def test_get_payoffs_by_strategy_empty_nan(self):
        """Test get_payoffs_by_strategy with empty or NaN data."""
        empty_df = pd.DataFrame(columns=['round', 'strategy', 'payoff'])
        assert get_payoffs_by_strategy(empty_df).empty

        nan_df = pd.DataFrame({'round': [0], 'strategy': ['A'], 'payoff': [np.nan]})
        # If all inputs are NaN, the result of mean() is NaN, which reset_index keeps
        result_nan = get_payoffs_by_strategy(nan_df)
        assert not result_nan.empty
        assert pd.isna(result_nan['payoff'].iloc[0])

        missing_col_df = pd.DataFrame({'round': [0], 'strategy': ['A']}) # Missing 'payoff'
        assert get_payoffs_by_strategy(missing_col_df).empty # Should return empty with warning

    def test_get_strategy_scores_valid(self, sample_agents_df):
         # ... (Keep existing test - seems okay) ...
         scores = get_strategy_scores(sample_agents_df)
         assert not scores.empty
         tft_row = scores[scores["strategy"] == "tit_for_tat"]
         assert tft_row["mean"].values[0] == pytest.approx(43.5)

    def test_get_strategy_scores_empty_nan(self):
        """Test get_strategy_scores with empty or NaN data."""
        empty_df = pd.DataFrame(columns=['strategy', 'final_score'])
        assert get_strategy_scores(empty_df).empty

        nan_df = pd.DataFrame({'strategy': ['A'], 'final_score': [np.nan]})
        result_nan = get_strategy_scores(nan_df)
        assert not result_nan.empty
        assert result_nan['mean'].iloc[0] == 0 # fillna(0) applied

        missing_col_df = pd.DataFrame({'strategy': ['A']}) # Missing 'final_score'
        assert get_strategy_scores(missing_col_df).empty

    def test_get_strategy_colors(self):
        # ... (Keep existing test - seems okay) ...
        colors = get_strategy_colors()
        assert "always_cooperate" in colors
        assert colors['always_cooperate'].startswith("#")

    def test_prepare_network_data_valid(self, sample_agents_df, sample_rounds_df):
        # ... (Keep existing test - seems okay) ...
        round_data = sample_rounds_df[sample_rounds_df["round"] == 1]
        network_data = prepare_network_data(sample_agents_df, round_data, round_num=1)
        assert network_data['round'] == 1
        assert len(network_data['nodes']) == 4
        node0 = next(n for n in network_data['nodes'] if n['id']==0)
        assert node0['strategy'] == 'tit_for_tat'
        assert node0['score'] == 45.0
        assert node0['move'] == 'cooperate'
        assert node0['payoff'] == 1.5

    def test_prepare_network_data_empty_missing(self):
        """Test prepare_network_data with missing/empty data."""
        empty_agents = pd.DataFrame(columns=['agent_id', 'strategy', 'final_score'])
        empty_rounds = pd.DataFrame(columns=['round', 'agent_id', 'move', 'payoff'])

        # Empty inputs
        res = prepare_network_data(empty_agents, empty_rounds, round_num=0)
        assert res['nodes'] == [] and res['edges'] == [] and res['round'] == 0

        # Round doesn't exist
        res = prepare_network_data(sample_agents_df(), sample_rounds_df(), round_num=99)
        assert res['nodes'] == [] and res['edges'] == [] and res['round'] == 99

        # Missing columns
        res = prepare_network_data(sample_agents_df().drop('strategy', axis=1), sample_rounds_df(), round_num=0)
        assert res['nodes'] == [] and res['edges'] == [] and res['round'] == 0


@pytest.mark.visualization
@pytest.mark.unit
class TestDataLoading:
    """Test data loading functions."""

    def test_get_available_scenarios(self, mock_results_dir: Path):
        """Test finding scenario directories."""
        scenarios = get_available_scenarios(str(mock_results_dir))
        assert scenarios == ["AnotherScenario", "MockScenario"] # Sorted alphabetically
        # Test non-existent dir
        assert get_available_scenarios("non_existent_dir") == []

    def test_load_scenario_results_single_run(self, mock_results_dir: Path):
        """Test loading results for a specific run."""
        agents_df, rounds_df = load_scenario_results("MockScenario", str(mock_results_dir), run_number=0)
        assert not agents_df.empty
        assert not rounds_df.empty
        assert agents_df['scenario_name'].iloc[0] == "test" # From sample data
        assert rounds_df['run_number'].unique() == [0]

    def test_load_scenario_results_aggregate(self, mock_results_dir: Path):
        """Test loading and aggregating results from all runs (only 1 run in mock)."""
        # In the mock setup, run_number=None should just load run 0
        agents_df, rounds_df = load_scenario_results("MockScenario", str(mock_results_dir), run_number=None)
        assert not agents_df.empty
        assert not rounds_df.empty
        assert agents_df['run_number'].unique() == [0]

    def test_load_scenario_results_errors(self, mock_results_dir: Path):
        """Test error handling for loading results."""
        # Scenario not found
        with pytest.raises(FileNotFoundError):
            load_scenario_results("NotFoundScenario", str(mock_results_dir))
        # Run not found
        with pytest.raises(FileNotFoundError):
            load_scenario_results("MockScenario", str(mock_results_dir), run_number=99)
        # No runs found (create empty scenario dir)
        empty_scenario_dir = mock_results_dir / "EmptyScenario"
        empty_scenario_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No run directories found"):
             load_scenario_results("EmptyScenario", str(mock_results_dir))

    def test_get_cooperation_rates_valid(self, sample_rounds_df):
         # ... (Keep existing test - seems okay) ...
         coop_rates = get_cooperation_rates(sample_rounds_df)
         assert coop_rates[coop_rates["round"] == 2]["cooperation_rate"].iloc[0] == pytest.approx(0.75)

    def test_get_strategy_cooperation_rates_valid(self, sample_rounds_df):
         # ... (Keep existing test - seems okay) ...
         coop_rates = get_strategy_cooperation_rates(sample_rounds_df)
         r0_tft = coop_rates[(coop_rates["round"] == 0) & (coop_rates["strategy"] == "tit_for_tat")]
         assert r0_tft["cooperation_rate"].iloc[0] == pytest.approx(1.0)

    def test_load_network_structure_valid(self, mock_results_dir: Path):
        """Test loading a valid network file."""
        G, net_data = load_network_structure("MockScenario", str(mock_results_dir), run_number=0)
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 4
        assert G.number_of_edges() == 4
        assert isinstance(net_data, dict)
        assert net_data['network_type'] == 'sample'
        assert 'metrics' in net_data

    def test_load_network_structure_errors(self, mock_results_dir: Path):
        """Test loading network structure with missing files or invalid JSON."""
        # File not found
        G, net_data = load_network_structure("MockScenario", str(mock_results_dir), run_number=99)
        assert G.number_of_nodes() == 0 and not net_data

        # Invalid JSON
        run_dir = mock_results_dir / "MockScenario" / "run_00"
        invalid_net_file = run_dir / "experiment_results_network.json"
        invalid_net_file.write_text("invalid json")
        G, net_data = load_network_structure("MockScenario", str(mock_results_dir), run_number=0)
        assert G.number_of_nodes() == 0 and not net_data

    def test_get_available_runs(self, mock_results_dir: Path):
        """Test finding available run numbers."""
        # Add another run dir
        (mock_results_dir / "MockScenario" / "run_01").mkdir()
        runs = get_available_runs("MockScenario", str(mock_results_dir))
        assert runs == [0, 1]

        # Test non-existent scenario
        assert get_available_runs("NotFound", str(mock_results_dir)) == []


@pytest.mark.visualization
@pytest.mark.unit
class TestNetworkVisualization:
    """Test network visualization functions."""

    def test_generate_network_positions(self, sample_network):
        # ... (Keep existing test - seems okay) ...
        positions = generate_network_positions(sample_network, layout_type="spring")
        assert len(positions) == sample_network.number_of_nodes()

    def test_create_network_figure_structure(self, sample_network):
        """Test creating a network visualization figure and inspect its structure."""
        node_data = { # Simplified node data for structure check
            0: {'strategy': 'A', 'score': 10, 'move': 'c', 'payoff': 1},
            1: {'strategy': 'A', 'score': 12, 'move': 'c', 'payoff': 1},
            2: {'strategy': 'B', 'score': 20, 'move': 'd', 'payoff': 3},
            3: {'strategy': 'A', 'score': 15, 'move': 'c', 'payoff': 2}
        }
        strategy_colors = {'A': '#ff0000', 'B': '#0000ff'}

        fig = create_network_figure(
            sample_network,
            node_data=node_data,
            color_by='strategy',
            size_by='score', # Use score for size
            layout_type='kamada_kawai',
            strategy_colors=strategy_colors
        )

        assert fig is not None
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2 # Edge trace, Node trace

        # Inspect Edge trace
        edge_trace = fig.data[0]
        assert edge_trace.type == 'scatter'
        assert edge_trace.mode == 'lines'
        # Calculate expected number of points in edge trace (2 per edge + 1 None) * num_edges
        num_edges = sample_network.number_of_edges()
        assert len(edge_trace.x) == num_edges * 3

        # Inspect Node trace
        node_trace = fig.data[1]
        assert node_trace.type == 'scatter'
        assert node_trace.mode == 'markers'
        num_nodes = sample_network.number_of_nodes()
        assert len(node_trace.x) == num_nodes
        assert len(node_trace.y) == num_nodes
        assert len(node_trace.marker.color) == num_nodes
        assert len(node_trace.marker.size) == num_nodes
        assert len(node_trace.text) == num_nodes

        # Check colors match strategy (approximate check)
        node_ids_in_order = [fig.layout.xaxis.tickvals[i] if fig.layout.xaxis.tickvals else i for i in range(num_nodes)] # Need a way to map trace index to node ID if possible, complex.
        # Easier check: count expected colors
        expected_a_count = 3
        expected_b_count = 1
        actual_a_count = sum(1 for c in node_trace.marker.color if c == strategy_colors['A'])
        actual_b_count = sum(1 for c in node_trace.marker.color if c == strategy_colors['B'])
        assert actual_a_count == expected_a_count
        assert actual_b_count == expected_b_count

        # Check sizes correspond to score (min size check)
        assert all(s >= 5 for s in node_trace.marker.size) # Check base size application

    def test_create_network_figure_empty(self):
        """Test creating figure with an empty graph."""
        empty_graph = nx.Graph()
        fig = create_network_figure(empty_graph)
        assert fig is not None
        assert len(fig.data) == 0 # Should have no traces
        assert len(fig.layout.annotations) > 0 # Should have annotation message